"""Score Brisbane-Event-VPR at more than one input resolution — the blocker on adopting 322².

Run from the repo root::

    pixi run python3 scripts/brisbane_resolution.py \
        --eventlab-dir /media/adam/vprdatasets/eventgem \
        --out-dir /media/adam/vprdatasets/megaevent/brisbane_res

`scripts/tokyo_trajectory.py` showed Tokyo 24/7 gains +10.5 R@1 from evaluating at 322² instead of
224², and another +2.9 from using ViT-B — but every Brisbane number on record is at 224², so the
paper's two tables would sit at different input resolutions. This closes that: the same checkpoints,
the same Brisbane protocol (ref **sunset2** → {sunset1, morning, daytime, sunrise}, stride 1, the
shipped hot-pixel filter), at both resolutions.

Structure mirrors `tokyo_trajectory.py`: HDF5 slicing plus countmask rendering costs more than the
forward pass, so all checkpoints are resident on the GPU and each rendered batch is fanned out to
every one of them — N checkpoints for roughly the price of one pass. The **reference** traverse is
extracted once per resolution and reused by all four query conditions, which is what
`gept/src/vpr/evalsuite.py::run_suite` does for the same reason.

Ground truth, whitening, and recall all come from `src/inference.py`, so the numbers are comparable
to this repo's published Brisbane figure (sunset2→sunset1 R@1 .929 native / .938 whitened). Two
known differences from `gept`'s in-training `eval_history`, neither of which affects a
same-harness resolution comparison: this path applies eventcv's hot-pixel filter (gept's
`evalsuite` does not), and `inference.recall_at_k` discards queries with no ground-truth positive
where `evalsuite.recall_at_k_cross` counts them as misses.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src import inference as inf  # noqa: E402
from src import scoring  # noqa: E402
from src import traversenpz as tnpz  # noqa: E402
from tokyo_trajectory import load_all  # noqa: E402  (identical fan-out / dedup logic)

DEFAULT_EVENTLAB = "/media/adam/vprdatasets/eventgem"
DEFAULT_OUT = "/media/adam/vprdatasets/megaevent/brisbane_res"
# The countmask sweep's protocol: no `night` (its R@1 floors at ~.02 for every arm, so it moves
# with nothing), stride 1, sunset2 as the reference.
REF = "sunset2"
QUERIES = ("sunset1", "morning", "daytime", "sunrise")
PCA_SETTINGS = ((2048, 0.5), (4096, 0.5))
KS = (1, 5, 10, 20)
# Which checkpoints the Tokyo work made decision-relevant: the one `best.pt` selected, ViT-S's best
# Tokyo checkpoint, and the overall best (ViT-B). label -> path, resolved against --ckpt-root.
DEFAULT_CKPTS = (
    ("s_step2000", "s_salad_ft4/step2000.pt"),     # what Brisbane selection actually picked
    ("s_step8000", "s_salad_ft4/step8000.pt"),     # ViT-S best on Tokyo @322 (.7810)
    ("b_step16000", "b_salad_ft4/step16000.pt"),   # best on Tokyo overall (.8095)
)


class _Args:
    """The handful of attributes `src.inference`'s helpers read off an argparse namespace."""

    def __init__(self, eventlab_dir, dataset, dt_ms, no_hot_pixel, no_event_filter,
                 source="real", npz_root=None):
        self.eventlab_dir = eventlab_dir
        self.dataset = dataset
        # Which arm the frames come from, and where the dumped arms live. Default `real`
        # keeps every existing caller on the eventcv reader with no behaviour change.
        self.source = source
        self.npz_root = npz_root
        self.dt_ms = dt_ms
        self.no_hot_pixel = no_hot_pixel
        self.no_event_filter = no_event_filter
        # eventcv's background-activity window is in the stream's raw timestamp unit —
        # microseconds — so a dt_ms of 50 means 50_000 here. Held as one attribute rather
        # than re-derived at the call site, which is how it came to be passed in the wrong
        # unit in the first place. None disables the filter.
        self.filter_dt_us = None if no_event_filter else dt_ms * 1000
        self.ref = REF
        self.query = None


def _slice_source(traverse, transform, args):
    """The traverse's frames, from the HDF5 recording or from a dumped ``.npz`` arm.

    ``args.source`` defaults to ``real``, which is the eventcv reader this script has always
    used and the only source the four traverses without an npz tree have. A Sim2Real arm has
    no recording — its events were simulated by I2E from the APS frames — so it comes off
    disk instead. Everything downstream (rendering, model, banks, scoring) is unchanged, so
    the arms differ in exactly one thing.
    """
    source = getattr(args, "source", "real")
    if source == "real":
        return inf.EventStreamDataset(
            args.dataset, traverse, inf.sequence_path(args, traverse), transform,
            args.representation, args.dt_ms, inf.sensor_size(args.dataset),
            hot_pixel=not args.no_hot_pixel,
            filter_dt_us=args.filter_dt_us,
        )
    paths = tnpz.frame_paths(args.npz_root, args.dataset, source, traverse)
    # frame_loader raises on a representation src.npzdata cannot render, which is the
    # guard that used to live here when countmask was the only one it supplied.
    return tnpz.NpzTraverseDataset(paths, tnpz.frame_loader(transform, args.representation))


def _forward(model, frames, autocast, oom_backoff, tag=""):
    """``model(frames)`` -> ``[B, D]`` float32 numpy, halving the batch on a CUDA OOM.

    Splitting the batch is only sound for models whose descriptor depends on one frame at a
    time — then it is a pure memory knob and the rows come back bit-identical. CricaVPR's
    cross-image encoder attends over the *batch axis*, so there a smaller batch is a
    different measurement rather than a smaller footprint: those callers pass
    ``oom_backoff=False`` and the OOM propagates instead of quietly changing the numbers.
    """
    try:
        with autocast:
            return model(frames).float().cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        if not oom_backoff or frames.shape[0] == 1:
            raise
        torch.cuda.empty_cache()
        half = frames.shape[0] // 2
        print(f"    {tag}: CUDA OOM at batch {frames.shape[0]} — retrying as "
              f"{half}+{frames.shape[0] - half}", flush=True)
        first = _forward(model, frames[:half], autocast, oom_backoff, tag)
        second = _forward(model, frames[half:], autocast, oom_backoff, tag)
        return np.concatenate([first, second], axis=0)


def extract(loaded, transform, traverse, args, out_dir, tag, device, batch_size, workers,
            oom_backoff=False):
    """One pass over a traverse's slices, fanned out to every checkpoint. -> {label: path}

    ``oom_backoff`` halves the batch and retries when the card is out of memory, for models
    where batch size is a memory knob only — see :func:`_forward`. Off by default so every
    existing caller keeps its current all-or-nothing behaviour.
    """
    dataset = _slice_source(traverse, transform, args)
    n = len(dataset)
    files = {name: os.path.join(out_dir, f"{tag}_{traverse}_{name}.npy") for name, _, _ in loaded}
    if all(os.path.exists(p) for p in files.values()):
        print(f"    {tag} {traverse}: {n} slices — all banks cached, skipping")
        return files, n

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, drop_last=False)
    autocast = torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda"))
    banks = {}
    written = 0
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            frames = batch.to(device, non_blocking=True)
            size = frames.shape[0]
            for name, _, model in loaded:
                desc = _forward(model, frames, autocast, oom_backoff, f"{tag} {traverse}")
                if name not in banks:
                    banks[name] = np.lib.format.open_memmap(
                        files[name], mode="w+", dtype=np.float32, shape=(n, desc.shape[1]))
                banks[name][written:written + size] = desc
            written += size
            if written % (batch_size * 40) == 0 or written == n:
                rate = written / (time.time() - start)
                print(f"    {tag} {traverse}: {written}/{n}  {rate:.1f} img/s  "
                      f"eta {(n - written) / rate / 60:.1f} min", flush=True)
    for bank in banks.values():
        bank.flush()
    return files, n


def score(ref_full, q_full, gt_path, device, stride=1):
    """{'native': {...}, 'pca<dim>p<power>': {...}, 'scorable': int} for one (ref, query) pair.

    Takes the full-rate banks **already in memory** rather than paths: each is 433 MB at
    8448-d, and re-reading them per query and per stride made this disk-bound (62 GB of reads
    for a 3-stride sweep, ~110 min). Hoisting the loads to one per bank cuts that ~5x.

    ``stride`` subsamples **both** banks before scoring, which is exactly what
    ``gept/src/vpr/evalsuite.py::run_suite`` does with ``--eval-stride``: the ground truth is
    resampled onto the new (n_ref, n_query) grid, so a strided score is the score that
    protocol would have reported. Since the banks are already on disk this is pure
    re-scoring — no extraction, no model.

    Why it matters: Brisbane R@1 is monotone in gallery density (.359 @5 s -> .833 at full
    rate for *one unchanged model*), because at stride 1 the nearest reference frame is 50 ms
    / ~0.5 m away and the task is largely near-duplicate retrieval. A model that becomes more
    viewpoint- and illumination-invariant should get *worse* at that, with no domain drift
    involved at all — so the whole "Brisbane decays while Tokyo climbs" trade-off may be an
    artefact of the stride rather than sim-to-real drift.

    **Read the native column, not the whitened one, across strides.** Whitening is fit on the
    reference bank, so striding shrinks the fit set (12,825 -> ~640 rows at stride 20) and the
    transform degrades for reasons that have nothing to do with gallery density. The effective
    rank is clamped to the fit set and recorded in ``pca_dim_eff`` so that confound is visible
    rather than silent.
    """
    ref, query = ref_full[::stride], q_full[::stride]
    gt, gt_shape = inf.load_gt(gt_path, ref.size(0), query.size(0))
    out = {"n_ref": ref.size(0), "n_query": query.size(0), "gt_shape": list(gt_shape),
           "stride": stride,
           "gt_density": float(gt.mean()), "scorable": int((gt.sum(0) > 0).sum()),
           "native": inf.recall_at_k(inf.sim_matrix(ref, query, device), gt, KS)}
    torch.cuda.empty_cache()
    fit = ref
    if ref.size(0) > scoring.PCA_FIT_SAMPLES:
        generator = torch.Generator().manual_seed(scoring.PCA_FIT_SEED)
        fit = ref[torch.randperm(ref.size(0), generator=generator)[:scoring.PCA_FIT_SAMPLES]]
    for dim, power in PCA_SETTINGS:
        # svd_lowrank(q=dim) is undefined for dim > min(n_fit, D); a strided bank can easily
        # have fewer rows than the requested dim.
        dim_eff = min(dim, fit.size(0) - 1)
        pca = inf.pca_fit(fit, device, dim=dim_eff, power=power)
        out[f"pca{dim}p{power}"] = inf.recall_at_k(
            inf.sim_matrix(inf.pca_apply(ref, pca, device),
                           inf.pca_apply(query, pca, device), device), gt, KS)
        out[f"pca{dim}p{power}_dim_eff"] = dim_eff
        del pca
        torch.cuda.empty_cache()
    return out


def _write(args_cli, all_results):
    """Dump results so far; returns the path. Called after every label, not just at the end."""
    out_json = os.path.join(args_cli.out_dir, args_cli.out_json or "results.json")
    tmp = out_json + ".tmp"
    with open(tmp, "w") as handle:
        json.dump({"dataset": args_cli.dataset, "ref": REF, "queries": args_cli.queries,
                   "dt_ms": args_cli.dt_ms, "hot_pixel": not args_cli.no_hot_pixel,
                   "strides": args_cli.strides, "results": all_results}, handle, indent=2)
    os.replace(tmp, out_json)      # atomic: a reader never sees a half-written file
    return out_json


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eventlab-dir", default=DEFAULT_EVENTLAB)
    ap.add_argument("--ckpt-root", default="/media/adam/vprdatasets/megaevent")
    ap.add_argument("--ckpt", nargs="*", metavar="LABEL=PATH",
                    help="override the default checkpoint set")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--resolutions", type=int, nargs="+", default=[224, 322])
    ap.add_argument("--strides", type=int, nargs="+", default=[1],
                    help="gallery strides to score the SAME cached banks at (1 = the paper "
                         "protocol). Extraction is unaffected — the banks are extracted once "
                         "at stride 1 and every stride is a re-score of those rows, so adding "
                         "strides costs minutes and no GPU passes.")
    ap.add_argument("--out-json", default=None,
                    help="results filename inside --out-dir (default results.json). Use a "
                         "different name to avoid overwriting an earlier run's artefact.")
    ap.add_argument("--dataset", default="brisbane_event")
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--queries", nargs="+", default=list(QUERIES))
    ap.add_argument("--no-hot-pixel", action="store_true",
                    help="drop eventcv's hot-pixel filter (gept's in-training eval had none)")
    ap.add_argument("--no-event-filter", action="store_true", default=True)
    ap.add_argument("--batch-size", type=int, default=inf.BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=inf.NUM_WORKERS)
    args_cli = ap.parse_args()

    pairs = ([tuple(spec.split("=", 1)) for spec in args_cli.ckpt] if args_cli.ckpt
             else [(label, os.path.join(args_cli.ckpt_root, rel)) for label, rel in DEFAULT_CKPTS])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args_cli.out_dir, exist_ok=True)

    args = _Args(args_cli.eventlab_dir, args_cli.dataset, args_cli.dt_ms,
                 args_cli.no_hot_pixel, args_cli.no_event_filter)
    # The representation is a property of the checkpoints, and `load_all` already refuses a set
    # whose input geometry or normalisation differs — read it once, without building a model.
    head = torch.load(pairs[0][1], map_location="cpu", weights_only=False)
    args.representation = inf.cfg_from_ckpt(head).representation
    del head
    print(f"{args_cli.dataset}: ref {REF} -> {args_cli.queries}, dt {args_cli.dt_ms} ms, "
          f"stride 1, representation {args.representation}, "
          f"hot-pixel filter {not args_cli.no_hot_pixel}")

    all_results = {}
    for resolution in args_cli.resolutions:
        tag = f"r{resolution}"
        print(f"\n{'=' * 78}\n{tag}\n{'=' * 78}")
        loaded, transform = load_all(pairs, device, resolution)
        print(f"{len(loaded)} models resident, "
              f"{torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB of weights\n")

        banks = {}
        for traverse in [REF, *args_cli.queries]:
            files, n = extract(loaded, transform, traverse, args, args_cli.out_dir, tag,
                               device, args_cli.batch_size, args_cli.workers)
            banks[traverse] = files
            print(f"    {traverse}: {n} slices done", flush=True)

        for _, _, model in loaded:
            del model
        loaded.clear()
        torch.cuda.empty_cache()

        # Bank loads hoisted out of the stride/query loops: the ref bank serves all four
        # queries and every stride, so it is read once per (label, resolution) rather than
        # 3 strides x 4 queries times.
        print(f"\n  scoring {tag} at strides {args_cli.strides}")
        for label, _ in pairs:
            if label not in banks[REF]:
                continue                                # deduplicated away by load_all
            ref_full = torch.from_numpy(np.load(banks[REF][label], mmap_mode="r").copy())
            for query in args_cli.queries:
                q_full = torch.from_numpy(np.load(banks[query][label], mmap_mode="r").copy())
                args.query = query
                gt_path = inf.ground_truth_path(args)
                for stride in args_cli.strides:
                    # tag stays `r<res>` at stride 1 so a re-run is directly comparable, cell
                    # for cell, with the results.json an earlier run wrote — that equality is
                    # the harness check that makes every other stride trustworthy.
                    stag = tag if stride == 1 else f"{tag}s{stride}"
                    res = score(ref_full, q_full, gt_path, device, stride=stride)
                    all_results.setdefault(label, {}).setdefault(stag, {})[query] = res
                    print(f"    {stag:10s} {label:12s} {query:8s} "
                          f"{res['n_query']}q/{res['n_ref']}r  "
                          f"native R@1 {res['native'][1]:.4f}   "
                          + "   ".join(f"pca{d}p{p} {res[f'pca{d}p{p}'][1]:.4f}"
                                       for d, p in PCA_SETTINGS), flush=True)
                del q_full
            del ref_full
            # Persist after every label: a 3-stride sweep is long enough that losing it all
            # to a timeout is a real cost, and the file is small.
            _write(args_cli, all_results)

    out_json = _write(args_cli, all_results)

    print(f"\n{'=' * 78}\nR@1 by resolution (native | best-whitened), and the 4-condition mean")
    for label in all_results:
        print(f"\n  {label}")
        for tag, per_query in all_results[label].items():
            nat = [per_query[q]["native"][1] for q in args_cli.queries]
            wht = [max(per_query[q][f"pca{d}p{p}"][1] for d, p in PCA_SETTINGS)
                   for q in args_cli.queries]
            print(f"    {tag}  " + "  ".join(
                f"{q} {n:.3f}/{w:.3f}" for q, n, w in zip(args_cli.queries, nat, wht))
                + f"   | mean {np.mean(nat):.3f}/{np.mean(wht):.3f}")
    print(f"\n-> {out_json}")


if __name__ == "__main__":
    main()
