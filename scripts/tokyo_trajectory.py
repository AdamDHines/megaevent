"""Score Tokyo 24/7 at every saved training step of one run — the selection-protocol test.

Run from the repo root::

    pixi run python3 scripts/tokyo_trajectory.py \
        --ckpt-dir /media/adam/vprdatasets/megaevent/s_salad_ft4 \
        --out-dir /media/adam/vprdatasets/megaevent/tokyo_traj

Why this exists (see ``../gept/docs/report_2026-07-30_training_regime_audit.md`` §3): training
selected ``best.pt`` on Brisbane ``sunset1`` R@1, which saturates at ~.91 by step 1000 and
then moves 1.5 points over the next 11k steps, and which holds illumination and viewpoint
fixed — the two things Tokyo 24/7 varies. Tokyo is also the *I2E domain* the model trains on,
where Brisbane is real events, so the sim-to-real drift that Brisbane-based selection
suppresses is the domain fit Tokyo rewards. The prediction is therefore that **Tokyo recall
keeps climbing over the steps where Brisbane recall decays.** Nothing had ever measured it,
because only one checkpoint was ever scored.

The pipeline is data-loading bound — rendering countmask from ``.npz`` costs more than a
ViT-S/14 forward, so seven models resident on the GPU and evaluated in a single pass over the
frames costs ~27 % more wall time than one model, not 7x (measured: 66.5 img/s for one model,
48.7 img/s = 341 desc/s for seven). Hence phase 1 streams the frames once and fans each batch
out to every checkpoint, writing one memmapped bank per checkpoint; phase 2 scores them.

``--resolution`` also makes this the test for the FixRes question (report F1): training used
``RandomResizedCrop(224, scale=(0.5,1.0))`` while evaluation resizes the whole frame, so a
higher test resolution should help. DINOv2 interpolates its position embedding, so any
multiple of 14 works. Run the script once per resolution and compare.

Ground truth, recall, whitening and the transform are all this repo's shipped code paths, so
the numbers are comparable to ``results_megaevent.json`` cell for cell.
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import inference as inf  # noqa: E402
from src import scoring  # noqa: E402
from src.imagevpr import build_gt  # noqa: E402
from src.methods import NpzFrameDataset  # noqa: E402
from src.npzdata import list_npz, load_countmask, utm_from_paths  # noqa: E402

DEFAULT_CKPTS = "/media/adam/vprdatasets/megaevent/s_salad_ft4"
DEFAULT_NPZ = "/media/adam/vprdatasets/megaevent/tokyo247/numpy"
DEFAULT_OUT = "/media/adam/vprdatasets/megaevent/tokyo_traj"
# (dim, power) settings to report. The first is what src/inference.py ships, so it is the
# number directly comparable to results_megaevent.json; the second dominates it on every
# cutoff (see docs/tokyo247_headroom.md).
PCA_SETTINGS = ((2048, 0.5), (4096, 0.5))
KS = (1, 5, 10, 20)


def order_key(name):
    """Sort ``step2000`` < ``step12000`` < ``latest``, numerically not lexically."""
    match = re.fullmatch(r"step(\d+)", name)
    return (0, int(match.group(1))) if match else (1, 0)


def discover(ckpt_dir, only=None):
    """[(name, path)] for the checkpoints to score, in training order."""
    names = sorted((f[:-3] for f in os.listdir(ckpt_dir) if f.endswith(".pt")), key=order_key)
    if only:
        missing = set(only) - set(names)
        if missing:
            raise SystemExit(f"no such checkpoint(s) in {ckpt_dir}: {sorted(missing)}")
        names = [n for n in names if n in only]
    return [(n, os.path.join(ckpt_dir, f"{n}.pt")) for n in names]


def identical(a, b):
    """True if two state dicts hold the same tensors — lets duplicates be skipped.

    ``a`` is a cached CPU copy and ``b`` comes off the GPU, so compare on one device.
    """
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k].cpu()) for k in a)


def load_all(ckpts, device, resolution):
    """([(name, step, model)], shared transform), dropping weight-identical duplicates.

    Every checkpoint of one run shares the normalisation constants and the input size, so
    they share one transform — which is what lets a single pass over the frames feed all of
    them. A mismatch would mean the banks were not comparable, so it is an error, not a
    silent per-model re-render.
    """
    loaded, seen, spec = [], [], None
    for name, path in ckpts:
        model, cfg, step = inf.load_model(path, device)
        if resolution:
            cfg.H = cfg.W = resolution
        state = model.state_dict()
        dup = next((n for n, s in seen if identical(s, state)), None)
        if dup is not None:
            print(f"  {name:10s} step {step:>6}  identical weights to {dup} — skipping")
            del model
            continue
        this = (cfg.H, cfg.W, tuple(cfg.tencode_mean), tuple(cfg.tencode_std))
        if spec is None:
            spec, transform = this, inf.eval_transform(cfg)
        elif this != spec:
            raise SystemExit(
                f"{name} wants input/normalisation {this} but earlier checkpoints want "
                f"{spec} — these banks would not be comparable. Score it separately.")
        seen.append((name, {k: v.cpu() for k, v in state.items()}))
        loaded.append((name, step, model))
        print(f"  {name:10s} step {step:>6}  in={cfg.H}x{cfg.W}  desc={cfg.desc_dim}")
    return loaded, transform


def extract(loaded, transform, paths, out_dir, tag, device, batch_size, workers):
    """One pass over ``paths``, fanning each batch out to every model. -> {name: [N, D] path}

    ``render`` is byte-for-byte what ``MegaEventMethod.descriptors`` does — render countmask,
    scale to [0,1], then ``Normalize`` + ``Resize`` **in the worker** — so these descriptors
    are comparable to ``results_megaevent.json`` cell for cell. Doing the resize on the GPU
    instead would be faster to transfer but would put bicubic interpolation on a different
    implementation. Banks are memmapped because seven full Tokyo banks are ~18 GB.
    """
    def render(path):
        frame = load_countmask(path)
        return transform(torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0))

    dataset = NpzFrameDataset(paths, render)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, drop_last=False)
    autocast = torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda"))

    files = {name: os.path.join(out_dir, f"{tag}_{name}.npy") for name, _, _ in loaded}
    banks = {}
    written = 0
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            frames = batch.to(device, non_blocking=True)
            n = frames.shape[0]
            for name, _, model in loaded:
                with autocast:
                    desc = model(frames).float().cpu().numpy()
                if name not in banks:
                    banks[name] = np.lib.format.open_memmap(
                        files[name], mode="w+", dtype=np.float32,
                        shape=(len(paths), desc.shape[1]))
                banks[name][written:written + n] = desc
            written += n
            if written % (batch_size * 40) == 0 or written == len(paths):
                rate = written / (time.time() - start)
                print(f"    {tag}: {written}/{len(paths)}  {rate:.1f} img/s  "
                      f"({rate * len(loaded):.0f} desc/s)  "
                      f"eta {(len(paths) - written) / rate / 60:.1f} min", flush=True)
    for bank in banks.values():
        bank.flush()
    return files


def score_bank(db_path, q_path, gt, device):
    """{'native': {...}, '(dim, power)': {...}} for one checkpoint's banks."""
    db = torch.from_numpy(np.load(db_path, mmap_mode="r").copy())
    queries = torch.from_numpy(np.load(q_path, mmap_mode="r").copy())
    out = {"native": inf.recall_at_k(inf.sim_matrix(db, queries, device), gt, KS)}
    torch.cuda.empty_cache()
    fit = db
    if db.size(0) > scoring.PCA_FIT_SAMPLES:
        generator = torch.Generator().manual_seed(scoring.PCA_FIT_SEED)
        fit = db[torch.randperm(db.size(0), generator=generator)[:scoring.PCA_FIT_SAMPLES]]
    for dim, power in PCA_SETTINGS:
        pca = inf.pca_fit(fit, device, dim=dim, power=power)
        out[f"pca{dim}p{power}"] = inf.recall_at_k(
            inf.sim_matrix(inf.pca_apply(db, pca, device),
                           inf.pca_apply(queries, pca, device), device), gt, KS)
        del pca
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPTS)
    ap.add_argument("--npz-root", default=DEFAULT_NPZ)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--resolution", type=int, default=0,
                    help="override the checkpoint's H=W (multiple of 14); 0 keeps 224")
    ap.add_argument("--only", nargs="*", help="checkpoint stems to score (default: all)")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only the N database images nearest a query (0 = full bank)")
    ap.add_argument("--threshold-m", type=float, default=25.0)
    ap.add_argument("--batch-size", type=int, default=inf.BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=inf.NUM_WORKERS)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"r{args.resolution or 224}" + (f"_limit{args.limit}" if args.limit else "")
    os.makedirs(args.out_dir, exist_ok=True)

    db_paths = list_npz(f"{args.npz_root}/database")
    q_paths = list_npz(f"{args.npz_root}/queries")
    if args.limit:
        from src.imagevpr import _limit_database
        db_paths = _limit_database(db_paths, utm_from_paths(q_paths), args.limit)
    gt = build_gt(utm_from_paths(db_paths), utm_from_paths(q_paths), args.threshold_m)
    print(f"{len(db_paths)} database x {len(q_paths)} queries, GT @ {args.threshold_m:g} m: "
          f"{gt.sum(0).mean():.2f} positives/query, "
          f"{int((gt.sum(0) > 0).sum())}/{len(q_paths)} scorable\n")

    print(f"checkpoints in {args.ckpt_dir}:")
    loaded, transform = load_all(discover(args.ckpt_dir, args.only), device, args.resolution)
    print(f"\n{len(loaded)} models resident, "
          f"{torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB of weights\n")

    print("phase 1 — one pass over the frames, fanned out to every checkpoint")
    q_files = extract(loaded, transform, q_paths, args.out_dir, f"{tag}_queries", device,
                      args.batch_size, args.workers)
    db_files = extract(loaded, transform, db_paths, args.out_dir, f"{tag}_database", device,
                       args.batch_size, args.workers)

    steps = {name: step for name, step, _ in loaded}
    for _, _, model in loaded:
        del model
    loaded.clear()
    torch.cuda.empty_cache()

    print("\nphase 2 — scoring")
    results = {}
    for name in db_files:
        results[name] = {"step": steps[name],
                         **score_bank(db_files[name], q_files[name], gt, device)}
        row = results[name]
        print(f"  {name:10s} step {steps[name]:>6}  " + "   ".join(
            f"{key} R@1 {val[1]:.4f}" for key, val in row.items() if key != "step"), flush=True)

    out_json = os.path.join(args.out_dir, f"results_{tag}.json")
    with open(out_json, "w") as handle:
        json.dump({"resolution": args.resolution or 224, "limit": args.limit,
                   "n_database": len(db_paths), "n_queries": len(q_paths),
                   "threshold_m": args.threshold_m, "results": results}, handle, indent=2)

    print(f"\n{'checkpoint':<11}{'step':>7}" + "".join(
        f"{k:>28}" for k in ("native", *(f"pca{d}p{p}" for d, p in PCA_SETTINGS))))
    print(f"{'':<11}{'':>7}" + "".join(f"{'R@1    R@5   R@10   R@20':>28}"
                                       for _ in range(1 + len(PCA_SETTINGS))))
    for name, row in sorted(results.items(), key=lambda kv: order_key(kv[0])):
        line = f"{name:<11}{row['step']:>7}"
        for key in ("native", *(f"pca{d}p{p}" for d, p in PCA_SETTINGS)):
            line += "  " + " ".join(f"{row[key][k]:.4f}" for k in KS)
        print(line)
    print(f"\n-> {out_json}")


if __name__ == "__main__":
    main()
