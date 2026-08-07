"""Brisbane-Event-VPR the way the paper describes it: one query traverse, everything else pooled.

Run from the repo root::

    pixi run python3 scripts/brisbane_pooled.py \
        --ckpt P64_s10000=/media/adam/vprdatasets/megaevent/runs/b_full_P64_v4_vpr/step10000.pt

Every Brisbane number this repo has published is *pairwise* — `sunset2` against one query
condition at a time (`scripts/brisbane_resolution.py`) — and scored against Event-LAB's ±70 m
**along-track band**. The method section describes something else: a single query traverse, all
other traverses pooled into one reference database, and a **25 m Euclidean radius** following
`VPR-methods-evaluation`. That is what `src/imagevpr.py` already does for Tokyo 24/7, so this
script is mostly the plumbing that lets a *traverse* reach it: `src/traversegps.py` puts a metric
coordinate on every descriptor row, after which `imagevpr.build_gt` applies unchanged.

Three consequences of the switch, all of them intended:

* **The radius is stricter than the band it replaces**, not looser — 0.78% of the pooled database
  is positive per query, against 1.81% for the 70 m band. These numbers are not comparable to the
  pairwise ones and are not meant to be.
* **A Euclidean radius admits revisits.** Brisbane's route self-intersects (11.5% of `sunset1`
  frames have another `sunset1` frame within 25 m more than 30 s away), and the along-track model
  is structurally unable to call those the same place. For place recognition, they are.
* **Scoring cannot materialise the similarity matrix.** Pooled, it is 67,208 x 14,269 = 3.8 GB, and
  `recallAtK` takes a full `argsort(0)` over it — a 7.7 GB int64 intermediate. `topk_ranked` below
  streams the top-20 on the GPU instead and never allocates the matrix.

Both filter arms are extracted. The shipped checkpoint was *trained* on unfiltered countmask
frames, so running it on background-activity-filtered ones is a test-time domain shift; the
unfiltered arm is what says whether that shift costs anything, and it also keeps continuity with
every recorded Brisbane figure.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src import inference as inf  # noqa: E402
from src import scoring  # noqa: E402
from src import traversegps as tg  # noqa: E402
from src import traversenpz as tnpz  # noqa: E402
from src.imagevpr import build_gt, figure_error_map  # noqa: E402
from brisbane_resolution import _Args, extract  # noqa: E402
from tokyo_trajectory import load_all, parse_pca  # noqa: E402

DEFAULT_EVENTLAB = "/media/adam/vprdatasets/eventgem"
DEFAULT_NPZ_ROOT = "/media/adam/vprdatasets/megaevent/brisbane_npz"
DEFAULT_OUT = "/media/adam/vprdatasets/megaevent/brisbane_pooled"
# The unfiltered banks already extracted by scripts/brisbane_resolution.py. Reused for the
# `off` arm under their existing `r{res}` tag so only `night` has to be rendered there.
DEFAULT_UNFILTERED_BANKS = "/media/adam/vprdatasets/megaevent/brisbane_v4"
DEFAULT_CKPT = ("P64_s10000",
                "/media/adam/vprdatasets/megaevent/runs/b_full_P64_v4_vpr/step10000.pt")

QUERY = "sunset1"
DATABASE = ("sunset2", "daytime", "morning", "night", "sunrise")
PCA_SETTINGS = ((4096, 0.5), (2048, 0.5))       # v4 ships at 4096/0.5
KS = (1, 5, 10, 20)
STATIONARY_MS = 0.5                             # m/s below which a frame counts as stopped


# ---------------------------------------------------------------------------
# Scoring: streamed, never materialising [n_db, n_q]
# ---------------------------------------------------------------------------
def topk_ranked(db, q, device, k=max(KS), chunk=256, db_chunk=None, return_scores=False):
    """``[k, n_q]`` int32 database row indices, best first.

    The queries stream past a resident database, so the peak allocation is one
    ``[n_db, chunk]`` similarity block rather than the whole matrix. Both banks are
    L2-normalised (the model's head normalises, and so does :func:`pca_apply`), so a dot
    product is cosine similarity.

    ``db_chunk`` splits the *database* as well, merging each chunk's top-k into a running
    best. That is exactly equal to the single-pass result — top-k of the concatenated
    per-chunk top-k is the global top-k — but it caps resident VRAM at one chunk instead of
    the whole gallery. Brisbane's pooled bank is 54,620 x 8448 (1.8 GB) and fits whole;
    NSAVP's is 101k x 8448 (3.4 GB) and does not, on an 8 GB card. Default None keeps the
    gallery in one piece, so nothing changes for a bank that already fitted.
    """
    n_db, n_q = db.size(0), q.size(0)
    step = n_db if db_chunk is None else int(db_chunk)
    with torch.no_grad():
        # The running best is [k, n_q] — 20 x 14k is ~1 MB, so it stays on the GPU while
        # the gallery streams past it. Database chunks are the outer loop so each one is
        # transferred exactly once; queries re-stream per chunk, which is the far cheaper
        # of the two directions.
        best_v = torch.full((k, n_q), float("-inf"), device=device)
        best_i = torch.zeros((k, n_q), dtype=torch.long, device=device)
        for d in range(0, n_db, step):
            part = db[d:d + step].to(device, non_blocking=True)
            for s in range(0, n_q, chunk):
                block = q[s:s + chunk].to(device, non_blocking=True)
                sim = part @ block.T                        # [chunk_db, chunk_q]
                v, i = torch.topk(sim, min(k, sim.size(0)), dim=0)
                cols = slice(s, s + block.size(0))
                cv = torch.cat([best_v[:, cols], v], 0)
                cidx = torch.cat([best_i[:, cols], i + d], 0)   # d: back to global rows
                best_v[:, cols], sel = torch.topk(cv, k, dim=0)
                best_i[:, cols] = torch.gather(cidx, 0, sel)
                del sim, v, i, cv, cidx, block
            del part
            torch.cuda.empty_cache()
        out = best_i.cpu().numpy().astype(np.int32)
        scores = best_v.cpu().numpy().astype(np.float32) if return_scores else None
        del best_v, best_i
    torch.cuda.empty_cache()
    return (out, scores) if return_scores else out


def recall_from_ranked(ranked, gt, ks=KS):
    """``({k: recall}, {n: recall}, scorable)`` — R@k and the full curve from one ranking.

    Matches ``recallAtK``'s denominator: queries with no ground-truth positive anywhere in the
    database are *discarded*, not counted as misses. With a pooled database at 25 m every
    Brisbane query is scorable, so the two conventions coincide here — but the mask is applied
    explicitly rather than assumed.
    """
    scorable = gt.sum(0) > 0
    hit = gt[ranked, np.arange(gt.shape[1])[None, :]]       # [k, n_q] bool
    found = np.cumsum(hit, axis=0) > 0
    curve = {n: float(found[n - 1, scorable].mean()) for n in range(1, ranked.shape[0] + 1)}
    return {k: curve[k] for k in ks}, curve, scorable


def assert_filter_active(args, traverse, filter_dt_us, n_probe=12):
    """Guard the two bugs this work uncovered: filter silently off, or off by 1000x in units.

    eventcv's ``dt`` is in raw timestamp units (microseconds). Passing ``dt_ms`` asked for a
    50 us correlation window and kept ~6% of the active pixels; 50_000 us keeps ~84%. A bank
    extracted under either mistake looks perfectly healthy from the outside, so the render is
    checked here rather than trusted.
    """
    from torchvision import transforms

    identity = transforms.Compose([])
    sensor, path = inf.sensor_size(args.dataset), inf.sequence_path(args, traverse)
    frac = {}
    for label, dt_us in (("off", None), ("on", filter_dt_us)):
        ds = inf.EventStreamDataset(args.dataset, traverse, path, identity,
                                    args.representation, args.dt_ms, sensor,
                                    hot_pixel=not args.no_hot_pixel, filter_dt_us=dt_us)
        idx = np.linspace(500, len(ds) - 500, n_probe).astype(int)
        frac[label] = np.array([float((ds[i].numpy() != 0).mean()) for i in idx])
    retention = float((frac["on"] / frac["off"]).mean())
    if not 0.50 <= retention <= 0.95:
        raise SystemExit(
            f"background-activity filter at dt={filter_dt_us} us retains {retention:.1%} of "
            f"active pixels on {traverse}. Expected ~84%. Below ~10% means the window is being "
            f"read as milliseconds; ~100% means the filter is not running at all.")
    print(f"  filter check: dt={filter_dt_us} us retains {retention:.1%} of active pixels  OK")
    return retention


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def traverse_geometry(args, sequences, npz_root, aps_align=False):
    """{seq: (xy, covered, speed, info)} in one shared local ENU frame.

    ``aps_align`` additionally drops every frame with no APS frame within half a slice. That
    mask describes the *alignment* between the recorded and simulated arms, not the event
    source, so it has to be applied to both or neither — masking only the synthetic arm would
    score a real frame against a duplicated one and report the difference as a domain gap.
    """
    lat0, lon0 = tg.track_origin(args.eventlab_dir, args.dataset, sequences)
    print(f"  projection origin ({lat0:.6f}, {lon0:.6f})")
    geom = {}
    for seq in sequences:
        geom[seq] = tg.frame_coords(args.eventlab_dir, args.dataset, seq, lat0, lon0,
                                    args.dt_ms, npz_root=npz_root)
        info = geom[seq][3]
        if aps_align:
            xy, covered, speed, info = geom[seq]
            aps = tnpz.aps_validity_mask(npz_root, args.dataset, seq, len(covered))
            dropped = int((covered & ~aps).sum())
            info = {**info, "aps_unaligned": dropped,
                    "aps_unaligned_fraction": float(dropped / max(len(covered), 1))}
            geom[seq] = (xy, covered & aps, speed, info)
            print(f"    {seq:9s} APS alignment drops {dropped} of {int(covered.sum())} "
                  f"GPS-covered frames ({dropped / max(int(covered.sum()), 1):.1%})")
        print(f"    {seq:9s} {info['n_frames']:6d} frames  {info['n_gps_fixes']:4d} GPS fixes  "
              f"{info['uncovered']:4d} outside GPS span  route {info['route_len_m']:.0f} m"
              + ("  [slice_times verified]" if info["slice_times_checked"] else ""))
    lengths = [g[3]["route_len_m"] for g in geom.values()]
    spread = (max(lengths) - min(lengths)) / float(np.mean(lengths))
    if spread > 0.01:
        raise SystemExit(f"route lengths disagree by {spread:.1%} — the traverses do not share "
                         f"a route, or the projection origin is wrong")
    return geom


def pool_database(bank_files, geom, sequences, label):
    """``(desc [n_db, D] torch, xy [n_db, 2], source [n_db] int)`` — the concatenated gallery.

    Filled into a preallocated array one memmap at a time: five 450 MB banks through
    ``np.concatenate`` would hold both the parts and the whole at once.
    """
    shapes, keeps = [], []
    for seq in sequences:
        bank = np.load(bank_files[seq][label], mmap_mode="r")
        xy, covered, _, _ = geom[seq]
        if bank.shape[0] != len(xy):
            raise SystemExit(
                f"{seq}: bank has {bank.shape[0]} rows but the GPS grid has {len(xy)}. The "
                f"descriptor framing and metadata.json have drifted apart; every coordinate "
                f"after the first would be attached to the wrong frame.")
        shapes.append(bank.shape)
        keeps.append(covered)

    n_db, dim = int(sum(k.sum() for k in keeps)), shapes[0][1]
    desc = np.empty((n_db, dim), dtype=np.float32)
    xy_out = np.empty((n_db, 2), dtype=np.float64)
    source = np.empty(n_db, dtype=np.int16)
    at = 0
    for i, seq in enumerate(sequences):
        bank = np.load(bank_files[seq][label], mmap_mode="r")
        keep = keeps[i]
        n = int(keep.sum())
        desc[at:at + n] = np.asarray(bank[keep])
        xy_out[at:at + n] = geom[seq][0][keep]
        source[at:at + n] = i
        at += n
        del bank
    return torch.from_numpy(desc), xy_out, source


# ---------------------------------------------------------------------------
# One (resolution, filter arm) configuration
# ---------------------------------------------------------------------------
def score_configuration(bank_files, geom, args, cli, label, device, tag):
    db_desc, db_xy, source = pool_database(bank_files, geom, list(cli.database), label)
    q_bank = np.load(bank_files[cli.query][label], mmap_mode="r")
    q_xy, q_covered, q_speed, q_info = geom[cli.query]
    if q_bank.shape[0] != len(q_xy):
        raise SystemExit(f"{cli.query}: bank {q_bank.shape[0]} rows vs GPS grid {len(q_xy)}")
    q_desc = torch.from_numpy(np.asarray(q_bank[q_covered]))
    q_xy = q_xy[q_covered]
    q_speed = q_speed[q_covered]
    del q_bank

    stationary = q_speed < STATIONARY_MS
    print(f"  {tag}: {db_desc.shape[0]} database x {q_desc.shape[0]} queries, "
          f"{int(stationary.sum())} stationary")

    out = {"tag": tag, "n_database": int(db_desc.shape[0]), "n_queries": int(q_desc.shape[0]),
           "threshold_m": cli.threshold_m, "thresholds_m": list(cli.thresholds_m),
           "query": cli.query, "database": list(cli.database),
           "stationary_query_fraction": float(stationary.mean()),
           "dropped_uncovered": {s: geom[s][3]["uncovered"] for s in
                                 [cli.query, *cli.database]},
           "database_rows_per_traverse": {s: int((source == i).sum())
                                          for i, s in enumerate(cli.database)},
           "gt": {}, "recall": {}, "recall_curve": {}, "recall_moving": {},
           "recall_by_threshold": {}, "top1_distance_m": {}, "top1_source": {}}

    curves, first = {}, None
    spaces = [("native", db_desc, q_desc)]
    for dim, power in cli.pca:
        fit = db_desc
        if fit.size(0) > scoring.PCA_FIT_SAMPLES:
            gen = torch.Generator().manual_seed(scoring.PCA_FIT_SEED)
            fit = db_desc[torch.randperm(db_desc.size(0), generator=gen)
                          [:scoring.PCA_FIT_SAMPLES]]
        dim_eff = min(dim, fit.size(0) - 1)
        try:
            pca = inf.pca_fit(fit, device, dim=dim_eff, power=power)
        except torch.cuda.OutOfMemoryError:
            # Recorded rather than swallowed: a silently smaller basis would look like a
            # weaker model. `brisbane_resolution.py:159` sets the same precedent.
            torch.cuda.empty_cache()
            dim_eff = min(2048, fit.size(0) - 1)
            print(f"    pca {dim} OOM -> refitting at {dim_eff}")
            pca = inf.pca_fit(fit, device, dim=dim_eff, power=power)
        out[f"pca{dim}p{power}_dim_eff"] = dim_eff
        spaces.append((f"pca{dim}p{power}",
                       inf.pca_apply(db_desc, pca, device),
                       inf.pca_apply(q_desc, pca, device)))
        del pca, fit
        torch.cuda.empty_cache()

    # The ranking depends only on the descriptors, so every tolerance is a re-read of the
    # same top-20. That makes the sensitivity sweep essentially free — and it is the thing
    # worth looking at here, because Brisbane's 1 Hz GPS puts consecutive fixes a median
    # 13.9 m apart, so a 25 m radius sits close to the ground truth's own noise floor.
    rankings = {}
    # ``rank_fn`` lets a method contribute more than one ranking per descriptor space — Event-GeM
    # returns its global ranking *and* the homography-verified one, so both are scored side by
    # side from a single pass. Default: the plain top-k this script has always used.
    rank_fn = getattr(cli, "rank_fn", None)
    for name, db_space, q_space in spaces:
        produced = (rank_fn(name, db_space, q_space, device) if rank_fn else
                    {name: topk_ranked(db_space, q_space, device, chunk=cli.score_chunk,
                                       db_chunk=getattr(cli, "db_chunk", None))})
        for rname, ranked in produced.items():
            rankings[rname] = ranked
            # How far the top-1 actually lands, in metres — a GT-free read on retrieval quality.
            d = np.linalg.norm(db_xy[ranked[0]] - q_xy, axis=1)
            out["top1_distance_m"][rname] = {
                "median": float(np.median(d)), "p25": float(np.percentile(d, 25)),
                "p75": float(np.percentile(d, 75)), "p90": float(np.percentile(d, 90))}
            # Which traverse actually served the top-1. A pooled gallery can be carried by one
            # easy condition — here sunset1's query shares its lighting with sunset2 — and a
            # single R@1 cannot show that. Compared against each traverse's share of the
            # gallery, so an over-represented traverse is not mistaken for a preferred one.
            won = np.bincount(source[ranked[0]], minlength=len(cli.database))
            out["top1_source"][rname] = {
                s: {"share_of_top1": float(won[i] / won.sum()),
                    "share_of_database": float((source == i).sum() / len(source))}
                for i, s in enumerate(cli.database)}
        if name != "native":
            del db_space, q_space

    for threshold in cli.thresholds_m:
        gt = build_gt(db_xy, q_xy, threshold)
        per_query = gt.sum(0)
        key = f"{threshold:g}"
        out["gt"][key] = {"scorable": int((per_query > 0).sum()),
                          "density": float(gt.mean()),
                          "positives_per_query": {
                              "mean": float(per_query.mean()),
                              "median": float(np.median(per_query)),
                              "min": int(per_query.min()), "max": int(per_query.max())}}
        print(f"    @{threshold:>5g} m: {int((per_query > 0).sum())}/{len(per_query)} scorable, "
              f"positives/query median {np.median(per_query):.0f}")
        for name, ranked in rankings.items():
            rec, curve, scorable = recall_from_ranked(ranked, gt)
            moving = (~stationary) & scorable
            hit = gt[ranked, np.arange(gt.shape[1])[None, :]]
            found = np.cumsum(hit, axis=0) > 0
            out["recall_by_threshold"].setdefault(key, {})[name] = {
                str(k): rec[k] for k in KS}
            if threshold == cli.threshold_m:
                out["scorable"] = int((per_query > 0).sum())
                out["gt_density"] = float(gt.mean())
                out["recall"][name] = {str(k): rec[k] for k in KS}
                out["recall_curve"][name] = {str(n): v for n, v in curve.items()}
                out["recall_moving"][name] = {str(k): float(found[k - 1, moving].mean())
                                              for k in KS}
                curves[name] = curve
                if first is None:
                    first = (ranked, scorable, gt.copy())
            print(f"      {name:14s} " + "  ".join(f"R@{k}={rec[k]:.4f}" for k in KS))
        del gt

    subtitle = (f"{cli.query} -> {'+'.join(cli.database)}  |  {out['n_database']} database x "
                f"{out['scorable']} scorable queries, {cli.threshold_m:g} m")
    # ``fig_tag`` separates the figure filename from the result key. They are the same thing for
    # a single-dataset run, but a caller that scores several datasets into one directory needs
    # them apart: the tag encodes only the run config (resolution, filter arm), so two datasets
    # produce the same tag and the second silently overwrites the first's figures.
    fig_tag = getattr(cli, "fig_tag", None) or tag
    scoring.figure_recall_curve(curves, subtitle,
                                os.path.join(cli.out_dir, f"recall_curve_{fig_tag}.png"))
    ranked, scorable, gt = first
    figure_error_map(db_xy, q_xy, ranked[:, scorable], gt, scorable, cli.threshold_m,
                     os.path.join(cli.out_dir, f"error_map_{fig_tag}.png"))
    del db_desc, q_desc, gt
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", metavar="LABEL=PATH", nargs="+", default=["=".join(DEFAULT_CKPT)],
                    help="one or more checkpoints. Slicing and rendering cost more than the "
                         "forward pass, so every checkpoint given here is fed from a single "
                         "pass over the frames rather than one pass each.")
    ap.add_argument("--query", default=QUERY)
    ap.add_argument("--database", nargs="+", default=list(DATABASE))
    ap.add_argument("--resolutions", type=int, nargs="+", default=[322, 224])
    ap.add_argument("--pca", nargs="+", metavar="DIM,POWER", default=None,
                    help="whitening settings to report beside the native descriptor "
                         f"(default: {' '.join(f'{d},{p}' for d, p in PCA_SETTINGS)}). "
                         "power 0.5 is the SALAD optimum and 0.25 the GeM one, so a table "
                         "mixing aggregators wants both — otherwise one arm is scored with "
                         "the other's hyperparameter.")
    ap.add_argument("--arms", nargs="+", default=["on", "off"], choices=["on", "off"],
                    help="background-activity filter arms to run")
    ap.add_argument("--threshold-m", type=float, default=25.0,
                    help="the headline tolerance, and the one the figures describe")
    ap.add_argument("--thresholds-m", type=float, nargs="+",
                    default=[25.0, 50.0, 75.0, 100.0],
                    help="tolerances to report alongside it. Free: the ranking does not "
                         "depend on the ground truth, so each extra radius is one KD-tree "
                         "query and a re-read of the same top-20.")
    ap.add_argument("--event-filter-dt-us", type=int, default=50000,
                    help="eventcv background-activity window in MICROseconds (its native "
                         "timestamp unit). 50000 = the 50 ms window, retaining ~84%% of "
                         "active pixels.")
    ap.add_argument("--source", default="real", choices=list(tnpz.SOURCES),
                    help="which arm the events come from. `real` (default) reads the HDF5 "
                         "recording through eventcv and is the only source the traverses "
                         "without an npz tree have. The others read "
                         "<npz-root>/<dataset>/<source>/<seq>/frame_*.npz — I2E micro-"
                         "saccades over the co-recorded DAVIS APS frames, and the vignette-"
                         "masked variants of both. Anything but `real` implies "
                         "--aps-aligned and forces the filter arm off.")
    ap.add_argument("--aps-aligned", action="store_true",
                    help="drop frames with no APS frame within half a slice. Implied by a "
                         "non-real --source, and available on `real` so the two arms of a "
                         "Sim2Real comparison are scored on exactly the same rows.")
    ap.add_argument("--eventlab-dir", default=DEFAULT_EVENTLAB)
    ap.add_argument("--npz-root", default=DEFAULT_NPZ_ROOT)
    ap.add_argument("--unfiltered-bank-dir", default=DEFAULT_UNFILTERED_BANKS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--out-json", default="results.json")
    ap.add_argument("--dataset", default="brisbane_event")
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--no-hot-pixel", action="store_true")
    ap.add_argument("--score-chunk", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=inf.BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=inf.NUM_WORKERS)
    cli = ap.parse_args()

    if cli.threshold_m not in cli.thresholds_m:
        cli.thresholds_m = sorted([cli.threshold_m, *cli.thresholds_m])
    cli.pca = parse_pca(cli.pca, PCA_SETTINGS)
    pairs = [tuple(spec.split("=", 1)) for spec in cli.ckpt]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cli.out_dir, exist_ok=True)
    sequences = [cli.query, *cli.database]

    synthetic = cli.source != "real"
    if synthetic:
        # The background-activity filter removes sensor read noise. I2E has none to remove,
        # and assert_filter_active would abort on a retention ratio that cannot be met.
        if cli.arms != ["off"]:
            print(f"  --source {cli.source}: forcing the filter arm off — background-"
                  f"activity denoising is meaningless on simulated events")
            cli.arms = ["off"]
        cli.aps_aligned = True

    args = _Args(cli.eventlab_dir, cli.dataset, cli.dt_ms, cli.no_hot_pixel, True,
                 source=cli.source, npz_root=cli.npz_root)
    head = torch.load(pairs[0][1], map_location="cpu", weights_only=False)
    args.representation = inf.cfg_from_ckpt(head).representation
    del head

    print(f"{cli.dataset}: query {cli.query} -> database {'+'.join(cli.database)}")
    for label, path in pairs:
        print(f"  {label} = {path}")
    print(f"  {cli.threshold_m:g} m radius, dt {cli.dt_ms} ms, {args.representation}, "
          f"hot-pixel {not cli.no_hot_pixel}, arms {cli.arms}")

    geom = traverse_geometry(args, sequences, cli.npz_root, aps_align=cli.aps_aligned)

    all_results = {}
    for arm in cli.arms:
        args.no_event_filter = (arm == "off")
        args.filter_dt_us = None if arm == "off" else cli.event_filter_dt_us
        if arm == "on":
            assert_filter_active(args, cli.query, cli.event_filter_dt_us)
        for resolution in cli.resolutions:
            if synthetic:
                # A distinct tag and the run's own directory. Without both, a synthetic arm
                # would be written as `r322` into --unfiltered-bank-dir and silently
                # overwrite — or worse, silently reuse — the real unfiltered banks that
                # scripts/brisbane_resolution.py put there.
                tag = f"r{resolution}_{cli.source}"
                bank_dir = cli.out_dir
            else:
                tag = f"r{resolution}" if arm == "off" else f"r{resolution}ba{cli.dt_ms}"
                bank_dir = cli.unfiltered_bank_dir if arm == "off" else cli.out_dir
            print(f"\n{'=' * 78}\n{tag}  (filter {arm}, banks in {bank_dir})\n{'=' * 78}")

            loaded, transform = load_all(pairs, device, resolution)
            bank_files = {}
            for seq in sequences:
                files, n = extract(loaded, transform, seq, args, bank_dir, tag, device,
                                   cli.batch_size, cli.workers)
                bank_files[seq] = files
            for _, _, model in loaded:
                del model
            loaded.clear()
            torch.cuda.empty_cache()

            for label, path in pairs:
                if label not in bank_files[cli.query]:
                    continue                        # deduplicated away by load_all
                # One tag per (arm, resolution, checkpoint) so a multi-checkpoint run does
                # not overwrite its own rows, but a single-checkpoint run keeps the plain
                # tag it has always written.
                key = tag if len(pairs) == 1 else f"{tag}_{label}"
                res = score_configuration(bank_files, geom, args, cli, label, device, key)
                res["filter_arm"] = arm
                res["event_filter_dt_us"] = None if arm == "off" else cli.event_filter_dt_us
                res["source"] = cli.source
                res["aps_aligned"] = bool(cli.aps_aligned)
                res["resolution"] = resolution
                res["checkpoint"] = path
                res["label"] = label
                all_results[key] = res

                out_json = os.path.join(cli.out_dir, cli.out_json)
                tmp = out_json + ".tmp"
                with open(tmp, "w") as handle:
                    json.dump({"dataset": cli.dataset,
                               "checkpoints": {a: b for a, b in pairs},
                               "query": cli.query, "database": list(cli.database),
                               "threshold_m": cli.threshold_m, "dt_ms": cli.dt_ms,
                               "pca": [list(s) for s in cli.pca],
                               "source": cli.source, "aps_aligned": bool(cli.aps_aligned),
                               "hot_pixel": not cli.no_hot_pixel, "results": all_results},
                              handle, indent=2)
                os.replace(tmp, out_json)           # a reader never sees a half-written file

    print(f"\n{'=' * 78}\nR@1 / R@10 by tolerance (best descriptor space)")
    for tag, res in all_results.items():
        best = max(res["recall"], key=lambda s: res["recall"][s]["1"])
        cells = "  ".join(
            f"{t:g}m {res['recall_by_threshold'][f'{t:g}'][best]['1']:.3f}/"
            f"{res['recall_by_threshold'][f'{t:g}'][best]['10']:.3f}"
            for t in cli.thresholds_m)
        print(f"  {tag:12s} [{best}]  {cells}")
        print(f"  {'':12s} top-1 lands a median "
              f"{res['top1_distance_m'][best]['median']:.1f} m from the query "
              f"(p90 {res['top1_distance_m'][best]['p90']:.1f} m)")
    print(f"\n-> {os.path.join(cli.out_dir, cli.out_json)}")


if __name__ == "__main__":
    main()
