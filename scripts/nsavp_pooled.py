"""NSAVP under the pooled protocol: one query traverse, every other traverse pooled.

Run from the repo root::

    pixi run python3 scripts/nsavp_pooled.py \
        --ckpt b_ship=/media/adam/vprdatasets/megaevent/runs/b_full_P64_v4_vpr/step10000.pt

The Brisbane protocol (``scripts/brisbane_pooled.py``) applied to NSAVP: 322^2 frames, the
background-activity filter on, a 25 m Euclidean radius, PCA whitening fit on the database bank
alone, and a streamed top-20 that never materialises the similarity matrix. Everything after the
geometry is imported from that script rather than reimplemented, so the two datasets cannot drift
apart in how they are scored. Only the coordinate source differs, and that is
:mod:`src.nsavpgps`.

**The database is route 0 only, because R0 and R1 are different routes.** Measured over the pose
tracks before anything was extracted: every ``R0_FA0`` query frame has an ``R0_*`` pose within
25 m (median nearest neighbour 0.3-4.8 m), and *none* has an ``R1_*`` pose within 25 m (median
825-848 m). The R1 traverses can supply no positives at all, so pooling them in would add ~77k
pure distractors and measure a different thing — how well the model rejects another part of Ann
Arbor, not how well it recognises this route under changed conditions. They are left out.

That also means excluding ``R1_FA0`` for sharing the query's illumination does not do what it
looks like it does: it is not a near-duplicate of ``R0_FA0`` but 843 m away on the other route.
There is no NSAVP analogue of dropping Brisbane's ``sunset2``, because no traverse re-runs route
0 in the query's own lighting — the closest conditions in the database are ``R0_FN0`` (night) and
``R0_FS0`` (sunset), both genuine appearance changes.

The five database traverses are the full condition/direction grid: forward and reverse
(``F``/``R``) crossed with night, sunset and afternoon (``N``/``S``/``A``), minus the forward
afternoon run that is the query itself. The three reverse traverses are the hard half — the
camera sees the route from the opposite heading — and they cover 98% of the query at 25 m.
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
from src import nsavpgps as ng  # noqa: E402
from brisbane_resolution import _Args, extract  # noqa: E402
from brisbane_pooled import assert_filter_active, score_configuration  # noqa: E402
from tokyo_trajectory import load_all, parse_pca  # noqa: E402

DEFAULT_EVENTLAB = "/media/adam/vprdatasets/eventlab"
DEFAULT_OUT = "/media/adam/vprdatasets/megaevent/nsavp_pooled"
# The two checkpoints this project ships, scored together from one pass over the frames.
DEFAULT_CKPT = (
    "b_ship=/media/adam/vprdatasets/megaevent/runs/b_full_P64_v4_vpr/step10000.pt",
    "s_ship=/media/adam/vprdatasets/megaevent/runs/s_salad_ft4_v2_vpr/step10000.pt",
)

QUERY = "R0_FA0"
# Route 0 in every other condition and direction. R1_DA0/R1_RA0/R1_FA0 are a different route
# and contribute zero positives — see the module docstring.
DATABASE = ("R0_FN0", "R0_FS0", "R0_RA0", "R0_RN0", "R0_RS0")
PCA_SETTINGS = ((4096, 0.5), (2048, 0.5))
STATIONARY_MS = 0.5

# NSAVP's pooled gallery is ~101k x 8448 (3.4 GB at float32) against Brisbane's 1.8 GB, so the
# database is scored in slices that fit an 8 GB card with room for the similarity block.
DB_CHUNK = 40000


def traverse_geometry(root, sequences, dt_ms, bank_frames=None):
    """{seq: (xy, covered, speed, info)} in one shared local ENU frame."""
    lat0, lon0 = ng.track_origin(root, sequences)
    print(f"  projection origin ({lat0:.6f}, {lon0:.6f})")
    geom = {}
    for seq in sequences:
        geom[seq] = ng.frame_coords(root, seq, lat0, lon0, dt_ms)
        info = geom[seq][3]
        print(f"    {seq:8s} {info['n_frames']:6d} frames  {info['n_gps_fixes']:7d} poses  "
              f"{info['uncovered']:4d} outside pose span  route {info['route_len_m']:.0f} m  "
              f"clock {info['clock_start_offset_s']:+.3f} s")
        if bank_frames and seq in bank_frames and bank_frames[seq] != info["n_frames"]:
            raise SystemExit(
                f"{seq}: eventcv yields {bank_frames[seq]} slices but the pose grid implies "
                f"{info['n_frames']}. Every coordinate after the first would be attached to "
                f"the wrong frame.")
    return geom


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", metavar="LABEL=PATH", nargs="+", default=list(DEFAULT_CKPT),
                    help="one or more checkpoints. Rendering costs more than the forward "
                         "pass, so all of them are fed from a single pass over the frames.")
    ap.add_argument("--query", default=QUERY)
    ap.add_argument("--database", nargs="+", default=list(DATABASE))
    ap.add_argument("--resolutions", type=int, nargs="+", default=[322])
    ap.add_argument("--pca", nargs="+", metavar="DIM,POWER", default=None,
                    help=f"whitening settings (default: "
                         f"{' '.join(f'{d},{p}' for d, p in PCA_SETTINGS)})")
    ap.add_argument("--arms", nargs="+", default=["on"], choices=["on", "off"],
                    help="background-activity filter arms to run")
    ap.add_argument("--threshold-m", type=float, default=25.0)
    ap.add_argument("--thresholds-m", type=float, nargs="+",
                    default=[25.0, 50.0, 75.0, 100.0])
    ap.add_argument("--event-filter-dt-us", type=int, default=50000,
                    help="eventcv background-activity window in MICROseconds. Verified on "
                         "NSAVP: retention plateaus at 75.7%% from 50000 upward, and falls to "
                         "13.9%% at 50 — the same units and the same plateau as Brisbane, "
                         "despite NSAVP timestamping in nanoseconds.")
    ap.add_argument("--eventlab-dir", default=DEFAULT_EVENTLAB)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--out-json", default="results.json")
    ap.add_argument("--dataset", default="nsavp")
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--no-hot-pixel", action="store_true")
    ap.add_argument("--score-chunk", type=int, default=256)
    ap.add_argument("--db-chunk", type=int, default=DB_CHUNK,
                    help="database rows resident on the GPU at once while ranking")
    ap.add_argument("--batch-size", type=int, default=inf.BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=inf.NUM_WORKERS)
    ap.add_argument("--geometry-only", action="store_true",
                    help="check the pose grid and clock alignment, then stop — no extraction")
    cli = ap.parse_args()

    if cli.threshold_m not in cli.thresholds_m:
        cli.thresholds_m = sorted([cli.threshold_m, *cli.thresholds_m])
    cli.pca = parse_pca(cli.pca, PCA_SETTINGS)
    pairs = [tuple(spec.split("=", 1)) for spec in cli.ckpt]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cli.out_dir, exist_ok=True)
    sequences = [cli.query, *cli.database]
    root = os.path.join(cli.eventlab_dir, cli.dataset)

    args = _Args(cli.eventlab_dir, cli.dataset, cli.dt_ms, cli.no_hot_pixel, True)
    head = torch.load(pairs[0][1], map_location="cpu", weights_only=False)
    args.representation = inf.cfg_from_ckpt(head).representation
    del head

    print(f"{cli.dataset}: query {cli.query} -> database {'+'.join(cli.database)}")
    for label, path in pairs:
        print(f"  {label} = {path}")
    print(f"  {cli.threshold_m:g} m radius, dt {cli.dt_ms} ms, {args.representation}, "
          f"hot-pixel {not cli.no_hot_pixel}, arms {cli.arms}")

    geom = traverse_geometry(root, sequences, cli.dt_ms)
    if cli.geometry_only:
        report_overlap(geom, cli)
        return

    all_results = {}
    for arm in cli.arms:
        args.no_event_filter = (arm == "off")
        args.filter_dt_us = None if arm == "off" else cli.event_filter_dt_us
        if arm == "on":
            assert_filter_active(args, cli.query, cli.event_filter_dt_us)
        for resolution in cli.resolutions:
            tag = f"r{resolution}" if arm == "off" else f"r{resolution}ba{cli.dt_ms}"
            print(f"\n{'=' * 78}\n{tag}  (filter {arm})\n{'=' * 78}")

            loaded, transform = load_all(pairs, device, resolution)
            bank_files, bank_frames = {}, {}
            for seq in sequences:
                files, n = extract(loaded, transform, seq, args, cli.out_dir, tag, device,
                                   cli.batch_size, cli.workers)
                bank_files[seq], bank_frames[seq] = files, n
            for _, _, model in loaded:
                del model
            loaded.clear()
            torch.cuda.empty_cache()

            # The bank row count is only knowable once eventcv has framed the recording, so
            # the grid is re-derived against it here rather than trusted from the ceil().
            geom = traverse_geometry(root, sequences, cli.dt_ms, bank_frames)

            for label, path in pairs:
                if label not in bank_files[cli.query]:
                    continue                        # deduplicated away by load_all
                key = tag if len(pairs) == 1 else f"{tag}_{label}"
                res = score_configuration(bank_files, geom, args, cli, label, device, key)
                res["filter_arm"] = arm
                res["event_filter_dt_us"] = None if arm == "off" else cli.event_filter_dt_us
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
        print(f"  {tag:20s} [{best}]  {cells}")
        print(f"  {'':20s} top-1 lands a median "
              f"{res['top1_distance_m'][best]['median']:.1f} m from the query "
              f"(p90 {res['top1_distance_m'][best]['p90']:.1f} m)")
    print(f"\n-> {os.path.join(cli.out_dir, cli.out_json)}")


def report_overlap(geom, cli):
    """How many positives each database traverse can actually supply, before anything is run.

    A traverse on the other route contributes none, and pooling it in only adds distractors.
    Printing this before a multi-hour extraction is what turns that from a surprise in the
    final number into a protocol decision.
    """
    from scipy.spatial import cKDTree

    q_xy, q_cov, q_speed, _ = geom[cli.query]
    q_xy = q_xy[q_cov]
    print(f"\n  {cli.query}: {len(q_xy)} covered query frames, "
          f"{float((q_speed[q_cov] < STATIONARY_MS).mean()):.1%} stationary")
    print(f"  {'traverse':10s} {'frames':>7s}  {'@25m':>7s}  {'median nn':>10s}")
    for seq in cli.database:
        xy, cov, _, _ = geom[seq]
        xy = xy[cov]
        d, _ = cKDTree(xy).query(q_xy, k=1)
        print(f"  {seq:10s} {len(xy):7d}  {float((d < cli.threshold_m).mean()):7.3f}  "
              f"{float(np.median(d)):9.1f} m")


if __name__ == "__main__":
    main()
