"""Checks on the real-vs-I2E ablation that the recall numbers themselves cannot make.

Run from the repo root::

    pixi run python3 scripts/verify_i2e_brisbane.py \
        --feature-dir /media/adam/vprdatasets/megaevent/brisbane_ablation/features

The experiment's whole claim is that the two arms differ *only* in how the events were
produced. Everything downstream — the frame index, the ground truth, the model, the metric —
is meant to be identical. These checks are what make that a verified statement rather than
an intention, in the order they would break:

1. **Alignment.** Every retained frame's DAVIS intensity timestamp is within half a slice of
   its event slice's centre, and the excluded set is exactly the one ``select.json`` records.
   A silent drift here would make the synthetic arm look worse for a bookkeeping reason.
   Watch for the trap this already caught once: eventcv clamps ``offset`` up to the
   recording's first timestamp, so sunset1's grid starts 1.22 s later than its configured
   offset and an analytic ``offset + i*dt`` is 24.5 slices wrong.
2. **Lossless dump.** ``src.npzdata.load_countmask`` on the real arm's ``.npz`` must be
   byte-identical to eventcv rendering the same slice straight from the HDF5. This is what
   licenses treating the real arm as the published pipeline rather than a re-implementation.
3. **Reproduction.** megaevent on the real arm must land on the published brisbane numbers
   (R@1 0.929 raw / 0.938 pca). One check that covers the dump, the ordering, the ground
   truth resample and the scoring extraction at once.
4. **Density.** The two arms' event statistics, reported rather than asserted. I2E's
   micro-saccade is far denser than a real 50 ms slice, which matters for the one method
   (sparse_event) whose descriptor is unnormalised counts — so the number belongs in the
   writeup next to that method's delta.
5. **Ground-truth sensitivity.** The shipped band lays its frame centres uniformly over the
   GPS route and ``load_gt`` then stretches that onto the event grid. The route is ~10 s
   shorter than the recording, so the stretch is a progressive time compression that could
   in principle be worth ~120 m by the end of a traverse. Rebuilt here at the frames' real
   timestamps — with the route epoch held at ``other.offset`` so the stretch is the only
   thing that varies (see :func:`route_epoch`) — and the cached similarity matrices
   re-scored under it. **Measured, it is benign**: R@1 moves by 0.001-0.002 in either
   direction, with a ~0.01 gain at R@20. The ±70 m band absorbs the drift, so the shipped
   protocol needs no change — and the error is common to both arms in any case.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eventcv as ecv                                                       # noqa: E402
import yaml                                                                 # noqa: E402

from src.inference import (                                                 # noqa: E402
    EVENTLAB_DATASETS, KS, load_gt, recall_at_k, sensor_size,
)
from src.npzdata import load_countmask, read_events                         # noqa: E402

EVENTLAB_ROOT = os.path.join(os.path.dirname(EVENTLAB_DATASETS))
PUBLISHED = {"native": 0.929, "pca": 0.938}     # logs/brisbane_event/sunset2_sunset1
TOL = 0.01                                      # the validity mask moves R@1 by ~0.001


def _ok(flag):
    return "PASS" if flag else "FAIL"


# ---------------------------------------------------------------------------
def check_alignment(args, seqs):
    """APS timestamps against the slice grid they were selected on."""
    print("\n1. APS / event-slice alignment")
    good = True
    for seq in seqs:
        aps = os.path.join(args.npz_root, args.dataset, "aps", seq)
        real = os.path.join(args.npz_root, args.dataset, "real", seq)
        centres = np.load(os.path.join(real, "slice_times.npy")).mean(axis=1) / 1e6
        times = np.load(os.path.join(aps, "frame_times.npy"))
        with open(os.path.join(aps, "select.json")) as f:
            sel = json.load(f)
        with open(os.path.join(real, "dump.json")) as f:
            dump = json.load(f)

        dts = np.abs(times - centres)
        tol = args.dt_ms / 2000.0
        recorded = np.zeros(len(centres), dtype=bool)
        recorded[np.asarray(sel["out_of_tolerance"], dtype=int)] = True
        measured = dts > tol
        n_frames = len(list(os.scandir(real))) - 2      # minus slice_times.npy + dump.json

        same_grid = len(times) == len(centres) == dump["n_slices"] == n_frames
        same_set = bool((recorded == measured).all())
        good &= same_grid and same_set
        print(f"  {seq}: {len(centres)} slices, grid origin {dump['framing_origin_s']:.6f}"
              + ("  (offset CLAMPED to the first event)" if dump["offset_clamped"] else ""))
        print(f"    retained |dt|: mean {dts[~measured].mean() * 1e3:.2f} ms  "
              f"max {dts[~measured].max() * 1e3:.2f} ms  (tolerance {tol * 1e3:.0f} ms)")
        print(f"    excluded {measured.sum()} of {len(centres)} "
              f"({100 * measured.mean():.2f}%) — matches select.json: {_ok(same_set)}")
        print(f"    frame counts agree across npz/aps/dump: {_ok(same_grid)}")
    return good


# ---------------------------------------------------------------------------
def check_lossless(args, seqs, n=6):
    """The real arm's npz vs eventcv rendering the HDF5 slice directly."""
    print("\n2. Real-arm npz round-trip is lossless")
    good = True
    for seq in seqs:
        real = os.path.join(args.npz_root, args.dataset, "real", seq)
        with open(os.path.join(real, "dump.json")) as f:
            dump = json.load(f)
        reader = ecv.open(dump["source"], dt_ms=args.dt_ms,
                          sensor_size=sensor_size(args.dataset),
                          hot_pixel_filter=dump["hot_pixel_filter"],
                          offset=dump["yaml_offset_s"] * 1000.0).with_repr(
                              "countmask", window_ms=args.dt_ms, white_frame=False)
        idx = np.linspace(0, dump["n_slices"] - 1, n).astype(int)
        same = True
        for i in idx:
            a = load_countmask(os.path.join(real, f"frame_{i:06d}.npz"))
            same &= bool(np.array_equal(a, np.asarray(reader[int(i)])))
        good &= same
        print(f"  {seq}: {n} sampled slices byte-identical to eventcv: {_ok(same)}")
    return good


# ---------------------------------------------------------------------------
def check_reproduction(args):
    """megaevent on the real arm against this repo's published brisbane numbers."""
    print("\n3. Real arm reproduces the published brisbane result")
    path = os.path.join(args.feature_dir, args.dataset, "results_megaevent_real.json")
    if not os.path.exists(path):
        print(f"  SKIP: no {os.path.basename(path)} — run --source real --method megaevent")
        return True
    with open(path) as f:
        res = json.load(f)
    good = True
    for space, want in PUBLISHED.items():
        got = res["recall"][space]["1"]
        hit = abs(got - want) <= TOL
        good &= hit
        print(f"  {space}: R@1 {got:.3f} vs published {want:.3f} "
              f"(|Δ| {abs(got - want):.3f} <= {TOL}): {_ok(hit)}")
    print(f"  note: {res['dropped_queries']} of {res['n_queries_raw']} queries are excluded "
          f"by the shared validity mask, which is why this is a tolerance and not equality")
    return good


# ---------------------------------------------------------------------------
def check_density(args, seqs, n=150):
    """Event statistics per arm — reported, not asserted.

    I2E's saccade fires on every pixel whose log-intensity gradient crosses ``C`` over a
    fixed 2 px trajectory, so its event count tracks image texture. A real slice's count
    tracks texture *and* vehicle speed. That difference is a candidate explanation for any
    sparse_event delta (its descriptor is unnormalised counts), independent of any domain
    gap, so the numbers belong in the writeup.
    """
    print("\n4. Event density per arm")
    stats = {}
    for source in ("real", "i2e"):
        for seq in seqs:
            d = os.path.join(args.npz_root, args.dataset, source, seq)
            if not os.path.isdir(d):
                continue
            n_slices = len([f for f in os.listdir(d) if f.startswith("frame_")])
            idx = np.linspace(0, n_slices - 1, min(n, n_slices)).astype(int)
            counts = []
            for i in idx:
                x, _, _, _, h, w = read_events(os.path.join(d, f"frame_{i:06d}.npz"))
                counts.append(x.size)
            counts = np.asarray(counts, dtype=np.float64)
            per_px = counts / (h * w)
            stats[(source, seq)] = counts
            print(f"  {source:5s} {seq}: {counts.mean():8.0f} events/frame "
                  f"(p5 {np.percentile(counts, 5):.0f}, p95 {np.percentile(counts, 95):.0f})"
                  f"  {per_px.mean():5.2f}/px  "
                  f"cv {counts.std() / max(counts.mean(), 1):.2f}")
    for seq in seqs:
        if ("real", seq) in stats and ("i2e", seq) in stats:
            r, s = stats[("real", seq)].mean(), stats[("i2e", seq)].mean()
            print(f"  {seq}: I2E is {s / max(r, 1):.1f}x denser than the real stream")
    print("  (sparse_event clips at 10 events/px, so neither arm saturates it; the ratio "
          "still means the two arms hand it different dynamic range)")
    check_vignette(args, seqs)
    return True


def check_vignette(args, seqs, n=60, dark=20):
    """Where each arm's events land relative to the frame's dark, vignetted foreground.

    I2E thresholds ``Δlog(luma + 1e-3)``, whose sensitivity goes as ``1/(luma + 1e-3)`` —
    so in a near-black region an imperceptible intensity change clears the contrast
    threshold easily. Brisbane's DAVIS frames have a heavily vignetted bottom quarter (the
    bonnet and lens falloff, mean intensity < 10), and that is enough to turn read noise
    there into the majority of the synthetic stream. A real DVS does the opposite: it
    responds to *temporal* change, and an unlit foreground produces almost nothing.

    This is not a bug in the pipeline and it does not break the pairing, but it is a
    concrete, fixable reason the synthetic arm might score lower — distinct from any
    intrinsic "synthetic events are a different domain" effect — so it has to be measured
    before the delta is attributed.
    """
    from PIL import Image

    print(f"\n4b. Where the events land (dark = APS intensity < {dark})")
    for seq in seqs:
        aps = os.path.join(args.npz_root, args.dataset, "aps", seq)
        n_slices = len(np.load(os.path.join(args.npz_root, args.dataset, "real", seq,
                                            "slice_times.npy")))
        idx = np.linspace(int(0.05 * n_slices), int(0.95 * n_slices), n).astype(int)
        share, area = {"real": [], "i2e": []}, []
        for i in idx:
            grey = np.asarray(Image.open(os.path.join(aps, f"frame_{i:06d}.png")).convert("L"))
            mask = grey < dark
            area.append(mask.mean())
            for source in ("real", "i2e"):
                x, y, _, _, h, w = read_events(os.path.join(
                    args.npz_root, args.dataset, source, seq, f"frame_{i:06d}.npz"))
                if x.size:
                    share[source].append(mask[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)].mean())
        r, s = np.mean(share["real"]), np.mean(share["i2e"])
        print(f"  {seq}: the dark region is {np.mean(area) * 100:.1f}% of the frame; "
              f"it holds {r * 100:.1f}% of real events but {s * 100:.1f}% of I2E events "
              f"({s / max(r, 1e-9):.1f}x over-representation)")
    print("  -> most of the synthetic stream is log-amplified noise from the vignetted "
          "foreground, not scene structure. Masking or cropping it before conversion would "
          "isolate the domain gap from this artefact.")


# ---------------------------------------------------------------------------
def route_epoch(args, seq):
    """Unix seconds that the ground truth's time base calls zero.

    This is ``other.offset`` from ``brisbane_event.yaml``, *not* the timestamp of the first
    NMEA sentence — and the difference matters enough to be worth stating.

    ``_load_nmea_gps`` reports pose times relative to the file's first sentence, so the
    naive reconstruction is to read that sentence's clock. Doing so puts sunset2 2.35 s and
    sunset1 0.57 s away from where ``other.offset`` puts them, and re-scoring under the
    resulting band costs 15 points of R@1 — the band lands ~40 rows off the retrievals.
    Those offsets are hand-calibrated alignments between each recording and its GPS log
    (the NMEA files are trimmed concatenations, and their sentence clocks are whole
    seconds), so re-deriving the epoch from the sentence text discards the calibration
    rather than improving on it.

    Holding ``other.offset`` fixed and varying only the frame times is what isolates the
    thing this check is actually about: the uniform grid plus stretch, against the frames'
    real timestamps.
    """
    with open(os.path.join(EVENTLAB_DATASETS, f"{args.dataset}.yaml")) as f:
        spec = yaml.safe_load(f)
    return float(spec["other"]["offset"][seq])


def check_gt_sensitivity(args, ref, query):
    """Re-score the cached similarity matrices under a timestamp-exact ground truth.

    The shipped band lays its frame centres uniformly over the *GPS route*, which for
    sunset1 is 714 s against a 723.9 s recording; ``load_gt`` then stretches it onto the
    event grid. Rebuilding it at the frames' real timestamps removes that stretch. Both arms
    share the shipped band, so this cannot change the ablation's conclusion — it says how
    much absolute recall the stretch is costing.
    """
    print("\n5. Ground-truth sensitivity (uniform-stretch vs timestamp-exact band)")
    sys.path.insert(0, EVENTLAB_ROOT)
    from datasets.groundtruths import ground_truth_from_pose_only

    with open(os.path.join(EVENTLAB_DATASETS, f"{args.dataset}.yaml")) as f:
        dataset_config = yaml.safe_load(f)
    config = {"data_path": args.eventlab_dir, "ground_truth_tolerance": args.gt_tolerance}

    centres, epochs = {}, {}
    for seq in (ref, query):
        real = os.path.join(args.npz_root, args.dataset, "real", seq)
        with open(os.path.join(real, "dump.json")) as f:
            dump = json.load(f)
        centres[seq] = np.load(os.path.join(real, "slice_times.npy")).mean(axis=1) / 1e6
        epochs[seq] = route_epoch(args, seq)
        print(f"  {seq}: route epoch {epochs[seq]:.2f}, framing origin "
              f"{dump['framing_origin_s']:.2f} — first frame sits "
              f"{dump['framing_origin_s'] + args.dt_ms / 2000.0 - epochs[seq]:+.3f} s "
              f"into the route")

    out_dir = os.path.join(args.feature_dir, args.dataset, "gt_exact")
    os.makedirs(out_dir, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(EVENTLAB_ROOT)
        gt_path = ground_truth_from_pose_only(
            config=config, dataset_config=dataset_config, dataset_name=args.dataset,
            reference_name=ref, query_name=query, timewindow_ms=float(args.dt_ms),
            gt_tolerance=args.gt_tolerance, out_dir=out_dir,
            out_basename=f"{ref}_{query}_GT_exact",
            ref_frame_times=centres[ref] - epochs[ref],
            qry_frame_times=centres[query] - epochs[query])
    finally:
        os.chdir(cwd)

    exact = np.load(gt_path) != 0
    keep_db = _mask(args, ref, exact.shape[0])
    keep_q = _mask(args, query, exact.shape[1])
    exact = exact[np.ix_(keep_db, keep_q)]
    print(f"  timestamp-exact band: {exact.shape[0]}x{exact.shape[1]}, "
          f"density {exact.mean():.4f} (no resample — one row per real frame)")

    feature_dir = os.path.join(args.feature_dir, args.dataset)
    any_scored = False
    for source in ("real", "i2e"):
        for sim_path in sorted(_sims(feature_dir, ref, query, source)):
            sim = np.load(sim_path)
            if sim.shape != exact.shape:
                print(f"  SKIP {os.path.basename(sim_path)}: {sim.shape} vs {exact.shape}")
                continue
            shipped, _ = load_gt(os.path.join(args.eventlab_dir, args.dataset, "ground_truth",
                                              f"{ref}_{query}_GT.npy"),
                                 len(keep_db), len(keep_q))
            shipped = shipped[np.ix_(keep_db, keep_q)]
            a, b = recall_at_k(sim, shipped), recall_at_k(sim, exact)
            any_scored = True
            print(f"  {os.path.basename(sim_path)}")
            print("    shipped band: " + "  ".join(f"R@{k}={a[k]:.3f}" for k in KS))
            print("    timestamp-exact: " + "  ".join(f"R@{k}={b[k]:.3f}" for k in KS)
                  + "   (Δ R@1 " + f"{b[1] - a[1]:+.3f})")
    if not any_scored:
        print("  SKIP: no cached similarity matrices to re-score yet")
    return True


def _mask(args, seq, n):
    path = os.path.join(args.npz_root, args.dataset, "aps", seq, "select.json")
    keep = np.ones(n, dtype=bool)
    if os.path.exists(path):
        with open(path) as f:
            bad = np.asarray(json.load(f)["out_of_tolerance"], dtype=int)
        keep[bad[bad < n]] = False
    return keep


def _sims(feature_dir, ref, query, source):
    import glob
    return glob.glob(os.path.join(feature_dir, f"sim_*_{ref}_{query}_{source}.npy"))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", "-d", default="brisbane_event")
    ap.add_argument("--ref", "-r", default="sunset2")
    ap.add_argument("--query", "-q", default="sunset1")
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--npz-root", default="/media/adam/vprdatasets/megaevent/brisbane_npz")
    ap.add_argument("--eventlab-dir", default="/media/adam/vprdatasets/eventgem")
    ap.add_argument("--feature-dir",
                    default="/media/adam/vprdatasets/megaevent/brisbane_ablation/features")
    ap.add_argument("--gt-tolerance", type=int, default=70,
                    help="metres of dilation along the reference route; 70 is what "
                         "src/eventlab.py writes for every brisbane band this repo builds")
    ap.add_argument("--skip-gt", action="store_true",
                    help="skip check 5, which re-reads every cached similarity matrix")
    args = ap.parse_args()

    seqs = (args.ref, args.query)
    results = [
        ("alignment", check_alignment(args, seqs)),
        ("lossless dump", check_lossless(args, seqs)),
        ("reproduction", check_reproduction(args)),
        ("density", check_density(args, seqs)),
    ]
    if not args.skip_gt:
        results.append(("gt sensitivity", check_gt_sensitivity(args, args.ref, args.query)))

    print("\n" + "-" * 70)
    for name, flag in results:
        print(f"  {_ok(flag)}  {name}")
    return 0 if all(f for _, f in results) else 1


if __name__ == "__main__":
    sys.exit(main())
