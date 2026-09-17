"""Measure the GoPro video's frame-0 time in the event clock, from the pictures alone.

Run from the repo root::

    pixi run python3 scripts/gopro_align.py --seq sunset1
    pixi run python3 scripts/gopro_align.py --seq morning

The wide-window (default +-15 s) version of ``extract_gopro.py --verify``: for a sample of
moving slices, decode a window of video around the position the candidate time base predicts
and cross-correlate Sobel-magnitude thumbnails against that slice's APS frame. Each sampled
slice yields one independent lag estimate; the median (with a bootstrap CI) is the
measurement. Searching +-15 s instead of +-1.5 s means an offset of 6-11 s lands inside the
window rather than at its clamped edge, so this can adjudicate between the two candidate
tables rather than assume one:

* the Event-LAB yaml ``other.offset`` (read via ``dump_event_npz.stream_offset``, the base
  every lag here is expressed against — measured lag ~0 means the yaml is right);
* the repudiated ``VIDEO_BEGINNING`` table of 2026-09-10, which sat 11 s (sunset1) / 6 s
  (morning) above the yaml and was chosen by maximising R@1 on cached descriptor banks —
  fitted to the test metric, so it is treated here as a *candidate to test*, never a prior.

The adopted value is ``v0 = v0_yaml - median_lag`` regardless of which candidate it lands
near; the verdict field only records the comparison. The measurement must PASS its own
acceptance criteria (tight CI, consistent per-slice peaks, real NCC contrast) or the run
stops — a weak peak is reported, not rounded to the nearest hypothesis.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_aps import slice_centres                              # noqa: E402
from extract_gopro import (DEFAULT_VIDEO_ROOT, VERIFY_CROPS, decode_window,  # noqa: E402
                           probe, resolve_video, sobel_ncc_prep)
from dump_event_npz import stream_offset                           # noqa: E402
from src import traversegps as tg                                  # noqa: E402

# The 2026-09-10 table this measurement replaces, kept ONLY as a labelled comparison
# candidate. Do not import it from extract_gopro: that module's VIDEO_BEGINNING is
# overwritten with the measured values once this script has passed.
FITTED_2026_09_10 = {
    "sunset1": 1587452593.35,
    "daytime": 1587705136.80,
    "morning": 1588029271.73,
    "sunrise": 1588105240.91,
}

PROXY = (480, 270)      # decode size of the search window
THUMB = (87, 65)        # NCC thumbnail, matches extract_gopro.verify_lag

# The per-window lag distribution is a mixture: a tight core where the NCC found the true
# pair, plus mismatches (a window that locked onto the wrong building is off by seconds).
# The raw median is not robust once mismatches are a third of the sample, so the estimator
# is the mode cluster: histogram the lags, take the densest 0.2 s bin, keep every window
# within +-CLUSTER_TOL_S of it, and quote the cluster median with a bootstrap CI.
CLUSTER_BIN_S = 0.2
CLUSTER_TOL_S = 0.75

# Acceptance criteria — all must hold for PASS.
MIN_CLUSTER_FRACTION = 0.5     # majority of windows must agree on one lag
MIN_CLUSTER_N = 12
MAX_CI95_WIDTH_S = 0.30        # ~+-0.15 s = ~2 m at route speed, well inside the 25 m radius
MIN_NCC_MEDIAN = 0.20          # true-pair NCC measured 0.28-0.44 (sobel_ncc_prep docstring)
MAX_DRIFT_SPAN_S = 0.5         # lag-vs-video-time slope x duration (chapter-join stutter)


def measure(seq, video_path, centres, v0_yaml, fps, duration, covered, speed,
            win_s, n_sample, speed_min, rho=1.0):
    """(lags[n], peaks[n], crops[n], slices[n]) — one lag estimate per sampled slice.

    ``rho`` = wall_fps / container_fps. GoPros record NTSC 29.97 fps while the mp4
    container labels the stream 30.0, so a frame's *container* time runs 0.1% fast
    against the wall clock. The predicted container time of the slice's match is
    ``(centre - v0) * rho``, and the returned lag is the residual from THAT — with the
    right rho the residuals are drift-free, which is exactly what the drift check
    downstream verifies.
    """
    dw, dh = PROXY
    tw, th = THUMB
    aps_dir = os.path.join(NPZ_ROOT_DS, "aps", seq)
    usable = np.flatnonzero(covered & (speed > speed_min)
                            & (centres > v0_yaml + win_s + 2.0)
                            & (centres < v0_yaml + duration - win_s - 2.0))
    if len(usable) < 8:
        raise SystemExit(f"only {len(usable)} usable moving slices — cannot measure")
    pick = usable[np.linspace(0, len(usable) - 1, min(n_sample, len(usable))).astype(int)]
    n_frames = int(round(2 * win_s * fps)) + 1

    import cv2
    lags, peaks, crops, used = [], [], [], []
    for at, i in enumerate(pick):
        aps = cv2.imread(os.path.join(aps_dir, f"frame_{int(i):06d}.png"),
                         cv2.IMREAD_GRAYSCALE)
        if aps is None:
            raise FileNotFoundError(f"{aps_dir}/frame_{int(i):06d}.png missing — run "
                                    f"scripts/extract_aps.py first")
        ref = sobel_ncc_prep(aps, tw, th)
        pred = (centres[i] - v0_yaml) * rho          # container time of the match
        t0 = pred - win_s
        frames = decode_window(video_path, t0, n_frames, dw, dh)
        if len(frames) < n_frames // 2:
            continue
        best = (-9.0, 0.0, -1)
        for k, frame in enumerate(frames):
            for ci, (x, y, cw, ch) in enumerate(VERIFY_CROPS):
                s = float((ref * sobel_ncc_prep(frame[y:y + ch, x:x + cw], tw, th)).sum())
                if s > best[0]:
                    best = (s, t0 + k / fps - pred, ci)
        peaks.append(best[0])
        lags.append(best[1])
        crops.append(best[2])
        used.append(int(i))
        if (at + 1) % 4 == 0:
            print(f"    {at + 1}/{len(pick)} windows  (last lag {best[1]:+.2f} s, "
                  f"ncc {best[0]:.3f})", flush=True)
    return np.asarray(lags), np.asarray(peaks), np.asarray(crops), np.asarray(used)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--dataset", default="brisbane_event")
    ap.add_argument("--win-s", type=float, default=15.0)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--speed-min", type=float, default=6.0)
    ap.add_argument("--wall-fps", type=float, default=None,
                    help="hypothesised TRUE frame rate (wall clock) when the container "
                         "lies about it — e.g. 29.97002997 (NTSC 30000/1001) for a GoPro "
                         "whose mp4 claims 30.0. Default: trust the container (rho=1). "
                         "The drift check is the test of the hypothesis: the right rate "
                         "leaves no lag-vs-time slope.")
    ap.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--npz-root", default="/media/adam/vprdatasets/megaevent/brisbane_npz")
    ap.add_argument("--eventlab-dir", default="/media/adam/vprdatasets/eventgem")
    ap.add_argument("--out-json", default=None,
                    help="default logs/gopro2/align_<seq>.json under the repo root")
    args = ap.parse_args()

    global NPZ_ROOT_DS
    NPZ_ROOT_DS = os.path.join(args.npz_root, args.dataset)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_json = args.out_json or os.path.join(repo, "logs", "gopro2",
                                             f"align_{args.seq}.json")

    video_path = resolve_video(args.seq, None, args.video_root)
    width, height, fps, n_frames, duration = probe(video_path)
    v0_yaml = stream_offset(args.dataset, args.seq) / 1000.0
    centres = slice_centres(os.path.join(NPZ_ROOT_DS, "real", args.seq, "slice_times.npy"))
    lat0, lon0 = tg.track_origin(args.eventlab_dir, args.dataset, [args.seq])
    _, covered, speed, _ = tg.frame_coords(args.eventlab_dir, args.dataset, args.seq,
                                           lat0, lon0, 50, npz_root=args.npz_root)

    fitted = FITTED_2026_09_10.get(args.seq)
    pred_fitted = -(fitted - v0_yaml) if fitted is not None else None
    wall_fps = args.wall_fps or fps
    rho = wall_fps / fps
    print(f"{args.seq}: video {os.path.basename(video_path)} {width}x{height} @ {fps:g} fps"
          f" (container), wall hypothesis {wall_fps:.6f} (rho {rho:.6f})  {duration:.1f} s")
    print(f"  yaml base v0={v0_yaml:.3f}; predicted lag 0.00 if yaml right, "
          f"{pred_fitted:+.2f} if the fitted table was right" if fitted is not None
          else f"  yaml base v0={v0_yaml:.3f}")

    lags, peaks, crops, used = measure(args.seq, video_path, centres, v0_yaml, fps,
                                       duration, covered, speed, args.win_s,
                                       args.samples, args.speed_min, rho=rho)

    # Mode cluster: densest CLUSTER_BIN_S bin, then everything within CLUSTER_TOL_S of it.
    edges = np.arange(lags.min() - CLUSTER_BIN_S, lags.max() + 2 * CLUSTER_BIN_S,
                      CLUSTER_BIN_S)
    hist, _ = np.histogram(lags, bins=edges)
    mode_centre = float(edges[int(np.argmax(hist))] + CLUSTER_BIN_S / 2)
    in_cluster = np.abs(lags - mode_centre) <= CLUSTER_TOL_S
    cl_lags, cl_peaks = lags[in_cluster], peaks[in_cluster]
    cluster_fraction = float(in_cluster.mean())

    med = float(np.median(cl_lags))
    rng = np.random.default_rng(0)
    boot = np.median(rng.choice(cl_lags, size=(2000, len(cl_lags)), replace=True), axis=1)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    iqr = [float(np.percentile(cl_lags, 25)), float(np.percentile(cl_lags, 75))]
    ncc_med = float(np.median(cl_peaks))

    # Drift diagnostic: the four videos are chapter CONCATENATIONS, so a stutter at a
    # join would make the true lag a function of video time rather than a constant. A
    # least-squares line over the cluster members measures that; its span over the whole
    # video must be small or no single v0 is right and the framing model itself is wrong.
    vt = (centres[used] - v0_yaml)[in_cluster]          # video time of each estimate
    slope = float(np.polyfit(vt, cl_lags, 1)[0]) if len(cl_lags) >= 8 else 0.0
    drift_span = slope * duration

    checks = {
        "cluster_fraction_ok": cluster_fraction >= MIN_CLUSTER_FRACTION,
        "cluster_n_ok": int(in_cluster.sum()) >= MIN_CLUSTER_N,
        "ci95_width_ok": (ci[1] - ci[0]) <= MAX_CI95_WIDTH_S,
        "ncc_median_ok": ncc_med >= MIN_NCC_MEDIAN,
        "drift_ok": abs(drift_span) <= MAX_DRIFT_SPAN_S,
    }
    ok = all(checks.values())

    # Verdict: which candidate the measured median is consistent with (0.5 s slack — well
    # under the 6-11 s the candidates disagree by, well over the CI of a passing run).
    if abs(med - 0.0) <= 0.5:
        verdict = "yaml"
    elif pred_fitted is not None and abs(med - pred_fitted) <= 0.5:
        verdict = "fitted"
    else:
        verdict = "neither"

    summary = {
        "sequence": args.seq,
        "video": video_path,
        "win_s": args.win_s,
        "n_sampled": int(len(lags)),
        "v0_yaml_s": v0_yaml,
        "median_lag_s": med,
        "bootstrap_ci95_s": ci,
        "iqr_s": iqr,
        "cluster": {"bin_s": CLUSTER_BIN_S, "tol_s": CLUSTER_TOL_S,
                    "mode_centre_s": mode_centre, "n": int(in_cluster.sum()),
                    "fraction": cluster_fraction},
        "ncc_median": ncc_med,
        "ncc_max": float(peaks.max()),
        "crop_histogram": np.bincount(crops, minlength=len(VERIFY_CROPS)).tolist(),
        "drift": {"slope_s_per_s": slope, "span_s": float(drift_span)},
        "per_slice_index": [int(v) for v in used],
        "per_slice_video_t_s": [float(centres[i] - v0_yaml) for i in used],
        "per_slice_speed_ms": [float(speed[i]) for i in used],
        "per_slice_lag_s": [float(v) for v in lags],
        "per_slice_ncc": [float(v) for v in peaks],
        "predicted_lag_if_yaml": 0.0,
        "predicted_lag_if_fitted": pred_fitted,
        "container_fps": float(fps),
        "wall_fps": float(wall_fps),
        "rho": float(rho),
        # lag is the residual against the rho-scaled prediction, measured in container
        # seconds; dividing by rho converts it back to the wall/event clock.
        "v0_measured_s": v0_yaml - med / rho,
        "verdict": verdict,
        "checks": checks,
        "pass": ok,
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  cluster {int(in_cluster.sum())}/{len(lags)} windows "
          f"({cluster_fraction:.0%}) at mode {mode_centre:+.2f} s   drift span "
          f"{drift_span:+.2f} s over the video")
    print(f"  measured lag {med:+.3f} s  CI95 [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
          f"IQR [{iqr[0]:+.2f}, {iqr[1]:+.2f}]  ncc median {ncc_med:.3f}")
    print(f"  v0_measured = {v0_yaml - med / rho:.3f}  (wall fps {wall_fps:.6f})  "
          f"verdict: {verdict}")
    print(f"  {'PASS' if ok else 'FAIL'} {checks}")
    print(f"  -> {out_json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
