"""Fit the empirical video->event-clock time warp from gopro_align.py's measurements.

    pixi run python3 scripts/gopro_timewarp.py --seq sunset1
    pixi run python3 scripts/gopro_timewarp.py --seq morning

Why a warp and not a (v0, fps) pair: the measured video-vs-APS lag is not a line in
video time. morning sits flat at +0.43 s for its first 300 s (bin IQRs of 0.1 s — the
measurement is that precise) and then steps down by ~1.0 s in discrete drops; sunset1
steps by ~1.1 s. That is the signature of frames lost at the shipped ``_concat.mp4``
chapter joins, not of a clock rate, so no single frame-rate correction (NTSC 29.97
included) can place every frame. The only model that fits the data is the data:

    container_time(c) = (c - v0_base) + delta(c - v0_base)

with ``delta`` a piecewise-linear interpolation through knots fitted as running medians
of the measured lag. ``v0_base`` is the Event-LAB yaml offset the measurements are
expressed against — it carries no claim of being the true video start; the warp absorbs
whatever it is off by.

Inputs are the align JSONs written by ``scripts/gopro_align.py`` (raw, rho=1 runs and
--wall-fps runs both accepted — a rho run's residuals are converted back to the raw
convention with ``lag_raw = lag + t*(rho-1)``). Several JSONs average together: the
same slice sampled twice gives two independent NCC measurements.

Acceptance separates model error from sample noise. Each knot is a median over its
window's ~10-25 samples, so its bootstrap CI is the warp's own accuracy (criteria:
median CI95 half-width <= 0.15 s, p90 <= 0.30 s ~= 2-4 m at route speed, inside the
8.86 m NMEA residual). The even/odd holdout residual additionally guards against a
systematically biased fit (median <= 0.25 s) — its raw value contains each sample's
own NCC matching noise (~0.2-0.35 s sd) and therefore overstates the warp error. No
knot gap over 90 s inside the covered span. FAIL stops the pipeline; the JSON keeps
every sample for diagnosis.

Output: ``logs/gopro2/warp_<seq>.json`` with the knots, a fingerprint (sha1 of the knot
arrays) that ``extract_gopro.py --timewarp`` copies into ``select.json`` so downstream
provenance checks can prove which warp framed a tree.
"""

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

MEDIAN_WIN_S = 30.0        # running-median half-window for cleaning and knots
KNOT_STEP_S = 20.0
CLEAN_TOL_S = 0.6          # reject samples this far from the running median
MIN_NCC = 0.28             # documented true-pair NCC floor (extract_gopro.sobel_ncc_prep):
                           # a window peaking below it matched noise, not the scene
MIN_KNOT_SAMPLES = 3
# Acceptance. The holdout per-sample residual CONTAINS each sample's own NCC matching
# noise (~0.2-0.35 s sd, worse at dusk), which more sampling never removes — so it
# bounds the warp error only very loosely and is kept as a BIAS guard (its median).
# The accuracy of the warp itself — the knots actually used — is each knot's bootstrap
# CI as a median over its ~10-25 window samples, and that is what the width criteria
# test.
MAX_HOLDOUT_MED_S = 0.25       # a biased fit shifts the median; noise does not
MAX_KNOT_CI_MED_S = 0.15       # median knot CI95 half-width
MAX_KNOT_CI_P90_S = 0.30       # p90 knot CI95 half-width
MAX_KNOT_GAP_S = 90.0


def load_samples(paths):
    t, lag, ncc, doc = [], [], [], None
    for path in paths:
        with open(path) as handle:
            doc = json.load(handle)
        ti = np.asarray(doc["per_slice_video_t_s"], dtype=np.float64)
        li = np.asarray(doc["per_slice_lag_s"], dtype=np.float64)
        rho = float(doc.get("rho", 1.0))
        if rho != 1.0:
            li = li + ti * (rho - 1.0)      # back to the raw (container==wall) convention
        t.append(ti)
        lag.append(li)
        ncc.append(np.asarray(doc["per_slice_ncc"], dtype=np.float64))
    return np.concatenate(t), np.concatenate(lag), np.concatenate(ncc), doc


def running_median(t, lag, at, win):
    out = np.full(len(at), np.nan)
    for i, c in enumerate(at):
        m = np.abs(t - c) <= win
        if m.sum() >= MIN_KNOT_SAMPLES:
            out[i] = np.median(lag[m])
    return out


def clean(t, lag):
    """Iteratively drop samples far from the local running median (the wrong-building
    matches sit seconds off the true-match manifold)."""
    keep = np.ones(len(t), dtype=bool)
    for _ in range(3):
        med = running_median(t[keep], lag[keep], t, MEDIAN_WIN_S)
        good = np.abs(lag - med) <= CLEAN_TOL_S
        good &= ~np.isnan(med)
        if good.sum() == keep.sum() and bool(np.all(good == keep)):
            break
        keep = good
    return keep


def fit_knots(t, lag, ci=False):
    grid = np.arange(np.floor(t.min()), np.ceil(t.max()) + KNOT_STEP_S, KNOT_STEP_S)
    delta = running_median(t, lag, grid, MEDIAN_WIN_S)
    ok = ~np.isnan(delta)
    if not ci:
        return grid[ok], delta[ok]
    rng = np.random.default_rng(0)
    half = np.zeros(int(ok.sum()))
    for i, c in enumerate(grid[ok]):
        vals = lag[np.abs(t - c) <= MEDIAN_WIN_S]
        boot = np.median(rng.choice(vals, size=(500, len(vals)), replace=True), axis=1)
        half[i] = (np.percentile(boot, 97.5) - np.percentile(boot, 2.5)) / 2
    return grid[ok], delta[ok], half


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seq", required=True)
    ap.add_argument("--align-json", nargs="+", default=None,
                    help="align JSONs to pool; default: every "
                         "logs/gopro2/align*_<seq>.json")
    ap.add_argument("--out-json", default=None,
                    help="default logs/gopro2/warp_<seq>.json")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(repo, "logs", "gopro2")
    paths = args.align_json or sorted(
        glob.glob(os.path.join(log_dir, f"align*_{args.seq}.json")))
    if not paths:
        raise SystemExit(f"no align JSONs for {args.seq} under {log_dir}")
    out_json = args.out_json or os.path.join(log_dir, f"warp_{args.seq}.json")

    t, lag, ncc, last_doc = load_samples(paths)
    strong = ncc >= MIN_NCC
    order = np.argsort(t[strong])
    t, lag = t[strong][order], lag[strong][order]
    keep = clean(t, lag)
    tc, lc = t[keep], lag[keep]
    print(f"{args.seq}: {len(strong)} samples from {len(paths)} run(s), "
          f"{int(strong.sum())} above NCC {MIN_NCC}, {int(keep.sum())} clean")

    # Held-out residuals: fit on evens, evaluate on odds (and vice versa), pooled.
    resid = []
    for half in (0, 1):
        fit = np.arange(len(tc)) % 2 == half
        kt, kd = fit_knots(tc[fit], lc[fit])
        pred = np.interp(tc[~fit], kt, kd)
        resid.append(lc[~fit] - pred)
    resid = np.abs(np.concatenate(resid))
    hold_med, hold_p90 = float(np.median(resid)), float(np.percentile(resid, 90))

    knots_t, knots_d, knots_ci = fit_knots(tc, lc, ci=True)
    gaps = np.diff(knots_t)
    max_gap = float(gaps.max()) if len(gaps) else np.inf
    ci_med, ci_p90 = float(np.median(knots_ci)), float(np.percentile(knots_ci, 90))

    checks = {
        "holdout_median_ok": hold_med <= MAX_HOLDOUT_MED_S,
        "knot_ci_median_ok": ci_med <= MAX_KNOT_CI_MED_S,
        "knot_ci_p90_ok": ci_p90 <= MAX_KNOT_CI_P90_S,
        "knot_gap_ok": max_gap <= MAX_KNOT_GAP_S,
    }
    ok = all(checks.values())

    fingerprint = hashlib.sha1(
        knots_t.tobytes() + knots_d.tobytes()).hexdigest()[:16]
    payload = {
        "sequence": args.seq,
        "v0_base_s": float(last_doc["v0_yaml_s"]),
        "container_fps": float(last_doc.get("container_fps", 30.0)),
        "align_jsons": [os.path.basename(p) for p in paths],
        "n_samples": int(len(t)), "n_clean": int(keep.sum()),
        "knots_t_s": [float(v) for v in knots_t],
        "knots_delta_s": [float(v) for v in knots_d],
        "knots_ci95_half_s": [float(v) for v in knots_ci],
        "knot_ci95_half_median_s": ci_med,
        "knot_ci95_half_p90_s": ci_p90,
        "span_s": [float(knots_t[0]), float(knots_t[-1])],
        "max_knot_gap_s": max_gap,
        "holdout": {"median_abs_s": hold_med, "p90_abs_s": hold_p90,
                    "n": int(len(resid))},
        "fingerprint": fingerprint,
        "checks": checks,
        "pass": ok,
    }
    with open(out_json, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"  warp: {len(knots_t)} knots over [{knots_t[0]:.0f}, {knots_t[-1]:.0f}] s, "
          f"delta [{knots_d.min():+.2f}, {knots_d.max():+.2f}] s, max gap {max_gap:.0f} s")
    print(f"  knot CI95 half-width: median {ci_med:.3f} s  p90 {ci_p90:.3f} s")
    print(f"  holdout |resid| (contains per-sample NCC noise): median {hold_med:.3f} s  "
          f"p90 {hold_p90:.3f} s  (n {len(resid)})")
    print(f"  {'PASS' if ok else 'FAIL'} {checks}  fingerprint {fingerprint}")
    print(f"  -> {out_json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
