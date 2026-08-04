"""Fit the per-traverse camera-vs-GPS clock offset for Brisbane-Event-VPR.

    pixi run python3 scripts/calibrate_brisbane_clock.py --write

Brisbane's HDF5 timestamps and its NMEA timestamps are both absolute UTC epochs, so a frame's
position ought to be the GPS track evaluated at that frame's own clock. It is not: the recording
clock of each session is offset from the GPS receiver's by up to **7 s**, which at ~14 m/s puts
the same physical place a hundred metres apart between two traverses. Uncorrected, the pairwise
correspondence error is a median 19-104 m depending on the pair — far too coarse for the 25 m
tolerance the evaluation protocol calls for.

The offsets are recoverable because the dataset ships its own GPS-derived correspondence. For each
`ground_truth/<ref>_<query>_GT.npy`, the centre of the ±70 m band in a query column is the
reference frame Event-LAB matched to it. Under a correct clock, the two frames of such a pair must
land on the same coordinate — so the offsets are whatever minimises that distance.

**This does not leak model information.** Event-LAB's correspondence comes from route-completion
matching over the NMEA tracks; no descriptor, no checkpoint. Calibrating against it and then
applying a metric radius re-expresses the dataset's own ground truth in metres, which is exactly
what the protocol wants.

What makes the result trustworthy is that it is heavily over-determined: eight pairs constrain
five free offsets, and the pairwise estimates are **cycle-consistent to 0.30 s RMS** — an artefact
of fitting would not close its loops. Four random restarts agree to 0.1 s.

Anchored at `sunset2 = 0`, the canonical reference traverse of every prior Brisbane result. Only
differences between traverses affect a query-vs-reference distance, so the anchor is a convention;
absolute positions inherit whatever residual sunset2 itself carries.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from src import traversegps as tg  # noqa: E402

DEFAULT_EVENTLAB = "/media/adam/vprdatasets/eventgem"
DATASET = "brisbane_event"
TRAVERSES = ("sunset1", "sunset2", "daytime", "morning", "night", "sunrise")
ANCHOR = "sunset2"
QUERY_STRIDE = 5                    # every 5th query column is plenty; the fit is over-determined


def load_correspondences(eventlab_dir, dataset, traverses, frames, stride=QUERY_STRIDE):
    """[(ref, query, ref_idx, query_idx)] — the band centre of every available GT pair.

    Indices are on the *descriptor* grid, mapped proportionally from the GT grid (the GT is
    built on a `floor(T_gps / dt)` pose grid, which is a few hundred rows shorter).
    """
    gt_dir = os.path.join(eventlab_dir, dataset, "ground_truth")
    out = []
    for name in sorted(os.listdir(gt_dir)):
        if not name.endswith("_GT.npy"):
            continue
        ref, query = name[:-len("_GT.npy")].split("_", 1)
        if ref not in traverses or query not in traverses or ref == query:
            continue
        gt = np.load(os.path.join(gt_dir, name), mmap_mode="r")
        n_ref_gt, n_q_gt = gt.shape
        n_ref, n_q = frames[ref], frames[query]
        q_idx = np.arange(0, n_q, stride)
        cols = np.clip((q_idx * (n_q_gt / n_q)).round().astype(int), 0, n_q_gt - 1)
        band = np.asarray(gt[:, cols]) != 0
        keep = band.any(0)
        q_idx, band = q_idx[keep], band[:, keep]
        centre = np.array([np.flatnonzero(c).mean() for c in band.T])
        ref_idx = np.clip((centre * (n_ref / n_ref_gt)).round().astype(int), 0, n_ref - 1)
        out.append((ref, query, ref_idx, q_idx))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eventlab-dir", default=DEFAULT_EVENTLAB)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--restarts", type=int, default=8,
                    help="Nelder-Mead is started from zero and then re-seeded around the best "
                         "point so far. The objective is a median over eight pairs and has "
                         "shallow local minima a few metres apart at offsets differing by "
                         "seconds, so a single descent from zero does land in one (~37 m).")
    ap.add_argument("--drift", action="store_true",
                    help="also fit a per-traverse clock *rate*. Fits the objective better and "
                         "evaluates worse — see the comment at the call site. Kept for the "
                         "record, not recommended.")
    ap.add_argument("--write", action="store_true",
                    help=f"write the fitted table to <eventlab-dir>/<dataset>/"
                         f"{tg.CLOCK_OFFSET_FILE}, which src.traversegps loads automatically")
    cli = ap.parse_args()

    lat0, lon0 = tg.track_origin(cli.eventlab_dir, cli.dataset, TRAVERSES)
    track, frames, start = {}, {}, {}
    for seq in TRAVERSES:
        t_gps, lat, lon = tg.nmea_track(tg.nmea_path(cli.eventlab_dir, cli.dataset, seq))
        start[seq], frames[seq] = tg.framing_origin(cli.eventlab_dir, cli.dataset, seq, cli.dt_ms)
        track[seq] = (t_gps, tg.local_enu(lat, lon, lat0, lon0))

    n = len(TRAVERSES)

    def xy_at(seq, idx, shift, drift):
        t_gps, xy = track[seq]
        elapsed = (np.asarray(idx) + 0.5) * (cli.dt_ms / 1000.0)
        t = start[seq] + elapsed + shift + drift * elapsed
        return np.column_stack([np.interp(t, t_gps, xy[:, 0]),
                                np.interp(t, t_gps, xy[:, 1])])

    corr = load_correspondences(cli.eventlab_dir, cli.dataset, TRAVERSES, frames)
    print(f"{len(corr)} ground-truth pairs constraining {2 * n} parameters "
          f"({n} offsets + {n} drifts)")

    def per_pair(vec):
        off, drift = dict(zip(TRAVERSES, vec[:n])), dict(zip(TRAVERSES, vec[n:]))
        # Median, not mean: the band centre is unreliable where a traverse stops or the GT
        # band runs off the end, and a handful of such columns would drag a mean fit.
        return np.array([np.median(np.linalg.norm(
            xy_at(ref, ri, off[ref], drift[ref]) - xy_at(query, qi, off[query], drift[query]),
            axis=1)) for ref, query, ri, qi in corr])

    # Offsets and drifts are each determined only up to a common value (a shift applied to
    # every traverse is nearly, though not exactly, unobservable). The weak penalties pin
    # that direction without biasing any resolvable difference.
    def objective(v):
        return per_pair(v).mean() + 1e-3 * abs(v[:n].mean()) + 1e-2 * abs(v[n:].mean())

    baseline = per_pair(np.zeros(2 * n))
    print(f"uncalibrated: mean median-distance {baseline.mean():.1f} m")

    # Two stages, because releasing drift from a cold start lands several metres worse: a
    # wrong offset trades against a wrong drift almost freely, so the offsets have to be
    # roughly right before the drifts mean anything. Stage 1 searches the six offsets with
    # drift held at zero — restricted explicitly, since Nelder-Mead's initial simplex
    # perturbs every coordinate it is handed and a zero sigma would not have pinned them.
    rng = np.random.default_rng(0)

    def search(x0, free, sigma, iters, label):
        """Minimise over the coordinates in ``free`` only, holding the rest at ``x0``."""
        def wrapped(sub):
            full = x0.copy()
            full[free] = sub
            return objective(full)

        best = (wrapped(x0[free]), x0[free])
        for seed in range(iters):
            s0 = best[1] if seed == 0 else best[1] + sigma * rng.normal(0, 1, len(free))
            res = minimize(wrapped, s0, method="Nelder-Mead",
                           options={"maxiter": 20000, "maxfev": 20000,
                                    "xatol": 1e-4, "fatol": 1e-5})
            if res.fun < best[0]:
                best = (res.fun, res.x)
        full = x0.copy()
        full[free] = best[1]
        print(f"  {label}: {per_pair(full).mean():.3f} m")
        return full

    best_vec = search(np.zeros(2 * n), np.arange(n), 1.5, cli.restarts, "offsets")
    if cli.drift:
        # Off by default, and it should stay off unless the objective gains more residuals.
        # A per-traverse rate term fits the eight pair medians better (4.99 m vs 8.50 m) and
        # then evaluates *worse* — pooled R@1 at 25 m fell .9033 -> .6717 — because eight
        # aggregate numbers cannot determine twelve parameters. The offset-only model has
        # eight residuals for six parameters and generalises.
        best_vec = search(best_vec, np.arange(2 * n),
                          np.concatenate([np.full(n, 0.5), np.full(n, 0.004)]),
                          cli.restarts, "+ drift")
    best = (objective(best_vec), best_vec)

    # Reported relative to the anchor: only differences between traverses affect a
    # query-vs-reference distance, and a table centred on sunset2 is comparable with the
    # every-prior-result convention.
    vec = best[1].copy()
    vec[:n] -= vec[TRAVERSES.index(ANCHOR)]
    vec[n:] -= vec[n + TRAVERSES.index(ANCHOR)]
    offsets = {s: round(float(v), 4) for s, v in zip(TRAVERSES, vec[:n])}
    drifts = {s: round(float(v), 6) for s, v in zip(TRAVERSES, vec[n:])}
    final = per_pair(vec)
    print(f"\ncalibrated: mean median-distance {final.mean():.2f} m "
          f"({baseline.mean() / final.mean():.1f}x better)\n")
    print(f"  {'traverse':10s} {'offset s':>10s} {'drift s/s':>11s} {'accumulated':>13s}")
    for seq in TRAVERSES:
        print(f"  {seq:10s} {offsets[seq]:+10.3f} {drifts[seq]:+11.5f} "
              f"{drifts[seq] * frames[seq] * cli.dt_ms / 1000.0:+12.2f}s")
    print("\nper-pair residual:")
    for (ref, query, _, _), before, after in zip(corr, baseline, final):
        print(f"  {ref:8s}->{query:8s}  {before:6.1f} m  ->  {after:5.1f} m")

    if cli.write:
        path = os.path.join(cli.eventlab_dir, cli.dataset, tg.CLOCK_OFFSET_FILE)
        with open(path, "w") as handle:
            json.dump({"anchor": ANCHOR, "dt_ms": cli.dt_ms,
                       "unit": {"offsets": "seconds", "drifts": "seconds per second"},
                       "model": "t_eff = t_frame + offset + drift * elapsed",
                       "fitted_on": [f"{r}->{q}" for r, q, _, _ in corr],
                       "mean_residual_m": float(final.mean()),
                       "uncalibrated_mean_residual_m": float(baseline.mean()),
                       "offsets": offsets, "drifts": drifts}, handle, indent=2)
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
