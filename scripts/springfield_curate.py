"""Springfield curation: telemetry-driven cleanup of the query set and the database.

Run from the repo root, stage by stage::

    pixi run python3 scripts/springfield_curate.py --stage evidence   # measure, cut nothing
    pixi run python3 scripts/springfield_curate.py --stage rules      # write the mask
    pixi run python3 scripts/springfield_curate.py --stage sheets     # eyeball what went
    pixi run python3 scripts/springfield_curate.py --stage report --after <results.json>

Why this exists: the eval's only slice filter is "inside GPS coverage and not bridging a
dropout" (``springfield_eval.partition_geometry``). Nothing gates capture quality, so the
0.312 headline mixes designed difficulty (reversals, viewpoint offsets — the dataset's
whole point) with capture junk (the rig being raised and lowered at stream boundaries,
GPS that has not converged, attitude excursions). This script separates the two under
two standing rules:

* **No model in the curation loop.** Every rule that *drops* a slice reads capture
  telemetry only — ``derived/location.csv``, ``derived/orientation.csv``, the GPS track —
  never a descriptor or a retrieval outcome. The one model-derived signal (scene
  self-similarity, which finds the vegetation corridors) is flag-only by user decision:
  those slices stay in the query set and are reported as their own split.
* **No rule ships unmeasured.** ``--stage evidence`` scores every candidate rule against
  the cached per-slice retrieval dump *before* it is adopted. A rule whose removed slices
  score like the slices it keeps is trimming difficulty rather than junk, and is rejected
  in writing.

Capture facts established by probe on 2026-09-01, which the rule set encodes:

* **The phone was never bolted to the rig** (user, 2026-09-01): pocket for the database
  passes, mostly *in hand* for the queries, so the operator could start and stop a short
  capture quickly. The telemetry agrees — on the database passes a yaw-to-travel fit
  leaves 69-89 deg of residual (i.e. uncorrelated) over a ~2 Hz, 15 deg gait bob, while
  on a clean query pass it leaves 5.6 deg with tilt inside ~3 deg, which is a hand
  carried level and pointed the way the walk goes.
* That makes attitude a **proxy for operator handling, not for camera pointing**. It is
  used here only where handling is what we mean: at the stream boundaries, where a
  90-120 deg swing is the phone being picked up to press start or put away after stop
  (query ``041152Z``: 118 deg for the first ~2 s and last ~2 s of an 11.7 s session; every
  database pass shows the same at 140 deg for its first 2.5-6.5 s and last 3-8 s).
  A *mid-session* tilt excursion is measured and reported as its own candidate rule but
  is not assumed to mean anything about the camera — the operator glancing at the phone
  looks identical — and it ships only if the evidence stage shows the slices it removes
  score far below the ones it keeps.
* Every database pass also opens with unconverged GPS: up to 43 m of reported accuracy on
  the first few fixes, settling to ~4 m.
* 52 of the 84 query sessions carry at least one fix with reported accuracy above 10 m,
  the worst 150 m. Against a 25 m match radius such a slice cannot adjudicate a
  retrieval at all: it is unscorable, not merely hard.

Stage outputs live in ``output/springfield_curation/``. The mask itself is indexed on the
**full slice grid** — the same indexing the descriptor banks and ``keep`` masks use — so
``springfield_full.py --curation`` can AND it straight into ``keep`` with no re-extraction.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src.traversegps import local_enu                          # noqa: E402
from springfield_eval import list_partitions                   # noqa: E402
from springfield_full import DB_ARMS, SWEEPS, discover_queries  # noqa: E402
from springfield_diag import load_cached_session, wrap180      # noqa: E402

DEFAULT_RESULTS = ("output/springfield_full/"
                   "results_b_v8_accum_s750_s750_3f4eb54780_accumulate_r322_hp1_baoff.json")
DEFAULT_ROOT = "/media/adam/vprdatasets/megaevent/springfield/sessions"
DEFAULT_BANK_DIR = "/media/adam/vprdatasets/megaevent/evaluations/springfield"
DEFAULT_DIAG_DIR = "output/springfield_diag"
DEFAULT_OUT = "output/springfield_curation"

# Rule parameters. Defaults set from the stage-0 evidence curves (see
# evidence_{tag}.json); every one of them is echoed into the curation JSON.
PROFILE = {
    # --- GPS validity: a slice whose own position is uncertain cannot adjudicate a
    # 25 m retrieval, whatever the image shows. Both gates are nearly inert on this
    # capture (63 and 31 query slices) because the camera starts several seconds after
    # the phone, by which time the receiver has converged — worth keeping as a guard,
    # not worth claiming as cleanup.
    "hacc_max_m": 10.0,
    # Leading/trailing GPS warmup trim: REJECTED on measurement, hence 0 (disabled).
    # Unlike the plain accuracy gate, this one extends the trim from the stream start
    # until the reported accuracy has *held* under the gate, so it removes slices whose
    # own accuracy is fine. Scored against the baseline dump it took 89 slices at R@1
    # 0.416 against a 0.312 average — above-average data. It cost dawn 0.005 overall and
    # gutted one session (201349Z: 161 -> 111 slices, R@1 0.453 -> 0.270, because the 50
    # it removed were scoring 0.86). A high *reported* accuracy is not a wrong position.
    "hacc_settle_s": 0.0,
    "gps_resid_max_m": 4.0,      # raw fix vs its own 5-fix rolling median; every query
                                 # slice past 4 m scored R@1 0.000 (31 of them)
    # --- Handling at the stream boundaries. REJECTED on evidence, hence 0: the last two
    # seconds of a query score *above* the session average (0.571 vs 0.479 in the
    # reference cell) and the first two are only 0.08 below it, which is a difficulty
    # gradient, not junk. Set a value here to re-enable.
    "boundary_tilt_deg": 0.0,
    "boundary_speed_mps": 0.0,   # stationary slices also score above average (0.65)
    "boundary_max_frac": 0.25,   # never trim more than this fraction off one end
    # --- Attitude excursions. R@1 is flat (0.28-0.42) from 0 to 90 deg of tilt and then
    # collapses to 0.087 past 90 deg: below 90 the phone is merely being held at an
    # angle, past it the phone is being pocketed or flipped, which is when the capture
    # actually suffers. 90 is the knee, not a round number.
    "attitude_stable_p25_deg": 6.0,   # see attitude_trustworthy()
    "tilt_max_deg": 90.0,
    "tilt_min_run": 2,           # slices; shorter is gait bob, not a real excursion
    "tilt_dilate": 1,            # slices either side (a 50 ms slice smears the exit)
    "omega_max_dps": 300.0,      # 32 slices at R@1 0.19; trivial mass, kept as a guard
    # --- Camera-side quality (frames stage). Zero disables. Both gates were measured on
    # the gallery by what they absorb against what they cost (stage_db_absorption).
    #
    # Blank rows are a free removal and the knee is sharp: at active_frac < 0.08 the gate
    # takes 95 rows (0.16% of the scanned gallery, well below its 0.1th percentile of
    # 0.055), absorbs 2.29% of all wrong top-1s and gives up *zero* correct ones. Push to
    # 0.12 and it starts costing correct matches for almost no extra absorption.
    "active_frac_min": 0.08,
    "db_active_frac_min": 0.08,
    # Sharpness is REJECTED on both sides, in opposite directions and both decisive. On
    # the gallery it is actively harmful: the least-sharp 5% of rows absorb 4.4% of the
    # wrong top-1s while giving up 7.7% of the right ones. On queries the least-sharp 1%
    # score 0.625 against 0.492 for the rest — blurred queries do *better*. The top-300
    # alias magnets are statistically indistinguishable from the gallery at large
    # (sharp_ratio 25.8 vs 26.4, active_frac 0.336 vs 0.334): they are concentrated
    # because many queries pass aliasing-prone *places*, not because the frames are bad.
    "sharp_ratio_min": 0.0,
    "db_sharp_ratio_min": 0.0,
    # --- Review, never automatic
    "min_keep_frac": 0.30,       # sessions below this are listed for the user to judge
    "self_sim_flag": 0.55,       # flag-only: vegetation-corridor scene degeneracy
}


# ---------------------------------------------------------------------------
# 1. Telemetry -> per-slice features (full slice grid, both roles)
# ---------------------------------------------------------------------------
def _csv(sdir, name):
    return np.atleast_1d(np.genfromtxt(os.path.join(sdir, "derived", name),
                                       delimiter=",", names=True, dtype=np.float64))


def gravity_vec(o):
    """World-up expressed in device axes — the yaw-invariant half of the attitude.

    This is the rotation matrix's third row. Turning a corner rotates the device about
    the world vertical, which leaves this vector untouched, so any change in it is tilt.
    Checked on the forward database pass, which sweeps 21585 deg of yaw: this vector's
    dispersion about its median is 17 deg, the third *column*'s is 83 deg.
    """
    qw, qx, qy, qz = o["qw"], o["qx"], o["qy"], o["qz"]
    v = np.column_stack([2 * (qx * qz - qy * qw),
                         2 * (qy * qz + qx * qw),
                         1 - 2 * (qx * qx + qy * qy)])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def ang_rate(t_ns, q, base_s):
    """``(t_ns, |omega| deg/s)`` over a fixed baseline, from quaternion deltas."""
    dt_med = max(float(np.median(np.diff(t_ns))) / 1e9, 1e-6)
    step = max(1, int(round(base_s / dt_med)))
    if len(t_ns) <= step:
        return t_ns, np.zeros(len(t_ns))
    dot = np.abs(np.sum(q[step:] * q[:-step], axis=1)).clip(0.0, 1.0)
    dth = 2.0 * np.degrees(np.arccos(dot))
    dt = np.maximum((t_ns[step:] - t_ns[:-step]) / 1e9, 1e-6)
    return 0.5 * (t_ns[step:] + t_ns[:-step]), dth / dt


def _rolling_median(x, k):
    """Centred rolling median, edges held — for spotting single-fix GPS jumps."""
    if len(x) < k:
        return np.full_like(x, np.median(x))
    win = np.lib.stride_tricks.sliding_window_view(x, k)
    med = np.median(win, axis=1)
    pad = k // 2
    return np.concatenate([np.full(pad, med[0]), med, np.full(len(x) - len(med) - pad,
                                                              med[-1])])


def _yaw_travel_resid(travel, yaw, moving):
    """Mean |residual| of the best yaw->travel-bearing fit — the carry-mode test.

    A few degrees means the phone turns with the walk: it was held in hand, facing the
    way the operator walked, so its attitude tracks the operator's handling. Near 90 deg
    means the two are unrelated — the pocket case, where attitude says nothing at all.
    Neither case makes attitude a measurement of where the *camera* points; the phone was
    never bolted to the rig.
    """
    if moving.sum() < 20:
        return None
    best = None
    for sign in (1.0, -1.0):
        d = np.radians(travel[moving] - sign * yaw[moving])
        off = np.degrees(np.arctan2(np.sin(d).mean(), np.cos(d).mean()))
        r = float(np.abs(wrap180(travel[moving] - (sign * yaw[moving] + off))).mean())
        if best is None or r < best:
            best = r
    return best


def session_features(cli, sdir, sid):
    """Per-slice capture-quality features on the session's full slice grid.

    Every array is length ``n_slices`` and indexed exactly like the descriptor bank and
    the eval's ``keep`` mask, so a curation mask built here drops the rows it means to.
    """
    s = load_cached_session(cli, sdir, sid, need_bank=False)
    keep0 = s["keep"]
    n = len(keep0)
    c = s["centres_ns"]
    lat, lon = s["latlon"][:, 0], s["latlon"][:, 1]
    xy = local_enu(lat, lon, float(np.median(lat)), float(np.median(lon)))

    # --- travel: speed and bearing over a +-1 s baseline on the slice grid
    w = max(1, int(round(1000.0 / cli.dt_ms)))
    idx = np.arange(n)
    lo, hi = np.maximum(idx - w, 0), np.minimum(idx + w, n - 1)
    step = xy[hi] - xy[lo]
    dt_s = np.maximum((c[hi] - c[lo]) / 1e9, 1e-6)
    speed = np.linalg.norm(step, axis=1) / dt_s
    travel = np.degrees(np.arctan2(step[:, 0], step[:, 1])) % 360.0

    # --- GPS quality from the raw fixes
    loc = _csv(sdir, "location.csv")
    order = np.argsort(loc["time_laptop_ns"])
    t_l = loc["time_laptop_ns"][order]
    hacc = np.interp(c, t_l, loc["horizontalAccuracy"][order])
    f_lat, f_lon = loc["latitude"][order], loc["longitude"][order]
    m_per_deg = 111320.0
    sm_lat, sm_lon = _rolling_median(f_lat, 5), _rolling_median(f_lon, 5)
    resid_fix = np.hypot((f_lat - sm_lat) * m_per_deg,
                         (f_lon - sm_lon) * m_per_deg
                         * np.cos(np.radians(float(np.median(f_lat)))))
    gps_resid = np.interp(c, t_l, resid_fix)

    # --- attitude
    o = _csv(sdir, "orientation.csv")
    oo = np.argsort(o["time_laptop_ns"])
    t_o = o["time_laptop_ns"][oo]
    g = gravity_vec(o)[oo]
    q = np.column_stack([o["qw"], o["qx"], o["qy"], o["qz"]])[oo]
    cov = (c >= t_o[0]) & (c <= t_o[-1])
    gs = np.column_stack([np.interp(c, t_o, g[:, j]) for j in range(3)])
    gs /= np.linalg.norm(gs, axis=1, keepdims=True)
    t_f, w_fast = ang_rate(t_o, q, cli.dt_ms / 1000.0)     # smear within one slice
    t_sl, w_slow = ang_rate(t_o, q, 0.5)                   # sustained turning
    omega = np.interp(c, t_f, w_fast)
    omega_slow = np.interp(c, t_sl, w_slow)

    # Reference attitude = the pose held while walking steadily. Taking the session
    # median instead breaks on sessions where the excursion is half the session (one
    # night query sits 53 deg from its own median).
    steady = cov & keep0 & (speed > 0.6) & (omega_slow < 40.0)
    base = gs[steady] if steady.sum() >= 20 else gs[cov & keep0] if (cov & keep0).any() \
        else gs
    ref = np.median(base, axis=0)
    ref = ref / max(float(np.linalg.norm(ref)), 1e-9)
    tilt = np.degrees(np.arccos(np.clip(gs @ ref, -1.0, 1.0)))

    yaw = np.degrees(np.interp(c, t_o, np.unwrap(o["yaw"][oo])))
    resid = _yaw_travel_resid(travel, yaw, keep0 & cov & (speed > 0.6))

    return {
        "n_slices": n,
        "keep0": keep0,
        "lat": lat.astype(np.float64),
        "lon": lon.astype(np.float64),
        "t_rel_s": ((c - c[0]) / 1e9).astype(np.float32),
        "t_end_s": ((c[-1] - c) / 1e9).astype(np.float32),
        "speed_mps": speed.astype(np.float32),
        "hacc_m": hacc.astype(np.float32),
        "gps_resid_m": gps_resid.astype(np.float32),
        "tilt_deg": np.where(cov, tilt, np.nan).astype(np.float32),
        "omega_dps": np.where(cov, omega, np.nan).astype(np.float32),
        "omega_slow_dps": np.where(cov, omega_slow, np.nan).astype(np.float32),
        "orient_cov": cov,
        "scalars": {
            "duration_s": round(float((c[-1] - c[0]) / 1e9), 1),
            "n_fixes": int(len(t_l)),
            "hacc_p50": round(float(np.median(loc["horizontalAccuracy"])), 1),
            "hacc_max": round(float(loc["horizontalAccuracy"].max()), 1),
            "yaw_travel_resid_deg": round(resid, 1) if resid is not None else None,
            "hand_carried": bool(resid is not None and resid < 45.0),
            "tilt_p50": round(float(np.nanmedian(tilt[cov])), 1) if cov.any() else None,
            "orient_cov_frac": round(float(cov.mean()), 3),
        },
    }


def build_telemetry(cli):
    """Feature tables for all 91 sessions, cached to one npz + one JSON."""
    npz_path = os.path.join(cli.out_dir, f"telemetry_{cli.tag}.npz")
    json_path = os.path.join(cli.out_dir, f"telemetry_{cli.tag}.json")
    if os.path.exists(npz_path) and os.path.exists(json_path) and not cli.force:
        with open(json_path) as f:
            meta = json.load(f)
        print(f"telemetry: cached ({len(meta['sessions'])} sessions)")
        return dict(np.load(npz_path, allow_pickle=False)), meta

    os.makedirs(cli.out_dir, exist_ok=True)
    sessions = [("database", sid, os.path.join(cli.root, "database", sid))
                for sid in DB_ARMS]
    sessions += [(sweep, sid, sdir)
                 for sweep, sid, sdir in discover_queries(cli.root, None, SWEEPS)]
    arrays, meta = {}, {"tag": cli.tag, "sessions": {}}
    for n, (role, sid, sdir) in enumerate(sessions):
        print(f"[{n + 1}/{len(sessions)}] {role}/{sid}")
        f = session_features(cli, sdir, sid)
        for k, v in f.items():
            if k not in ("scalars", "n_slices"):
                arrays[f"{sid}|{k}"] = v
        meta["sessions"][sid] = {"role": role, "n_slices": f["n_slices"],
                                 "n_keep0": int(f["keep0"].sum()), **f["scalars"]}
    np.savez_compressed(npz_path, **arrays)
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"-> {npz_path}\n-> {json_path}")
    return arrays, meta


def feat(arrays, sid, name):
    return arrays[f"{sid}|{name}"]


def attitude_trustworthy(arrays, meta, sid, p):
    """Whether this session's phone attitude can be read as operator handling at all.

    The phone was never on the rig, so a swing only means "the operator picked it up" if
    the phone was otherwise sitting still. The lower quartile of tilt says whether it
    was: carried in a pocket the walking gait alone never lets it settle — all seven
    database passes sit at 7.8-12.0 deg, against 0.3-5.3 for a query held in hand. Six
    day queries fall in the pocket band too, and are treated exactly like the database:
    no attitude rule at all, rather than one built on gait. Deciding this per session
    from the data beats trusting the role, because the carry mode genuinely varied.

    A yaw-to-travel fit cannot do this job: a phone in a trouser pocket still points
    along the walk, so it scores 14-32 deg residual, as good as a hand-held one.
    """
    if meta["sessions"][sid]["role"] == "database":
        return False
    t = feat(arrays, sid, "tilt_deg")
    k = feat(arrays, sid, "keep0") & feat(arrays, sid, "orient_cov")
    t = t[k & np.isfinite(t)]
    if len(t) < 20:
        return False
    return bool(np.percentile(t, 25) < p["attitude_stable_p25_deg"])


# ---------------------------------------------------------------------------
# 2. Stage: frames — camera-side quality, which the phone cannot see
# ---------------------------------------------------------------------------
# The phone was in a pocket or a hand, never on the rig, so no amount of telemetry
# reveals what the camera did. These are statistics of the *rendered slice* — the exact
# array the model is fed — and nothing more: no descriptor, no learned weight, no
# retrieval outcome enters them, so a rule built on them still keeps the model out of the
# curation loop. On the accumulate representation the background is white, so "deviation
# from white" is event activity.
FRAME_COLS = ("active_frac", "strong_frac", "mean_dev", "lap_var", "sharp_ratio")
FRAME_STATS_VERSION = 2       # bump to invalidate every cached per-partition scan


def _stats_from_d(d):
    """The five statistics, given the accumulate frame's deviation-from-white map.

    * ``active_frac`` — fraction of pixels carrying any event. Near zero is a blank
      frame: the single worst alias magnet in this gallery (162 wrong top-1s, never
      right) is one of those.
    * ``strong_frac`` / ``sharp_ratio`` — how concentrated that activity is. A swung
      camera spreads an edge over dozens of pixels, so its energy survives while its
      high-frequency content collapses; a sharp frame keeps both. ``sharp_ratio``
      divides by ``mean_dev**2`` so it measures concentration, not how much happened.
    """
    lap = (4.0 * d[1:-1, 1:-1] - d[:-2, 1:-1] - d[2:, 1:-1]
           - d[1:-1, :-2] - d[1:-1, 2:])
    mean_dev = float(d.mean())
    lap_var = float(lap.var())
    active = d > 0.05
    # 0.6, not 0.5: the normalisation clips at the 99th percentile of nonzero counts, so
    # on a sparse slice (clip = 2) every single-event pixel lands on exactly 0.5 and a
    # threshold there splits a mass point — the same frame reads 0.02 or 0.06 depending
    # on whether a uint8 round tips it. 0.6 falls between the 1/2 and 2/3 levels instead.
    n_act = int(active.sum())
    return (float(active.mean()), float((d > 0.6).sum() / max(n_act, 1)), mean_dev,
            lap_var, float(lap_var / max(mean_dev * mean_dev, 1e-9)))


def frame_stats(frame):
    """Statistics of an already-rendered ``[3,H,W]`` accumulate frame in [0,1]."""
    return _stats_from_d(1.0 - frame.min(axis=0))


def _norm99(a):
    """``src.npzdata.accumulate_numpy``'s per-polarity normalisation, op for op."""
    if a.max() == 0:
        return a
    thr = np.percentile(a[a > 0], 99.0) if np.any(a > 0) else 1.0
    if thr <= 0:
        thr = float(a.max())
    return np.clip(a, 0, thr) / thr


def frame_stats_from_events(x, y, p, height, width):
    """The same five statistics, straight from one slice's events.

    Rendering the frame first costs 76% of the scan's wall time, nearly all of it in
    ``accumulate_numpy``'s two ``np.add.at`` calls, and the gallery is ~800 GB of events
    on a spinning disk. This skips the image without approximating it: in that renderer
    green is the only channel both polarities darken, so ``1 - min(R,G,B)`` is always
    ``1 - G``, which is exactly ``max(pos_n, neg_n)`` — the winner-take-all step is a
    pixelwise maximum once you write it out. Building the two polarity histograms with
    ``bincount`` instead of ``add.at`` gives that array identically, so the only
    difference from the rendered path is the uint8 quantisation this one does not do.
    """
    m = (x < width) & (y < height) & (x >= 0) & (y >= 0)
    idx = (y[m].astype(np.int64) * width + x[m].astype(np.int64))
    pol = p[m] > 0
    n = height * width
    pos = np.bincount(idx[pol], minlength=n).astype(np.float32).reshape(height, width)
    neg = np.bincount(idx[~pol], minlength=n).astype(np.float32).reshape(height, width)
    return _stats_from_d(np.maximum(_norm99(pos), _norm99(neg)))


def _frames_manifest(cli, n_slices):
    return {"n_slices": int(n_slices), "representation": cli.representation,
            "dt_ms": cli.dt_ms, "hot_pixel": not cli.no_hot_pixel,
            "filter_dt_us": None if cli.no_event_filter else cli.event_filter_dt_us,
            "cols": list(FRAME_COLS), "version": FRAME_STATS_VERSION}


def _frames_stem(cli, sid, path):
    return os.path.join(cli.bank_dir, sid,
                        f"framestats_{cli.tag}_"
                        f"{os.path.splitext(os.path.basename(path))[0]}")


def _frames_worker(job):
    """One partition -> ``[n_slices, len(FRAME_COLS)]`` float32, cached beside the bank."""
    import numpy as _np
    from torchvision import transforms as _tf
    cli, sid, path = job
    stem = _frames_stem(cli, sid, path)
    if os.path.exists(stem + ".npy") and os.path.exists(stem + ".json"):
        with open(stem + ".json") as f:
            cached = json.load(f)
        if (cached.get("cols") == list(FRAME_COLS)
                and cached.get("version") == FRAME_STATS_VERSION):
            return sid, os.path.basename(path), int(_np.load(stem + ".npy").shape[0]), 1
    from springfield_eval import make_dataset as _mk
    ds = _mk(path, _tf.Compose([]), cli.representation, cli.dt_ms,
             not cli.no_hot_pixel,
             None if cli.no_event_filter else cli.event_filter_dt_us)
    out = _np.empty((len(ds), len(FRAME_COLS)), _np.float32)
    # Take the events straight from eventcv's reader where the local accumulate renderer
    # is in play — same reader, same filters, same slice indices, just without paying for
    # an image nothing looks at. Anything else keeps the rendered path.
    inner = getattr(ds.reader, "_reader", None)
    W, H = ds.sensor
    if inner is not None and hasattr(inner, "slice"):
        import eventcv as _ecv
        for i in range(len(ds)):
            ev = _ecv.numpy(inner.slice(i))
            out[i] = frame_stats_from_events(ev[:, 0], ev[:, 1], ev[:, 3], H, W)
    else:
        for i in range(len(ds)):
            out[i] = frame_stats(ds[i].numpy())
    _np.save(stem + ".npy", out)
    with open(stem + ".json", "w") as f:
        json.dump(_frames_manifest(cli, len(ds)), f)
    return sid, os.path.basename(path), len(ds), 0


def stage_frames(cli):
    import multiprocessing as mp
    sessions = []
    roles = [r.strip() for r in cli.roles.split(",") if r.strip()]
    if "database" in roles:
        sessions += [(sid, os.path.join(cli.root, "database", sid)) for sid in DB_ARMS]
    qsweeps = [r for r in roles if r in SWEEPS]
    if qsweeps:
        sessions += [(sid, sdir)
                     for _, sid, sdir in discover_queries(cli.root, None, qsweeps)]
    jobs = [(cli, sid, p) for sid, sdir in sessions for p in list_partitions(sdir)]
    # Scanning the whole gallery is hours of spinning disk, but the question it answers
    # is decided by a small minority of rows: 0.23% of the gallery absorbs 47% of all
    # wrong top-1s. Doing the partitions that hold those rows first means the verdict on
    # a gallery-side rule lands early, and the remainder only firms up the mask.
    if cli.magnets_first and os.path.exists(
            os.path.join(cli.diag_dir, f"dump_{cli.tag}.npz")):
        d = np.load(os.path.join(cli.diag_dir, f"dump_{cli.tag}.npz"), allow_pickle=False)
        top1 = d["top_i"][0]
        bad = np.linalg.norm(d["db_xy"][top1] - d["q_xy"], axis=1) > cli.threshold_m
        cnt = np.bincount(top1[bad], minlength=len(d["db_xy"]))
        weight = {}
        for sid, part, n in zip(d["db_sid"], d["db_part"], cnt):
            if n:
                k = (str(sid), str(part))
                weight[k] = weight.get(k, 0) + int(n)
        jobs.sort(key=lambda j: -weight.get((j[1], os.path.basename(j[2])), 0))
        top = [(j[1][9:15], os.path.basename(j[2])[8:13],
                weight.get((j[1], os.path.basename(j[2])), 0)) for j in jobs[:5]]
        print("magnet-first order; leading partitions (session, part, wrong top-1s): "
              + ", ".join(f"{a}/{b}={c}" for a, b, c in top))
    print(f"frame stats: {len(jobs)} partitions over {len(sessions)} sessions, "
          f"{cli.workers} workers", flush=True)
    # The gallery is ~800 GB of events on a spinning disk, so this is IO bound and the
    # worker count is a seek-thrashing tradeoff, not a core count: five concurrent
    # readers ran slower in aggregate than one did alone. The running rate is logged so
    # the tradeoff can be re-checked rather than guessed at.
    import time
    t0 = time.time()
    done = n_slices = 0
    with mp.get_context("fork").Pool(cli.workers) as pool:
        for sid, part, n, cached in pool.imap_unordered(_frames_worker, jobs):
            done += 1
            n_slices += 0 if cached else n
            el = time.time() - t0
            print(f"  [{done}/{len(jobs)}] {sid}/{part}: {n} slices"
                  f"{' (cached)' if cached else ''}  |  {el / 60:.1f} min, "
                  f"{n_slices / max(el, 1):.1f} slices/s, eta "
                  f"{(len(jobs) - done) * el / max(done, 1) / 3600:.1f} h", flush=True)
    print("frame stats complete", flush=True)


def load_frame_stats(cli, sid, sdir, by_part=False):
    """Frame stats for one session, or ``None`` if it has not been scanned.

    ``by_part`` keys them by partition basename — the dump records database provenance as
    (session, partition, *partition-local* index), so that is the form it needs.
    """
    out, parts = {}, list_partitions(sdir)
    for p in parts:
        stem = _frames_stem(cli, sid, p)
        if not os.path.exists(stem + ".npy"):
            return None
        out[os.path.basename(p)] = np.load(stem + ".npy")
    if not out:
        return None
    return out if by_part else np.concatenate([out[os.path.basename(p)] for p in parts])


# ---------------------------------------------------------------------------
# 3. Stage: evidence — measure every candidate rule before adopting it
# ---------------------------------------------------------------------------
def _query_frame(cli, arrays, meta):
    """Per query slice: telemetry joined to the cached retrieval outcome.

    Rows are the dump's rows (the kept slices, in the eval's own order), so ``hit``
    is exactly the number the headline is made of.
    """
    d = np.load(os.path.join(cli.diag_dir, f"dump_{cli.tag}.npz"), allow_pickle=False)
    o = np.load(os.path.join(cli.diag_dir, f"orient_{cli.tag}.npz"), allow_pickle=False)
    sids = [str(s) for s in d["sids"]]
    sweeps = [str(s) for s in d["sweeps"]]
    q_sid, q_slice, q_xy = d["q_sid"], d["q_slice"], d["q_xy"]
    hit = (np.linalg.norm(d["db_xy"][d["top_i"][0]] - q_xy, axis=1) <= cli.threshold_m)

    cols = {}
    for name in ("t_rel_s", "t_end_s", "speed_mps", "hacc_m", "gps_resid_m",
                 "tilt_deg", "omega_dps", "omega_slow_dps"):
        v = np.empty(len(q_sid), np.float32)
        for i, sid in enumerate(sids):
            m = q_sid == i
            v[m] = feat(arrays, sid, name)[q_slice[m]]
        cols[name] = v
    cols["hit"] = hit
    cols["bc_rank"] = d["bc_rank"]
    cols["sweep"] = np.array([sweeps[i] for i in q_sid])
    cols["sid"] = np.array([sids[i] for i in q_sid])
    cols["psi"] = np.concatenate([o[f"{s}_psi"] for s in sids])
    trust = {s: attitude_trustworthy(arrays, meta, s, cli.profile) for s in sids}
    cols["hand"] = np.array([trust[s] for s in cols["sid"]])

    # camera-side statistics, if the frames stage has run for these sessions
    missing = []
    fs = np.full((len(q_sid), len(FRAME_COLS)), np.nan, np.float32)
    for i, sid in enumerate(sids):
        sdir = os.path.join(cli.root, f"query_{sweeps[i]}", sid)
        a = load_frame_stats(cli, sid, sdir)
        if a is None:
            missing.append(sid)
            continue
        m = q_sid == i
        fs[m] = a[q_slice[m]]
    for j, name in enumerate(FRAME_COLS):
        cols[name] = fs[:, j]
    if missing:
        print(f"  frame stats missing for {len(missing)} query session(s) — run "
              f"--stage frames; their rows are NaN and drop out of those tests")
    return cols, sids, sweeps


def db_frame_stats(cli, d):
    """``[n_db, len(FRAME_COLS)]`` aligned to the dump's gallery rows; NaN where unscanned.

    Partial coverage is returned rather than refused: the scan is hours of spinning disk
    and the gallery verdict is legible long before it finishes, so every caller reports
    the covered fraction next to the numbers it computes from it.
    """
    per_part = {}
    for sid in DB_ARMS:
        for p in list_partitions(os.path.join(cli.root, "database", sid)):
            stem = _frames_stem(cli, sid, p)
            if os.path.exists(stem + ".npy"):
                per_part[(sid, os.path.basename(p))] = np.load(stem + ".npy")
    if not per_part:
        return None
    db_sid = [str(s) for s in d["db_sid"]]
    db_part = [str(s) for s in d["db_part"]]
    db_slice = d["db_slice"]
    out = np.full((len(db_sid), len(FRAME_COLS)), np.nan, np.float32)
    for i in range(len(db_sid)):
        a = per_part.get((db_sid[i], db_part[i]))
        if a is not None and db_slice[i] < len(a):
            out[i] = a[db_slice[i]]
    return out


def _bin_curve(x, hit, edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        out.append({"lo": float(a), "hi": float(b), "n": int(m.sum()),
                    "r1": round(float(hit[m].mean()), 4) if m.any() else None})
    return out


def _rule_test(name, drop, hit, base):
    """Recall of what a rule removes vs what it keeps, inside the reference cell."""
    dm, km = drop & base, (~drop) & base
    r_drop = float(hit[dm].mean()) if dm.any() else None
    r_keep = float(hit[km].mean()) if km.any() else None
    return {"rule": name, "n_removed": int(dm.sum()),
            "frac_removed": round(float(dm.sum() / max(base.sum(), 1)), 4),
            "r1_removed": None if r_drop is None else round(r_drop, 4),
            "r1_kept": None if r_keep is None else round(r_keep, 4),
            "gap": None if (r_drop is None or r_keep is None)
                   else round(r_keep - r_drop, 4)}


def stage_evidence(cli):
    arrays, meta = build_telemetry(cli)
    cols, sids, sweeps = _query_frame(cli, arrays, meta)
    hit = cols["hit"]
    p = cli.profile

    # Reference cell for threshold-setting: day+dawn, and not against-route. Against
    # cells are ~0 by construction (the camera is 90 deg minimum from every database arm
    # on a reversed pass), so including them would let any rule look like it removes junk.
    ref = (np.isin(cols["sweep"], ("day", "dawn")) & (np.abs(cols["psi"]) < 135.0))
    print(f"\nreference cell (day+dawn, |psi|<135): {int(ref.sum())} of {len(hit)} "
          f"slices, R@1 {hit[ref].mean():.3f}")

    ev = {"tag": cli.tag, "n_query_slices": int(len(hit)),
          "reference_cell": {"n": int(ref.sum()), "r1": round(float(hit[ref].mean()), 4)},
          "profile": p}

    # --- curves -------------------------------------------------------------------
    ev["curves"] = {
        "t_from_start_s": _bin_curve(cols["t_rel_s"], hit,
                                     np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 9, 15, 1e9])),
        "t_to_end_s": _bin_curve(cols["t_end_s"], hit,
                                 np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 9, 15, 1e9])),
        "hacc_m": _bin_curve(cols["hacc_m"], hit,
                             np.array([0, 4, 5, 6, 8, 10, 15, 20, 30, 60, 1e9])),
        "gps_resid_m": _bin_curve(cols["gps_resid_m"], hit,
                                  np.array([0, 1, 2, 4, 8, 12, 20, 40, 1e9])),
        "tilt_deg": _bin_curve(cols["tilt_deg"], hit,
                               np.array([0, 5, 10, 15, 20, 30, 45, 60, 90, 1e9])),
        "omega_dps": _bin_curve(cols["omega_dps"], hit,
                                np.array([0, 20, 40, 60, 100, 150, 200, 300, 1e9])),
        "speed_mps": _bin_curve(cols["speed_mps"], hit,
                                np.array([0, .2, .4, .6, .9, 1.2, 1.5, 1.8, 1e9])),
    }
    # the same curves inside the reference cell — these are what set thresholds
    ev["curves_ref"] = {
        k: _bin_curve(cols[k if k != "t_from_start_s" else "t_rel_s"][ref]
                      if k != "t_to_end_s" else cols["t_end_s"][ref],
                      hit[ref], np.array([e["lo"] for e in v] + [v[-1]["hi"]]))
        for k, v in ev["curves"].items() if k not in ("t_from_start_s", "t_to_end_s")}
    ev["curves_ref"]["t_from_start_s"] = _bin_curve(
        cols["t_rel_s"][ref], hit[ref],
        np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 9, 15, 1e9]))
    ev["curves_ref"]["t_to_end_s"] = _bin_curve(
        cols["t_end_s"][ref], hit[ref],
        np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 9, 15, 1e9]))

    # --- candidate rules ----------------------------------------------------------
    # Boundary and interior tilt are tested apart on purpose: at the ends a swing is the
    # operator working the phone (start/stop), in the middle it is just as likely to be
    # a glance at the screen while the rig kept filming cleanly. Only the evidence can
    # tell them apart, so neither inherits the other's justification.
    hand = cols["hand"]
    edge = (cols["t_rel_s"] < 2.0) | (cols["t_end_s"] < 2.0)
    tilt = cols["tilt_deg"]
    tests = [
        _rule_test(f"gps_hacc>{p['hacc_max_m']:g}m", cols["hacc_m"] > p["hacc_max_m"],
                   hit, ref),
        _rule_test("gps_hacc>15m", cols["hacc_m"] > 15.0, hit, ref),
        _rule_test("gps_hacc>20m", cols["hacc_m"] > 20.0, hit, ref),
        _rule_test(f"gps_resid>{p['gps_resid_max_m']:g}m",
                   cols["gps_resid_m"] > p["gps_resid_max_m"], hit, ref),
        _rule_test("start<1.0s", cols["t_rel_s"] < 1.0, hit, ref),
        _rule_test("start<2.0s", cols["t_rel_s"] < 2.0, hit, ref),
        _rule_test("end<1.0s", cols["t_end_s"] < 1.0, hit, ref),
        _rule_test("end<2.0s", cols["t_end_s"] < 2.0, hit, ref),
        _rule_test("tilt>35 at edge (<2s)", hand & edge & (tilt > 35.0), hit, ref),
        _rule_test("tilt>35 interior", hand & ~edge & (tilt > 35.0), hit, ref),
        _rule_test(f"tilt>{p['tilt_max_deg']:g} interior",
                   hand & ~edge & (tilt > p["tilt_max_deg"]), hit, ref),
        _rule_test("tilt>60 interior", hand & ~edge & (tilt > 60.0), hit, ref),
        _rule_test("omega>200dps", cols["omega_dps"] > 200.0, hit, ref),
        _rule_test(f"omega>{p['omega_max_dps']:g}dps",
                   cols["omega_dps"] > p["omega_max_dps"], hit, ref),
        _rule_test("speed<0.35 m/s", cols["speed_mps"] < 0.35, hit, ref),
    ]
    if not np.all(np.isnan(cols["active_frac"])):
        af, sr = cols["active_frac"], cols["sharp_ratio"]
        for thr in (0.005, 0.01, 0.02, 0.04, 0.08):
            tests.append(_rule_test(f"active_frac<{thr:g}", af < thr, hit, ref))
        for q in (1, 2, 5, 10):
            t = float(np.nanpercentile(sr, q))
            tests.append(_rule_test(f"sharp_ratio<p{q} ({t:.1f})", sr < t, hit, ref))
        ev["curves"]["active_frac"] = _bin_curve(
            af, hit, np.array([0, .005, .01, .02, .04, .08, .15, .3, 1.0]))
        ev["curves"]["sharp_ratio"] = _bin_curve(
            sr, hit, np.nanpercentile(sr, [0, 1, 2, 5, 10, 25, 50, 75, 100]))
    ev["rule_tests"] = tests
    print("\ncandidate rules (reference cell): "
          "removed / R@1 of removed vs kept / gap")
    for t in tests:
        print(f"  {t['rule']:24s} n={t['n_removed']:6d} ({t['frac_removed']:6.1%})  "
              f"removed {str(t['r1_removed']):6s}  kept {str(t['r1_kept']):6s}  "
              f"gap {t['gap']}")

    # --- database-side mass: how much of the gallery each rule would take -----------
    db = {}
    for sid in DB_ARMS:
        k0 = feat(arrays, sid, "keep0")
        db[sid] = {
            "n_keep0": int(k0.sum()),
            "hacc": int((k0 & (feat(arrays, sid, "hacc_m") > p["hacc_max_m"])).sum()),
            "resid": int((k0 & (feat(arrays, sid, "gps_resid_m")
                                > p["gps_resid_max_m"])).sum()),
        }
    ev["database_mass"] = db
    print("\ndatabase mass a GPS rule would touch (kept slices):")
    for sid, v in db.items():
        print(f"  {DB_ARMS[sid][0]:11s} {sid}  keep0 {v['n_keep0']:6d}  "
              f"hacc {v['hacc']:5d}  resid {v['resid']:5d}")

    ev["db_absorption"] = stage_db_absorption(cli, ev, p)
    ev["sessions"] = meta["sessions"]

    os.makedirs(cli.out_dir, exist_ok=True)
    out = os.path.join(cli.out_dir, f"evidence_{cli.tag}.json")
    with open(out, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\n-> {out}")
    figure_evidence(ev, os.path.join(cli.out_dir, f"evidence_{cli.tag}.png"))


def stage_db_absorption(cli, ev, p):
    """What a gallery-side rule would absorb, and what it would cost.

    A database row cannot be scored the way a query slice can — removing it changes the
    retrieval for every query, so there is no held-out recall to compare. The honest
    paired measure is what the row is *doing*: how many of the dataset's wrong top-1s
    land on it, against how many of its correct top-1s would be given up. A junk row
    absorbs wrong matches and returns nothing; a hard-but-real place does both.
    """
    d = np.load(os.path.join(cli.diag_dir, f"dump_{cli.tag}.npz"), allow_pickle=False)
    fs = db_frame_stats(cli, d)
    top1 = d["top_i"][0]
    miss = np.linalg.norm(d["db_xy"][top1] - d["q_xy"], axis=1) > cli.threshold_m
    n_db = len(d["db_xy"])
    cnt_bad = np.bincount(top1[miss], minlength=n_db)
    cnt_good = np.bincount(top1[~miss], minlength=n_db)

    # concentration: this is what says whether a gallery rule can matter at all
    order = np.argsort(-cnt_bad)
    conc = {str(k): round(float(cnt_bad[order[:k]].sum() / max(cnt_bad.sum(), 1)), 4)
            for k in (10, 50, 100, 300, 1000, 3000)}
    out = {"n_db": int(n_db), "n_miss": int(miss.sum()),
           "wrong_top1_concentration": conc,
           "rows_ever_top1": int((cnt_bad + cnt_good > 0).sum())}
    print(f"\ngallery: {int(miss.sum())} wrong top-1s; the worst 300 rows "
          f"({300 / n_db:.2%} of it) absorb {conc['300']:.1%} of them")
    if fs is None:
        print("  (no database frame stats yet — run --stage frames --roles database)")
        return out

    seen = np.isfinite(fs[:, 0])
    out["frame_stat_coverage"] = round(float(seen.mean()), 4)
    out["wrong_top1_coverage"] = round(float(cnt_bad[seen].sum()
                                             / max(cnt_bad.sum(), 1)), 4)
    print(f"  frame stats cover {seen.mean():.1%} of gallery rows, holding "
          f"{out['wrong_top1_coverage']:.1%} of the wrong top-1s")
    if seen.sum() < 1000:
        return out

    def test(name, flag):
        """Rows are only ever counted against the *scanned* population."""
        flag = flag & seen
        return {"rule": name, "n_rows": int(flag.sum()),
                "frac_scanned": round(float(flag.sum() / max(seen.sum(), 1)), 5),
                "wrong_absorbed": round(float(cnt_bad[flag].sum()
                                              / max(cnt_bad[seen].sum(), 1)), 4),
                "right_given_up": round(float(cnt_good[flag].sum()
                                              / max(cnt_good[seen].sum(), 1)), 4)}

    af, sr = fs[:, 0], fs[:, 4]
    tests = []
    for thr in (0.005, 0.01, 0.02, 0.04, 0.08):
        tests.append(test(f"db active_frac<{thr:g}", af < thr))
    for q in (1, 2, 5, 10):
        thr = float(np.nanpercentile(sr, q))
        tests.append(test(f"db sharp_ratio<p{q} ({thr:.1f})", sr < thr))
    out["rule_tests"] = tests
    print("  candidate gallery rules (of scanned): rows / % / % wrong absorbed / "
          "% right lost")
    for t in tests:
        print(f"    {t['rule']:28s} {t['n_rows']:6d} {t['frac_scanned']:7.3%} "
              f"{t['wrong_absorbed']:8.2%} {t['right_given_up']:8.2%}")

    # what the magnets look like statistically, against the gallery as a whole
    mag = np.zeros(n_db, bool)
    mag[order[:300]] = True
    mag &= seen
    if mag.sum() >= 20:
        out["magnet_profile"] = {
            c: {"n_magnets": int(mag.sum()),
                "magnets_p50": round(float(np.median(fs[mag, j])), 5),
                "gallery_p50": round(float(np.nanmedian(fs[seen, j])), 5),
                "gallery_p05": round(float(np.nanpercentile(fs[seen, j], 5)), 5)}
            for j, c in enumerate(FRAME_COLS)}
        print(f"  top-300 magnets ({int(mag.sum())} scanned) vs gallery, median:")
        for c, v in out["magnet_profile"].items():
            print(f"    {c:12s} magnets {v['magnets_p50']:10.4f}   gallery "
                  f"{v['gallery_p50']:10.4f}  (gallery p05 {v['gallery_p05']:10.4f})")
    return out


def figure_evidence(ev, out_png):
    """The threshold-setting curves: R@1 against each capture-quality signal."""
    keys = ["t_from_start_s", "t_to_end_s", "hacc_m", "gps_resid_m", "tilt_deg",
            "omega_dps"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), constrained_layout=True)
    for ax, k in zip(axes.ravel(), keys):
        for src, style in (("curves", dict(color="#bbbbbb", lw=1.2, label="all")),
                           ("curves_ref", dict(color="#2b6cb0", lw=2.0,
                                               label="day+dawn, not against"))):
            b = [e for e in ev[src][k] if e["r1"] is not None]
            x = [0.5 * (e["lo"] + min(e["hi"], e["lo"] * 2 + 5)) for e in b]
            ax.plot(x, [e["r1"] for e in b], marker="o", ms=3, **style)
        ax.set_title(k, fontsize=9, loc="left")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0, 0].legend(fontsize=7, loc="best")
    for ax in axes[:, 0]:
        ax.set_ylabel("R@1", fontsize=8)
    fig.suptitle("Springfield curation evidence: per-slice R@1 vs capture-quality signal",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out_png}")


# ---------------------------------------------------------------------------
# 3. Stage: rules — build the mask
# ---------------------------------------------------------------------------
def _runs(mask, min_run):
    """Start/stop index pairs of ``True`` runs at least ``min_run`` long."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_run:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _dilate(mask, k):
    if k <= 0:
        return mask
    out = mask.copy()
    for s in range(1, k + 1):
        out[s:] |= mask[:-s]
        out[:-s] |= mask[s:]
    return out


def _boundary_trim(bad, keep0, max_frac):
    """Leading and trailing runs of ``bad``, capped at a fraction of the session.

    Only the ends are trimmed: a stop in the middle of a route is a legitimate pause,
    the same signal at the very start is the operator still setting up.
    """
    n = len(bad)
    out = np.zeros(n, bool)
    cap = max(1, int(round(max_frac * n)))
    live = np.flatnonzero(keep0)
    if not len(live):
        return out
    a, b = live[0], live[-1]
    i = a
    while i <= b and bad[i] and i - a < cap:
        out[i] = True
        i += 1
    j = b
    while j >= a and bad[j] and b - j < cap:
        out[j] = True
        j -= 1
    return out


def _settle_trim(x, gate, keep0, hold_s, dt_ms, max_frac):
    """Leading/trailing trim extended until ``x`` has held under ``gate`` for ``hold_s``.

    A GPS receiver that reports 25 m, then 4 m, then 25 m again has not converged; the
    hold requirement stops the trim ending on the first lucky sample.
    """
    n = len(x)
    hold = max(1, int(round(hold_s * 1000.0 / dt_ms)))
    bad = x > gate
    ok = ~bad
    # position i qualifies as "settled" if the next `hold` samples are all under the gate
    csum = np.concatenate([[0], np.cumsum(ok)])
    settled = np.zeros(n, bool)
    for i in range(n):
        j = min(i + hold, n)
        settled[i] = (csum[j] - csum[i]) == (j - i)
    out = np.zeros(n, bool)
    live = np.flatnonzero(keep0)
    if not len(live):
        return out
    a, b = live[0], live[-1]
    cap = max(1, int(round(max_frac * n)))
    i = a
    while i <= b and not settled[i] and i - a < cap:
        out[i] = True
        i += 1
    j = b
    while j >= a and bad[j] and b - j < cap:
        out[j] = True
        j -= 1
    return out


def session_rules(sid, role, arrays, meta, p, fstats=None):
    """``{reason_code: bool mask}`` on the full slice grid for one session."""
    k0 = feat(arrays, sid, "keep0")
    hacc = feat(arrays, sid, "hacc_m")
    resid = feat(arrays, sid, "gps_resid_m")
    speed = feat(arrays, sid, "speed_mps")
    tilt = feat(arrays, sid, "tilt_deg")
    omega = feat(arrays, sid, "omega_dps")
    hand = attitude_trustworthy(arrays, meta, sid, p)
    dt_ms = meta.get("dt_ms", 50)
    r = {}

    # 1. GPS validity — the slice's own position must be able to adjudicate a 25 m match
    r["gps_hacc"] = hacc > p["hacc_max_m"]
    r["gps_warmup"] = (_settle_trim(hacc, p["hacc_max_m"], k0, p["hacc_settle_s"],
                                    dt_ms, p["boundary_max_frac"])
                       if p["hacc_settle_s"] > 0 else np.zeros(len(k0), bool))
    r["gps_outlier"] = resid > p["gps_resid_max_m"]

    # 2. Boundary handling — the operator starting and stopping the capture. The
    # predicate follows the carry mode: a hand-held phone shows the start/stop swing in
    # its attitude, a pocketed one (all seven database passes) shows nothing, so there
    # only the walk itself is evidence and the trim runs on speed. A gate of 0 means the
    # rule is off; it cannot mean "trim everything above zero tilt", which is what a bare
    # `>` comparison would do to every slice in the session.
    off = np.zeros(len(k0), bool)
    gate = p["boundary_tilt_deg"] if hand else p["boundary_speed_mps"]
    if gate > 0:
        bad = (np.nan_to_num(tilt, nan=0.0) > gate) if hand else (speed < gate)
        r["boundary"] = _boundary_trim(bad, k0, p["boundary_max_frac"])
    else:
        r["boundary"] = off.copy()

    # 3. Mid-session attitude excursions, and extreme angular rate. Both read the phone,
    # so both are restricted to the sessions where the phone was in the operator's hand;
    # on a pocketed database pass the same numbers describe a walking gait and a trouser
    # pocket, and gating only one of the two would quietly curate the gallery on noise.
    if hand and p["tilt_max_deg"] > 0:
        hi = np.nan_to_num(tilt, nan=0.0) > p["tilt_max_deg"]
        run = off.copy()
        for a, b in _runs(hi, int(p["tilt_min_run"])):
            run[a:b] = True
        r["tilt"] = _dilate(run, int(p["tilt_dilate"]))
    else:
        r["tilt"] = off.copy()
    if hand and p["omega_max_dps"] > 0:
        r["smear"] = _dilate(np.nan_to_num(omega, nan=0.0) > p["omega_max_dps"],
                             int(p["tilt_dilate"]))
    else:
        r["smear"] = off.copy()

    # 4. Camera-side quality, the only signal that actually saw through the lens. The
    # gallery gets its own (stricter or looser) gates because a junk row there is not
    # merely an unscorable slice, it is an attractor for every query that passes it.
    if fstats is not None and len(fstats) == len(k0):
        af_min = p["db_active_frac_min"] if role == "database" else p["active_frac_min"]
        sr_min = p["db_sharp_ratio_min"] if role == "database" else p["sharp_ratio_min"]
        r["blank"] = (fstats[:, 0] < af_min) if af_min else np.zeros(len(k0), bool)
        r["blurred"] = (fstats[:, 4] < sr_min) if sr_min else np.zeros(len(k0), bool)
    else:
        r["blank"] = np.zeros(len(k0), bool)
        r["blurred"] = np.zeros(len(k0), bool)

    # only ever report drops among slices the eval was scoring in the first place
    return {k: (v & k0) for k, v in r.items()}


def forest_flags(cli, arrays, meta, p):
    """Label (never drop) the query slices whose scene barely changes as the walk goes on.

    A vegetation corridor looks the same 20 m further along, which is what makes it
    unlocalizable — so the measure is a slice's mean descriptor similarity to the *other*
    slices of its own pass at least ``forest_baseline_m`` further down the path. This is
    the one signal here computed from descriptors, so by the user's decision (2026-09-01)
    it is reporting only: these slices stay in the query set and get their own split in
    the results, exactly so the paper can say how much of the route is like this and how
    the model does there.
    """
    base_m = 10.0
    out = {}
    for sid, s in meta["sessions"].items():
        if s["role"] == "database":
            continue
        sdir = os.path.join(cli.root, f"query_{s['role']}", sid)
        cs = load_cached_session(cli, sdir, sid, need_bank=True)
        k = np.flatnonzero(cs["keep"])
        if len(k) < 8:
            continue
        b = cs["bank"][k]
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
        lat, lon = feat(arrays, sid, "lat")[k], feat(arrays, sid, "lon")[k]
        xy = local_enu(lat, lon, float(np.median(lat)), float(np.median(lon)))
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0),
                                                              axis=1))])
        far = np.abs(cum[:, None] - cum[None, :]) >= base_m
        sim = b @ b.T
        n_far = far.sum(axis=1)
        with np.errstate(invalid="ignore"):
            self_sim = np.where(n_far > 0, (sim * far).sum(axis=1)
                                / np.maximum(n_far, 1), np.nan)
        idx = k[self_sim > p["self_sim_flag"]]
        out[sid] = {"role": s["role"], "n_kept": int(len(k)),
                    "self_sim_p50": (round(float(np.nanmedian(self_sim)), 4)
                                     if np.isfinite(self_sim).any() else None),
                    "n_flagged": int(len(idx)),
                    "frac_flagged": round(len(idx) / max(len(k), 1), 4),
                    "flagged": idx.astype(int).tolist()}
    return out


def stage_rules(cli):
    arrays, meta = build_telemetry(cli)
    meta["dt_ms"] = cli.dt_ms
    p = cli.profile
    out = {"profile": cli.profile_name, "tag": cli.tag, "params": p,
           "threshold_m": cli.threshold_m, "sessions": {}, "review": []}
    tot = {"n_keep0": 0, "n_drop": 0}
    per_code = {}
    n_no_frames = 0
    for sid, s in meta["sessions"].items():
        role = s["role"]
        sdir = os.path.join(cli.root, "database" if role == "database"
                            else f"query_{role}", sid)
        fstats = load_frame_stats(cli, sid, sdir)
        n_no_frames += fstats is None
        hand = attitude_trustworthy(arrays, meta, sid, p)
        rules = session_rules(sid, role, arrays, meta, p, fstats)
        k0 = feat(arrays, sid, "keep0")
        union = np.zeros(len(k0), bool)
        codes = {}
        for code, m in rules.items():
            if m.any():
                codes[code] = np.flatnonzero(m).astype(int).tolist()
                per_code[code] = per_code.get(code, 0) + int(m.sum())
            union |= m
        n_keep = int((k0 & ~union).sum())
        out["sessions"][sid] = {
            "role": role, "n_slices": s["n_slices"], "n_keep0": int(k0.sum()),
            "n_dropped": int(union.sum()), "n_kept": n_keep,
            "keep_frac": round(n_keep / max(int(k0.sum()), 1), 4),
            "attitude_used": bool(hand), "drop": codes,
        }
        tot["n_keep0"] += int(k0.sum())
        tot["n_drop"] += int(union.sum())
        if role != "database" and n_keep < p["min_keep_frac"] * max(int(k0.sum()), 1):
            out["review"].append({"session": sid, "sweep": role, "n_keep0": int(k0.sum()),
                                  "n_kept": n_keep,
                                  "keep_frac": round(n_keep / max(int(k0.sum()), 1), 3),
                                  "reason": "survived below min_keep_frac"})
    out["totals"] = {**tot, "per_code": per_code,
                     "frac_dropped": round(tot["n_drop"] / max(tot["n_keep0"], 1), 4)}
    out["flags"] = {"self_sim_forest": forest_flags(cli, arrays, meta, p)}
    ff = out["flags"]["self_sim_forest"]
    n_fl = sum(v["n_flagged"] for v in ff.values())
    n_all = sum(v["n_kept"] for v in ff.values())
    print(f"\nflag-only (never dropped): {n_fl} of {n_all} query slices "
          f"({n_fl / max(n_all, 1):.1%}) are low-change scenes at "
          f"self-sim > {p['self_sim_flag']:g}; "
          f"{sum(1 for v in ff.values() if v['frac_flagged'] > 0.5)} session(s) are "
          f"majority such")
    os.makedirs(cli.out_dir, exist_ok=True)
    path = os.path.join(cli.out_dir, f"curation_{cli.profile_name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    if n_no_frames and (p["active_frac_min"] or p["db_active_frac_min"]
                        or p["sharp_ratio_min"] or p["db_sharp_ratio_min"]):
        raise SystemExit(f"{n_no_frames} session(s) have no frame stats but a "
                         f"camera-side gate is enabled — run --stage frames first, or "
                         f"the mask would silently spare exactly those sessions")
    print(f"\ncuration '{cli.profile_name}': {tot['n_drop']} of {tot['n_keep0']} "
          f"previously-kept slices dropped ({out['totals']['frac_dropped']:.1%})")
    print("  by reason: " + "  ".join(f"{k}={v}" for k, v in sorted(per_code.items())))
    for role in ("database", "day", "dawn", "night"):
        rows = [v for v in out["sessions"].values() if v["role"] == role]
        if rows:
            k0 = sum(r["n_keep0"] for r in rows)
            dr = sum(r["n_dropped"] for r in rows)
            print(f"  {role:9s} {len(rows):3d} sessions  {k0:7d} -> {k0 - dr:7d} "
                  f"({dr / max(k0, 1):5.1%} dropped)")
    if out["review"]:
        print(f"\n{len(out['review'])} session(s) below the {p['min_keep_frac']:.0%} "
              f"survival floor — flagged for review, NOT dropped:")
        for r in out["review"]:
            print(f"    {r['session']} ({r['sweep']}) {r['n_keep0']} -> {r['n_kept']} "
                  f"({r['keep_frac']:.0%})")
    print(f"-> {path}")


# ---------------------------------------------------------------------------
# 4. Stage: sheets — look at what the rules actually removed
# ---------------------------------------------------------------------------
def _flipbook_index(cli, sweep, spot, sid):
    """``{slice_idx: jpeg path}`` from the already-rendered per-slice flick-through set.

    Day and dawn queries were rendered slice-by-slice for the forensics report; reusing
    the left panel of those sheets costs nothing, so only night and database drops need
    the event reader.
    """
    d = os.path.join(cli.diag_dir, "all_slices", f"{sweep}_{spot}_{sid}")
    if not os.path.isdir(d):
        return {}
    out = {}
    for name in os.listdir(d):
        if name.startswith("q") and name.endswith(".jpg"):
            try:
                out[int(name[1:6])] = os.path.join(d, name)
            except ValueError:
                continue
    return out


def stage_sheets(cli):
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager
    from springfield_diag import _LRUFrames, _slice_to_part

    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
    arrays, meta = build_telemetry(cli)
    with open(os.path.join(cli.out_dir, f"curation_{cli.profile_name}.json")) as f:
        cur = json.load(f)
    with open(cli.results) as f:
        spot = {r["session"]: r["spot"] for r in json.load(f)["per_session"]}

    # gather every dropped slice, grouped by reason code
    by_code = {}
    for sid, s in cur["sessions"].items():
        for code, idxs in s["drop"].items():
            by_code.setdefault(code, []).append((sid, s["role"], idxs))

    frames = _LRUFrames(cli, cli.representation)
    part_paths = {}
    out_dir = os.path.join(cli.out_dir, f"sheets_{cli.profile_name}")
    os.makedirs(out_dir, exist_ok=True)

    def thumb(sid, role, idx):
        """160x90 RGB thumbnail of one slice, from the flipbook if it was rendered."""
        if role != "database":
            fb = _flipbook_index(cli, role, spot.get(sid, "spot-00"), sid)
            if idx in fb:
                im = Image.open(fb[idx])
                # sheet layout: query | top-1 | GPS-closest, 3 equal panels
                w = im.width // 3
                return im.crop((3, 27, w - 4, im.height - 5)).resize((160, 90))
        sdir = os.path.join(cli.root, "database" if role == "database"
                            else f"query_{role}", sid)
        if sid not in part_paths:
            part_paths[sid] = {os.path.basename(pp): pp for pp in list_partitions(sdir)}
        path, loc = _slice_to_part(cli, sdir, sid, idx)
        return Image.fromarray(frames.frame(path, loc)).resize((160, 90))

    rng = np.random.default_rng(0)
    for code, groups in sorted(by_code.items()):
        picks = []
        for sid, role, idxs in groups:
            take = min(len(idxs), max(1, cli.max_per_code // max(len(groups), 1)))
            for i in rng.choice(len(idxs), size=take, replace=False):
                picks.append((sid, role, int(idxs[i])))
        rng.shuffle(picks)
        picks = picks[:cli.max_per_code]
        if not picks:
            continue
        ncol = 6
        nrow = int(np.ceil(len(picks) / ncol))
        sheet = Image.new("RGB", (ncol * 168 + 8, nrow * 116 + 34), "white")
        drw = ImageDraw.Draw(sheet)
        drw.text((6, 8), f"dropped by '{code}'  —  {sum(len(g[2]) for g in groups)} "
                         f"slices over {len(groups)} sessions, {len(picks)} shown",
                 fill=(20, 20, 20), font=font)
        for n, (sid, role, idx) in enumerate(picks):
            row, col = divmod(n, ncol)
            try:
                im = thumb(sid, role, idx)
            except Exception as exc:                                   # noqa: BLE001
                print(f"  {code} {sid}#{idx}: {exc}")
                continue
            x, y = 4 + col * 168, 30 + row * 116
            sheet.paste(im, (x, y + 16))
            tilt = feat(arrays, sid, "tilt_deg")[idx]
            drw.text((x, y + 2),
                     f"{role[:3]} {sid[9:15]} #{idx}  h{feat(arrays, sid, 'hacc_m')[idx]:.0f}"
                     f" t{0 if np.isnan(tilt) else tilt:.0f}",
                     fill=(90, 90, 90), font=font)
        out = os.path.join(out_dir, f"dropped_{code}.jpg")
        sheet.save(out, quality=88)
        print(f"-> {out}  ({len(picks)} of {sum(len(g[2]) for g in groups)})")

    _sheets_map(cli, arrays, cur, out_dir)


def _sheets_map(cli, arrays, cur, out_dir):
    """Every dropped slice on the route, coloured by reason — the geographic sanity check."""
    import folium
    colour = {"gps_hacc": "#e6550d", "gps_warmup": "#fd8d3c", "gps_outlier": "#a63603",
              "boundary": "#3182bd", "tilt": "#756bb1", "smear": "#31a354"}
    lat0 = np.median(np.concatenate([feat(arrays, s, "lat")[feat(arrays, s, "keep0")]
                                     for s in DB_ARMS]))
    lon0 = np.median(np.concatenate([feat(arrays, s, "lon")[feat(arrays, s, "keep0")]
                                     for s in DB_ARMS]))
    m = folium.Map(location=[lat0, lon0], zoom_start=16, tiles="cartodbpositron")
    kept = folium.FeatureGroup(name="kept (database)", show=True)
    for sid in DB_ARMS:
        k0 = feat(arrays, sid, "keep0")
        lat, lon = feat(arrays, sid, "lat")[k0], feat(arrays, sid, "lon")[k0]
        folium.PolyLine(list(zip(lat[::20], lon[::20])), color="#999999", weight=1.5,
                        opacity=0.6).add_to(kept)
    kept.add_to(m)
    groups = {}
    for sid, s in cur["sessions"].items():
        for code, idxs in s["drop"].items():
            g = groups.setdefault(code, folium.FeatureGroup(name=f"dropped: {code}"))
            lat, lon = feat(arrays, sid, "lat"), feat(arrays, sid, "lon")
            for i in idxs[::max(1, len(idxs) // 400)]:
                folium.CircleMarker(
                    [float(lat[i]), float(lon[i])], radius=2.5,
                    color=colour.get(code, "#444444"), fill=True, fill_opacity=0.7,
                    popup=f"{sid} #{i} — {code} ({s['role']})").add_to(g)
    for g in groups.values():
        g.add_to(m)
    folium.LayerControl().add_to(m)
    out = os.path.join(out_dir, f"dropped_map_{cli.profile_name}.html")
    m.save(out)
    print(f"-> {out}")


# ---------------------------------------------------------------------------
# 5. Stage: report — what curation changed, and what it cost
# ---------------------------------------------------------------------------
def _res_block(res):
    mic = res["recall_micro"]
    mac = res["recall_macro"]
    th = res["theta_curve_r1"]
    return {
        "micro": {k: (round(v["1"], 4) if v else None) for k, v in mic.items()},
        "macro": {k: (round(v["1"], 4) if v else None) for k, v in mac.items()},
        "any": {k: (round(v["any"], 4) if v.get("any") is not None else None)
                for k, v in th.items()},
        "majority": {k: (round(v["0.50"], 4) if v.get("0.50") is not None else None)
                     for k, v in th.items()},
        "n_scorable": sum(res["scorable"].values()),
        "n_database": res["n_database"],
    }


def stage_report(cli):
    if not cli.after:
        raise SystemExit("--stage report needs --after <curated results json>")
    with open(cli.results) as f:
        before = json.load(f)
    with open(cli.after) as f:
        after = json.load(f)
    with open(os.path.join(cli.out_dir, f"curation_{cli.profile_name}.json")) as f:
        cur = json.load(f)
    ev_path = os.path.join(cli.out_dir, f"evidence_{cli.tag}.json")
    ev = json.load(open(ev_path)) if os.path.exists(ev_path) else {}

    b, a = _res_block(before), _res_block(after)
    rep = {"profile": cli.profile_name, "params": cur["params"],
           "before": b, "after": a,
           "removed": cur["totals"], "review": cur["review"],
           "rule_evidence": ev.get("rule_tests", [])}

    # per-session deltas — a session that vanished entirely is a finding, not a rounding
    rb = {r["session"]: r for r in before["per_session"]}
    ra = {r["session"]: r for r in after["per_session"]}
    moved = []
    for sid, r in rb.items():
        r2 = ra.get(sid)
        if r2 is None:
            moved.append({"session": sid, "sweep": r["sweep"], "status": "GONE",
                          "r1_before": r["r1_slice"]})
            continue
        if r["r1_slice"] is None or r2["r1_slice"] is None:
            continue
        moved.append({"session": sid, "sweep": r["sweep"], "spot": r.get("spot"),
                      "n_kept_before": r["n_kept"], "n_kept_after": r2["n_kept"],
                      "r1_before": r["r1_slice"], "r1_after": r2["r1_slice"],
                      "delta": round(r2["r1_slice"] - r["r1_slice"], 4)})
    moved.sort(key=lambda x: -(x.get("delta") or 0))
    rep["per_session"] = moved

    print(f"\ncuration '{cli.profile_name}': "
          f"{cur['totals']['n_drop']} slices removed "
          f"({cur['totals']['frac_dropped']:.1%} of previously kept)")
    print(f"  database {b['n_database']} -> {a['n_database']} rows; "
          f"scorable query slices {b['n_scorable']} -> {a['n_scorable']}")
    print(f"\n  {'scope':9s} {'micro R@1':>20s} {'macro R@1':>20s} {'any-hit':>20s}")
    for scope in ("overall", "day", "dawn", "night"):
        if b["micro"].get(scope) is None:
            continue
        def cell(d1, d2, k):
            x, y = d1.get(k), d2.get(k)
            return f"{x:.3f} -> {y:.3f} ({y - x:+.3f})" if (x is not None
                                                            and y is not None) else "-"
        print(f"  {scope:9s} {cell(b['micro'], a['micro'], scope):>20s} "
              f"{cell(b['macro'], a['macro'], scope):>20s} "
              f"{cell(b['any'], a['any'], scope):>20s}")
    print("\n  biggest per-session movers:")
    for r in moved[:8] + moved[-5:]:
        if "delta" in r:
            print(f"    {r['session']} ({r['sweep']:5s}) {r['n_kept_before']:4d}->"
                  f"{r['n_kept_after']:4d} slices  R@1 {r['r1_before']:.3f} -> "
                  f"{r['r1_after']:.3f}  ({r['delta']:+.3f})")
    gone = [r for r in moved if r.get("status") == "GONE"]
    if gone:
        print(f"\n  {len(gone)} session(s) disappeared entirely: "
              + ", ".join(r["session"] for r in gone))

    out = os.path.join(cli.out_dir, f"delta_{cli.profile_name}.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\n-> {out}")


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True,
                    choices=("frames", "evidence", "rules", "sheets", "report"))
    ap.add_argument("--roles", default="database,day,dawn,night",
                    help="frames stage: which session roles to scan")
    ap.add_argument("--workers", type=int, default=2,
                    help="frames stage: parallel partitions. IO bound on a spinning "
                         "disk — five readers ran slower in aggregate than two")
    ap.add_argument("--no-magnets-first", dest="magnets_first", action="store_false",
                    help="frames stage: scan partitions in file order instead of "
                         "most-wrong-top-1s first")
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help="uncurated results json — supplies tag, dt, radius")
    ap.add_argument("--after", default=None,
                    help="report stage: the curated results json to compare against")
    ap.add_argument("--profile-name", default="v1")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    ap.add_argument("--diag-dir", default=DEFAULT_DIAG_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="rebuild the telemetry cache")
    ap.add_argument("--max-per-code", type=int, default=60,
                    help="sheets stage: thumbnails per reason code")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a rule parameter, e.g. --set tilt_max_deg=60")
    cli = ap.parse_args()

    with open(cli.results) as f:
        res = json.load(f)
    cli.tag = res["tag"]
    cli.dt_ms = res["dt_ms"]
    cli.threshold_m = res["threshold_m"]
    cli.representation = res["representation"]
    # fields the shared FrameCache/make_dataset plumbing reads off the cli namespace
    cli.no_hot_pixel = not res["hot_pixel"]
    cli.no_event_filter = res["filter_dt_us"] is None
    cli.event_filter_dt_us = res["filter_dt_us"] or 50_000
    cli.profile = dict(PROFILE)
    for kv in cli.set:
        k, v = kv.split("=", 1)
        if k not in cli.profile:
            raise SystemExit(f"unknown parameter {k}; known: {sorted(cli.profile)}")
        cli.profile[k] = float(v)

    {"frames": stage_frames, "evidence": stage_evidence, "rules": stage_rules,
     "sheets": stage_sheets, "report": stage_report}[cli.stage](cli)


if __name__ == "__main__":
    main()
