"""Per-frame metric coordinates for NSAVP traverses.

    <seq>_ground_truth.h5  --200 Hz pose_base_link-->  (t_ns, ECEF xyz metres)
    <seq>.{hdf5,h5}        --first event timestamp-->  frame centre times
      --interpolate--> [n_frames, 2] metres in a shared local ENU frame

The NSAVP counterpart of :mod:`src.traversegps`, producing the identical
``(xy, covered, speed, info)`` tuple so ``scripts/brisbane_pooled.py``'s pooling and scoring
apply unchanged. Three things differ from Brisbane, and all three make this the easier dataset:

* **No clock calibration.** Brisbane's camera and GPS clocks disagree by up to 7 s per
  recording, which had to be fitted before a 25 m tolerance meant anything
  (``scripts/calibrate_brisbane_clock.py``). NSAVP timestamps its poses and its events off one
  hardware clock: measured across all nine traverses, the two streams start within 5 ms of each
  other and end within 50 ms — under one 50 ms frame. There is no offset to fit, so none is
  applied, and :func:`assert_clocks_aligned` keeps that an asserted invariant rather than an
  assumption.
* **No ``metadata.json``.** Brisbane reads its framing origin out of a dumped frame directory.
  NSAVP has no such dump, so slice 0's leading edge is the recording's first event timestamp —
  the origin eventcv itself uses at ``offset=0``. The frame count that implies is checked
  against the count eventcv actually reports before any coordinate is trusted.
* **Poses are ECEF, not lat/lon.** ``pose_base_link/positions`` is earth-centred earth-fixed
  metres, so it is converted to geodetic and then through :func:`src.traversegps.local_enu`,
  which keeps NSAVP and Brisbane on one projection convention.

The 200 Hz pose rate is the other reason this is the easier dataset: consecutive fixes are
~5 cm apart, against Brisbane's 1 Hz and 13.9 m. A 25 m tolerance sits nowhere near this ground
truth's noise floor.
"""

import os

import h5py
import numpy as np

from src.traversegps import local_enu

POSE_GROUP = "pose_base_link"
# WGS84. ``local_enu`` uses the equatorial radius alone for its equirectangular projection;
# the flattening is needed here only to get geodetic latitude out of ECEF.
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563


def sequence_file(root, seq):
    """The raw recording for one traverse — NSAVP ships both extensions in one release."""
    for ext in (".hdf5", ".h5"):
        path = os.path.join(root, seq, seq + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no recording for '{seq}' under {os.path.join(root, seq)}")


def ground_truth_file(root, seq):
    for ext in (".h5", ".hdf5"):
        path = os.path.join(root, seq, f"{seq}_ground_truth{ext}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no ground truth for '{seq}' under {os.path.join(root, seq)}")


def ecef_to_geodetic(xyz):
    """``[N, 3]`` ECEF metres -> ``(lat, lon)`` degrees, by Bowring's closed form.

    Geodetic rather than geocentric latitude: the two differ by up to 0.19 deg, which would
    tilt the local tangent plane and leak the route's altitude change into its horizontal
    coordinates. One evaluation of Bowring's formula is accurate to well under a millimetre at
    these altitudes, so no iteration is needed.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    b = WGS84_A * (1.0 - WGS84_F)
    e2 = WGS84_F * (2.0 - WGS84_F)                      # first eccentricity squared
    ep2 = (WGS84_A ** 2 - b ** 2) / b ** 2              # second eccentricity squared
    p = np.hypot(x, y)
    theta = np.arctan2(z * WGS84_A, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(theta) ** 3,
                     p - e2 * WGS84_A * np.cos(theta) ** 3)
    lon = np.arctan2(y, x)
    return np.rad2deg(lat), np.rad2deg(lon)


def pose_track(root, seq):
    """``(t_ns [N] int64, lat [N], lon [N])`` — the traverse's pose track, in time order."""
    with h5py.File(ground_truth_file(root, seq), "r") as handle:
        group = handle[POSE_GROUP]
        t_ns = np.asarray(group["timestamps"][:], dtype=np.int64)
        lat, lon = ecef_to_geodetic(group["positions"][:])
    if not np.all(np.diff(t_ns) >= 0):
        order = np.argsort(t_ns)
        t_ns, lat, lon = t_ns[order], lat[order], lon[order]
    return t_ns, lat, lon


def event_span(root, seq):
    """``(t0_ns, t1_ns)`` of the recording — slice 0's leading edge is ``t0_ns``.

    Read from the file's own timestamps rather than from eventcv, because this is what the
    frame grid is anchored to and it must not depend on how the reader was configured.
    Only the two endpoints are touched; the arrays are billions of events long.
    """
    with h5py.File(sequence_file(root, seq), "r") as handle:
        stamps = handle["events/timestamps"]
        return int(stamps[0]), int(stamps[-1])


def assert_clocks_aligned(root, seq, tol_s=0.25):
    """Assert the pose and event streams share a clock. -> the start offset in seconds.

    A silent disagreement here is the failure this dataset could plausibly have and the one
    nothing downstream would reveal: every coordinate would be shifted by a constant, and the
    recall would merely look like a weaker model. Brisbane needed a fitted per-traverse offset
    for exactly this reason; NSAVP is asserted to need none.
    """
    e0, e1 = event_span(root, seq)
    t_ns, _, _ = pose_track(root, seq)
    start = (e0 - int(t_ns[0])) / 1e9
    span = abs((e1 - e0) - (int(t_ns[-1]) - int(t_ns[0]))) / 1e9
    if abs(start) > tol_s or span > tol_s:
        raise SystemExit(
            f"{seq}: event and pose clocks disagree — start {start:+.3f} s, span {span:+.3f} s "
            f"(tolerance {tol_s} s). NSAVP is supposed to timestamp both off one hardware "
            f"clock; if it does not here, this traverse needs a fitted offset the way Brisbane "
            f"does and every coordinate is currently wrong by {abs(start):.3f} s of travel.")
    return start


def n_frames_for(root, seq, dt_ms=50):
    """The number of ``dt_ms`` slices eventcv yields for this recording.

    ``ceil``, not ``floor``: the final partial slice is emitted and the descriptor bank has a
    row for it. The caller checks this against the reader's own ``n_slices`` — an off-by-one
    here shifts nothing at the start and everything at the end.
    """
    t0, t1 = event_span(root, seq)
    return int(np.ceil((t1 - t0) / (dt_ms * 1e6)))


def track_origin(root, sequences):
    """``(lat0, lon0)`` — one projection origin shared by every traverse.

    The mean of each traverse's mean fix. NSAVP's two routes sit ~800 m apart, so a single
    equirectangular origin covers both to well under the 25 m tolerance.
    """
    lats, lons = [], []
    for seq in sequences:
        _, lat, lon = pose_track(root, seq)
        lats.append(float(lat.mean()))
        lons.append(float(lon.mean()))
    return float(np.mean(lats)), float(np.mean(lons))


def frame_coords(root, seq, lat0, lon0, dt_ms=50):
    """``(xy[n, 2], covered[n], speed[n], info)`` for one traverse's descriptor grid.

    Frame *i* spans ``[t0 + i*dt, t0 + (i+1)*dt)``, so the instant its descriptor describes is
    the centre ``t0 + (i + 0.5)*dt``. Poses are interpolated onto those centres.

    ``covered`` is False where a centre falls outside the pose track's own span: ``np.interp``
    clamps there, which would pin trailing frames onto the last fix and invent positives around
    it. The caller drops them. ``speed`` is the finite difference in m/s, used to report what
    fraction of the query set was stopped — NSAVP is a car on public roads, so traffic lights
    put genuinely stationary frames in the query set.
    """
    start = assert_clocks_aligned(root, seq)
    t_ns, lat, lon = pose_track(root, seq)
    t0, _ = event_span(root, seq)
    n = n_frames_for(root, seq, dt_ms)

    centres = t0 + (np.arange(n) + 0.5) * (dt_ms * 1e6)     # nanoseconds
    track = local_enu(lat, lon, lat0, lon0)
    xy = np.column_stack([np.interp(centres, t_ns, track[:, 0]),
                          np.interp(centres, t_ns, track[:, 1])])
    covered = (centres >= t_ns[0]) & (centres <= t_ns[-1])

    step = np.linalg.norm(np.diff(xy, axis=0), axis=1) / (dt_ms / 1000.0)
    speed = np.concatenate([step[:1], step]) if n > 1 else np.zeros(n)
    info = {"n_frames": int(n),
            "n_gps_fixes": int(len(t_ns)),
            "uncovered": int((~covered).sum()),
            "route_len_m": float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum()),
            "clock_start_offset_s": float(start),
            "slice_times_checked": True}       # the clock assertion above is the equivalent
    return xy, covered, speed, info
