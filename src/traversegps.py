"""Per-frame geographic coordinates for a continuous traverse.

    <seq>_ground_truth.nmea  --1 Hz GGA fixes-->  (epoch, lat, lon)
    <seq>-frames-50/metadata.json  --start_tick-->  frame centre times
      --interpolate--> [n_frames, 2] metres in a shared local ENU frame

This is what a metric ground truth needs and the repo did not have. :mod:`src.imagevpr`
already scores an image set against a 25 m UTM radius, but it reads the coordinate out of
each filename; a traverse has no such per-frame label — only a 1 Hz GPS track and a
recording clock. Bridging those two is the whole job of this module, after which
:func:`src.imagevpr.build_gt` applies to a traverse unchanged.

The alternative already in the tree, Event-LAB's ``ground_truth_from_pose_only``, models
position as *route completion percentage* and dilates along the reference arc. That cannot
express a revisit: Brisbane's route self-intersects, and 11.5% of ``sunset1`` frames have
another frame of the same traverse within 25 m more than 30 s away. A Euclidean radius
counts those as the same place, which for place recognition is the point.

Two things about the timing are easy to get wrong, and both are asserted rather than
assumed:

* the descriptor grid has ``total_frames + 1`` rows — eventcv emits a final partial slice
  that the frame dumper does not write, so a naive ``total_frames`` is off by one and
  silently shifts nothing at the start and everything at the end;
* ``metadata.json``'s ``start_tick`` must equal the framing origin eventcv actually used.
  Where ``slice_times.npy`` was dumped it is checked; a mismatch means the offset handling
  in :meth:`src.inference.EventStreamDataset._open` has drifted and every coordinate is
  wrong by a constant, which no downstream number would reveal.
"""

import datetime
import json
import os

import numpy as np

# WGS84 equatorial radius. The traverses span ~2.2 x 2.8 km, where an equirectangular
# projection about a local origin differs from a proper UTM one by well under 0.1 m —
# three orders of magnitude below the 25 m tolerance it feeds, and it keeps `utm`/`pyproj`
# (neither of which is in this environment) out of the dependency list.
EARTH_RADIUS_M = 6378137.0

# Per-traverse camera-vs-GPS clock offset, in seconds, written by
# scripts/calibrate_brisbane_clock.py into <eventlab-dir>/<dataset>/. Both clocks report absolute
# UTC, but each recording session's differs from its GPS receiver's by up to 7 s — ~100 m of
# travel — which is what put "the same place" in two traverses a median 19-104 m apart. Without
# this the 25 m tolerance is meaningless; with it the residual is ~8.5 m.
CLOCK_OFFSET_FILE = "clock_offsets.json"


def clock_offsets(eventlab_dir, dataset, required=None):
    """``({seq: offset_s}, {seq: drift_s_per_s})`` from the fitted calibration file.

    NB the file actually on disk is the **offset-only** fit (all drifts 0.0,
    ``mean_residual_m`` 8.86); a drift term was measured to tighten pairs to 3.7-6.2 m but
    has not been re-fitted into the shipped file, so every current Brisbane number rests on
    the 8.86 m residual — safely inside a 25 m radius either way.

    ``required``: raise unless the file exists AND covers these sequences. Brisbane MUST
    pass its traverses here — two Brisbane trees exist on this machine and only one carries
    the calibration; falling back silently to zero offsets puts "the same place" a median
    19-104 m apart and turns a 25 m benchmark into a clock measurement.
    """
    path = os.path.join(eventlab_dir, dataset, CLOCK_OFFSET_FILE)
    if not os.path.exists(path):
        if required:
            raise SystemExit(
                f"missing clock calibration: {path}\nThis dataset's camera and GPS clocks "
                f"disagree by up to 7 s (~100 m); without the fitted offsets every metric "
                f"tolerance is meaningless. Either --eventlab-dir points at the wrong tree "
                f"(the calibrated one carries {CLOCK_OFFSET_FILE}) or the fit was never "
                f"run: scripts/calibrate_brisbane_clock.py --write.")
        return {}, {}
    with open(path) as handle:
        table = json.load(handle)
    offsets, drifts = table.get("offsets", {}), table.get("drifts", {})
    missing = sorted(set(required or ()) - set(offsets))
    if missing:
        raise SystemExit(f"{path} has no clock offset for {missing} — re-run "
                         f"scripts/calibrate_brisbane_clock.py --write to cover them.")
    return offsets, drifts


def nmea_track(path):
    """``(t_epoch[N], lat[N], lon[N])`` — every usable GGA fix, in absolute UTC epoch seconds.

    GGA carries a time of day but no date; RMC carries the date. The most recent RMC
    datestamp is therefore carried forward onto each subsequent GGA, which is what makes
    the track directly comparable to the recording clock without a per-traverse offset.

    Deliberately *not* ``groundtruths.py::_load_nmea_gps``: that returns time relative to
    the first sentence and drops any fix within 1e-4 deg of its predecessor. The
    de-duplication is right for a route-percentage model, which only needs the shape of
    the path, and wrong here — dropping the stationary fixes is exactly what destroys the
    time base this module interpolates against.
    """
    import pynmea2

    lat, lon, ts, date = [], [], [], None
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                msg = pynmea2.parse(line)
            except Exception:                       # NMEA files carry partial/garbled lines
                continue
            if msg.sentence_type == "RMC" and getattr(msg, "datestamp", None):
                date = msg.datestamp
            if msg.sentence_type != "GGA" or date is None:
                continue
            if not msg.latitude or not msg.longitude:       # 0,0 = no fix
                continue
            stamp = datetime.datetime.combine(date, msg.timestamp,
                                              tzinfo=datetime.timezone.utc)
            ts.append(stamp.timestamp())
            lat.append(msg.latitude)
            lon.append(msg.longitude)

    t = np.asarray(ts, dtype=np.float64)
    if t.size < 2:
        raise ValueError(f"{path}: only {t.size} usable GGA fixes — cannot interpolate")
    if not np.all(np.diff(t) > 0):
        # np.interp needs an increasing x. Ties happen when two GGA share a second.
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, lat, lon = t[keep], np.asarray(lat)[keep], np.asarray(lon)[keep]
        return t, lat, lon
    return t, np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)


def framing_origin(eventlab_dir, dataset, seq, dt_ms=50):
    """``(start_s, n_frames)`` — the epoch of slice 0's leading edge, and the slice count.

    ``n_frames`` is ``total_frames + 1``: eventcv yields a final partial slice that the
    frame dumper drops, and it is the eventcv count the descriptor banks have.

    Where ``brisbane_npz/.../real/<seq>/slice_times.npy`` exists it is authoritative — it
    was written by the same eventcv framing that produced the banks — and ``metadata.json``
    is asserted against it, turning the alignment into a runtime invariant.
    """
    meta_path = os.path.join(eventlab_dir, dataset, seq, f"{seq}-frames-{dt_ms}",
                             "metadata.json")
    with open(meta_path) as handle:
        meta = json.load(handle)
    if float(meta["timewindow_ms"]) != float(dt_ms):
        raise ValueError(
            f"{meta_path} was dumped at {meta['timewindow_ms']} ms but this run uses "
            f"{dt_ms} ms — the frame centres, and so every coordinate, would be wrong.")
    start_s = float(meta["start_tick"]) / float(meta["ticks_per_second"])
    n_frames = int(meta["total_frames"]) + 1
    return start_s, n_frames


def check_slice_times(npz_root, dataset, seq, start_s, n_frames, tol_s=1e-3):
    """Assert ``metadata.json`` agrees with a dumped ``slice_times.npy``. -> bool (checked?).

    Only sunset1 and sunset2 were ever materialised to ``.npz``, so this verifies the two
    traverses it can and reports that the rest went unchecked rather than passing silently.
    """
    path = os.path.join(npz_root, dataset, "real", seq, "slice_times.npy")
    if not os.path.exists(path):
        return False
    times = np.load(path)                                   # [n, 2] float64, microseconds
    drift = times[0, 0] / 1e6 - start_s
    if abs(drift) > tol_s:
        raise ValueError(
            f"{seq}: metadata.json start_tick is {drift:+.6f} s from {path}'s first slice. "
            f"The descriptor banks were framed from the latter, so every GPS coordinate "
            f"would be offset by {abs(drift):.3f} s of travel.")
    if times.shape[0] != n_frames:
        raise ValueError(
            f"{seq}: {path} has {times.shape[0]} slices, metadata.json implies {n_frames}")
    return True


def local_enu(lat, lon, lat0, lon0):
    """``[N, 2]`` metres east/north about ``(lat0, lon0)`` — equirectangular.

    All traverses must share one origin, or their coordinates are not comparable and the
    pooled database is meaningless.
    """
    east = np.deg2rad(np.asarray(lon) - lon0) * EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))
    north = np.deg2rad(np.asarray(lat) - lat0) * EARTH_RADIUS_M
    return np.column_stack([east, north])


def track_origin(eventlab_dir, dataset, sequences):
    """``(lat0, lon0)`` — the shared projection origin, the mean fix over every traverse."""
    lats, lons = [], []
    for seq in sequences:
        _, lat, lon = nmea_track(nmea_path(eventlab_dir, dataset, seq))
        lats.append(lat.mean())
        lons.append(lon.mean())
    return float(np.mean(lats)), float(np.mean(lons))


def nmea_path(eventlab_dir, dataset, seq):
    return os.path.join(eventlab_dir, dataset, seq, f"{seq}_ground_truth.nmea")


def frame_coords(eventlab_dir, dataset, seq, lat0, lon0, dt_ms=50, npz_root=None,
                 offsets=None):
    """``(xy[n, 2], covered[n], speed[n], info)`` for one traverse's descriptor grid.

    Frame *i* spans ``[start + i*dt, start + (i+1)*dt)``, so its centre — the instant the
    descriptor describes — is ``start + (i + 0.5)*dt``, plus this traverse's clock offset
    (see :data:`CLOCK_OFFSET_FILE`). GPS east/north are interpolated onto those centres.

    ``covered`` is False where a frame centre falls outside the GPS track's own time span.
    ``np.interp`` clamps there, which would silently pin up to a few hundred frames onto
    the first or last fix and invent positives around it; the caller drops them instead.
    ``speed`` is the finite difference in m/s, for reporting how much of the query set was
    stationary.
    """
    t_gps, lat, lon = nmea_track(nmea_path(eventlab_dir, dataset, seq))
    start_s, n_frames = framing_origin(eventlab_dir, dataset, seq, dt_ms)
    checked = (check_slice_times(npz_root, dataset, seq, start_s, n_frames)
               if npz_root else False)
    if offsets is None:
        # required: a Brisbane tree without the calibration must stop the run, not score
        # with zero offsets (two trees exist locally; only one is calibrated).
        offsets, drifts = clock_offsets(eventlab_dir, dataset, required=(seq,))
    else:
        offsets, drifts = offsets
    shift = float(offsets.get(seq, 0.0))
    drift = float(drifts.get(seq, 0.0))

    # Elapsed recording time is what the drift acts on, so it is applied to the offset from
    # the framing origin rather than to the absolute epoch.
    elapsed = (np.arange(n_frames) + 0.5) * (dt_ms / 1000.0)
    centres = start_s + elapsed + shift + drift * elapsed
    track = local_enu(lat, lon, lat0, lon0)
    xy = np.column_stack([np.interp(centres, t_gps, track[:, 0]),
                          np.interp(centres, t_gps, track[:, 1])])
    covered = (centres >= t_gps[0]) & (centres <= t_gps[-1])

    step = np.linalg.norm(np.diff(xy, axis=0), axis=1) / (dt_ms / 1000.0)
    speed = np.concatenate([step[:1], step]) if n_frames > 1 else np.zeros(n_frames)

    info = {"n_frames": n_frames, "n_gps_fixes": int(t_gps.size),
            "start_s": start_s, "clock_offset_s": shift, "clock_drift_s_per_s": drift,
            "slice_times_checked": bool(checked),
            "uncovered": int((~covered).sum()),
            "route_len_m": float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum()),
            "gps_gap_max_s": float(np.diff(t_gps).max())}
    return xy, covered, speed, info
