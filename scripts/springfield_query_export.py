"""Export the published Springfield query cell as a standalone, redistributable dataset.

Run from the repo root::

    CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python3 \
        scripts/springfield_query_export.py --stage all

The paper's headline Springfield cell is *day+dawn, reversals excluded*: every 50 ms query
slice whose sweep is not ``night`` and whose camera-vs-route bearing satisfies
``|psi| < 135 deg``. That is 5,557 of the 15,886 scored query slices, spread over 38 of the
84 query sessions (18 day/dawn sessions are pure reversal passes and contribute nothing;
10 more are walked partly in each direction). This script writes that cell out as event
data plus labels, so the queries can be published without shipping the 63 GB of raw
day+dawn recordings — or the night and reversal slices the cell deliberately excludes.

Which slices are in the cell is *not* re-derived here. The membership test reads the two
artefacts the evaluation itself produced:

* ``dump_<tag>.npz`` (``springfield_diag.py --stage dump``) — the scored query set, in
  scoring order, with each slice's session, native-grid slice index and pooled ENU
  coordinate, plus the 132,569-row pooled gallery;
* ``orient_<tag>.npz`` (``--stage orient``) — per-slice ``psi``, the camera bearing
  relative to the local route direction, which is what ``|psi| >= 135`` thresholds.

Geometry is then recomputed from the sessions through ``springfield_diag``'s own
``load_cached_session`` — the same slice grid, clock fit and GPS interpolation the eval
used — and checked against the dump's ENU coordinates before anything is written. A
mismatch beyond ``--xy-tol-m`` is fatal: it would mean the exported labels describe a
different slice grid than the published numbers, which is exactly the failure this export
must not ship silently.

Stages:

``manifest``
    ``manifest.csv`` (one row per exported query slice), ``database_index.csv`` (the pooled
    gallery, so the positives are interpretable), ``positives_25m.npz`` (CSR ground truth),
    ``protocol.json`` and ``README.md``. Cheap — no event payload is read.
``events``
    One HDF5 per contributing session under ``events/query_<sweep>/<session>.h5``, holding
    only the cell's slices. Resumable: a session whose output already carries the expected
    slice count and a ``complete`` attr is skipped, so an interrupted run (this box's
    systemd-oomd kills long ones) resumes with ``--stage events`` and no ``--force``.
``all``
    ``manifest`` then ``events``.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src.traversegps import local_enu                  # noqa: E402
from springfield_diag import load_cached_session       # noqa: E402
from springfield_full import DB_ARMS                   # noqa: E402

DEFAULT_ROOT = "/media/adam/vprdatasets/megaevent/springfield/sessions"
DEFAULT_BANK_DIR = "/media/adam/vprdatasets/megaevent/evaluations/springfield"
DEFAULT_OUT = "/media/adam/vprdatasets/megaevent/springfield/queries"
DEFAULT_DIAG = "output/springfield_diag"
DEFAULT_TAG = "b_v8_accum_s750_s750_3f4eb54780_accumulate_r322_hp1_baoff"
DEFAULT_PER_QUERY = ("output/springfield_full/"
                     "per_query_b_v8_accum_s750_s750_3f4eb54780_accumulate_r322_hp1_baoff.csv")

# Read the source `events/t` in blocks this large when locating slice boundaries. 20 M
# int64 is 160 MB resident; the whole array would be 3.3 GB for the longest session, and
# this box's oomd scope does not forgive that on top of an output buffer.
SCAN_BLOCK = 20_000_000
# Slices per write batch. A 50 ms slice is ~500 k events here, so 32 slices is ~200 MB
# through the compressor at a time.
WRITE_BATCH = 32


# ---------------------------------------------------------------------------
# 1. The cell
# ---------------------------------------------------------------------------
def load_cell(cli):
    """The exported query set: the dump's scored slices, filtered to the paper's cell.

    Returns the dump handle, the per-slice ``psi``, and the boolean cell mask over the
    dump's own scoring order — every later stage indexes off that order, so the exported
    dataset and the published recall numbers are talking about the same rows.
    """
    dump_path = os.path.join(cli.diag_dir, f"dump_{cli.tag}.npz")
    orient_path = os.path.join(cli.diag_dir, f"orient_{cli.tag}.npz")
    for path in (dump_path, orient_path):
        if not os.path.exists(path):
            raise SystemExit(f"no {path} — run scripts/springfield_diag.py --stage "
                             f"{'dump' if 'dump_' in path else 'orient'} --tag {cli.tag}")
    d = np.load(dump_path, allow_pickle=False)
    o = np.load(orient_path, allow_pickle=False)

    sids = [str(s) for s in d["sids"]]
    psi, moving = [], []
    for sid in sids:
        if f"{sid}_psi" not in o:
            raise SystemExit(f"{orient_path} has no psi for {sid}")
        psi.append(o[f"{sid}_psi"])
        moving.append(o[f"{sid}_moving"])
    psi, moving = np.concatenate(psi), np.concatenate(moving)
    if len(psi) != len(d["q_sid"]):
        raise SystemExit(f"orient holds {len(psi)} slices but the dump scored "
                         f"{len(d['q_sid'])} — the two were built on different keep masks")

    keep_sweeps = tuple(s for s in cli.sweeps.split(",") if s)
    sweep_of = d["sweeps"][d["q_sid"]]
    cell = np.isin(sweep_of, keep_sweeps) & (np.abs(psi) < cli.exclude_reversals_deg)
    print(f"dump {cli.tag}: {len(psi)} scored query slices over {len(sids)} sessions")
    print(f"cell (sweeps {list(keep_sweeps)}, |psi| < {cli.exclude_reversals_deg:g} deg): "
          f"{int(cell.sum())} slices")
    return d, psi, moving, cell


def session_order(d, cell):
    """``[(sid, sweep, rows)]`` for every session contributing to the cell.

    ``rows`` are indices into the dump's scoring order, ascending, so a session's exported
    slices come out in native time order.
    """
    out = []
    for i, sid in enumerate(d["sids"]):
        rows = np.flatnonzero(cell & (d["q_sid"] == i))
        if len(rows):
            out.append((str(sid), str(d["sweeps"][i]), rows))
    return out


def load_spots(path):
    """``{session: spot}`` from the evaluation's per-query CSV.

    The spot label is a connected component of session centroids at 50 m linkage, computed
    over all 84 sessions at once, so it cannot be recovered from the exported subset alone
    — it is read from the run that defined it rather than re-clustered here.
    """
    if not os.path.exists(path):
        raise SystemExit(f"no {path} — needed for the spot labels; run "
                         f"scripts/springfield_full.py or pass --per-query-csv")
    import csv
    with open(path) as f:
        return {r["session"]: r["spot"] for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# 2. Geometry, re-derived and checked against the dump
# ---------------------------------------------------------------------------
def geometry(cli, sdir, sid, want_slices, ref_xy, lat0, lon0):
    """Per-slice lat/lon, camera-clock windows and GPS-clock centres for one session.

    ``want_slices`` are native-grid indices; the returned arrays are aligned to them.
    The ENU projection of the result is compared against ``ref_xy`` — the dump's own
    coordinates for the same rows — so a clock or GPS re-read that lands anywhere else
    stops the export instead of mislabelling it.
    """
    s = load_cached_session(cli, sdir, sid, need_bank=False)
    n = len(s["keep"])
    if want_slices.max() >= n:
        raise SystemExit(f"{sid}: cell wants slice {want_slices.max()} but the grid has {n}")
    latlon = s["latlon"][want_slices]
    xy = local_enu(latlon[:, 0], latlon[:, 1], lat0, lon0)
    err = np.abs(xy - ref_xy).max()
    if err > cli.xy_tol_m:
        raise SystemExit(f"{sid}: re-derived ENU differs from the dump by {err:.4f} m "
                         f"(tolerance {cli.xy_tol_m} m) — the slice grid or clock fit has "
                         f"moved since the scored run; do not publish this export")

    parts = sorted({p for _, p, _ in s["prov"]})
    if len(parts) != 1:
        raise SystemExit(f"{sid}: {len(parts)} partitions — query sessions are single-file "
                         f"by capture design and the exporter assumes it")
    part = parts[0]
    with h5py.File(os.path.join(sdir, part), "r") as f:
        t0 = int(f["events"]["t"][0])
        n_events_src = int(f["events"]["t"].shape[0])
        wh = (int(f.attrs["width"]), int(f.attrs["height"]))
    dt_us = cli.dt_ms * 1000
    t_start = t0 + want_slices.astype(np.int64) * dt_us
    return {"latlon": latlon, "xy": xy, "t_start_us": t_start,
            "t_end_us": t_start + dt_us, "centres_ns": s["centres_ns"][want_slices],
            "clock_flag": s["clock_flag"], "partition": part,
            "n_events_src": n_events_src, "sensor_wh": wh, "err_m": float(err)}


def database_index(cli, d, lat0, lon0):
    """The pooled gallery's lat/lon, in the dump's row order, ENU-checked the same way.

    The positives are row indices into this table, so it ships alongside them: without it
    a ``positives`` array published on its own would name rows nobody can resolve.
    """
    rows, off = [], 0
    for sid in DB_ARMS:
        arm, facing = DB_ARMS[sid]
        s = load_cached_session(cli, os.path.join(cli.root, "database", sid), sid,
                                need_bank=False)
        kept = np.flatnonzero(s["keep"])
        latlon = s["latlon"][kept]
        xy = local_enu(latlon[:, 0], latlon[:, 1], lat0, lon0)
        ref = d["db_xy"][off:off + len(kept)]
        if len(ref) != len(kept):
            raise SystemExit(f"{sid}: {len(kept)} kept slices but the dump's gallery has "
                             f"{len(ref)} rows left at offset {off}")
        err = np.abs(xy - ref).max()
        if err > cli.xy_tol_m:
            raise SystemExit(f"{sid}: gallery ENU differs from the dump by {err:.4f} m — "
                             f"the pooled row order or keep mask has changed")
        for j, i in enumerate(kept):
            sid_j, part_j, slice_j = s["prov"][i]
            rows.append((off + j, sid_j, arm, facing, part_j, slice_j,
                         latlon[j, 0], latlon[j, 1], xy[j, 0], xy[j, 1],
                         float(d["db_bear"][off + j])))
        print(f"    [db/{arm}] {sid}: {len(kept)} rows, ENU max err {err:.5f} m")
        off += len(kept)
    if off != len(d["db_xy"]):
        raise SystemExit(f"gallery rebuilt to {off} rows, dump holds {len(d['db_xy'])}")
    return rows


# ---------------------------------------------------------------------------
# 3. Ground truth
# ---------------------------------------------------------------------------
def positives_csr(q_xy, db_xy, threshold_m, chunk=256):
    """CSR ``(indptr, indices)`` of gallery rows within ``threshold_m`` of each query.

    Chunked because the dense form is 5,557 x 132,569 — 2.9 GB as float32, and this box
    kills what it cannot fit.
    """
    ind, counts = [], np.zeros(len(q_xy), np.int64)
    t2 = threshold_m ** 2
    for a in range(0, len(q_xy), chunk):
        b = min(a + chunk, len(q_xy))
        d2 = ((q_xy[a:b, None, :] - db_xy[None, :, :]) ** 2).sum(axis=2)
        for r in range(b - a):
            hit = np.flatnonzero(d2[r] <= t2).astype(np.int32)
            ind.append(hit)
            counts[a + r] = len(hit)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return indptr, np.concatenate(ind) if ind else np.zeros(0, np.int32)


# ---------------------------------------------------------------------------
# 4. Event payload
# ---------------------------------------------------------------------------
def slice_bounds(t_ds, targets, block=SCAN_BLOCK):
    """Row index of the first event with ``t >= target``, for ascending ``targets``.

    One sequential pass over the on-disk timestamps, stopping as soon as every target is
    placed — a session whose cell ends early (a pass reversed half way) never reads its
    tail. A per-target binary search would instead touch ~30 scattered chunks each.
    """
    n = int(t_ds.shape[0])
    out = np.full(len(targets), n, np.int64)
    filled = np.zeros(len(targets), bool)
    off = 0
    while off < n and not filled.all():
        blk = t_ds[off:off + block]
        idx = np.searchsorted(blk, targets, side="left")
        hit = (~filled) & (idx < len(blk))
        out[hit] = off + idx[hit]
        filled |= hit
        off += len(blk)
    return out


def export_session(cli, sid, sweep, sdir, geo, labels, out_path):
    """Write one session's cell slices to ``out_path``; returns ``(n_slices, n_events)``."""
    n_slices = len(geo["t_start_us"])
    tmp = out_path + ".part"
    src = os.path.join(sdir, geo["partition"])

    # Both edges of every slice, interleaved and ascending, in one scan.
    targets = np.empty(2 * n_slices, np.int64)
    targets[0::2] = geo["t_start_us"]
    targets[1::2] = geo["t_end_us"]
    if np.any(np.diff(targets) < 0):
        raise SystemExit(f"{sid}: slice windows are not ascending — cannot scan once")

    t_start_clock = time.time()
    with h5py.File(src, "r") as f:
        bounds = slice_bounds(f["events"]["t"], targets)
        lo, hi = bounds[0::2], bounds[1::2]
        per_slice = (hi - lo).astype(np.int64)
        total = int(per_slice.sum())
        print(f"    {n_slices} slices, {total / 1e6:.1f} Mev "
              f"({time.time() - t_start_clock:.0f} s to locate)")

        comp = ({} if cli.compression == "none" else
                {"compression": cli.compression, "shuffle": True,
                 **({"compression_opts": cli.compression_level}
                    if cli.compression == "gzip" else {})})
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with h5py.File(tmp, "w") as g:
            g.attrs.update({"session": sid, "sweep": sweep, "spot": labels["spot"],
                            "width": geo["sensor_wh"][0], "height": geo["sensor_wh"][1],
                            "timestamp_scale_ms": 0.001, "dt_ms": cli.dt_ms,
                            "source_partition": geo["partition"],
                            "source_n_events": geo["n_events_src"],
                            "n_slices": n_slices, "n_events": total,
                            "exclude_reversals_deg": cli.exclude_reversals_deg,
                            "complete": 0})
            ev = g.create_group("events")
            dsets = {k: ev.create_dataset(
                        k, shape=(total,), dtype=dt,
                        chunks=(min(1 << 20, max(total, 1)),), **comp)
                     for k, dt in (("x", "u2"), ("y", "u2"),
                                   ("t", "i8"), ("p", "u1"))}
            ptr = np.concatenate([[0], np.cumsum(per_slice)]).astype(np.int64)
            written = 0
            for a in range(0, n_slices, WRITE_BATCH):
                b = min(a + WRITE_BATCH, n_slices)
                if hi[b - 1] - lo[a] == int(per_slice[a:b].sum()):
                    # The batch's slices are adjacent in the source: one contiguous read.
                    buf = {k: f["events"][k][lo[a]:hi[b - 1]] for k in dsets}
                else:
                    buf = {k: np.concatenate([f["events"][k][lo[j]:hi[j]]
                                              for j in range(a, b)]) for k in dsets}
                m = len(buf["t"])
                for k, ds in dsets.items():
                    ds[written:written + m] = buf[k]
                written += m
            if written != total:
                raise SystemExit(f"{sid}: wrote {written} events, expected {total}")

            sl = g.create_group("slices")
            sl.create_dataset("slice_index", data=labels["slice_index"].astype(np.int32))
            sl.create_dataset("ptr", data=ptr)
            sl.create_dataset("n_events", data=per_slice.astype(np.int64))
            sl.create_dataset("t_start_us", data=geo["t_start_us"])
            sl.create_dataset("t_end_us", data=geo["t_end_us"])
            sl.create_dataset("t_centre_gps_ns", data=geo["centres_ns"])
            sl.create_dataset("lat", data=geo["latlon"][:, 0])
            sl.create_dataset("lon", data=geo["latlon"][:, 1])
            sl.create_dataset("x_enu_m", data=geo["xy"][:, 0])
            sl.create_dataset("y_enu_m", data=geo["xy"][:, 1])
            sl.create_dataset("psi_deg", data=labels["psi"].astype(np.float32))
            sl.create_dataset("moving", data=labels["moving"])
            sl.create_dataset("n_positives_25m", data=labels["n_pos"].astype(np.int32))
            g.attrs["complete"] = 1
    os.replace(tmp, out_path)
    dur = time.time() - t_start_clock
    print(f"    -> {os.path.basename(out_path)}  "
          f"{os.path.getsize(out_path) / 1e9:.2f} GB  ({dur:.0f} s)")
    return n_slices, total


def already_done(out_path, n_slices):
    """True when a previous run finished this session — the resume test."""
    if not os.path.exists(out_path):
        return False
    try:
        with h5py.File(out_path, "r") as g:
            return bool(g.attrs.get("complete")) and int(g.attrs["n_slices"]) == n_slices
    except OSError:
        return False


# ---------------------------------------------------------------------------
# 5. Stages
# ---------------------------------------------------------------------------
def build(cli):
    """Everything the manifest stage needs, shared with the events stage."""
    d, psi, moving, cell = load_cell(cli)
    sessions = session_order(d, cell)
    spots = load_spots(cli.per_query_csv)

    orient_meta = os.path.join(cli.diag_dir, f"orient_{cli.tag}.json")
    with open(orient_meta) as f:
        lat0, lon0 = json.load(f)["__origin__"]
    print(f"projection origin ({lat0:.6f}, {lon0:.6f}) — the scored run's own")

    print("\nre-deriving query geometry against the dump")
    geos, max_err = {}, 0.0
    for sid, sweep, rows in sessions:
        sdir = os.path.join(cli.root, f"query_{sweep}", sid)
        geos[sid] = geometry(cli, sdir, sid, d["q_slice"][rows], d["q_xy"][rows],
                             lat0, lon0)
        max_err = max(max_err, geos[sid]["err_m"])
    print(f"  {len(sessions)} sessions, ENU max err {max_err:.5f} m "
          f"(tolerance {cli.xy_tol_m} m)")
    return d, psi, moving, cell, sessions, spots, geos, (lat0, lon0)


def stage_manifest(cli, built):
    import csv
    d, psi, moving, cell, sessions, spots, geos, (lat0, lon0) = built
    os.makedirs(cli.out, exist_ok=True)

    print("\nrebuilding the pooled gallery index")
    db_rows = database_index(cli, d, lat0, lon0)

    q_xy = np.concatenate([geos[sid]["xy"] for sid, _, _ in sessions])
    print(f"\nground truth: {cli.threshold_m:g} m radius over {len(q_xy)} queries "
          f"x {len(d['db_xy'])} gallery rows")
    indptr, indices = positives_csr(q_xy, d["db_xy"].astype(np.float64), cli.threshold_m)
    n_pos = np.diff(indptr)
    print(f"  positives/query: mean {n_pos.mean():.1f}  min {n_pos.min()}  "
          f"max {n_pos.max()}  zero-positive queries {int((n_pos == 0).sum())}")

    man_path = os.path.join(cli.out, "manifest.csv")
    cols = ["query_id", "session", "sweep", "spot", "slice_index", "row_in_session",
            "t_start_us", "t_end_us", "t_centre_gps_ns", "lat", "lon",
            "x_enu_m", "y_enu_m", "psi_deg", "moving", "n_positives_25m",
            "event_file", "clock_refit"]
    written, off = 0, 0
    with open(man_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for sid, sweep, rows in sessions:
            geo, k = geos[sid], len(rows)
            for j in range(k):
                w.writerow([
                    f"{sid}_{int(d['q_slice'][rows[j]]):06d}", sid, sweep,
                    spots.get(sid, ""), int(d["q_slice"][rows[j]]), j,
                    int(geo["t_start_us"][j]), int(geo["t_end_us"][j]),
                    int(round(geo["centres_ns"][j])),
                    "%.8f" % geo["latlon"][j, 0], "%.8f" % geo["latlon"][j, 1],
                    "%.3f" % geo["xy"][j, 0], "%.3f" % geo["xy"][j, 1],
                    "%.2f" % psi[rows[j]], int(moving[rows[j]]),
                    int(n_pos[off + j]),
                    f"events/query_{sweep}/{sid}.h5",
                    "" if geo["clock_flag"] is None else geo["clock_flag"]])
                written += 1
            off += k
    print(f"-> {man_path} ({written} rows)")

    db_path = os.path.join(cli.out, "database_index.csv")
    with open(db_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "session", "arm", "facing", "partition", "slice_index",
                    "lat", "lon", "x_enu_m", "y_enu_m", "travel_bearing_deg"])
        for r in db_rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], r[5],
                        "%.8f" % r[6], "%.8f" % r[7], "%.3f" % r[8], "%.3f" % r[9],
                        "%.2f" % r[10]])
    print(f"-> {db_path} ({len(db_rows)} rows)")

    gt_path = os.path.join(cli.out, f"positives_{int(cli.threshold_m)}m.npz")
    np.savez_compressed(gt_path, indptr=indptr, indices=indices,
                        threshold_m=np.float64(cli.threshold_m))
    print(f"-> {gt_path} ({len(indices)} pairs, "
          f"{os.path.getsize(gt_path) / 1e6:.1f} MB)")

    proto = {
        "dataset": "springfield-event queries (day+dawn, reversals excluded)",
        "cell": {"sweeps": [s for s in cli.sweeps.split(",") if s],
                 "exclude_reversals_deg": cli.exclude_reversals_deg,
                 "n_queries": int(written), "n_sessions": len(sessions),
                 "n_scored_query_slices_all_cells": int(len(psi))},
        "slice": {"dt_ms": cli.dt_ms,
                  "grid": "eventcv frames each recording from its first timestamp; "
                          "slice i spans [t0 + i*dt, t0 + (i+1)*dt)"},
        "retrieval": {"threshold_m": cli.threshold_m,
                      "n_database": int(len(d["db_xy"])),
                      "database_sessions": {s: list(a) for s, a in DB_ARMS.items()},
                      "positives_per_query_mean": round(float(n_pos.mean()), 2)},
        "projection": {"kind": "equirectangular local ENU, metres",
                       "lat0": lat0, "lon0": lon0},
        "psi": {"definition": "signed camera bearing minus local route bearing, degrees",
                "source": "GPS travel bearing while moving, per-session yaw fit otherwise",
                "reversal_rule": f"|psi| >= {cli.exclude_reversals_deg:g} is a reversal "
                                 f"and is excluded from this export"},
        "benchmark_pipeline": {
            "representation": "accumulate", "eval_resolution": 322,
            "hot_pixel_filter": "eventcv, std 3",
            "background_activity_filter": "off (the published baoff cell)",
            "note": "the exported events are RAW — no hot-pixel or BA filtering has been "
                    "applied, so any representation can be rebuilt from them"},
        "provenance": {"dump": f"dump_{cli.tag}.npz",
                       "orient": f"orient_{cli.tag}.npz",
                       "per_query_csv": os.path.basename(cli.per_query_csv),
                       "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime())},
    }
    proto_path = os.path.join(cli.out, "protocol.json")
    with open(proto_path, "w") as f:
        json.dump(proto, f, indent=2)
    print(f"-> {proto_path}")

    write_readme(cli, os.path.join(cli.out, "README.md"), proto, sessions, geos, n_pos)
    return n_pos


def write_readme(cli, path, proto, sessions, geos, n_pos):
    sweeps = {}
    for sid, sweep, rows in sessions:
        sweeps.setdefault(sweep, [0, 0])
        sweeps[sweep][0] += 1
        sweeps[sweep][1] += len(rows)
    lines = [
        "# Springfield-Event — query set (day + dawn, reversals excluded)",
        "",
        f"{proto['cell']['n_queries']} event queries of {cli.dt_ms} ms each, from "
        f"{len(sessions)} walking passes captured with a Prophesee EVK4 "
        f"({geos[sessions[0][0]]['sensor_wh'][0]}x"
        f"{geos[sessions[0][0]]['sensor_wh'][1]}).",
        "",
        "This is the cell the paper reports: every scored query slice whose sweep is "
        f"day or dawn and whose camera-vs-route bearing satisfies "
        f"`|psi| < {cli.exclude_reversals_deg:g} deg`. Night sweeps and reversed "
        "(back-along-the-route) passes are deliberately absent — the full capture has "
        f"{proto['cell']['n_scored_query_slices_all_cells']} scored query slices in total.",
        "",
        "| sweep | passes | queries |",
        "| --- | --- | --- |",
    ]
    for sw in sorted(sweeps):
        lines.append(f"| {sw} | {sweeps[sw][0]} | {sweeps[sw][1]} |")
    lines += [
        "",
        "## Files",
        "",
        "| file | what |",
        "| --- | --- |",
        "| `manifest.csv` | one row per query: session, sweep, spot cluster, native slice "
        "index, camera-clock window, GPS-clock centre, lat/lon, local ENU, `psi`, and the "
        "number of database rows inside the 25 m radius |",
        "| `events/query_<sweep>/<session>.h5` | the raw events of that session's queries |",
        "| `database_index.csv` | the pooled reference gallery in row order — "
        f"{proto['retrieval']['n_database']} rows — so the positives resolve to places |",
        f"| `positives_{int(cli.threshold_m)}m.npz` | ground truth as CSR (`indptr`, "
        "`indices`): gallery rows within 25 m of each query, in `manifest.csv` order |",
        "| `protocol.json` | the parameters above, machine-readable |",
        "",
        "## Event files",
        "",
        "Each HDF5 holds only the exported slices of one session:",
        "",
        "```",
        "events/x, y, t, p     concatenated events; t is the camera clock in microseconds",
        "slices/ptr            [n+1] offsets, so slice i is events[ptr[i]:ptr[i+1]]",
        "slices/slice_index    the index this slice had in the session's native grid",
        "slices/t_start_us     window start on the camera clock (= t0 + slice_index*dt)",
        "slices/t_end_us       window end, exclusive",
        "slices/t_centre_gps_ns  slice centre on the GPS/host clock",
        "slices/lat, lon       WGS84 position of the slice centre",
        "slices/x_enu_m, y_enu_m   local ENU metres about the shared origin",
        "slices/psi_deg        signed camera bearing minus route bearing",
        "slices/moving         whether psi came from GPS travel (True) or the yaw fit",
        "slices/n_positives_25m  database rows within 25 m",
        "```",
        "",
        "Events are **raw**: the benchmark's hot-pixel filter (eventcv, std 3) is applied "
        "at render time, not here, and the published cell runs with the background-activity "
        "filter off. Any representation can be rebuilt from these files.",
        "",
        "## Protocol",
        "",
        f"- retrieval radius {cli.threshold_m:g} m; "
        f"{proto['retrieval']['positives_per_query_mean']} database rows per query on "
        "average (min {0}, max {1})".format(int(n_pos.min()), int(n_pos.max())),
        "- the gallery is three full-route passes with the camera facing forward, left and "
        "right, plus two extra locations walked forward-then-right — the viewpoint axis "
        "this dataset exists to probe",
        f"- local ENU is equirectangular about "
        f"({proto['projection']['lat0']:.6f}, {proto['projection']['lon0']:.6f})",
        "- `psi` is the camera bearing relative to the local route direction, taken from "
        "the GPS travel bearing while the carrier is moving and from a per-session yaw fit "
        "when they are not. It is the quantity the reversal filter thresholds.",
        "",
        "## Caveats",
        "",
        "- The gallery is a continuous 50 ms walk, ~12 rows per metre, so a query has "
        "thousands of near-duplicate positives. Per-slice recall is the headline metric; "
        "per-pass rates weight each visit equally.",
        "- GPS positions come from a phone track interpolated onto the slice grid; slices "
        "outside GPS coverage or inside a dropout are not part of the scored set and are "
        "absent here.",
        "- Ten of the exported passes are only partly forward-facing: their reversed "
        "slices are excluded, so their `slice_index` values are not contiguous.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"-> {path}")


def stage_events(cli, built, n_pos):
    d, psi, moving, cell, sessions, spots, geos, _ = built
    tot_slices = tot_events = 0
    off = 0
    for n, (sid, sweep, rows) in enumerate(sessions, 1):
        out_path = os.path.join(cli.out, "events", f"query_{sweep}", f"{sid}.h5")
        k = len(rows)
        print(f"\n[{n}/{len(sessions)}] {sweep}/{sid}")
        if not cli.force and already_done(out_path, k):
            with h5py.File(out_path, "r") as g:
                ev = int(g.attrs["n_events"])
            print(f"    cached ({k} slices, {ev / 1e6:.1f} Mev)")
            tot_slices += k
            tot_events += ev
            off += k
            continue
        labels = {"slice_index": d["q_slice"][rows], "psi": psi[rows],
                  "moving": moving[rows], "spot": spots.get(sid, ""),
                  "n_pos": (n_pos[off:off + k] if n_pos is not None
                            else np.zeros(k, np.int32))}
        ns, ne = export_session(
            cli, sid, sweep, os.path.join(cli.root, f"query_{sweep}", sid),
            geos[sid], labels, out_path)
        tot_slices += ns
        tot_events += ne
        off += k
    print(f"\n{len(sessions)} sessions, {tot_slices} slices, "
          f"{tot_events / 1e6:.1f} Mev exported")
    if tot_slices != int(cell.sum()):
        raise SystemExit(f"exported {tot_slices} slices but the cell holds "
                         f"{int(cell.sum())}")


def stage_checksums(cli):
    """SHA-256 of every published file, so a HF upload can be verified end to end."""
    out = []
    for base, _, files in os.walk(cli.out):
        for name in sorted(files):
            if name.endswith(".part") or name == "SHA256SUMS":
                continue
            path = os.path.join(base, name)
            with open(path, "rb") as f:
                h = hashlib.file_digest(f, "sha256").hexdigest()
            out.append((h, os.path.relpath(path, cli.out)))
    path = os.path.join(cli.out, "SHA256SUMS")
    with open(path, "w") as f:
        for h, rel in sorted(out, key=lambda r: r[1]):
            f.write(f"{h}  {rel}\n")
    print(f"-> {path} ({len(out)} files)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="all",
                    choices=("manifest", "events", "all", "checksums"))
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    ap.add_argument("--diag-dir", default=DEFAULT_DIAG)
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="the scored run whose dump/orient define the query set")
    ap.add_argument("--per-query-csv", default=DEFAULT_PER_QUERY,
                    help="source of the spot-cluster labels")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sweeps", default="day,dawn")
    ap.add_argument("--exclude-reversals-deg", type=float, default=135.0)
    ap.add_argument("--threshold-m", type=float, default=25.0)
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--compression", default="gzip", choices=("gzip", "lzf", "none"))
    ap.add_argument("--compression-level", type=int, default=4)
    ap.add_argument("--xy-tol-m", type=float, default=0.01,
                    help="how far a re-derived slice coordinate may sit from the scored "
                         "run's before the export is refused")
    ap.add_argument("--force", action="store_true",
                    help="re-export sessions whose event file is already complete")
    cli = ap.parse_args()

    if cli.stage == "checksums":
        stage_checksums(cli)
        return
    built = build(cli)
    n_pos = None
    if cli.stage in ("manifest", "all"):
        n_pos = stage_manifest(cli, built)
    if cli.stage in ("events", "all"):
        if n_pos is None:
            import csv
            path = os.path.join(cli.out, "manifest.csv")
            if not os.path.exists(path):
                raise SystemExit(f"no {path} — run --stage manifest first")
            with open(path) as f:
                n_pos = np.array([int(r["n_positives_25m"])
                                  for r in csv.DictReader(f)], np.int32)
            if len(n_pos) != int(built[3].sum()):
                raise SystemExit(f"{path} holds {len(n_pos)} rows but the cell has "
                                 f"{int(built[3].sum())} — rerun --stage manifest")
        stage_events(cli, built, n_pos)


if __name__ == "__main__":
    main()
