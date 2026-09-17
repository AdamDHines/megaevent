"""Springfield retrieval diagnostics: per-slice forensics behind the recall numbers.

Run from the repo root, stage by stage::

    pixi run python3 scripts/springfield_diag.py --stage orient    # CPU, run any time
    pixi run python3 scripts/springfield_diag.py --stage dump     # GPU, banks cached
    pixi run python3 scripts/springfield_diag.py --stage analyze  # CPU, needs dump
    pixi run python3 scripts/springfield_diag.py --stage audit    # renders frame pairs

Everything reads the *cached* banks and sidecar CSVs — no model, no re-extraction.

* ``orient`` — per-slice camera bearing from each session's ``orientation.csv`` (100 Hz
  quaternion yaw placed on the GPS clock via the session's camera fit), calibrated against
  the three full database passes whose camera-vs-travel offsets are known (forward ~0,
  left/right ~±90). Replaces the net-displacement session bearing, labels rotation slices
  by angular rate, and audits the old heading column (the ``202103Z`` anomaly).
* ``dump`` — streamed ranking of every query slice against the pooled database: top-20
  rows + cosines, and the **best-correct** database row, cosine and rank (rank = 1 + how
  many database rows outscore it), plus a pooled-descriptor retrieval per visit.
* ``analyze`` — taxonomy (correct-outranked vs correct-nowhere), alias structure, the
  sweep x direction grid, translation-only recall, recall vs camera-offset curve,
  visit-level majority vote, tolerance sweep. Writes one JSON the report is built from.
* ``audit`` — the implementation spot-check: query frames rendered beside their
  GPS-nearest forward-arm database frames. Same scene = the pipeline is sound.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src.imagesets import build_radius_gt                       # noqa: E402
from src.traversegps import local_enu                           # noqa: E402
from springfield_eval import (                                  # noqa: E402
    list_partitions, load_gps, part_label, partition_geometry)
from springfield_full import (                                  # noqa: E402
    DB_ARMS, SWEEPS, _LRUFrames, discover_queries, load_clock, track_bearings)

DEFAULT_RESULTS = ("output/springfield_full/"
                   "results_b_v8_accum_s750_s750_3f4eb54780_accumulate_r322_hp1_baoff.json")
DEFAULT_ROOT = "/media/adam/vprdatasets/megaevent/springfield/sessions"
DEFAULT_BANK_DIR = "/media/adam/vprdatasets/megaevent/evaluations/springfield"
DEFAULT_OUT = "output/springfield_diag"


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# 1. Cached-session loading (no model, no eventcv)
# ---------------------------------------------------------------------------
def load_cached_session(cli, sdir, sid, need_bank=True):
    """Geometry (+ cached bank) for one session, slice grid identical to the eval's.

    Slice counts come from the bank manifests written at extraction time, so the grid is
    the extraction's own, not a re-derivation; ``partition_geometry`` still cross-checks
    each count against the file's timestamp span.
    """
    parts = list_partitions(sdir)
    clock, clock_flag = load_clock(sdir)
    gps = load_gps(sdir)
    latlon, keep, banks, prov, centres = [], [], [], [], []
    for path in parts:
        stem = os.path.join(cli.bank_dir, sid,
                            f"{cli.tag}_{os.path.splitext(os.path.basename(path))[0]}")
        with open(stem + ".json") as f:
            n = json.load(f)["n_slices"]
        ll, kp, _ = partition_geometry(path, n, cli.dt_ms, clock, gps)
        # Baseline banks (springfield_baselines.bank_subset) are stored compact —
        # [n_selected, D] plus a `.rows.npy` of the extracted row indices — because only
        # the rows in the eval's `keep` were ever extracted; the rest are absent, not
        # scored. Restrict keep to the selected rows so those never enter the ranking, and
        # do it whether or not the bank is loaded, so the orient and dump stages stay on
        # the same per-slice grid. The v8 banks are full and carry no `.rows.npy`, so this
        # is a no-op for them.
        rows_path = stem + ".rows.npy"
        idx = np.load(rows_path) if os.path.exists(rows_path) else None
        if idx is not None:
            sel = np.zeros(n, bool)
            sel[idx] = True
            kp = kp & sel
        latlon.append(ll); keep.append(kp)
        # slice-centre laptop time, for placing orientation samples
        with h5py.File(path, "r") as f:
            t0 = int(f["events"]["t"][0])
        c_us = t0 + (np.arange(n, dtype=np.float64) + 0.5) * cli.dt_ms * 1000
        alpha, beta, delta = clock
        centres.append(alpha * (c_us * 1e3) + beta + delta)
        prov.extend((sid, os.path.basename(path), i) for i in range(n))
        if need_bank:
            arr = np.load(stem + ".npy")
            if idx is not None:
                full = np.zeros((n, arr.shape[1]), np.float32)
                full[idx] = arr
                arr = full
            elif arr.shape[0] != n:
                raise SystemExit(f"{stem}: bank rows != manifest n_slices")
            banks.append(arr)
    return {"latlon": np.concatenate(latlon), "keep": np.concatenate(keep),
            "bank": np.concatenate(banks) if need_bank else None,
            "prov": prov, "centres_ns": np.concatenate(centres),
            "clock": clock, "clock_flag": clock_flag, "dir": sdir}


def yaw_at(sdir, centres_ns):
    """(yaw_deg, rate_dps) at each slice centre, from orientation.csv on the laptop clock."""
    rows = np.genfromtxt(os.path.join(sdir, "derived", "orientation.csv"),
                         delimiter=",", names=True, dtype=np.float64)
    t = rows["time_laptop_ns"]
    order = np.argsort(t)
    t, yaw = t[order], np.unwrap(rows["yaw"][order])
    y = np.interp(centres_ns, t, yaw)
    rate = np.gradient(yaw, t / 1e9, edge_order=1)
    r = np.interp(centres_ns, t, rate)
    covered = (centres_ns >= t[0]) & (centres_ns <= t[-1])
    return np.degrees(y), np.abs(np.degrees(r)), covered


# ---------------------------------------------------------------------------
# 2. Stage: orient
# ---------------------------------------------------------------------------
def stage_orient(cli):
    """Per-slice camera-vs-route offset for every query slice.

    The phone is carried, not rig-aligned (a global yaw calibration leaves 61 deg mean
    residual on the forward pass), so yaw is NOT trusted as an absolute heading. The
    query rig faces the walking direction, so while the person is *moving* the camera
    bearing IS the per-slice GPS travel bearing. Yaw contributes two things only: its
    rate labels rotation slices (offset-free), and a per-session offset fit on the
    moving slices lets yaw fill in the bearing while the person stands or spins.
    """
    fwd_xy, fwd_bear, xy0 = [], [], None
    for sid in list(DB_ARMS):
        if DB_ARMS[sid][1] != "forward":
            continue
        s = load_cached_session(cli, os.path.join(cli.root, "database", sid), sid,
                                need_bank=False)
        if xy0 is None:
            k = s["latlon"][s["keep"]]
            xy0 = (float(k[:, 0].mean()), float(k[:, 1].mean()))
        xy = local_enu(s["latlon"][:, 0], s["latlon"][:, 1], *xy0)
        bear = track_bearings(xy)
        fwd_xy.append(xy[s["keep"]])
        fwd_bear.append(bear[s["keep"]])
    fwd_xy = np.concatenate(fwd_xy)
    fwd_bear = np.concatenate(fwd_bear)

    # Design values: the camera mounts sit at 0/+90/-90 from travel by construction.
    out = {"__db_arm_angle__": {"forward": 0.0, "left": 90.0, "right": -90.0},
           "__origin__": list(xy0), "__psi_source__": "gps-travel (yaw fill-in)"}
    arrays = {"fwd_xy": fwd_xy, "fwd_bear": fwd_bear}
    for sweep, sid, sdir in discover_queries(cli.root, None, SWEEPS):
        s = load_cached_session(cli, sdir, sid, need_bank=False)
        xy = local_enu(s["latlon"][:, 0], s["latlon"][:, 1], *xy0)
        kept = np.flatnonzero(s["keep"])
        k_xy, k_ns = xy[kept], s["centres_ns"][kept]
        # per-slice speed + travel bearing over a ~1 s window
        w = 10
        lo = np.maximum(np.arange(len(kept)) - w, 0)
        hi = np.minimum(np.arange(len(kept)) + w, len(kept) - 1)
        step = k_xy[hi] - k_xy[lo]
        dt = np.maximum((k_ns[hi] - k_ns[lo]) / 1e9, 1e-6)
        speed = np.linalg.norm(step, axis=1) / dt
        travel = np.degrees(np.arctan2(step[:, 0], step[:, 1])) % 360.0
        moving = speed > 0.6

        yaw, rate, cov = yaw_at(sdir, s["centres_ns"])
        yaw_k, rate_k, cov_k = yaw[kept], rate[kept], cov[kept]
        # per-session yaw->bearing: sign and offset fit on the moving slices
        calib_resid = None
        cam = travel.copy()
        fit_m = moving & cov_k
        if fit_m.sum() >= 20:
            best = None
            for sign in (+1.0, -1.0):
                d = np.radians(travel[fit_m] - sign * yaw_k[fit_m])
                off = np.degrees(np.arctan2(np.sin(d).mean(), np.cos(d).mean()))
                r = wrap180(travel[fit_m] - (sign * yaw_k[fit_m] + off))
                if best is None or np.abs(r).mean() < best[0]:
                    best = (float(np.abs(r).mean()), sign, off)
            calib_resid, sign, off = best
            cam = np.where(moving, travel, wrap180(sign * yaw_k + off) % 360.0)

        near = np.argmin(np.linalg.norm(
            fwd_xy[None, :, :] - k_xy[:, None, :], axis=2), axis=1)
        route = fwd_bear[near]
        psi = wrap180(cam - route)                   # camera vs route, signed
        arrays[f"{sid}_psi"] = psi.astype(np.float32)
        arrays[f"{sid}_rate"] = rate_k.astype(np.float32)
        arrays[f"{sid}_cov"] = cov_k & (moving | (calib_resid is not None))
        arrays[f"{sid}_moving"] = moving
        steady = moving & (rate_k < 30.0)
        out[sid] = {
            "sweep": sweep, "n_kept": int(len(kept)),
            "moving_frac": round(float(moving.mean()), 3),
            "steady_frac": round(float(steady.mean()), 3),
            "yaw_fit_resid_deg": round(calib_resid, 1) if calib_resid is not None else None,
            "median_abs_psi_deg": (round(float(np.median(np.abs(psi[steady]))), 1)
                                   if steady.any() else None),
            "median_rate_dps": (round(float(np.median(rate_k[cov_k])), 1)
                                if cov_k.any() else None),
        }
    os.makedirs(cli.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(cli.out_dir, f"orient_{cli.tag}.npz"), **arrays)
    with open(os.path.join(cli.out_dir, f"orient_{cli.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> orient_{cli.tag}.json / .npz ({len(out) - 3} query sessions)")

    # Audit against the old net-displacement heading column
    with open(cli.results) as f:
        rows = {r["session"]: r for r in json.load(f)["per_session"]}
    print("\nheading audit (|old net-displacement offset| vs |new camera-route offset|):")
    for sid, o in out.items():
        if sid.startswith("__") or o["median_abs_psi_deg"] is None:
            continue
        old = rows.get(sid, {}).get("heading_offset_deg")
        if old is not None and abs(abs(old) - o["median_abs_psi_deg"]) > 45:
            print(f"  {sid} ({o['sweep']}): old {old:.0f} -> new "
                  f"{o['median_abs_psi_deg']:.0f} deg  DISAGREES "
                  f"(R@1 {rows[sid]['r1_slice']})")


# ---------------------------------------------------------------------------
# 3. Stage: dump (GPU)
# ---------------------------------------------------------------------------
def load_all(cli, need_bank=True):
    db, order = {}, list(DB_ARMS)
    xy0 = None
    for sid in order:
        s = load_cached_session(cli, os.path.join(cli.root, "database", sid), sid,
                                need_bank)
        if xy0 is None:
            k = s["latlon"][s["keep"]]
            xy0 = (float(k[:, 0].mean()), float(k[:, 1].mean()))
        s["xy"] = local_enu(s["latlon"][:, 0], s["latlon"][:, 1], *xy0)
        s["bear"] = track_bearings(s["xy"])
        db[sid] = s
        print(f"  db {DB_ARMS[sid][0]}: {int(s['keep'].sum())} kept")
    queries = {}
    for sweep, sid, sdir in discover_queries(cli.root, None, SWEEPS):
        s = load_cached_session(cli, sdir, sid, need_bank)
        s["xy"] = local_enu(s["latlon"][:, 0], s["latlon"][:, 1], *xy0)
        s["sweep"] = sweep
        queries[sid] = s
    return db, queries, xy0


def stage_dump(cli):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    db, queries, xy0 = load_all(cli)
    db_xy = np.concatenate([db[s]["xy"][db[s]["keep"]] for s in db])
    db_desc = np.concatenate([db[s]["bank"][db[s]["keep"]] for s in db])
    db_bear = np.concatenate([db[s]["bear"][db[s]["keep"]] for s in db])
    db_prov = [p for s in db for p, k in zip(db[s]["prov"], db[s]["keep"]) if k]
    for s in db:
        db[s]["bank"] = None
    db_t = torch.from_numpy(db_desc)
    print(f"database {len(db_xy)} rows")

    K = 20
    sids = list(queries)
    q_desc = np.concatenate([queries[s]["bank"][queries[s]["keep"]] for s in sids])
    q_xy = np.concatenate([queries[s]["xy"][queries[s]["keep"]] for s in sids])
    q_sid = np.concatenate([np.full(int(queries[s]["keep"].sum()), i, dtype=np.int32)
                            for i, s in enumerate(sids)])
    q_slice = np.concatenate([np.flatnonzero(queries[s]["keep"]).astype(np.int32)
                              for s in sids])
    pooled = np.stack([queries[s]["bank"][queries[s]["keep"]].mean(axis=0) for s in sids])
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
    for s in sids:
        queries[s]["bank"] = None
    n_q = len(q_desc)
    print(f"queries {n_q} slices over {len(sids)} sessions")

    gt = build_radius_gt(db_xy, q_xy, cli.threshold_m)
    top_i = np.zeros((K, n_q), np.int32)
    top_c = np.zeros((K, n_q), np.float32)
    bc_i = np.zeros(n_q, np.int32)
    bc_c = np.full(n_q, -2.0, np.float32)
    bc_rank = np.zeros(n_q, np.int64)
    qt = torch.from_numpy(q_desc)
    DBC, QC = 20000, 4096
    for pass_no in (0, 1):
        for d in range(0, len(db_xy), DBC):
            part = db_t[d:d + DBC].to(device)
            for s0 in range(0, n_q, QC):
                block = qt[s0:s0 + QC].to(device)
                sim = part @ block.T                          # [dbc, qc]
                cols = slice(s0, s0 + block.size(0))
                if pass_no == 0:
                    v, i = torch.topk(sim, min(K, sim.size(0)), dim=0)
                    cv = torch.cat([torch.from_numpy(top_c[:, cols]).to(device), v])
                    ci = torch.cat([torch.from_numpy(top_i[:, cols]).to(device),
                                    (i + d).to(torch.int32)])
                    nv, sel = torch.topk(cv, K, dim=0)
                    top_c[:, cols] = nv.cpu().numpy()
                    top_i[:, cols] = torch.gather(ci, 0, sel).cpu().numpy()
                    g = torch.from_numpy(gt[d:d + DBC, cols]).to(device)
                    masked = sim.masked_fill(~g, -2.0)
                    mv, mi = masked.max(dim=0)
                    upd = (mv > torch.from_numpy(bc_c[cols]).to(device))
                    mv_c, mi_c = mv.cpu().numpy(), (mi + d).cpu().numpy()
                    u = upd.cpu().numpy()
                    bc_c[cols] = np.where(u, mv_c, bc_c[cols])
                    bc_i[cols] = np.where(u, mi_c, bc_i[cols])
                else:
                    thr = torch.from_numpy(bc_c[cols]).to(device)
                    bc_rank[cols] += (sim > thr[None, :]).sum(dim=0).cpu().numpy()
                del sim
            del part
        torch.cuda.empty_cache()
        print(f"pass {pass_no} done")
    bc_rank += 1

    # pooled retrieval, streamed like the rest
    p_c = np.full(len(sids), -2.0, np.float32)
    p_i = np.zeros(len(sids), np.int32)
    pt = torch.from_numpy(pooled)
    for d in range(0, len(db_xy), DBC):
        part = db_t[d:d + DBC].to(device)
        sim = part @ pt.T.to(device)
        v, i = sim.max(dim=0)
        v, i = v.cpu().numpy(), (i.cpu().numpy() + d)
        u = v > p_c
        p_c, p_i = np.where(u, v, p_c), np.where(u, i, p_i).astype(np.int32)
        del sim, part
    torch.cuda.empty_cache()

    os.makedirs(cli.out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(cli.out_dir, f"dump_{cli.tag}.npz"),
        sids=np.array(sids), sweeps=np.array([queries[s]["sweep"] for s in sids]),
        q_sid=q_sid, q_slice=q_slice, q_xy=q_xy.astype(np.float32),
        top_i=top_i, top_c=top_c, bc_i=bc_i, bc_c=bc_c, bc_rank=bc_rank,
        pooled_i=p_i, pooled_c=p_c,
        db_xy=db_xy.astype(np.float32), db_bear=db_bear.astype(np.float32),
        db_arm=np.array([DB_ARMS[p[0]][0] for p in db_prov]),
        db_sid=np.array([p[0] for p in db_prov]),
        db_part=np.array([p[1] for p in db_prov]),
        db_slice=np.array([p[2] for p in db_prov], dtype=np.int32))
    print(f"-> dump_{cli.tag}.npz")


# ---------------------------------------------------------------------------
# 4. Stage: analyze
# ---------------------------------------------------------------------------
def stage_analyze(cli):
    d = np.load(os.path.join(cli.out_dir, f"dump_{cli.tag}.npz"), allow_pickle=False)
    o = np.load(os.path.join(cli.out_dir, f"orient_{cli.tag}.npz"), allow_pickle=False)
    with open(os.path.join(cli.out_dir, f"orient_{cli.tag}.json")) as f:
        orient = json.load(f)
    arm_angles = orient["__db_arm_angle__"]
    sids = [str(s) for s in d["sids"]]
    sweeps = [str(s) for s in d["sweeps"]]
    q_sid, q_xy = d["q_sid"], d["q_xy"]
    top_i, top_c = d["top_i"], d["top_c"]
    bc_c, bc_rank = d["bc_c"], d["bc_rank"]
    db_xy, db_bear, db_arm = d["db_xy"], d["db_bear"], d["db_arm"]

    top1_d = np.linalg.norm(db_xy[top_i[0]] - q_xy, axis=1)
    hit1 = top1_d <= cli.threshold_m
    topk_d = np.linalg.norm(db_xy[top_i] - q_xy[None], axis=2)   # [K, n]

    psi = np.concatenate([o[f"{s}_psi"] for s in sids])
    rate = np.concatenate([o[f"{s}_rate"] for s in sids])
    cov = np.concatenate([o[f"{s}_cov"] for s in sids])
    moving = np.concatenate([o[f"{s}_moving"] for s in sids])
    steady = moving & (rate < 30.0)
    arm_off = np.min(np.abs(wrap180(psi[:, None] -
                                    np.array(list(arm_angles.values()))[None, :])), axis=1)

    res = {"tag": cli.tag, "threshold_m": cli.threshold_m,
           "psi_source": orient["__psi_source__"], "db_arm_angles": arm_angles}
    sweep_arr = np.array([sweeps[i] for i in q_sid])

    res["micro_check"] = {sw: round(float(hit1[sweep_arr == sw].mean()), 4)
                          for sw in SWEEPS}
    res["tolerance_sweep"] = {
        str(r): {"r1": round(float((top1_d <= r).mean()), 4),
                 "r10": round(float((topk_d[:10] <= r).any(axis=0).mean()), 4)}
        for r in (10, 15, 20, 25, 30, 35, 40, 50)}
    res["translation_only_micro_r1"] = {
        sw: round(float(hit1[(sweep_arr == sw) & steady].mean()), 4) for sw in SWEEPS}
    res["rotation_slices_micro_r1"] = {
        sw: (round(float(hit1[(sweep_arr == sw) & cov & ~steady].mean()), 4)
             if ((sweep_arr == sw) & cov & ~steady).any() else None) for sw in SWEEPS}
    bins = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 90), (90, 180)]
    res["recall_vs_arm_offset"] = {
        f"{a}-{b}": {"n": int(((arm_off >= a) & (arm_off < b) & steady).sum()),
                     "r1": (round(float(hit1[(arm_off >= a) & (arm_off < b) & steady]
                                        .mean()), 4)
                            if ((arm_off >= a) & (arm_off < b) & steady).any() else None)}
        for a, b in bins}

    per_session, cells = [], {}
    for i, sid in enumerate(sids):
        m = q_sid == i
        st = steady[m]
        med_psi = float(np.median(np.abs(psi[m][st]))) if st.any() else None
        direction = (None if med_psi is None else
                     "with" if med_psi < 45 else "against" if med_psi > 135 else "cross")
        r1 = float(hit1[m].mean())
        ranks = bc_rank[m]
        margin = top_c[0][m] - bc_c[m]
        alias = {}
        if (~hit1[m]).any():
            w = np.flatnonzero(m)[~hit1[m]]
            pos = db_xy[top_i[0][w]]
            centre = pos.mean(axis=0)
            alias = {"n_wrong": len(w),
                     "modal_spread_m": round(float(np.median(
                         np.linalg.norm(pos - centre, axis=1))), 1),
                     "wrong_dist_med_m": round(float(np.median(top1_d[w])), 1),
                     "wrong_arm": {str(a): int(c) for a, c in
                                   zip(*np.unique(db_arm[top_i[0][w]],
                                                  return_counts=True))}}
        taxonomy = ("ok" if r1 >= 0.5 else
                    "outranked" if float(np.median(ranks)) <= 100 else "nowhere")
        row = {"session": sid, "sweep": sweeps[i], "r1": round(r1, 4),
               "direction": direction, "median_abs_psi": med_psi,
               "steady_frac": round(float(st.mean()), 3),
               "bc_rank_median": int(np.median(ranks)),
               "bc_cos_margin_median": round(float(np.median(margin)), 4),
               "taxonomy": taxonomy, "alias": alias}
        per_session.append(row)
        if direction:
            cells.setdefault(f"{sweeps[i]}/{direction}", []).append(r1)
    res["per_session"] = per_session
    res["grid"] = {k: {"n": len(v), "mean_r1": round(float(np.mean(v)), 4)}
                   for k, v in sorted(cells.items())}

    # visit-level: majority vote over slice top-1 positions
    votes = {}
    for i, sid in enumerate(sids):
        m = np.flatnonzero(q_sid == i)
        pos = db_xy[top_i[0][m]]
        support = (np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
                   <= cli.threshold_m).sum(axis=1)
        best = pos[np.argmax(support)]
        ok = bool((np.linalg.norm(q_xy[m] - best, axis=1) <= cli.threshold_m).any())
        votes[sid] = ok
    res["visit_majority_vote_acc"] = {
        sw: round(float(np.mean([votes[s] for i, s in enumerate(sids)
                                 if sweeps[i] == sw])), 4) for sw in SWEEPS}
    pooled_d = np.array([
        np.linalg.norm(q_xy[q_sid == i] - db_xy[d["pooled_i"][i]], axis=1).min()
        for i in range(len(sids))])
    res["visit_pooled_desc_acc"] = {
        sw: round(float((pooled_d[[i for i, s2 in enumerate(sweeps) if s2 == sw]]
                         <= cli.threshold_m).mean()), 4) for sw in SWEEPS}

    out = os.path.join(cli.out_dir, f"analysis_{cli.tag}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("micro_check", "grid",
                                          "visit_majority_vote_acc",
                                          "visit_pooled_desc_acc",
                                          "translation_only_micro_r1")}, indent=1))
    print(f"-> {out}")


# ---------------------------------------------------------------------------
# 5. Stage: figures (strips, alias maps, summary curves)
# ---------------------------------------------------------------------------
def _slice_to_part(cli, sdir, sid, slice_idx):
    """Session-level slice index -> (partition path, local index) via the manifests."""
    parts = list_partitions(sdir)
    n = []
    for p in parts:
        stem = os.path.join(cli.bank_dir, sid,
                            f"{cli.tag}_{os.path.splitext(os.path.basename(p))[0]}")
        with open(stem + ".json") as f:
            n.append(json.load(f)["n_slices"])
    edges = np.concatenate([[0], np.cumsum(n)])
    part = int(np.searchsorted(edges, slice_idx, side="right") - 1)
    return parts[part], int(slice_idx - edges[part])


def stage_figures(cli):
    d = np.load(os.path.join(cli.out_dir, f"dump_{cli.tag}.npz"))
    with open(os.path.join(cli.out_dir, f"analysis_{cli.tag}.json")) as f:
        analysis = json.load(f)
    sids = [str(s) for s in d["sids"]]
    sweeps = [str(s) for s in d["sweeps"]]
    q_sid, q_xy, q_slice = d["q_sid"], d["q_xy"], d["q_slice"]
    top_i, top_c = d["top_i"], d["top_c"]
    bc_i, bc_c, bc_rank = d["bc_i"], d["bc_c"], d["bc_rank"]
    db_xy = d["db_xy"]
    db_sid = [str(s) for s in d["db_sid"]]
    db_part = [str(s) for s in d["db_part"]]
    db_slice = d["db_slice"]
    top1_d = np.linalg.norm(db_xy[top_i[0]] - q_xy, axis=1)
    hit1 = top1_d <= cli.threshold_m

    frames = _LRUFrames(cli, cli.representation)
    strip_dir = os.path.join(cli.out_dir, f"strips_{cli.tag}")
    os.makedirs(strip_dir, exist_ok=True)
    db_paths = {sid: {os.path.basename(p): p for p in
                      list_partitions(os.path.join(cli.root, "database", sid))}
                for sid in DB_ARMS}
    only = {s.strip() for s in cli.sessions.split(",") if s.strip()} or set(sids)

    def small(img):
        return img[::4, ::4]                       # 720x1280 -> 180x320 panels

    for i, sid in enumerate(sids):
        if sid not in only:
            continue
        m = np.flatnonzero(q_sid == i)
        hits = m[hit1[m]]
        rep = hits[len(hits) // 2] if len(hits) else m[len(m) // 2]
        misses = m[~hit1[m]]
        picks = [("rep", rep)]
        if len(misses):
            picks.append(("miss", misses[np.argmax(top1_d[misses])]))
        sdir = os.path.join(cli.root, f"query_{sweeps[i]}", sid)
        for name, q in picks:
            q_path, q_loc = _slice_to_part(cli, sdir, sid, int(q_slice[q]))
            panels = [(small(frames.frame(q_path, q_loc)),
                       f"query #{q_loc}", "#333333")]
            for c in range(2):
                r = int(top_i[c, q])
                dist = float(np.linalg.norm(db_xy[r] - q_xy[q]))
                ok = dist <= cli.threshold_m
                panels.append((
                    small(frames.frame(db_paths[db_sid[r]][db_part[r]],
                                       int(db_slice[r]))),
                    f"top-{c + 1} {DB_ARMS[db_sid[r]][0]} {dist:.0f}m "
                    f"cos {top_c[c, q]:.3f}", "#2a9d4e" if ok else "#d43d3d"))
            r = int(bc_i[q])
            panels.append((
                small(frames.frame(db_paths[db_sid[r]][db_part[r]], int(db_slice[r]))),
                f"best correct: rank {int(bc_rank[q])} cos {bc_c[q]:.3f}", "#3572b0"))
            fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 2.15),
                                     constrained_layout=True)
            for ax, (img, title, col) in zip(axes, panels):
                ax.imshow(img)
                ax.set_title(title, fontsize=7, color=col)
                for spine in ax.spines.values():
                    spine.set_edgecolor(col); spine.set_linewidth(2)
                ax.set_xticks([]); ax.set_yticks([])
            fig.savefig(os.path.join(strip_dir, f"{sweeps[i]}_{sid}_{name}.jpg"),
                        dpi=96, bbox_inches="tight", facecolor="white",
                        pil_kwargs={"quality": 82})
            plt.close(fig)

        # alias mini-map for sessions with misses
        if len(misses):
            fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
            ax.scatter(db_xy[::20, 0], db_xy[::20, 1], s=1, c="#d5d4d0", linewidths=0)
            sub = misses[:: max(1, len(misses) // 60)]
            for q in sub:
                t1 = db_xy[top_i[0, q]]
                ax.plot([q_xy[q, 0], t1[0]], [q_xy[q, 1], t1[1]],
                        c="#d43d3d", lw=0.4, alpha=0.5)
            ax.scatter(q_xy[m, 0], q_xy[m, 1], s=4, c="#3572b0", linewidths=0,
                       label="query pass")
            ax.scatter(db_xy[top_i[0, misses], 0], db_xy[top_i[0, misses], 1], s=5,
                       c="#d43d3d", linewidths=0, label="wrong top-1")
            if len(hits):
                ax.scatter(db_xy[top_i[0, hits], 0], db_xy[top_i[0, hits], 1], s=5,
                           c="#2a9d4e", linewidths=0, label="correct top-1")
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(fontsize=6, loc="best", framealpha=0.9)
            ax.set_title(f"{sid} top-1 landings", fontsize=8)
            fig.savefig(os.path.join(strip_dir, f"{sweeps[i]}_{sid}_alias.jpg"),
                        dpi=110, bbox_inches="tight", facecolor="white",
                        pil_kwargs={"quality": 82})
            plt.close(fig)
        print(f"  {sid} done")

    # summary curves
    fig, ax = plt.subplots(figsize=(4.6, 3.4), constrained_layout=True)
    tol = analysis["tolerance_sweep"]
    xs = sorted(int(k) for k in tol)
    ax.plot(xs, [tol[str(x)]["r1"] for x in xs], marker="o", label="R@1")
    ax.plot(xs, [tol[str(x)]["r10"] for x in xs], marker="s", label="R@10")
    ax.axvline(cli.threshold_m, color="#bbbbbb", ls="--", lw=0.8)
    ax.set_xlabel("tolerance radius (m)"); ax.set_ylabel("recall"); ax.legend()
    fig.savefig(os.path.join(cli.out_dir, f"tolerance_{cli.tag}.png"), dpi=140,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.4), constrained_layout=True)
    curve = analysis["recall_vs_arm_offset"]
    ks = [k for k, v in curve.items() if v["r1"] is not None]
    ax.bar(range(len(ks)), [curve[k]["r1"] for k in ks], color="#3572b0")
    ax.set_xticks(range(len(ks)), ks, fontsize=8)
    ax.set_xlabel("camera offset to nearest DB arm (deg, steady slices)")
    ax.set_ylabel("R@1")
    fig.savefig(os.path.join(cli.out_dir, f"arm_offset_{cli.tag}.png"), dpi=140,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    margin = top_c[0] - bc_c
    fig, ax = plt.subplots(figsize=(4.6, 3.4), constrained_layout=True)
    ax.hist(margin[~hit1], bins=40, range=(0, 0.4), color="#d43d3d", alpha=0.7,
            label="missed slices")
    ax.set_xlabel("cosine margin: top-1 over best-correct"); ax.set_ylabel("slices")
    ax.legend()
    fig.savefig(os.path.join(cli.out_dir, f"margin_{cli.tag}.png"), dpi=140,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {strip_dir} + summary curves")


# ---------------------------------------------------------------------------
# 6. Stage: flipbook (query | top-1 | GPS-closest, one PNG per query session)
# ---------------------------------------------------------------------------
def stage_flipbook(cli):
    """A flick-through set: for every day/dawn query session, the median kept slice
    beside its top-1 match and the GPS-closest database frame — recording quality,
    match quality, and what the right answer looks like, in one glance."""
    d = np.load(os.path.join(cli.out_dir, f"dump_{cli.tag}.npz"))
    with open(cli.results) as f:
        spot = {r["session"]: r["spot"] for r in json.load(f)["per_session"]}
    sids = [str(s) for s in d["sids"]]
    sweeps = [str(s) for s in d["sweeps"]]
    q_sid, q_xy, q_slice = d["q_sid"], d["q_xy"], d["q_slice"]
    top_i, top_c = d["top_i"], d["top_c"]
    db_xy = d["db_xy"]
    db_sid = [str(s) for s in d["db_sid"]]
    db_part = [str(s) for s in d["db_part"]]
    db_slice = d["db_slice"]

    frames = _LRUFrames(cli, cli.representation)
    out_dir = os.path.join(cli.out_dir, "top1_vs_gps")
    os.makedirs(out_dir, exist_ok=True)
    db_paths = {sid: {os.path.basename(p): p for p in
                      list_partitions(os.path.join(cli.root, "database", sid))}
                for sid in DB_ARMS}
    for i, sid in enumerate(sids):
        if sweeps[i] not in ("day", "dawn"):
            continue
        m = np.flatnonzero(q_sid == i)
        mid = m[len(m) // 2]
        q_path, q_loc = _slice_to_part(
            cli, os.path.join(cli.root, f"query_{sweeps[i]}", sid), sid,
            int(q_slice[mid]))
        r1 = int(top_i[0, mid])
        d1 = float(np.linalg.norm(db_xy[r1] - q_xy[mid]))
        ok = d1 <= cli.threshold_m
        g = int(np.argmin(np.linalg.norm(db_xy - q_xy[mid], axis=1)))
        dg = float(np.linalg.norm(db_xy[g] - q_xy[mid]))

        panels = [
            (frames.frame(q_path, q_loc),
             f"query {sid} ({sweeps[i]}, {spot[sid]})  slice #{q_loc}", "#333333"),
            (frames.frame(db_paths[db_sid[r1]][db_part[r1]], int(db_slice[r1])),
             f"top-1  {DB_ARMS[db_sid[r1]][0]}  {d1:.0f} m  cos {top_c[0, mid]:.3f}",
             "#2a9d4e" if ok else "#d43d3d"),
            (frames.frame(db_paths[db_sid[g]][db_part[g]], int(db_slice[g])),
             f"GPS-closest DB  {DB_ARMS[db_sid[g]][0]}  {dg:.1f} m", "#3572b0"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(16.2, 3.4), constrained_layout=True)
        for ax, (img, title, col) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(title, fontsize=9, color=col)
            for spine in ax.spines.values():
                spine.set_edgecolor(col); spine.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])
        out = os.path.join(out_dir,
                           f"{sweeps[i]}_{spot[sid]}_{sid}_{'hit' if ok else 'miss'}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {os.path.basename(out)}")
    print(f"-> {out_dir}")


# ---------------------------------------------------------------------------
# 7. Stage: slices (every kept slice of every day/dawn query, PIL-composed)
# ---------------------------------------------------------------------------
def stage_slices(cli):
    """query | top-1 | GPS-closest for EVERY kept slice, one JPEG per slice.

    One subfolder per query session, files ordered by slice index, so flicking
    through a folder scrubs the pass in time. ~10k images — composed with PIL
    (matplotlib per-figure overhead would triple the wall time), DB frames deduped
    per session (consecutive slices usually match the same handful of frames),
    written to the data drive (default out lives under the bank dir, symlinked from
    --out-dir/all_slices).
    """
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 15)

    d = np.load(os.path.join(cli.out_dir, f"dump_{cli.tag}.npz"))
    with open(cli.results) as f:
        spot = {r["session"]: r["spot"] for r in json.load(f)["per_session"]}
    sids = [str(s) for s in d["sids"]]
    sweeps = [str(s) for s in d["sweeps"]]
    q_sid, q_xy, q_slice = d["q_sid"], d["q_xy"], d["q_slice"]
    top_i, top_c = d["top_i"], d["top_c"]
    db_xy = d["db_xy"]
    db_sid = [str(s) for s in d["db_sid"]]
    db_part = [str(s) for s in d["db_part"]]
    db_slice = d["db_slice"]

    root_out = os.path.join(cli.bank_dir, f"slices_{cli.tag}")
    os.makedirs(root_out, exist_ok=True)
    link = os.path.join(cli.out_dir, "all_slices")
    if not os.path.islink(link) and not os.path.exists(link):
        os.symlink(os.path.abspath(root_out), link)

    frames = _LRUFrames(cli, cli.representation)
    db_paths = {sid: {os.path.basename(p): p for p in
                      list_partitions(os.path.join(cli.root, "database", sid))}
                for sid in DB_ARMS}
    C = {"q": (51, 51, 51), "hit": (42, 157, 78), "miss": (212, 61, 61),
         "gps": (53, 114, 176)}
    only = {s.strip() for s in cli.sessions.split(",") if s.strip()} or set(sids)

    def panel(img, label, colour):
        img = Image.fromarray(img[::2, ::2])                 # 360 x 640
        tile = Image.new("RGB", (img.width + 6, img.height + 30), "white")
        drw = ImageDraw.Draw(tile)
        drw.rectangle([0, 24, tile.width - 1, tile.height - 1], outline=colour, width=3)
        drw.text((6, 4), label, fill=colour, font=font)
        tile.paste(img, (3, 27))
        return tile

    for i, sid in enumerate(sids):
        if sweeps[i] not in ("day", "dawn") or sid not in only:
            continue
        sdir = os.path.join(cli.root, f"query_{sweeps[i]}", sid)
        out_dir = os.path.join(root_out, f"{sweeps[i]}_{spot[sid]}_{sid}")
        os.makedirs(out_dir, exist_ok=True)
        m = np.flatnonzero(q_sid == i)
        # nearest DB row per slice, vectorised in chunks
        gps_near = np.empty(len(m), np.int64)
        for a in range(0, len(m), 512):
            blk = q_xy[m[a:a + 512]]
            gps_near[a:a + 512] = np.argmin(
                np.linalg.norm(db_xy[None, :, :] - blk[:, None, :], axis=2), axis=1)
        cache = {}

        def db_frame(row):
            key = (db_sid[row], db_part[row], int(db_slice[row]))
            if key not in cache:
                cache[key] = frames.frame(db_paths[key[0]][key[1]], key[2])[::1]
            return cache[key]

        n_done = 0
        for j, q in enumerate(m):
            q_path, q_loc = _slice_to_part(cli, sdir, sid, int(q_slice[q]))
            r1 = int(top_i[0, q])
            d1 = float(np.linalg.norm(db_xy[r1] - q_xy[q]))
            ok = d1 <= cli.threshold_m
            out_jpg = os.path.join(
                out_dir, f"q{q_loc:05d}_{'hit' if ok else 'miss'}_{d1:03.0f}m.jpg")
            if os.path.exists(out_jpg):        # resume: a killed run costs nothing
                n_done += 1
                continue
            g = int(gps_near[j])
            dg = float(np.linalg.norm(db_xy[g] - q_xy[q]))
            tiles = [
                panel(frames.frame(q_path, q_loc),
                      f"query #{q_loc}  t+{j * cli.dt_ms / 1000:.1f}s", C["q"]),
                panel(db_frame(r1),
                      f"top-1  {DB_ARMS[db_sid[r1]][0]}  {d1:.0f} m  "
                      f"cos {top_c[0, q]:.3f}", C["hit"] if ok else C["miss"]),
                panel(db_frame(g),
                      f"GPS-closest  {DB_ARMS[db_sid[g]][0]}  {dg:.1f} m", C["gps"]),
            ]
            sheet = Image.new("RGB", (sum(t.width for t in tiles) + 16,
                                      tiles[0].height + 8), "white")
            x = 4
            for t in tiles:
                sheet.paste(t, (x, 4))
                x += t.width + 4
            sheet.save(out_jpg, quality=85)
            n_done += 1
        cache.clear()
        print(f"  {sweeps[i]}_{spot[sid]}_{sid}: {n_done} slices")
    print(f"-> {root_out} (symlinked at {link})")


# ---------------------------------------------------------------------------
# 8. Stage: audit (frame pairs)
# ---------------------------------------------------------------------------
def stage_audit(cli):
    """Query frame vs GPS-nearest forward-arm database frame, side by side."""
    d = np.load(os.path.join(cli.out_dir, f"dump_{cli.tag}.npz"))
    sids = [str(s) for s in d["sids"]]
    sweeps = [str(s) for s in d["sweeps"]]
    fwd = np.array([str(a).endswith("forward") for a in d["db_arm"]])
    fwd_rows = np.flatnonzero(fwd)
    db_xy = d["db_xy"]
    pick = [s.strip() for s in cli.audit_sessions.split(",") if s.strip()]
    frames = _LRUFrames(cli, cli.representation)
    out_dir = os.path.join(cli.out_dir, "audit")
    os.makedirs(out_dir, exist_ok=True)
    db_paths = {sid: {os.path.basename(p): p for p in
                      list_partitions(os.path.join(cli.root, "database", sid))}
                for sid in DB_ARMS}
    for sid in pick:
        i = sids.index(sid)
        m = np.flatnonzero(d["q_sid"] == i)
        mid = m[len(m) // 2]
        q_pos = d["q_xy"][mid]
        near = fwd_rows[np.argmin(np.linalg.norm(db_xy[fwd_rows] - q_pos, axis=1))]
        sweep = sweeps[i]
        sdir = os.path.join(cli.root, f"query_{sweep}", sid)
        q_parts = {os.path.basename(p): p for p in list_partitions(sdir)}
        # provenance of both frames
        q_part = [p for p in q_parts][0]
        q_sl = int(d["q_slice"][mid])
        n_sid, n_part, n_sl = (str(d["db_sid"][near]), str(d["db_part"][near]),
                               int(d["db_slice"][near]))
        dist = float(np.linalg.norm(db_xy[near] - q_pos))
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), constrained_layout=True)
        ax[0].imshow(frames.frame(q_parts[q_part], q_sl))
        ax[0].set_title(f"query {sid} ({sweep}) {part_label(q_part)}#{q_sl}", fontsize=9)
        ax[1].imshow(frames.frame(db_paths[n_sid][n_part], n_sl))
        ax[1].set_title(f"GPS-nearest fwd DB frame  {DB_ARMS[n_sid][0]} "
                        f"{part_label(n_part)}#{n_sl}  ({dist:.1f} m away)", fontsize=9)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        out = os.path.join(out_dir, f"audit_{sweep}_{sid}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"-> {out}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True,
                    choices=("orient", "dump", "analyze", "figures", "flipbook",
                             "slices", "audit"))
    ap.add_argument("--sessions", default="",
                    help="figures stage: comma-separated session subset (default all)")
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help="results json of the arm to diagnose (supplies tag + settings)")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--audit-sessions", default="",
                    help="audit stage: comma-separated query session ids")
    cli = ap.parse_args()
    with open(cli.results) as f:
        res = json.load(f)
    cli.tag = res["tag"]
    cli.dt_ms = res["dt_ms"]
    cli.threshold_m = res["threshold_m"]
    cli.representation = res["representation"]
    # fields _LRUFrames/make_dataset read off the cli namespace
    cli.no_hot_pixel = not res["hot_pixel"]
    cli.no_event_filter = res["filter_dt_us"] is None
    cli.event_filter_dt_us = res["filter_dt_us"] or 50_000
    {"orient": stage_orient, "dump": stage_dump, "analyze": stage_analyze,
     "figures": stage_figures, "flipbook": stage_flipbook, "slices": stage_slices,
     "audit": stage_audit}[cli.stage](cli)


if __name__ == "__main__":
    main()
