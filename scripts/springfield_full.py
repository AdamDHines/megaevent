"""Springfield-Event full dataset: pooled 7-session database vs 84 spot queries.

Run from the repo root::

    CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python3 scripts/springfield_full.py \
        --allow-unrepacked

The 2026-08-29 full capture (``docs/springfield_capture_plan.md``): three same-route
reference passes with the camera facing forward / left / right, two extra locations each
covered forward-then-right, and Tokyo 24/7-style query sessions — one short walking pass
per query — swept at day, dawn and night. All database slices pool into one gallery; every
50 ms query slice retrieves from it under a 25 m radius, native cosine, the streamed
ranker from :mod:`scripts.brisbane_pooled`. Session plumbing (partition listing, clock
fits, geometry, per-partition bank caching, the retention guard, figures) is imported from
the pilot's :mod:`scripts.springfield_eval` so the two evaluations cannot drift apart.

What this adds over the pilot script:

* the fixed session -> arm map for the seven database sessions (user-confirmed capture
  order);
* query discovery from ``query_{day,dawn,night}/`` with per-sweep, per-spot-cluster and
  per-session (pass-level) scoring on top of the usual per-slice R@K;
* a clock sanity clamp — a short session can fit a wildly wrong camera-clock rate (one
  night query: -56% drift, 3.4 s residual); such fits are refit offset-only from
  ``camera_sync.csv``, the same model as sync.json's own single-sample fallback;
* per-query heading relative to the route (net-displacement bearing vs the nearest
  forward-facing database slice's travel bearing), so recall-vs-heading-offset is an
  analysis one-liner rather than a capture-side label.

The capture config snapshot says ``camera_rotate_180=false`` and the files carry no
``repacked`` attr; probe frames were rendered and visually confirmed upright on
2026-08-31, which is what justifies ``--allow-unrepacked`` here.
"""

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src import inference as inf                      # noqa: E402
from src.imagesets import build_radius_gt             # noqa: E402
from src.scoring import figure_recall_curve           # noqa: E402
from src.traversegps import local_enu                 # noqa: E402
from brisbane_pooled import topk_ranked, recall_from_ranked  # noqa: E402
from springfield_eval import (                        # noqa: E402
    FrameCache, bank_for_partition, check_repacked, figure_topk, figure_track_map,
    filter_retention, list_partitions, load_gps, make_dataset, part_label,
    partition_geometry, partition_manifest)

DEFAULT_ROOT = "/media/adam/vprdatasets/megaevent/springfield/sessions"
DEFAULT_CKPT = "/media/adam/vprdatasets/megaevent/v8_bench/ckpts/b_v8_accum_s750.pt"
DEFAULT_BANK_DIR = "/media/adam/vprdatasets/megaevent/evaluations/springfield"
DEFAULT_OUT = "output/springfield_full"
KS = (1, 5, 10, 20)
SWEEPS = ("day", "dawn", "night")

# Tokyo 24/7 as this repo evaluates it (docs/tokyo247_headroom.md, protocol re-verified
# there): the reference point for how a *place* dataset is organised, against which this
# capture's continuous-stream gallery is measured. Its 315 queries are 35 UTM locations x
# 9 images (3 times of day x 3 camera directions) and its 25 m radius is the one used here.
TOKYO247 = {"n_database": 75984, "n_queries": 315, "threshold_m": 25.0,
            "positives_per_query": 118.29, "chance_r1": 0.0016,
            "query_structure": "35 locations x 3 times x 3 directions"}

# Capture order, user-confirmed 2026-08-31: three full-route passes (forward, left,
# right), then two extra locations each walked forward-then-right. "Facing" is the
# camera's mount relative to travel — the axis this dataset exists to probe.
DB_ARMS = {
    "20260829T000024Z": ("forward", "forward"),
    "20260829T005503Z": ("left", "left"),
    "20260829T023529Z": ("right", "right"),
    "20260829T032741Z": ("xA-forward", "forward"),
    "20260829T033229Z": ("xA-right", "right"),
    "20260829T033917Z": ("xB-forward", "forward"),
    "20260829T034314Z": ("xB-right", "right"),
}


# ---------------------------------------------------------------------------
# 1. Clocks — the pilot's load_sync plus a sanity clamp
# ---------------------------------------------------------------------------
def load_clock(sdir, resid_max_ms=500.0, drift_max_ppm=20000.0):
    """``((alpha, beta_prime_ns, delta_ns), flag)`` from ``derived/sync.json``.

    The pilot's reader, with one addition: a camera fit whose residual or rate is
    physically impossible for this sensor (EVK4 drift is tens of ppm; one 29 s night
    session fitted -561038 ppm) is *refit offset-only* from ``camera_sync.csv`` —
    ``alpha = 1`` and the median offset over every sync sample, exactly the model
    sync.json itself falls back to when it holds a single sample.
    """
    path = os.path.join(sdir, "derived", "sync.json")
    if not os.path.exists(path):
        raise SystemExit(f"no {path} — run event-capture's `pixi run process -- {sdir}`")
    with open(path) as f:
        sync = json.load(f)
    cam = sync.get("camera")
    if not cam or cam.get("alpha") is None:
        raise SystemExit(f"{path}: no camera clock fit — camera_sync.csv missing or empty")
    for name in ("anchor", "camera", "phone"):
        warning = (sync.get(name) or {}).get("warning")
        if warning:
            print(f"    sync.json {name}: WARNING {warning}")
    delta = float(sync["anchor"]["delta_ns"])
    alpha, beta = float(cam["alpha"]), float(cam["beta_prime_ns"])
    resid, drift = cam.get("resid_p50_ms"), cam.get("drift_ppm") or 0.0

    flag = None
    if (resid is not None and resid > resid_max_ms) or abs(drift) > drift_max_ppm:
        rows = np.atleast_1d(np.genfromtxt(os.path.join(sdir, "camera_sync.csv"),
                                           delimiter=",", names=True, dtype=np.float64))
        alpha = 1.0
        beta = float(np.median(rows["host_epoch_ns"] - rows["cam_t_last_us"] * 1e3)) - delta
        flag = (f"offset-only refit over {rows.size} sync samples "
                f"(fit was resid {resid:.0f} ms, drift {drift:+.0f} ppm)")
        print(f"    camera clock: CLAMPED — {flag}")
    elif resid is not None:
        print(f"    camera clock: drift {drift:+.0f} ppm, residual p50 {resid:.1f} ms")
    return (alpha, beta, delta), flag


# ---------------------------------------------------------------------------
# 2. Session processing (shared by database and query roles)
# ---------------------------------------------------------------------------
def session_rate_mevs(sdir):
    """Session-average event rate — ~10.0 here means the ERC cap was pinning."""
    with open(os.path.join(sdir, "session.json")) as f:
        j = json.load(f)
    dur_s = (j["stopped_epoch_ns"] - j["started_epoch_ns"]) / 1e9
    return j["partitions"]["total_events"] / dur_s / 1e6


def cached_n_slices(cli, path, sid, tag, ckpt_sha, representation):
    """Slice count from the cached bank manifest, or ``None`` if it cannot be trusted.

    Opening an eventcv reader builds an index over a multi-GB recording, and a rerun with
    every bank already cached pays that on all 199 partitions only to learn a number the
    manifest wrote down at extraction time. The manifest is accepted only when every
    other field matches what this run would extract with — the same comparison
    ``bank_for_partition`` makes before reusing the bank — and ``partition_geometry``
    still cross-checks the count against the recording's own timestamp span, so a wrong
    number cannot slip through into the slice grid.
    """
    if cli.force_rebuild:
        return None
    stem = os.path.join(cli.bank_dir, sid,
                        f"{tag}_{os.path.splitext(os.path.basename(path))[0]}")
    if not (os.path.exists(stem + ".npy") and os.path.exists(stem + ".json")):
        return None
    with open(stem + ".json") as f:
        cached = json.load(f)
    n = cached.get("n_slices")
    if n is None:
        return None
    if cached != partition_manifest(cli, path, ckpt_sha, representation, n):
        return None
    return int(n)


def process_session(cli, model, device, sdir, sid, tag, ckpt_sha, cfg, transform,
                    limit=None):
    """Geometry + banks for every partition of one session, through the pilot cache."""
    parts = list_partitions(sdir, limit)
    check_repacked(parts, cli.allow_unrepacked)
    clock, clock_flag = load_clock(sdir)
    gps = load_gps(sdir)

    s_latlon, s_keep, s_banks, s_prov = [], [], [], []
    for path in parts:
        n = cached_n_slices(cli, path, sid, tag, ckpt_sha, cfg.representation)
        ds = None if n is not None else make_dataset(
            path, transform, cfg.representation, cli.dt_ms, not cli.no_hot_pixel,
            None if cli.no_event_filter else cli.event_filter_dt_us)
        if ds is not None:
            n = len(ds)
        ll, kp, info = partition_geometry(path, n, cli.dt_ms, clock, gps)
        manifest = partition_manifest(cli, path, ckpt_sha, cfg.representation, n)
        bank = bank_for_partition(cli, model, ds, path, tag, sid, manifest, device)
        if bank.shape[0] != len(kp):
            raise SystemExit(f"{path}: bank holds {bank.shape[0]} rows for "
                             f"{len(kp)} geometry slices")
        s_latlon.append(ll); s_keep.append(kp); s_banks.append(bank)
        s_prov.extend((sid, os.path.basename(path), i) for i in range(len(kp)))
    keep = np.concatenate(s_keep)
    rate = session_rate_mevs(sdir)
    print(f"    {len(parts)} partition(s), {len(keep)} slices, {int(keep.sum())} kept "
          f"({int((~keep).sum())} dropped), {rate:.1f} Mev/s")
    return {"latlon": np.concatenate(s_latlon), "keep": keep,
            "bank": np.concatenate(s_banks), "prov": s_prov,
            "clock_flag": clock_flag, "mev_s": round(rate, 2),
            "n_partitions": len(parts), "n_slices": int(len(keep)),
            "n_kept": int(keep.sum())}


def load_curation(path):
    """``({sid: bool drop mask}, meta)`` from ``springfield_curate.py``'s mask.

    The mask is indexed on the full slice grid — the same indexing the descriptor banks
    and ``keep`` use — so it ANDs straight into ``keep`` with nothing to re-extract.
    """
    if not path:
        return {}, None
    with open(path) as f:
        cur = json.load(f)
    masks = {}
    for sid, s in cur["sessions"].items():
        m = np.zeros(s["n_slices"], bool)
        for idxs in s["drop"].values():
            m[np.asarray(idxs, dtype=int)] = True
        masks[sid] = m
    meta = {"profile": cur["profile"], "params": cur["params"],
            "n_dropped": cur["totals"]["n_drop"], "per_code": cur["totals"]["per_code"]}
    return masks, meta


def apply_curation(session, sid, masks):
    """AND one session's curation mask into its keep mask; returns slices removed."""
    m = masks.get(sid)
    if m is None:
        return 0
    if len(m) != len(session["keep"]):
        raise SystemExit(f"{sid}: curation mask has {len(m)} slices, session has "
                         f"{len(session['keep'])} — the mask was built for a different "
                         f"slice grid")
    before = int(session["keep"].sum())
    session["keep"] = session["keep"] & ~m
    session["n_kept"] = int(session["keep"].sum())
    return before - session["n_kept"]


def stride_mask(xy, keep, stride_m):
    """Keep one slice per ``stride_m`` of travelled path, over the already-kept slices.

    Springfield's reference is a continuous walk sampled every 50 ms — 12.0 rows per metre,
    median step 8 cm — where a place dataset samples space. Tokyo 24/7's gallery holds
    118.29 reference images within a query's 25 m radius; Springfield's holds ~2,000 near
    duplicates of the same few viewpoints. This walks each session's track in time order
    and takes a slice whenever the path length since the last one reaches ``stride_m``.

    Applied per session, so the three viewpoint arms are sampled independently and every
    sampled place keeps all of its camera directions — the arms are this capture's
    analogue of the multiple perspective crops Tokyo takes from one panorama.

    No smoothing is needed before accumulating: the track comes from ``gps_interp.csv``
    interpolated onto the slice grid, whose consecutive steps are already well conditioned
    (p50 0.08 m, p90 0.11 m), so the sum does not inflate the way a raw noisy fix sequence
    would.
    """
    out = np.zeros(len(keep), bool)
    idx = np.flatnonzero(keep)
    if not len(idx) or stride_m <= 0:
        return keep.copy()
    step = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(xy[idx], axis=0), axis=1))])
    last = -np.inf
    for j, s in enumerate(step):
        if s - last >= stride_m:
            out[idx[j]] = True
            last = s
    return out


def apply_stride(session, stride_m):
    """Reset ``keep`` to the session's base mask thinned to ``stride_m`` (0 = no thinning)."""
    base = session["keep_base"]
    session["keep"] = base.copy() if stride_m <= 0 else stride_mask(
        session["xy"], base, stride_m)
    session["n_kept"] = int(session["keep"].sum())
    return session["n_kept"]


def stride_tag(db_stride, q_stride, always=False):
    """Filename suffix for one configuration; empty when nothing is thinned.

    ``always`` forces a suffix even for the unthinned configuration. A sweep passes it so
    its stride-0 run cannot silently overwrite the canonical unstrided results file, whose
    name carries no suffix — the contents happen to be identical, but a protocol sweep
    should never be able to rewrite the reference it is being compared against.
    """
    def part(v):
        return ("%g" % v).replace(".", "p")
    if always:
        return f"_db{part(db_stride)}m_q{part(q_stride)}m"
    bits = []
    if db_stride > 0:
        bits.append(f"db{part(db_stride)}m")
    if q_stride > 0:
        bits.append(f"q{part(q_stride)}m")
    return ("_" + "_".join(bits)) if bits else ""


def discover_queries(root, query_limit, sweeps, only=None):
    """``[(sweep, session_id, session_dir)]`` in sweep order, sessions sorted."""
    out = []
    for sweep in sweeps:
        gdir = os.path.join(root, f"query_{sweep}")
        sids = sorted(d for d in os.listdir(gdir)
                      if os.path.isdir(os.path.join(gdir, d)))
        if only is not None:
            sids = [s for s in sids if s in only]
        if query_limit:
            sids = sids[:query_limit]
        out.extend((sweep, sid, os.path.join(gdir, sid)) for sid in sids)
    return out


# ---------------------------------------------------------------------------
# 3. Geometry analysis: clusters and headings
# ---------------------------------------------------------------------------
def spot_clusters(centres_xy, linkage_m=50.0):
    """Connected components under a metric linkage — the physical query spots.

    The capture protocol spaced spots >= 50 m apart (2x the eval radius), so single-link
    components at 50 m recover them without any label having been recorded.
    """
    n = len(centres_xy)
    labels = -np.ones(n, dtype=int)
    d = np.linalg.norm(centres_xy[:, None] - centres_xy[None, :], axis=2)
    cluster = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack = [i]
        labels[i] = cluster
        while stack:
            j = stack.pop()
            for m in np.flatnonzero((d[j] <= linkage_m) & (labels < 0)):
                labels[m] = cluster
                stack.append(m)
        cluster += 1
    return labels, cluster


def track_bearings(xy, half_window=40):
    """Per-slice travel bearing (degrees, 0=N clockwise) over a ±2 s baseline."""
    n = len(xy)
    lo = np.maximum(np.arange(n) - half_window, 0)
    hi = np.minimum(np.arange(n) + half_window, n - 1)
    d = xy[hi] - xy[lo]
    return np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0


def bearing_offset(a, b):
    """Absolute angular difference in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
def figure_stride_sweep(rows, out_png):
    """Recall against gallery sampling, with Tokyo 24/7's density marked for scale."""
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for q in sorted({r["query_stride_m"] for r in rows}):
        sel = sorted((r for r in rows if r["query_stride_m"] == q),
                     key=lambda r: r["db_stride_m"] if r["db_stride_m"] > 0 else 1e-3)
        x = [r["db_stride_m"] if r["db_stride_m"] > 0 else 0.08 for r in sel]
        lbl = f"query stride {q:g} m" if q > 0 else "all query slices"
        ax[0].plot(x, [r["micro_r1"] for r in sel], marker="o", ms=4, label=lbl)
        ax[1].plot([r["positives_per_query_mean"] for r in sel],
                   [r["micro_r1"] for r in sel], marker="o", ms=4, label=lbl)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("database stride (m); 0.08 = the native 50 ms spacing", fontsize=9)
    ax[1].set_xscale("log")
    ax[1].axvline(TOKYO247["positives_per_query"], color="#cc4444", lw=1.0, ls="--")
    # axes-fraction y: a data-coordinate y sits far outside a 0.27-0.31 axis, and
    # bbox_inches="tight" then stretches the canvas to include it.
    ax[1].annotate(" Tokyo 24/7", xy=(TOKYO247["positives_per_query"], 0.03),
                   xycoords=("data", "axes fraction"), fontsize=8, color="#cc4444",
                   rotation=90, va="bottom")
    ax[1].set_xlabel("reference images within the 25 m radius (positives/query)",
                     fontsize=9)
    for a in ax:
        a.set_ylabel("micro R@1", fontsize=9)
        a.grid(alpha=0.25, lw=0.5)
        a.legend(fontsize=8)
    fig.suptitle("Springfield: recall vs how densely the reference route is sampled",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out_png}")


def figure_theta_curve(theta_curve, thetas, out_png):
    """Fraction of visits localized vs the required within-visit hit fraction.

    Every binary pass-level verdict is a point on this curve (any-hit at theta->0+,
    majority at 0.5), so it replaces them all without defending a threshold.
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    for scope, fr in theta_curve.items():
        if fr["any"] is None:
            continue
        style = {"lw": 2.2, "color": "#333333"} if scope == "overall" else {"lw": 1.4}
        ax.plot(thetas, [fr[f"{t:.2f}"] for t in thetas], marker="o", ms=3,
                label=f"{scope} (any {fr['any']:.0%})", **style)
    ax.axvline(0.5, color="#bbbbbb", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("required within-visit top-1 hit fraction θ", fontsize=9)
    ax.set_ylabel("fraction of visits localized", fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_title("pass-level localization vs verdict threshold", fontsize=10, loc="left")
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Top-k strips
# ---------------------------------------------------------------------------
class _LRUFrames(FrameCache):
    """FrameCache with an eviction cap — the full capture has 148 partitions, and a
    lazy reader per 7.5 GB file must not accumulate without bound."""

    MAX_READERS = 4

    def frame(self, h5path, i):
        if h5path not in self._readers and len(self._readers) >= self.MAX_READERS:
            self._readers.pop(next(iter(self._readers)))
        return super().frame(h5path, i)


def topk_figures(cli, tag, cfg, db_sids, q_sessions, session_rows, db_prov, db_xy,
                 db_t, device):
    """Retrieval strips: a representative slice per query session, plus its worst miss.

    The chosen slices are re-ranked here (a couple of rows each) rather than carrying
    every sweep's full ``[k, n_q]`` ranking through to the figure stage.
    """
    topk_dir = os.path.join(cli.out_dir, f"topk_{tag}")
    os.makedirs(topk_dir, exist_ok=True)
    frames = _LRUFrames(cli, cfg.representation)
    db_chunk = cli.db_chunk if len(db_xy) > cli.db_chunk else None
    part_paths = {sid: {os.path.basename(p): p
                        for p in list_partitions(os.path.join(cli.root, "database", sid),
                                                 cli.limit_partitions)}
                  for sid in db_sids}

    manifest = []
    for row in session_rows:
        sid = row["session"]
        s = q_sessions[sid]
        sdir = os.path.join(cli.root, f"query_{row['sweep']}", sid)
        part_paths[sid] = {os.path.basename(p): p for p in list_partitions(sdir)}

        picks = [("rep", row.get("rep_slice"))]
        if cli.topk_per_session > 1:
            picks.append(("miss", row.get("worst_miss_slice")))
        picks = [(name, i) for name, i in picks if i is not None][:cli.topk_per_session]
        if not picks:
            continue

        desc = torch.from_numpy(s["bank"][[i for _, i in picks]])
        ranked, scores = topk_ranked(db_t, desc, device, k=cli.topk_fig,
                                     chunk=cli.score_chunk, db_chunk=db_chunk,
                                     return_scores=True)
        keep_idx = np.flatnonzero(s["keep"])
        kept_xy = s["xy"][keep_idx]
        cum_m = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(kept_xy, axis=0), axis=1))])
        for col, (name, q_i) in enumerate(picks):
            _, q_part, q_slice = s["prov"][q_i]
            q_xy = s["xy"][q_i]
            metres = float(cum_m[np.searchsorted(keep_idx, q_i)])
            imgs = [frames.frame(part_paths[sid][q_part], q_slice)]
            rows_lbl, dists, sims, arms, correct = [], [], [], [], []
            for c in range(cli.topk_fig):
                db_i = int(ranked[c, col])
                d_sid, d_part, d_slice = db_prov[db_i]
                imgs.append(frames.frame(part_paths[d_sid][d_part], d_slice))
                rows_lbl.append(f"{part_label(d_part)}#{d_slice}")
                dist = float(np.linalg.norm(db_xy[db_i] - q_xy))
                dists.append(dist)
                sims.append(float(scores[c, col]))
                arms.append(DB_ARMS[d_sid][0])
                correct.append(dist <= cli.threshold_m)
            out_png = os.path.join(
                topk_dir, f"{row['sweep']}_{row['spot']}_{sid}_{name}.png")
            figure_topk(imgs, f"{sid}\n{part_label(q_part)}#{q_slice}", rows_lbl, dists,
                        sims, arms, correct, metres, out_png, cli.threshold_m)
            manifest.append({"session": sid, "sweep": row["sweep"], "spot": row["spot"],
                             "kind": name, "slice": int(q_slice),
                             "figure": os.path.basename(out_png),
                             "retrieved": [{"arm": arms[c], "distance_m": round(dists[c], 1),
                                            "cosine": round(sims[c], 4),
                                            "correct": correct[c]}
                                           for c in range(cli.topk_fig)]})
    with open(os.path.join(topk_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"-> {topk_dir} ({len(manifest)} figures + manifest.json)")


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--eval-resolution", type=int, default=322)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--threshold-m", type=float, default=25.0)
    ap.add_argument("--event-filter-dt-us", type=int, default=50_000,
                    help="background-activity window in MICROseconds (eventcv's raw unit)")
    ap.add_argument("--no-hot-pixel", action="store_true")
    ap.add_argument("--no-event-filter", action="store_true")
    ap.add_argument("--skip-filter-check", action="store_true")
    ap.add_argument("--allow-unrepacked", action="store_true",
                    help="accept partitions without repack attrs — justified for this "
                         "capture by the visually confirmed upright probe frames")
    ap.add_argument("--limit-partitions", type=int, default=None,
                    help="smoke test: first N partitions of each database session")
    ap.add_argument("--query-limit", type=int, default=None,
                    help="smoke test: first N query sessions per sweep")
    ap.add_argument("--query-sessions", default=None,
                    help="smoke test: comma-separated query session ids to keep")
    ap.add_argument("--sweeps", default=",".join(SWEEPS),
                    help="comma-separated subset of day,dawn,night")
    ap.add_argument("--db-sessions", default=",".join(DB_ARMS),
                    help="comma-separated subset of the database session ids")
    ap.add_argument("--topk-per-session", type=int, default=2,
                    help="retrieval strips per query session: 1 = representative slice, "
                         "2 = + worst miss, 0 = no figure stage")
    ap.add_argument("--topk-fig", type=int, default=3)
    ap.add_argument("--db-chunk", type=int, default=40_000)
    ap.add_argument("--score-chunk", type=int, default=256)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--db-stride-m", default="0",
                    help="comma list of database sampling strides in metres, e.g. "
                         "'0,1,2,5'. 0 keeps the native 50 ms stream (~12 rows/m); a "
                         "stride keeps one slice per that much travelled path, per "
                         "session, so every arm is sampled independently")
    ap.add_argument("--query-stride-m", default="0",
                    help="comma list of query sampling strides in metres; 0 scores every "
                         "kept slice. Crossed with --db-stride-m into a sweep")
    ap.add_argument("--curation", default=None,
                    help="curation mask json from scripts/springfield_curate.py; ANDed "
                         "into every session's keep mask (banks are untouched)")
    ap.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    cli = ap.parse_args()

    sweeps = [s for s in cli.sweeps.split(",") if s]
    db_sids = [s for s in cli.db_sessions.split(",") if s]
    unknown = [s for s in db_sids if s not in DB_ARMS]
    if unknown:
        raise SystemExit(f"unknown database session(s) {unknown}; known: {list(DB_ARMS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, step = inf.load_model(cli.ckpt, device)
    cfg.H = cfg.W = cli.eval_resolution
    transform = inf.eval_transform(cfg)
    with open(cli.ckpt, "rb") as f:
        ckpt_sha = hashlib.file_digest(f, "sha256").hexdigest()
    label = os.path.splitext(os.path.basename(cli.ckpt))[0]
    filter_tag = "baoff" if cli.no_event_filter else f"ba{cli.event_filter_dt_us // 1000}"
    tag = (f"{label}_s{step}_{ckpt_sha[:10]}_{cfg.representation}"
           f"_r{cli.eval_resolution}_hp{int(not cli.no_hot_pixel)}_{filter_tag}")
    # Curation changes which slices are scored, never which are extracted, so the bank
    # cache key stays `tag` and only the output files carry the profile suffix.
    cur_masks, cur_meta = load_curation(cli.curation)
    out_tag = f"{tag}_cur{cur_meta['profile']}" if cur_meta else tag
    print(f"{label} (step {step}, sha {ckpt_sha[:10]}): vit={cfg.vit} "
          f"rep={cfg.representation} desc={cfg.desc_dim} in={cfg.H}x{cfg.W}")
    print(f"database: {len(db_sids)} sessions   sweeps: {sweeps}   "
          f"{cli.threshold_m:g} m radius, dt {cli.dt_ms} ms, filter {filter_tag}")
    if cur_meta:
        print(f"curation '{cur_meta['profile']}': {cur_meta['n_dropped']} slices masked "
              f"({'  '.join(f'{k}={v}' for k, v in sorted(cur_meta['per_code'].items()))})")

    # ---- database ------------------------------------------------------------------
    db_sessions, retention = {}, {}
    for sid in db_sids:
        arm, _ = DB_ARMS[sid]
        sdir = os.path.join(cli.root, "database", sid)
        print(f"\n[db/{arm}] {sid}")
        if not cli.no_event_filter:
            retention[sid] = filter_retention(
                cli, list_partitions(sdir, cli.limit_partitions)[0], cfg.representation)
        db_sessions[sid] = process_session(cli, model, device, sdir, sid, tag, ckpt_sha,
                                           cfg, transform, limit=cli.limit_partitions)
        n_cut = apply_curation(db_sessions[sid], sid, cur_masks)
        if n_cut:
            print(f"    curation: -{n_cut} slices, {db_sessions[sid]['n_kept']} kept")

    # ---- queries -------------------------------------------------------------------
    only = set(cli.query_sessions.split(",")) if cli.query_sessions else None
    queries = discover_queries(cli.root, cli.query_limit, sweeps, only)
    if not queries:
        raise SystemExit("no query sessions selected")
    probe_sids = {next(sid for sw2, sid, _ in queries if sw2 == sw)
                  for sw in sweeps if any(q[0] == sw for q in queries)}
    q_sessions = {}
    for n, (sweep, sid, sdir) in enumerate(queries):
        print(f"\n[query_{sweep} {n + 1}/{len(queries)}] {sid}")
        if not cli.no_event_filter and sid in probe_sids:
            filter_retention(cli, list_partitions(sdir)[0], cfg.representation)
        q_sessions[sid] = process_session(cli, model, device, sdir, sid, tag,
                                          ckpt_sha, cfg, transform)
        n_cut = apply_curation(q_sessions[sid], sid, cur_masks)
        if n_cut:
            print(f"    curation: -{n_cut} slices, {q_sessions[sid]['n_kept']} kept")
        q_sessions[sid]["sweep"] = sweep

    # A session curated down to nothing has no position, no descriptor and no vote; it
    # cannot be scored, so drop it here and say so rather than let it reach the metrics
    # as a silent NaN.
    emptied = [sid for sid, s in q_sessions.items() if s["n_kept"] == 0]
    for sid in emptied:
        print(f"  {sid}: 0 slices survive curation — excluded from scoring")
        del q_sessions[sid]
    if not q_sessions:
        raise SystemExit("curation left no query slices at all")

    del model
    torch.cuda.empty_cache()

    # ---- shared metric frame -------------------------------------------------------
    anchor = db_sessions[db_sids[0]]
    fwd_kept = anchor["latlon"][anchor["keep"]]
    if not len(fwd_kept):
        raise SystemExit(f"{db_sids[0]} has no GPS-covered slices — check its clock fit")
    lat0, lon0 = float(fwd_kept[:, 0].mean()), float(fwd_kept[:, 1].mean())
    print(f"\nprojection origin ({lat0:.6f}, {lon0:.6f})")
    for s in list(db_sessions.values()) + list(q_sessions.values()):
        s["xy"] = local_enu(s["latlon"][:, 0], s["latlon"][:, 1], lat0, lon0)
        s["keep_base"] = s["keep"].copy()      # every configuration thins from this

    # ---- protocol sweep --------------------------------------------------------------
    # One load, many protocols. Configurations with a thinned gallery are cheap (a 1 m
    # stride leaves ~10.6k rows against 132.6k), so the unthinned one is scored LAST: it
    # is the only configuration whose pooled copy rivals the per-session banks in size,
    # and by then those banks can be released as it builds them.
    db_strides = [float(x) for x in cli.db_stride_m.split(",") if x.strip()]
    q_strides = [float(x) for x in cli.query_stride_m.split(",") if x.strip()]
    configs = [(d, q) for d in db_strides for q in q_strides]
    configs.sort(key=lambda c: (c[0] <= 0, c[1] <= 0))
    if len(configs) > 1:
        print(f"\nprotocol sweep: {len(configs)} configuration(s) "
              + "  ".join(f"db={d:g}/q={q:g}" for d, q in configs))

    sweep_summary = []
    for n, (db_stride, q_stride) in enumerate(configs):
        cfg_tag = out_tag + stride_tag(db_stride, q_stride, always=len(configs) > 1)
        n_db = sum(apply_stride(db_sessions[sid], db_stride) for sid in db_sids)
        n_q = sum(apply_stride(s, q_stride) for s in q_sessions.values())
        print(f"\n{'=' * 78}\n[config {n + 1}/{len(configs)}] database stride "
              f"{db_stride:g} m -> {n_db} rows   query stride {q_stride:g} m -> "
              f"{n_q} slices\n{'=' * 78}")
        empty = [sid for sid, s in q_sessions.items() if s["n_kept"] == 0]
        if empty:
            raise SystemExit(f"stride {q_stride:g} m emptied {len(empty)} query "
                             f"session(s): {empty[:5]} — sampling must not delete a visit")
        # The last configuration may release the per-session banks as it pools them.
        results = score_protocol(cli, tag, cfg_tag, cfg, label, step, ckpt_sha, retention,
                                 cur_meta, db_sessions, q_sessions, db_sids, list(sweeps),
                                 device, db_stride, q_stride,
                                 free_banks=(n == len(configs) - 1))
        sweep_summary.append({
            "db_stride_m": db_stride, "query_stride_m": q_stride,
            "out_tag": cfg_tag, "n_database": results["n_database"],
            "n_query_slices": int(sum(results["scorable"].values())),
            **{k: results["protocol"][k] for k in
               ("positives_per_query_mean", "positives_per_query_median",
                "chance_r1", "database_never_near_query")},
            "micro_r1": results["recall_micro"]["overall"]["1"],
            "macro_r1": (results["recall_macro"]["overall"] or {}).get("1"),
        })

    if len(configs) > 1:
        path = os.path.join(cli.out_dir, f"stride_sweep_{out_tag}.json")
        with open(path, "w") as f:
            json.dump({"tag": tag, "threshold_m": cli.threshold_m,
                       "tokyo247_reference": TOKYO247, "configs": sweep_summary},
                      f, indent=2)
        print(f"\n{'=' * 78}\nprotocol sweep summary "
              f"(Tokyo 24/7 for scale: {TOKYO247['n_database']} db, "
              f"{TOKYO247['positives_per_query']} pos/query, "
              f"chance {TOKYO247['chance_r1']:.4f})")
        print(f"  {'db_m':>6s} {'q_m':>5s} {'db rows':>8s} {'queries':>8s} "
              f"{'pos/q':>8s} {'chance':>7s} {'micro R@1':>10s} {'macro R@1':>10s}")
        for r in sweep_summary:
            print(f"  {r['db_stride_m']:6g} {r['query_stride_m']:5g} "
                  f"{r['n_database']:8d} {r['n_query_slices']:8d} "
                  f"{r['positives_per_query_mean']:8.1f} {r['chance_r1']:7.4f} "
                  f"{r['micro_r1']:10.4f} "
                  f"{(r['macro_r1'] if r['macro_r1'] is not None else float('nan')):10.4f}")
        print(f"-> {path}")
        figure_stride_sweep(sweep_summary,
                            os.path.join(cli.out_dir, f"stride_sweep_{out_tag}.png"))


def score_protocol(cli, tag, out_tag, cfg, label, step, ckpt_sha, retention, cur_meta,
                   db_sessions, q_sessions, db_sids, sweeps, device,
                   db_stride=0.0, q_stride=0.0, free_banks=True):
    """Score one protocol configuration and write its artefacts; returns the results dict.

    Split out of ``main`` so a stride sweep pays the 91-session load once. Everything here
    reads ``keep``, so a configuration is applied simply by setting each session's mask
    before the call. ``free_banks`` releases the per-session database banks while pooling —
    correct only for the last configuration, since a later one would need them again.
    """
    db_xy = np.concatenate([db_sessions[sid]["xy"][db_sessions[sid]["keep"]]
                            for sid in db_sids])
    db_desc = np.concatenate([db_sessions[sid]["bank"][db_sessions[sid]["keep"]]
                              for sid in db_sids])
    db_prov = [p for sid in db_sids
               for p, k in zip(db_sessions[sid]["prov"], db_sessions[sid]["keep"]) if k]
    db_arm = np.array([DB_ARMS[p[0]][0] for p in db_prov])
    if free_banks:                 # the pooled copy is the only one used from here on;
        for sid in db_sids:        # holding both doubled RSS and fed the oomd
            db_sessions[sid]["bank"] = None
    print(f"database {len(db_xy)} rows: " + "  ".join(
        f"{DB_ARMS[sid][0]}={db_sessions[sid]['n_kept']}" for sid in db_sids))

    # Travel bearings of the forward-facing database slices — the local route direction,
    # against which each query's heading offset is measured.
    fwd_xy, fwd_bear = [], []
    for sid in db_sids:
        if DB_ARMS[sid][1] != "forward":
            continue
        s = db_sessions[sid]
        bear = track_bearings(s["xy"])
        fwd_xy.append(s["xy"][s["keep"]])
        fwd_bear.append(bear[s["keep"]])
    fwd_xy = np.concatenate(fwd_xy) if fwd_xy else np.zeros((0, 2))
    fwd_bear = np.concatenate(fwd_bear) if fwd_bear else np.zeros(0)

    # ---- per-sweep ranking and scoring ---------------------------------------------
    os.makedirs(cli.out_dir, exist_ok=True)
    db_t = torch.from_numpy(db_desc)
    db_chunk = cli.db_chunk if len(db_xy) > cli.db_chunk else None
    sweep_curves, sweep_recalls, sweep_scorable = {}, {}, {}
    session_rows, all_found, all_scorable, all_top1 = [], [], [], []
    # Tokyo 24/7's own comparability statistics, so the two datasets can be set side by
    # side: how many reference images depict a query's place, and what chance alone scores.
    all_pos, near_any = [], np.zeros(len(db_xy), bool)

    for sweep in sweeps:
        sids = [sid for sid in q_sessions if q_sessions[sid]["sweep"] == sweep]
        if not sids:
            print(f"\n[{sweep}] no query sessions selected — skipped")
            continue
        spans, q_parts_xy, q_parts_desc = {}, [], []
        start = 0
        for sid in sids:
            s = q_sessions[sid]
            spans[sid] = (start, start + s["n_kept"])
            start += s["n_kept"]
            q_parts_xy.append(s["xy"][s["keep"]])
            q_parts_desc.append(s["bank"][s["keep"]])
        q_xy = np.concatenate(q_parts_xy)
        q_desc = np.concatenate(q_parts_desc)

        gt = build_radius_gt(db_xy, q_xy, cli.threshold_m)
        per_query = gt.sum(0)
        all_pos.append(per_query)
        near_any |= gt.any(axis=1)
        print(f"\n[{sweep}] {len(sids)} sessions, {len(q_xy)} slices; "
              f"{int((per_query > 0).sum())} scorable, positives/query mean "
              f"{per_query.mean():.1f}")

        ranked, scores = topk_ranked(db_t, torch.from_numpy(q_desc), device, k=max(KS),
                                     chunk=cli.score_chunk, db_chunk=db_chunk,
                                     return_scores=True)
        recalls, curve, scorable = recall_from_ranked(ranked, gt, ks=KS)
        sweep_curves[sweep], sweep_recalls[sweep] = curve, recalls
        sweep_scorable[sweep] = int(scorable.sum())
        print(f"[{sweep}] native cosine: " +
              "  ".join(f"R@{k}={recalls[k]:.3f}" for k in KS))

        hit = gt[ranked, np.arange(gt.shape[1])[None, :]]
        found = np.cumsum(hit, axis=0) > 0                       # [k, n_q]
        top1_dist = np.linalg.norm(db_xy[ranked[0]] - q_xy, axis=1)
        all_found.append(found[[k - 1 for k in KS]])
        all_scorable.append(scorable)
        all_top1.append(ranked[0])

        for sid in sids:
            a, b = spans[sid]
            sc, f1 = scorable[a:b], found[0, a:b]
            xy = q_xy[a:b]
            keep_idx = np.flatnonzero(q_sessions[sid]["keep"])
            path_m = (float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
                      if b - a > 1 else 0.0)
            net = xy[-1] - xy[0] if b - a > 1 else np.zeros(2)
            net_m = float(np.linalg.norm(net))
            q_bear = float(np.degrees(np.arctan2(net[0], net[1])) % 360.0)
            centre = xy.mean(axis=0)
            if len(fwd_xy) and net_m >= 5.0:
                near = int(np.argmin(np.linalg.norm(fwd_xy - centre, axis=1)))
                db_bear = float(fwd_bear[near])
                offset = round(bearing_offset(q_bear, db_bear), 1)
            else:
                db_bear = offset = None            # spin-in-place, or no forward track
            sc_pos = np.flatnonzero(sc)
            rep = int(keep_idx[sc_pos[len(sc_pos) // 2]]) if len(sc_pos) else None
            miss = sc & ~f1
            worst = (int(keep_idx[np.argmax(np.where(miss, top1_dist[a:b], -1.0))])
                     if miss.any() else None)
            arms_c, counts_c = np.unique(
                db_arm[ranked[0, a:b][np.flatnonzero(sc & f1)]], return_counts=True)
            session_rows.append({
                "session": sid, "sweep": sweep,
                "n_slices": q_sessions[sid]["n_slices"],
                "n_kept": b - a, "n_scorable": int(sc.sum()),
                "r1_slice": round(float(f1[sc].mean()), 6) if sc.any() else None,
                "r5_slice": round(float(found[4, a:b][sc].mean()), 6) if sc.any() else None,
                "r10_slice": round(float(found[9, a:b][sc].mean()), 6) if sc.any() else None,
                "r20_slice": round(float(found[19, a:b][sc].mean()), 6) if sc.any() else None,
                "pass_hit_any": bool(f1[sc].any()) if sc.any() else None,
                "pass_hit_majority": bool(f1[sc].mean() >= 0.5) if sc.any() else None,
                "top1_median_m": (round(float(np.median(top1_dist[a:b][sc])), 1)
                                  if sc.any() else None),
                "path_m": round(path_m, 1), "net_disp_m": round(net_m, 1),
                "q_bearing_deg": round(q_bear, 1) if net_m >= 5.0 else None,
                "db_bearing_deg": round(db_bear, 1) if db_bear is not None else None,
                "heading_offset_deg": offset,
                "centre_xy": [round(float(centre[0]), 1), round(float(centre[1]), 1)],
                "clock_flag": q_sessions[sid]["clock_flag"],
                "top1_arms_when_correct": {str(k): int(v)
                                           for k, v in zip(arms_c, counts_c)},
                "rep_slice": rep, "worst_miss_slice": worst,
            })

        figure_track_map(db_xy, q_xy[scorable], found[0][scorable], cli.threshold_m,
                         os.path.join(cli.out_dir, f"error_map_{sweep}_{out_tag}.png"))
        # The gt matrix alone is ~1 GB per sweep; release before the next one builds.
        del gt, hit, found, ranked, scores, top1_dist, q_desc

    # ---- clusters + aggregates -----------------------------------------------------
    centres = np.array([r["centre_xy"] for r in session_rows])
    labels, n_clusters = spot_clusters(centres)
    first_sid = {lab: min(session_rows[i]["session"]
                          for i in np.flatnonzero(labels == lab))
                 for lab in range(n_clusters)}
    rename = {lab: f"spot-{n + 1:02d}"
              for n, lab in enumerate(sorted(first_sid, key=first_sid.get))}
    for row, lab in zip(session_rows, labels):
        row["spot"] = rename[lab]
    print(f"\n{n_clusters} spot clusters: " + "  ".join(
        f"{rename[lab]}={int((labels == lab).sum())}"
        for lab in sorted(rename, key=rename.get)))

    found_ks = np.concatenate(all_found, axis=1)                 # [len(KS), n_all]
    scorable = np.concatenate(all_scorable)
    overall = {k: float(found_ks[i, scorable].mean()) for i, k in enumerate(KS)}
    print("overall micro (per-slice): " +
          "  ".join(f"R@{k}={overall[k]:.3f}" for k in KS))

    # Macro (per-visit) rates: the session is the independent unit — slices are ~200
    # correlated repeated measures of one visit, so the micro average silently weights
    # long sessions and overstates N. Macro averages each session's within-visit hit
    # fraction, one vote per visit. The theta curve generalises every binary pass-level
    # verdict: "any-hit" and "majority" are its theta->0+ and theta=0.5 points, so no
    # single threshold has to be defended.
    def macro_rows(scope):
        return [r for r in session_rows if r["n_scorable"] > 0
                and (scope == "overall" or r["sweep"] == scope)]

    sweeps = [s for s in sweeps if s in sweep_curves]    # drop empty sweeps everywhere
    macro, theta_curve = {}, {}
    thetas = [round(t, 2) for t in np.arange(0.05, 1.0001, 0.05)]
    for scope in ["overall"] + sweeps:
        rows = macro_rows(scope)
        macro[scope] = {k: (float(np.mean([r[f"r{k}_slice"] for r in rows]))
                            if rows else None) for k in KS}
        fr = np.array([r["r1_slice"] for r in rows], dtype=float)
        theta_curve[scope] = {
            "any": float((fr > 0).mean()) if len(fr) else None,
            **{f"{t:.2f}": (float((fr >= t).mean()) if len(fr) else None)
               for t in thetas}}
    for scope in ["overall"] + sweeps:
        if macro[scope][1] is not None:
            print(f"{scope} macro (per-visit): " +
                  "  ".join(f"R@{k}={macro[scope][k]:.3f}" for k in KS) +
                  f"   majority-localized {theta_curve[scope]['0.50']:.1%}"
                  f"   any {theta_curve[scope]['any']:.1%}")
    n_unscorable = sum(1 for r in session_rows if r["n_scorable"] == 0)
    if n_unscorable:
        print(f"{n_unscorable} session(s) with no scorable slices "
              f"excluded from the macro rates")

    top1_rows = np.concatenate(all_top1)
    correct1 = scorable & found_ks[0]
    prov_arm = {str(a): int(c) for a, c in
                zip(*np.unique(db_arm[top1_rows[correct1]], return_counts=True))}
    prov_facing = {}
    for a, c in prov_arm.items():
        facing = a.split("-")[-1]
        prov_facing[facing] = prov_facing.get(facing, 0) + c
    print(f"top-1 provenance when correct: {prov_arm}")

    per_spot = {}
    slice_spot = np.concatenate([np.full(r["n_kept"], labels[i])
                                 for i, r in enumerate(session_rows)])
    for lab, spot in sorted(rename.items(), key=lambda kv: kv[1]):
        m = (slice_spot == lab) & scorable
        rows = [r for r, l2 in zip(session_rows, labels) if l2 == lab]
        per_spot[spot] = {
            "n_sessions": len(rows),
            "sweeps": sorted({r["sweep"] for r in rows}),
            "r1": round(float(found_ks[0, m].mean()), 4) if m.any() else None,
            "r10": round(float(found_ks[2, m].mean()), 4) if m.any() else None,
        }

    # ---- protocol shape, in Tokyo 24/7's terms ---------------------------------------
    pos = np.concatenate(all_pos) if all_pos else np.zeros(0)
    protocol = {
        "db_stride_m": db_stride, "query_stride_m": q_stride,
        "rows_per_arm": {DB_ARMS[sid][0]: int(db_sessions[sid]["n_kept"])
                         for sid in db_sids},
        "positives_per_query_mean": round(float(pos.mean()), 2) if len(pos) else None,
        "positives_per_query_median": int(np.median(pos)) if len(pos) else None,
        "positives_per_query_min": int(pos.min()) if len(pos) else None,
        # chance R@1 = the probability a uniformly random gallery row is a positive
        "chance_r1": round(float(pos.mean() / max(len(db_xy), 1)), 6) if len(pos) else None,
        "database_never_near_query": round(float((~near_any).mean()), 4),
        "tokyo247_reference": TOKYO247,
    }
    print(f"\nprotocol: {len(db_xy)} gallery rows, "
          f"{protocol['positives_per_query_mean']} positives/query "
          f"(Tokyo 24/7: {TOKYO247['positives_per_query']}), chance R@1 "
          f"{protocol['chance_r1']:.4f} (Tokyo {TOKYO247['chance_r1']:.4f}), "
          f"{protocol['database_never_near_query']:.1%} of gallery never near a query")

    # ---- artefacts -----------------------------------------------------------------
    results = {
        "protocol": protocol,
        "dataset": "springfield-full", "checkpoint": os.path.abspath(cli.ckpt),
        "ckpt_sha256": ckpt_sha, "step": step, "tag": tag, "out_tag": out_tag,
        "curation": cur_meta,
        "representation": cfg.representation, "resolution": cli.eval_resolution,
        "dt_ms": cli.dt_ms, "threshold_m": cli.threshold_m,
        "hot_pixel": not cli.no_hot_pixel,
        "filter_dt_us": None if cli.no_event_filter else cli.event_filter_dt_us,
        "filter_retention": retention or None,
        "limit_partitions": cli.limit_partitions, "query_limit": cli.query_limit,
        "database": {sid: {"arm": DB_ARMS[sid][0], "facing": DB_ARMS[sid][1],
                           **{k: db_sessions[sid][k] for k in
                              ("n_partitions", "n_slices", "n_kept", "mev_s",
                               "clock_flag")}}
                     for sid in db_sids},
        "n_database": int(len(db_xy)), "n_queries": len(session_rows),
        "recall_micro": {"overall": {str(k): overall[k] for k in KS},
                         **{s: {str(k): sweep_recalls[s][k] for k in KS}
                            for s in sweeps}},
        "recall_macro": {scope: ({str(k): macro[scope][k] for k in KS}
                                 if macro[scope][1] is not None else None)
                         for scope in ["overall"] + sweeps},
        "theta_curve_r1": theta_curve,
        "n_sessions_unscorable": n_unscorable,
        "recall_curves": {s: {str(n): v for n, v in sweep_curves[s].items()}
                          for s in sweeps},
        "scorable": {s: sweep_scorable[s] for s in sweeps},
        "top1_provenance_when_correct": {"by_arm": prov_arm, "by_facing": prov_facing},
        "per_spot": per_spot,
        "per_session": session_rows,
    }
    out_json = os.path.join(cli.out_dir, f"results_{out_tag}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {out_json}")

    csv_path = os.path.join(cli.out_dir, f"per_query_{out_tag}.csv")
    cols = ["session", "sweep", "spot", "n_slices", "n_kept", "n_scorable", "r1_slice",
            "r5_slice", "r10_slice", "r20_slice", "pass_hit_any", "pass_hit_majority",
            "top1_median_m", "path_m", "net_disp_m", "q_bearing_deg", "db_bearing_deg",
            "heading_offset_deg", "clock_flag"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(session_rows)
    print(f"-> {csv_path}")

    figure_recall_curve(
        sweep_curves,
        f"springfield-full / {label}: {len(db_xy)} database x "
        f"{int(scorable.sum())} scorable query slices, {cli.threshold_m:g} m",
        os.path.join(cli.out_dir, f"recall_curves_{out_tag}.png"))
    figure_theta_curve(theta_curve, thetas,
                       os.path.join(cli.out_dir, f"theta_curve_{out_tag}.png"))

    if cli.topk_per_session:
        topk_figures(cli, out_tag, cfg, db_sids, q_sessions, session_rows, db_prov, db_xy,
                     db_t, device)
    return results




if __name__ == "__main__":
    main()
