"""Springfield-Event pilot: pooled forward+left database vs a reverse query traverse.

Run from the repo root::

    CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python3 scripts/springfield_eval.py \
        --db-forward 20260828T040109Z --db-left <session> --query <session>

The Brisbane/NSAVP pooled protocol applied to an event-capture pilot dataset: every 50 ms
query slice retrieves from the pooled database (forward-facing and left-facing traverses),
a 25 m Euclidean radius on iPhone GPS, native cosine only, and the streamed top-20 ranker
imported from :mod:`scripts.brisbane_pooled` so the datasets cannot drift apart in how they
are scored. What differs is the *source*: sessions recorded by ``~/repo/event-capture`` —
``events_pNNNN_<lat>_<lon>.h5`` partitions rotated every 100 m of GPS travel, with the
camera clock and the GPS track living in each session's ``derived/`` sidecars.

Three format facts this script leans on (see the event-capture README):

* Partitions of one session share a **gapless camera-µs clock** — the camera is never
  closed at a rotation — so one ``sync.json`` camera fit covers every partition, and
  eventcv frames each partition from its own first timestamp.
* ``derived/sync.json`` maps camera µs to laptop epoch ns:
  ``laptop_ns = alpha * (t_us * 1e3) + beta_prime_ns + delta_ns``.
* ``derived/gps_interp.csv`` is the cleaned, PCHIP-interpolated track on the same laptop
  clock (``time_laptop_ns, lat, lon, alt, in_gap``). Rows flagged ``in_gap`` bridge a GPS
  dropout longer than 15 s and are dropped here, as the README instructs.

The event pipeline is byte-identical to every recorded-event benchmark in this repo:
50 ms slices, eventcv hot-pixel filter (std 3), background-activity filter at 50 000 µs,
and the representation the checkpoint was trained on (``accumulate`` for v8) rendered by
:class:`src.inference.EventStreamDataset`. Partitions must have been through event-capture's
``pixi run repack`` first — that is what applies the ``camera_rotate_180`` mount correction,
and this script refuses files that have not.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src import inference as inf                      # noqa: E402
from src.imagesets import build_radius_gt             # noqa: E402
from src.scoring import figure_recall_curve           # noqa: E402
from src.traversegps import local_enu                 # noqa: E402
from brisbane_pooled import topk_ranked, recall_from_ranked  # noqa: E402

DEFAULT_ROOT = "/media/adam/vprdatasets/data/springfield-event/sessions"
DEFAULT_CKPT = "/media/adam/vprdatasets/megaevent/v8_bench/ckpts/b_v8_accum_s750.pt"
DEFAULT_BANK_DIR = "/media/adam/vprdatasets/megaevent/evaluations/springfield"
DEFAULT_OUT = "output/springfield"
KS = (1, 5, 10, 20)
ROLES = ("forward", "left", "query")


# ---------------------------------------------------------------------------
# 1. Session plumbing
# ---------------------------------------------------------------------------
def list_partitions(sdir, limit=None):
    """The session's event files, in capture order.

    Partitioned sessions name files ``events_pNNNN_<lat>_<lon>.h5`` with a zero-padded
    index, so the lexicographic sort is the capture order. A ``_nogps`` partition is kept:
    the name merely lacks a start fix — its slices still get coordinates from
    ``gps_interp.csv`` like everyone else's. Single-file sessions (``events.h5``) work too.
    """
    parts = sorted(glob.glob(os.path.join(sdir, "events_p*.h5")))
    if not parts:
        parts = sorted(glob.glob(os.path.join(sdir, "events.h5")) +
                       glob.glob(os.path.join(sdir, "*.hdf5")))
    if not parts:
        raise SystemExit(f"{sdir}: no events_p*.h5 (or events.h5/*.hdf5) found")
    leftovers = glob.glob(os.path.join(sdir, "events_p*.h5.part"))
    if leftovers:
        raise SystemExit(f"{sdir}: in-progress partition(s) present ({leftovers}); the "
                         f"session did not close cleanly — run event-capture's recovery "
                         f"(restart its server once) or repack first")
    return parts[:limit] if limit else parts


def check_repacked(paths, allow_unrepacked):
    """Refuse partitions the repack step has not touched.

    ``pixi run repack`` (event-capture) is what applies the ``camera_rotate_180`` mount
    correction from the session's config snapshot. An un-repacked file from this rig is
    upside-down, and a bank extracted from it would be silently, plausibly wrong.
    """
    for path in paths:
        with h5py.File(path, "r") as f:
            repacked = bool(f.attrs.get("repacked"))
        if not repacked and not allow_unrepacked:
            raise SystemExit(
                f"{os.path.basename(path)}: no 'repacked' attr — the 180-degree mount "
                f"correction has not been applied. Run\n"
                f"  cd ~/repo/event-capture && pixi run repack -- {os.path.dirname(path)}\n"
                f"or pass --allow-unrepacked if this capture genuinely was not rotated.")


def load_sync(sdir):
    """``(alpha, beta_prime_ns, delta_ns)`` from ``derived/sync.json``, warnings surfaced."""
    path = os.path.join(sdir, "derived", "sync.json")
    if not os.path.exists(path):
        raise SystemExit(f"no {path} — run event-capture's `pixi run process -- {sdir}` "
                         f"(needs camera_sync.csv + phone.jsonl in the session dir)")
    with open(path) as f:
        sync = json.load(f)
    cam = sync.get("camera")
    if not cam or cam.get("alpha") is None:
        raise SystemExit(f"{path}: no camera clock fit — camera_sync.csv missing or empty; "
                         f"event timestamps cannot be placed on the GPS clock")
    for name in ("anchor", "camera", "phone"):
        warning = (sync.get(name) or {}).get("warning")
        if warning:
            print(f"    sync.json {name}: WARNING {warning}")
    if cam.get("resid_p50_ms") is not None:
        print(f"    camera clock: drift {cam['drift_ppm']:+.0f} ppm over {cam.get('span_s', 0):.0f} s, "
              f"residual p50 {cam['resid_p50_ms']:.1f} ms")
    return float(cam["alpha"]), float(cam["beta_prime_ns"]), float(sync["anchor"]["delta_ns"])


def load_gps(sdir):
    """``(t_ns, lat, lon, in_gap)`` float64 arrays from ``derived/gps_interp.csv``."""
    path = os.path.join(sdir, "derived", "gps_interp.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no {path} — run event-capture's `pixi run process -- {sdir}`")
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if rows.size < 2:
        raise SystemExit(f"{path}: {rows.size} samples — no usable GPS track")
    return (rows["time_laptop_ns"], rows["lat"], rows["lon"],
            rows["in_gap"].astype(bool))


def cam_us_to_laptop_ns(t_us, alpha, beta_prime_ns, delta_ns):
    """Camera microseconds -> laptop epoch nanoseconds (event_capture/process.py:114)."""
    return alpha * (np.asarray(t_us, dtype=np.float64) * 1e3) + beta_prime_ns + delta_ns


# ---------------------------------------------------------------------------
# 2. Per-partition geometry
# ---------------------------------------------------------------------------
def partition_geometry(h5path, n_slices, dt_ms, clock, gps):
    """``(latlon [n,2], keep [n], info)`` for one partition's 50 ms slice grid.

    eventcv frames a recording from its first timestamp, so slice *i*'s centre is
    ``t[0] + (i + 0.5) * dt``. The count is cross-checked against the file's own span:
    a mismatch beyond one (eventcv's final partial slice) means the grid and the bank
    rows would disagree about what frame *i* is, which poisons every coordinate after it.
    """
    with h5py.File(h5path, "r") as f:
        t = f["events"]["t"]
        t0, t1 = int(t[0]), int(t[-1])
    dt_us = dt_ms * 1000
    n_expected = math.ceil((t1 - t0) / dt_us)
    if abs(n_expected - n_slices) > 1:
        raise SystemExit(
            f"{os.path.basename(h5path)}: eventcv yields {n_slices} slices but the "
            f"timestamp span implies {n_expected} — the slice grid cannot be trusted")

    centres_us = t0 + (np.arange(n_slices, dtype=np.float64) + 0.5) * dt_us
    alpha, beta, delta = clock
    centres_ns = cam_us_to_laptop_ns(centres_us, alpha, beta, delta)

    g_t, g_lat, g_lon, g_gap = gps
    covered = (centres_ns >= g_t[0]) & (centres_ns <= g_t[-1])
    lat = np.interp(centres_ns, g_t, g_lat)
    lon = np.interp(centres_ns, g_t, g_lon)
    # A slice between a flagged sample and a clean one is still bridging the dropout:
    # linear interpolation of the flag is nonzero anywhere either bracketing sample is.
    in_gap = np.interp(centres_ns, g_t, g_gap.astype(np.float64)) > 1e-9
    keep = covered & ~in_gap
    info = {"n_slices": int(n_slices), "uncovered": int((~covered).sum()),
            "in_gap": int((covered & in_gap).sum()),
            "span_s": (t1 - t0) / 1e6}
    return np.column_stack([lat, lon]), keep, info


# ---------------------------------------------------------------------------
# 3. Extraction (cached per partition)
# ---------------------------------------------------------------------------
def sensor_from_h5(h5path):
    """``(W, H)`` — eventcv's own axis order — from the capture sink's root attrs."""
    with h5py.File(h5path, "r") as f:
        return int(f.attrs["width"]), int(f.attrs["height"])


def make_dataset(h5path, transform, representation, dt_ms, hot_pixel, filter_dt_us):
    """One partition as 50 ms rendered slices — the exact benchmark pipeline."""
    return inf.EventStreamDataset(
        "springfield", os.path.basename(h5path), h5path, transform, representation,
        dt_ms, sensor_from_h5(h5path), hot_pixel=hot_pixel, filter_dt_us=filter_dt_us)


def partition_manifest(cli, h5path, ckpt_sha, representation, n_slices):
    with h5py.File(h5path, "r") as f:
        n_events = int(f["events"]["t"].shape[0])
        rotated = int(f.attrs.get("rotated_180", 0))
    return {"partition": os.path.basename(h5path), "n_events": n_events,
            "rotated_180": rotated,
            "n_slices": int(n_slices), "ckpt": os.path.abspath(cli.ckpt),
            "ckpt_sha256": ckpt_sha, "dt_ms": cli.dt_ms,
            "representation": representation, "resolution": cli.eval_resolution,
            "hot_pixel": not cli.no_hot_pixel,
            "filter_dt_us": None if cli.no_event_filter else cli.event_filter_dt_us}


def bank_for_partition(cli, model, ds, h5path, tag, sid, manifest, device):
    """``[n, D]`` float32 — cached under ``<bank-dir>/<session>/<tag>_<partition>.npy``."""
    out_dir = os.path.join(cli.bank_dir, sid)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{tag}_{os.path.splitext(os.path.basename(h5path))[0]}")
    arr_path, man_path = stem + ".npy", stem + ".json"

    if os.path.exists(arr_path) and os.path.exists(man_path) and not cli.force_rebuild:
        with open(man_path) as f:
            cached = json.load(f)
        if cached == manifest:
            bank = np.load(arr_path)
            print(f"    {os.path.basename(arr_path)}: cached ({bank.shape[0]} rows)")
            return bank
        raise SystemExit(f"{man_path}: manifest mismatch —\n  cached: {cached}\n  "
                         f"wanted: {manifest}\nPass --force-rebuild to re-extract.")

    desc = inf.extract_descriptors(model, ds, device,
                                   label=f"{sid}/{os.path.basename(h5path)}",
                                   batch_size=cli.batch_size, oom_backoff=True)
    bank = desc.numpy().astype(np.float32)
    np.save(arr_path, bank)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    {os.path.basename(arr_path)}: extracted {bank.shape} -> cached")
    return bank


def filter_retention(cli, h5path, representation, n_probe=12):
    """The brisbane_pooled retention guard, ported to a capture partition.

    Guards the two historical bugs: filter silently off (~100% retention) and the window
    read as milliseconds (~6%). ``bg`` is representation-aware — accumulate's background
    is white, so "active" means deviating from 1.0, not from 0.0.
    """
    identity = transforms.Compose([])
    bg = 1.0 if representation == "accumulate" else 0.0
    frac = {}
    for label, dt_us in (("off", None), ("on", cli.event_filter_dt_us)):
        ds = make_dataset(h5path, identity, representation, cli.dt_ms,
                          not cli.no_hot_pixel, dt_us)
        idx = np.linspace(min(500, len(ds) // 4), len(ds) - 1 - min(500, len(ds) // 4),
                          n_probe).astype(int)
        frac[label] = np.array([float((np.abs(ds[i].numpy() - bg) > 1e-6).mean())
                                for i in idx])
    # A recording dropout leaves probe frames with no events at all; their 0/0 ratio says
    # nothing about the filter and would drag the mean toward "broken". Only frames that
    # actually saw events vote.
    alive = frac["off"] > 1e-4
    if alive.sum() < 3:
        print(f"    filter check ({os.path.basename(h5path)}): only {int(alive.sum())} of "
              f"{n_probe} probe frames hold events — dropout-heavy partition, check skipped")
        return None
    retention = float((frac["on"][alive] / frac["off"][alive]).mean())
    line = (f"    filter check ({os.path.basename(h5path)}): dt={cli.event_filter_dt_us} us "
            f"retains {retention:.1%} of active pixels")
    if not 0.50 <= retention <= 0.95:
        if cli.skip_filter_check:
            print(line + "  OUTSIDE the [50%, 95%] band — continuing (--skip-filter-check)")
        else:
            raise SystemExit(
                line + f"\nExpected 50-95% (Brisbane ~84%, NSAVP ~76%). Below ~10% means the "
                f"window is being read as milliseconds; ~100% means the filter is not "
                f"running. Pass --skip-filter-check to proceed anyway on this new sensor.")
    else:
        print(line + "  OK")
    return retention


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
C_TP, C_FP = inf.C_TP, inf.C_FP


def figure_track_map(db_xy, q_xy, top1_hit, threshold_m, out_png):
    """Every query slice on the map, coloured by its top-1 verdict.

    The image-set error map aggregates per *unique* coordinate, which suits several
    photos per spot; a traverse has 6k unique coordinates and washes out. Here the
    query track is simply drawn point by point, so failure *segments* are visible.
    """
    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    ax.scatter(db_xy[:, 0], db_xy[:, 1], s=2, c="#dcdbd6", marker=".", linewidths=0,
               label=f"database track ({len(db_xy)})", zorder=1)
    hit, miss = top1_hit, ~top1_hit
    ax.scatter(q_xy[miss, 0], q_xy[miss, 1], s=5, c=C_FP, marker=".", linewidths=0,
               label=f"top-1 wrong ({int(miss.sum())})", zorder=2)
    ax.scatter(q_xy[hit, 0], q_xy[hit, 1], s=5, c=C_TP, marker=".", linewidths=0,
               label=f"top-1 within {threshold_m:g} m ({int(hit.sum())})", zorder=3)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("east (m)", fontsize=9)
    ax.set_ylabel("north (m)", fontsize=9)
    ax.set_title(f"top-1 retrieval along the query traverse "
                 f"({float(top1_hit.mean()):.1%} correct)", fontsize=10, loc="left")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def part_label(name):
    """``events_p0003_-27.68698_152.92084.h5`` -> ``p0003``; other names keep their stem."""
    m = re.match(r"events_(p\d+)", name)
    return m.group(1) if m else os.path.splitext(name)[0]


class FrameCache:
    """Rendered display frames at native sensor resolution, one lazy reader per partition.

    Identity transform on the same filtered reader the banks came from, so a figure shows
    exactly what the model saw (before the Normalize+Resize the model input adds).
    """

    def __init__(self, cli, representation):
        self.cli = cli
        self.representation = representation
        self._readers = {}

    def frame(self, h5path, i):
        if h5path not in self._readers:
            self._readers[h5path] = make_dataset(
                h5path, transforms.Compose([]), self.representation, self.cli.dt_ms,
                not self.cli.no_hot_pixel,
                None if self.cli.no_event_filter else self.cli.event_filter_dt_us)
        frame = self._readers[h5path][i].numpy()          # [3,H,W] float in [0,1]
        return (np.transpose(frame, (1, 2, 0)) * 255).astype(np.uint8)


def figure_topk(frames, q_row, db_rows, dists, scores, arms, correct, metres, out_png,
                threshold_m):
    """One query and its top-k retrievals, bordered by the 25 m verdict."""
    k = len(db_rows)
    fig, axes = plt.subplots(1, k + 1, figsize=(3.2 * (k + 1), 3.2),
                             constrained_layout=True)
    axes[0].imshow(frames[0])
    axes[0].set_title(f"query @ {metres:.0f} m\n{q_row}", fontsize=8)
    for ax, frame, row, dist, score, arm, ok in zip(
            axes[1:], frames[1:], db_rows, dists, scores, arms, correct):
        ax.imshow(frame)
        colour = C_TP if ok else C_FP
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(4)
        ax.set_title(f"{arm}  {dist:.1f} m {'<=' if ok else '>'} {threshold_m:g}\n"
                     f"cos {score:.3f}   {row}", fontsize=8,
                     color=colour)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_png, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sample_every_metres(q_xy_kept, every_m):
    """Kept-query indices at each ``every_m`` mark of cumulative along-track distance."""
    step = np.linalg.norm(np.diff(q_xy_kept, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    picks, next_mark = [], 0.0
    for i, d in enumerate(cum):
        if d >= next_mark:
            picks.append((i, d))
            next_mark += every_m
    return picks, float(cum[-1])


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="directory holding the extracted session dirs")
    ap.add_argument("--db-forward", required=True, metavar="SESSION",
                    help="session id of the forward-facing database traverse")
    ap.add_argument("--db-left", default=None, metavar="SESSION",
                    help="session id of the left-facing database traverse (omit to just "
                         "extract banks for the sessions that have arrived)")
    ap.add_argument("--query", default=None, metavar="SESSION",
                    help="session id of the reverse query traverse (omit to just "
                         "extract banks for the sessions that have arrived)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--eval-resolution", type=int, default=322)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--threshold-m", type=float, default=25.0)
    ap.add_argument("--figure-every-m", type=float, default=5.0)
    ap.add_argument("--topk-fig", type=int, default=3)
    ap.add_argument("--event-filter-dt-us", type=int, default=50_000,
                    help="background-activity window in MICROseconds (eventcv's raw unit)")
    ap.add_argument("--no-hot-pixel", action="store_true")
    ap.add_argument("--no-event-filter", action="store_true")
    ap.add_argument("--skip-filter-check", action="store_true",
                    help="soften the retention guard to a warning (new sensor, no "
                         "measured band yet)")
    ap.add_argument("--allow-unrepacked", action="store_true",
                    help="accept partitions without the repack attrs (mount rotation "
                         "will NOT be applied)")
    ap.add_argument("--limit-partitions", type=int, default=None,
                    help="smoke test: first N partitions of each session only")
    ap.add_argument("--db-chunk", type=int, default=40_000,
                    help="database rows resident on the GPU at once while ranking")
    ap.add_argument("--score-chunk", type=int, default=256)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    cli = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sessions = {"forward": cli.db_forward, "left": cli.db_left, "query": cli.query}
    roles = [r for r in ROLES if sessions[r]]
    given = [sessions[r] for r in roles]
    if len(set(given)) != len(given):
        raise SystemExit(f"sessions must be distinct: {sessions}")

    model, cfg, step = inf.load_model(cli.ckpt, device)
    cfg.H = cfg.W = cli.eval_resolution
    transform = inf.eval_transform(cfg)
    with open(cli.ckpt, "rb") as f:
        ckpt_sha = hashlib.file_digest(f, "sha256").hexdigest()
    label = os.path.splitext(os.path.basename(cli.ckpt))[0]
    filter_tag = "baoff" if cli.no_event_filter else f"ba{cli.event_filter_dt_us // 1000}"
    tag = (f"{label}_s{step}_{ckpt_sha[:10]}_{cfg.representation}"
           f"_r{cli.eval_resolution}_hp{int(not cli.no_hot_pixel)}_{filter_tag}")
    print(f"{label} (step {step}, sha {ckpt_sha[:10]}): vit={cfg.vit} "
          f"rep={cfg.representation} desc={cfg.desc_dim} in={cfg.H}x{cfg.W}")
    print(f"pooled database: forward={cli.db_forward} + left={cli.db_left}   "
          f"query={cli.query}   {cli.threshold_m:g} m radius, dt {cli.dt_ms} ms")

    # ---- per-session: partitions, clocks, geometry, banks --------------------------
    latlon, keep, banks, prov = {}, {}, {}, {}
    infos, retention = {}, {}
    for role in roles:
        sid = sessions[role]
        sdir = os.path.join(cli.root, sid)
        if not os.path.isdir(sdir):
            raise SystemExit(f"no session directory at {sdir}")
        print(f"\n[{role}] {sid}")
        parts = list_partitions(sdir, cli.limit_partitions)
        check_repacked(parts, cli.allow_unrepacked)
        clock = load_sync(sdir)
        gps = load_gps(sdir)
        print(f"    {len(parts)} partition(s); GPS track "
              f"{(gps[0][-1] - gps[0][0]) / 1e9:.0f} s, {int(gps[3].sum())} in-gap samples")
        if not cli.no_event_filter:
            retention[role] = filter_retention(cli, parts[0], cfg.representation)

        s_latlon, s_keep, s_banks, s_prov, s_info = [], [], [], [], []
        for path in parts:
            ds = make_dataset(path, transform, cfg.representation, cli.dt_ms,
                              not cli.no_hot_pixel,
                              None if cli.no_event_filter else cli.event_filter_dt_us)
            ll, kp, info = partition_geometry(path, len(ds), cli.dt_ms, clock, gps)
            manifest = partition_manifest(cli, path, ckpt_sha, cfg.representation, len(ds))
            bank = bank_for_partition(cli, model, ds, path, tag, sid, manifest, device)
            if bank.shape[0] != len(kp):
                raise SystemExit(f"{path}: bank holds {bank.shape[0]} rows for "
                                 f"{len(kp)} geometry slices")
            s_latlon.append(ll); s_keep.append(kp); s_banks.append(bank)
            s_prov.extend((role, os.path.basename(path), i) for i in range(len(kp)))
            s_info.append({**info, "partition": os.path.basename(path)})
            print(f"    {os.path.basename(path)}: {info['n_slices']} slices over "
                  f"{info['span_s']:.0f} s, {info['uncovered']} uncovered, "
                  f"{info['in_gap']} in-gap")
        latlon[role] = np.concatenate(s_latlon)
        keep[role] = np.concatenate(s_keep)
        banks[role] = np.concatenate(s_banks)
        prov[role] = s_prov
        infos[role] = s_info
        print(f"    total {len(keep[role])} slices, {int(keep[role].sum())} kept "
              f"({int((~keep[role]).sum())} dropped)")

    del model
    torch.cuda.empty_cache()

    if len(roles) < len(ROLES):
        missing = [r for r in ROLES if r not in roles]
        print(f"\nbanks + geometry cached for {roles}; no {'/'.join(missing)} session "
              f"given — rerun with all three to score. (Cached banks will be reused.)")
        return

    # ---- shared metric frame -------------------------------------------------------
    fwd_kept = latlon["forward"][keep["forward"]]
    if not len(fwd_kept):
        raise SystemExit("the forward session has no GPS-covered slices — check its "
                         "sync.json camera fit against its gps_interp.csv time span")
    lat0, lon0 = float(fwd_kept[:, 0].mean()), float(fwd_kept[:, 1].mean())
    print(f"\nprojection origin ({lat0:.6f}, {lon0:.6f})")
    xy = {r: local_enu(latlon[r][:, 0], latlon[r][:, 1], lat0, lon0) for r in ROLES}

    db_xy = np.concatenate([xy["forward"][keep["forward"]], xy["left"][keep["left"]]])
    db_desc = np.concatenate([banks["forward"][keep["forward"]],
                              banks["left"][keep["left"]]])
    db_prov = ([p for p, k in zip(prov["forward"], keep["forward"]) if k] +
               [p for p, k in zip(prov["left"], keep["left"]) if k])
    n_fwd = int(keep["forward"].sum())
    q_xy = xy["query"][keep["query"]]
    q_desc = banks["query"][keep["query"]]
    q_prov = [p for p, k in zip(prov["query"], keep["query"]) if k]
    print(f"database {len(db_xy)} rows ({n_fwd} forward + {len(db_xy) - n_fwd} left) x "
          f"{len(q_xy)} queries")

    # ---- ground truth + ranking ----------------------------------------------------
    gt = build_radius_gt(db_xy, q_xy, cli.threshold_m)          # bool [db, q]
    per_query = gt.sum(0)
    scorable_n = int((per_query > 0).sum())
    print(f"ground truth @ {cli.threshold_m:g} m: {scorable_n}/{len(q_xy)} queries "
          f"scorable, positives/query mean {per_query.mean():.1f} "
          f"min {per_query.min()} max {per_query.max()}")
    if scorable_n == 0:
        raise SystemExit("no query has a database slice within tolerance — check the "
                         "clock fits and the GPS tracks before anything else")

    db_t = torch.from_numpy(db_desc)
    q_t = torch.from_numpy(q_desc)
    db_chunk = cli.db_chunk if len(db_xy) > cli.db_chunk else None
    ranked, scores = topk_ranked(db_t, q_t, device, k=max(KS), chunk=cli.score_chunk,
                                 db_chunk=db_chunk, return_scores=True)
    recalls, curve, scorable = recall_from_ranked(ranked, gt, ks=KS)
    print("\nnative cosine: " + "  ".join(f"R@{k}={recalls[k]:.3f}" for k in KS))

    top1_dist = np.linalg.norm(db_xy[ranked[0]] - q_xy, axis=1)
    top1_fwd = np.array([db_prov[i][0] == "forward" for i in ranked[0]])
    top1_hit = gt[ranked[0], np.arange(gt.shape[1])]
    provenance = {
        "top1_forward_frac": float(top1_fwd[scorable].mean()),
        "top1_forward_frac_when_correct": (float(top1_fwd[scorable & top1_hit].mean())
                                           if (scorable & top1_hit).any() else None)}
    print(f"top-1 lands a median {np.median(top1_dist[scorable]):.1f} m from the query "
          f"(p90 {np.percentile(top1_dist[scorable], 90):.1f} m); "
          f"{provenance['top1_forward_frac']:.1%} of top-1s come from the forward arm")

    # ---- artefacts ----------------------------------------------------------------
    os.makedirs(cli.out_dir, exist_ok=True)
    results = {
        "dataset": "springfield-event", "checkpoint": os.path.abspath(cli.ckpt),
        "ckpt_sha256": ckpt_sha, "step": step, "tag": tag,
        "representation": cfg.representation, "resolution": cli.eval_resolution,
        "dt_ms": cli.dt_ms, "threshold_m": cli.threshold_m,
        "hot_pixel": not cli.no_hot_pixel,
        "filter_dt_us": None if cli.no_event_filter else cli.event_filter_dt_us,
        "filter_retention": retention,
        "sessions": sessions, "limit_partitions": cli.limit_partitions,
        "partitions": infos,
        "n_database": int(len(db_xy)), "n_database_forward": n_fwd,
        "n_queries": int(len(q_xy)),
        "dropped": {r: int((~keep[r]).sum()) for r in ROLES},
        "scorable_queries": scorable_n,
        "positives_per_query": {"mean": float(per_query.mean()),
                                "min": int(per_query.min()), "max": int(per_query.max())},
        "recall": {str(k): recalls[k] for k in KS},
        "recall_curve": {str(n): v for n, v in curve.items()},
        "top1_distance_m": {"median": float(np.median(top1_dist[scorable])),
                            "p90": float(np.percentile(top1_dist[scorable], 90))},
        "top1_provenance": provenance,
    }
    out_json = os.path.join(cli.out_dir, f"results_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {out_json}")

    figure_recall_curve(
        {"native": curve},
        f"springfield-event / {label}: {len(db_xy)} database (fwd+left) x "
        f"{scorable_n} scorable queries, {cli.threshold_m:g} m",
        os.path.join(cli.out_dir, f"recall_curve_{tag}.png"))
    figure_track_map(db_xy, q_xy[scorable], top1_hit[scorable], cli.threshold_m,
                     os.path.join(cli.out_dir, f"error_map_{tag}.png"))

    # ---- top-k figures every N metres of the query traverse ------------------------
    topk_dir = os.path.join(cli.out_dir, f"topk_{tag}")
    os.makedirs(topk_dir, exist_ok=True)
    picks, route_m = sample_every_metres(q_xy, cli.figure_every_m)
    print(f"query route {route_m:.0f} m -> {len(picks)} top-{cli.topk_fig} figures "
          f"every {cli.figure_every_m:g} m")
    frames = FrameCache(cli, cfg.representation)
    part_paths = {r: {os.path.basename(p): p
                      for p in list_partitions(os.path.join(cli.root, sessions[r]),
                                               cli.limit_partitions)}
                  for r in ROLES}
    manifest = []
    for q_i, metres in picks:
        _, q_part, q_slice = q_prov[q_i]
        imgs = [frames.frame(part_paths["query"][q_part], q_slice)]
        rows, dists, sims, arms, correct = [], [], [], [], []
        for c in range(cli.topk_fig):
            db_i = int(ranked[c, q_i])
            arm, part, sl = db_prov[db_i]
            imgs.append(frames.frame(part_paths[arm][part], sl))
            rows.append(f"{part_label(part)}#{sl}")
            dists.append(float(np.linalg.norm(db_xy[db_i] - q_xy[q_i])))
            sims.append(float(scores[c, q_i]))
            arms.append(arm)
            correct.append(bool(gt[db_i, q_i]))
        out_png = os.path.join(topk_dir, f"q{q_i:05d}_{metres:06.0f}m.png")
        figure_topk(imgs, f"{part_label(q_part)}#{q_slice}", rows, dists, sims, arms,
                    correct, metres, out_png, cli.threshold_m)
        manifest.append({"query_index": int(q_i), "metres": float(metres),
                         "query": list(q_prov[q_i]),
                         "retrieved": [{"row": int(ranked[c, q_i]),
                                        "provenance": list(db_prov[int(ranked[c, q_i])]),
                                        "cosine": float(scores[c, q_i]),
                                        "distance_m": dists[c],
                                        "correct": correct[c]}
                                       for c in range(cli.topk_fig)],
                         "figure": os.path.basename(out_png)})
    with open(os.path.join(topk_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"-> {topk_dir} ({len(picks)} figures + manifest.json)")


if __name__ == "__main__":
    main()
