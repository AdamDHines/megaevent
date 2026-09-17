"""EventCV-backed pooled Brisbane validation for v8 model selection."""

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..evaluation import recall_at_k_cross as recall_at_k_cross
from ..events import eval_transform

# Brisbane-Event / ED-VPR tencode bins are 50 ms (20 Hz); DAVIS346 is 346x260.
DEFAULT_DT_MS = 50.0
DEFAULT_SENSOR = (346, 260)

# Validated per-traverse leading-slice offsets (physical property of the raw recordings).
# Reproduce the measured offsets exactly (sunset2 202, daytime 147, morning 288, sunset1 0);
# sunrise 190 / night 222 derived from the GT start timestamps. Passed by default so the
# never-rendered conditions (sunrise/night) are not silently assumed 0. A local
# eval_offsets.json is only consulted when brisbane_tasks is called with offsets=None.
BRISBANE_OFFSETS = {
    "sunset1": 0,
    "sunset2": 202,
    "daytime": 147,
    "morning": 288,
    "sunrise": 190,
    "night": 222,
}


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------


class EventH5Dataset(Dataset):
    """Raw event HDF5 -> tencode frames, sliced lazily by ecv.

    ``ecv.open(path, dt_ms=...)`` binary-searches the on-disk timestamps, so a
    slice of a multi-GB recording costs a handful of reads and the file is never
    materialised. ``stride`` subsamples frames (the gallery-density knob of PLAN 1j:
    R@1 is monotone in density, so a recall number is only comparable at a fixed
    stride).

    Three things that would silently corrupt the frames if got wrong:

    1. ``window_ms`` **must** be set to ``dt_ms``. ``open(repr="tencode")`` applies the
       representation's own default of 30 ms, so with 50 ms slices every event older
       than 30 ms would be dropped from each frame.
    2. The reader owns an open HDF5 handle and is **not fork-safe**, so it is opened
       lazily on first use inside each DataLoader worker rather than in ``__init__``.
    3. ``start`` — **the raw recording and the published traverse do not always begin at
       the same instant.** eventcv slices from the recording's own ``t_min``, whereas the
       distributed frames may have been rendered from a trimmed sub-interval. Measured on
       Brisbane: sunset1 aligns 1:1 (14478 slices either way, per-frame IoU 1.000), but
       **sunset2 has 202 extra leading slices** (13027 vs 12825, and ``png[i]`` matches
       ``eventcv[i+202]`` at every probe). Since the ground truth is resized *proportionally*
       onto the descriptor grid, an untrimmed reference silently shifts the whole GT band —
       it cost R@1 .895 -> .608 on sunset1 before this was found. Offsets are per traverse;
       derive them with ``--derive-offsets``.
    """

    def __init__(
        self,
        path,
        transform,
        dt_ms=DEFAULT_DT_MS,
        stride=1,
        sensor_size=DEFAULT_SENSOR,
        n_slices=None,
        start=0,
        count=None,
        representation="tencode",
    ):
        self.path = path
        self.transform = transform
        self.dt_ms = float(dt_ms)
        self.stride = max(1, int(stride))
        self.sensor_size = tuple(sensor_size) if sensor_size else None
        self.start = int(start)
        # For non-tencode reps eventcv has no built-in renderer, so we slice the raw
        # events (eventcv's binary-searched slicing keeps the offsets valid) and render
        # ourselves. Verified: our tencode over raw slice(i) reproduces eventcv's
        # with_repr("tencode")[i] exactly on polarity (IoU 1.000) and to ~0.06/255 on age.
        self.representation = representation
        self._reader = None
        if n_slices is None:
            n_slices = self._open().n_slices
            self._reader = None  # don't carry the handle across a fork
        avail = int(n_slices) - self.start
        self._n_total = min(avail, int(count)) if count else avail
        if self._n_total <= 0:
            raise ValueError(f"{path}: start={self.start} leaves no slices of {n_slices}")

    def _open(self):
        import eventcv as ecv

        reader = ecv.open(self.path, dt_ms=self.dt_ms, sensor_size=self.sensor_size)
        if self.representation == "tencode":
            return reader.with_repr("tencode", window_ms=self.dt_ms)  # fast built-in path
        return reader  # raw reader; render per-slice in __getitem__

    @property
    def reader(self):
        if self._reader is None:  # first touch in this process/worker
            self._reader = self._open()
        return self._reader

    def __len__(self):
        return (self._n_total + self.stride - 1) // self.stride

    def __getitem__(self, i):
        idx = self.start + i * self.stride
        if self.representation == "tencode":
            frame = np.asarray(self.reader[idx])  # [3,H,W] uint8
        else:
            from ..events import render_stream

            frame = render_stream(self.reader.slice(idx), self.representation, self.dt_ms)
        x = torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0)
        return self.transform(x)

    def __getstate__(self):
        # the reader must never be pickled into a worker — rebuild it there
        return {**self.__dict__, "_reader": None}


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def pooled_gt(db_xy, q_xy, threshold_m=25.0):
    """Boolean ``[nq, nr]`` GT: query within ``threshold_m`` metres of a database frame.

    Both inputs are metre coordinates in one shared local projection (the coords file is
    exported by megaevent's ``scripts/export_brisbane_coords.py`` from the calibrated
    benchmark geometry, so this radius check IS the benchmark's ``build_gt`` semantics).
    """
    db = torch.as_tensor(np.asarray(db_xy), dtype=torch.float32)
    q = torch.as_tensor(np.asarray(q_xy), dtype=torch.float32)
    return (torch.cdist(q, db) <= threshold_m).numpy()


def run_pooled(
    model,
    root,
    coords_path,
    query,
    db_conds,
    cfg,
    device,
    ks=(1,),
    dt_ms=DEFAULT_DT_MS,
    sensor_size=DEFAULT_SENSOR,
    stride=1,
    num_workers=4,
    threshold_m=25.0,
    verbose=True,
):
    """The benchmark protocol in-loop: ``query`` traverse vs one POOLED database.

    Replaces the per-condition suite (each condition vs a single sunset2 reference) with
    the pooled retrieval the project actually reports: sunset1 queries against the union
    of every other traverse, 25 m Euclidean GT, native cosine — no whitening anywhere.

    Frame indexing: descriptors are extracted at ``stride`` over eventcv's dt=50 ms
    slicing of the raw HDF5 (``EventH5Dataset``, ``start=0``); the coords file is keyed
    by the same slicing, and the per-traverse lengths are asserted equal so any eventcv
    slicing drift is a hard error rather than a silently shifted GT. Frames outside the
    GPS span (``valid`` False) are dropped from both sides after extraction.
    """
    coords = np.load(coords_path)
    tf = eval_transform(cfg)
    rep = getattr(cfg, "representation", "tencode")
    # The coords are keyed to the PUBLISHED traverse framing (megaevent's calibrated
    # geometry), while eventcv slices from the raw recording's own t_min — sunset2 has
    # 202 extra leading raw slices (see EventH5Dataset's docstring). Trim with the same
    # measured offsets the per-condition suite uses; fall back to the count difference
    # as a leading-offset assumption, loudly. That fallback was verified 2026-08-20 by
    # count-series cross-correlation on all six Brisbane traverses: peak at lag 0
    # everywhere (daytime -1 slice = <1 m at driving speed, noise against the 25 m GT).
    # NB the published grid can sit at a sub-50 ms phase to the raw grid, so exact
    # window identity is NOT expected — center alignment within one slice is.
    offsets = {}
    cache = os.path.join(root, "eval_offsets.json")
    if os.path.exists(cache):
        with open(cache) as f:
            offsets = json.load(f)
    banks = {}
    for cond in [query, *db_conds]:
        spec = os.path.join(root, cond, f"{cond}.hdf5")
        xy = coords[f"{cond}_xy"]
        valid = coords[f"{cond}_valid"]
        probe = EventH5Dataset(spec, tf, dt_ms=dt_ms, sensor_size=sensor_size, representation=rep)
        n_raw = probe._n_total
        off = int(offsets.get(cond, n_raw - len(xy)))
        if cond not in offsets and off != 0:
            print(
                f"[eval] WARNING: pooled {cond}: no measured offset in {cache}; "
                f"assuming the {off} extra raw slices are LEADING (raw {n_raw} vs "
                f"coords {len(xy)}). Verify with evalsuite.py --derive-offsets."
            )
        if off < 0 or n_raw - off != len(xy):
            raise ValueError(
                f"[pooled] {cond}: coords file has {len(xy)} slices but eventcv produced "
                f"{n_raw} (offset {off}) — regenerate the coords "
                f"(export_brisbane_coords.py) with the eventcv version this trainer uses"
            )
        ds = EventH5Dataset(
            spec,
            tf,
            dt_ms=dt_ms,
            stride=stride,
            sensor_size=sensor_size,
            representation=rep,
            start=off,
            n_slices=n_raw,
        )
        t0 = time.time()
        desc = _descriptors(model, ds, cfg, device, num_workers=num_workers)
        idx = np.arange(0, ds._n_total, ds.stride)
        keep = valid[idx]
        banks[cond] = (desc[torch.as_tensor(keep)], xy[idx][keep])
        if verbose:
            print(
                f"[eval]   pooled {cond}: {int(keep.sum())}/{len(idx)} frames "
                f"({time.time() - t0:.0f}s)"
            )
    q_desc, q_xy = banks[query]
    db_desc = torch.cat([banks[c][0] for c in db_conds])
    db_xy = np.concatenate([banks[c][1] for c in db_conds])
    gt = pooled_gt(db_xy, q_xy, threshold_m)
    rec = recall_at_k_cross(q_desc, db_desc, gt, ks=ks, device=device)
    out = {f"R@{k}": v for k, v in rec.items()}
    out["nq"] = int(q_desc.size(0))
    out["nr"] = int(db_desc.size(0))
    out["scorable"] = int(gt.any(axis=1).sum())
    return out


# ---------------------------------------------------------------------------
# PCA-whitening post-processing (fit on the reference traverse, apply to both)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------


@torch.no_grad()
def _descriptors(model, ds, cfg, device, batch_size=None, num_workers=4, amp=True):
    """Descriptors for a frame-source Dataset, in order. [N, D] float32 on CPU."""
    model.eval()
    loader = DataLoader(
        ds,
        batch_size=batch_size or cfg.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    use_amp = amp and str(device).startswith("cuda")
    autocast = torch.amp.autocast(device_type="cuda", enabled=use_amp)
    out = []
    for imgs in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast:
            out.append(model(imgs).float().cpu())
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Checkpoint loading (shared with sweep_eval / edvpr_eval)
# ---------------------------------------------------------------------------
