"""LENS v2 descriptor extraction, run inside LENS's own pixi environment.

Not invoked by hand. :mod:`src.lens_bridge` writes a JSON job and launches this through
``pixi run --manifest-path <LENSV2>/pixi.toml -e cuda``; everything the child needs arrives
as a plain value in that job, so nothing about megaevent's environment has to be reachable
from here. See ``src/lens_bridge.py`` for why a second environment exists at all.

    <job>.json --> (2, 128, 128) event-count frames --> LENSv2 spiking conv net
      --> [N, 1024] L2-normalised float32 bank, written atomically

Two input modes, one renderer:

===============  ==============================================  ==========================
``mode``         rows                                            source
===============  ==============================================  ==========================
``npz``          one per ``.npz`` path, in the order given        image sets (Tokyo 24/7)
``traverse``     one per ``dt_ms`` slice of one recording         pooled Brisbane
===============  ==============================================  ==========================

**This is the one extractor that does not render through** :mod:`src.npzdata`. LENS's
input is ``(2, 128, 128)`` raw event **counts**, and the network's first spiking layer
fires on their absolute magnitude — an IAF threshold is not scale-invariant the way a
normalised ViT input is. ``load_onoff`` resamples to (260, 346), so reaching 128 through it
would resample twice and rescale every bin, changing the model rather than the protocol.
LENS therefore decodes natively with ``lens.src.i2e.load_event_frame`` (``.npz``) and
``events_to_frame_ecv`` (HDF5 slices) — the same eventcv rebin its training loader uses,
which preserves counts by construction. Everything upstream of the frame — which files, in
what order, sliced how, with which filters — is still megaevent's.

The row order *is* the contract. ``npz`` mode consumes the path list the caller enumerated
(so row *i* keeps its ground-truth coordinate) and ``traverse`` mode walks slice indices in
order (so row *i* is slice *i*, which is what lets the pose grid index it). Neither
re-sorts anything.

``quantise`` picks which network runs. ``chip`` is the int8 ``DynapcnnNetwork`` that
actually deploys and is generally the *better* model here (Brisbane sunset1 R@1
60.8 -> 67.1); ``fp32`` is the trained weights. They are different estimators, so a bank
records which one it came from and the two are never mixed inside one similarity matrix.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _add_paths(job):
    """Make ``lens.*`` importable. The environment already supplies sinabs and eventcv."""
    repo = job["lens_repo"]
    if repo not in sys.path:
        sys.path.insert(0, repo)


# ---------------------------------------------------------------------------
# 1. Datasets
# ---------------------------------------------------------------------------
class NpzCountDataset(Dataset):
    """One ``.npz`` event stream per row, rebinned to ``(2, 128, 128)`` counts."""

    def __init__(self, paths, out_size):
        from lens.src.i2e import load_event_frame

        self.paths = list(paths)
        self.out_size = int(out_size)
        self._render = load_event_frame
        # Render one up front so a broken tree fails *here*, with a useful path in the
        # traceback, rather than inside a DataLoader worker.
        probe = self._render(self.paths[0], out_size=self.out_size)
        if tuple(probe.shape) != (2, self.out_size, self.out_size):
            raise ValueError(f"{self.paths[0]} renders {tuple(probe.shape)}; LENS needs a "
                             f"2-channel {self.out_size}x{self.out_size} count frame")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # (frame, index) rather than a bare frame, so `eventcv_collate` applies and the
        # extractor can assert row order instead of trusting it.
        return self._render(self.paths[i], out_size=self.out_size), i


class SliceCountDataset(Dataset):
    """Fixed ``dt_ms`` slices of one recording, rebinned to ``(2, 128, 128)`` counts.

    The reader is opened lazily and never pickled (see ``__getstate__``): eventcv's is not
    fork-safe, so each DataLoader worker builds its own. Same contract and the same filter
    settings as ``scripts/spikevpr_extract.SliceOnOffDataset``; what differs is only the
    rendering of the raw ``EventStream`` the reader hands back.
    """

    def __init__(self, recording, sensor, dt_ms, offset_ms, hot_pixel, filter_dt_us,
                 out_size):
        self.recording = recording
        self.sensor = tuple(sensor)
        self.dt_ms = int(dt_ms)
        self.offset_ms = int(offset_ms)
        self.hot_pixel = bool(hot_pixel)
        self.filter_dt_us = int(filter_dt_us) if filter_dt_us else None
        self.out_size = int(out_size)
        self._reader = None

        reader = self._open()
        self._n = int(reader.n_slices)
        self._reader = None                 # don't carry the handle across a fork

    def _open(self):
        import eventcv as ecv

        reader = ecv.open(self.recording, dt_ms=self.dt_ms, sensor_size=self.sensor,
                          hot_pixel_filter=self.hot_pixel, offset=self.offset_ms)
        # eventcv's window is in the stream's raw timestamp unit (microseconds), not
        # milliseconds — the caller resolves that and passes it already converted.
        if self.filter_dt_us is not None:
            reader = reader.background_activity_filter(self.filter_dt_us)
        return reader

    @property
    def reader(self):
        if self._reader is None:            # first touch in this process/worker
            self._reader = self._open()
        return self._reader

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        # The same two eventcv calls `lens.src.i2e.load_event_frame` makes on a .npz:
        # an event-domain rebin (the frame then holds exactly as many events as the
        # sensor saw, where an area-resized count frame would only hold the same *mean*)
        # then a polarity split into [ON, OFF]. Going through `events_to_frame_ecv`
        # instead would round-trip the slice through numpy for the same answer.
        frame = self.reader.slice(int(i)).resize(self.out_size, self.out_size) \
                    .flatten("polarity")
        arr = np.asarray(frame.numpy() if hasattr(frame, "numpy") else frame)
        return torch.as_tensor(arr, dtype=torch.float32), i

    def __getstate__(self):
        return {**self.__dict__, "_reader": None}


def build_dataset(job):
    out_size = int(job["input_shape"][-1])
    if job["mode"] == "npz":
        return NpzCountDataset(job["paths"], out_size)
    if job["mode"] == "traverse":
        return SliceCountDataset(job["recording"], job["sensor"], job["dt_ms"],
                                 job["offset_ms"], job["hot_pixel"], job["filter_dt_us"],
                                 out_size)
    raise ValueError(f"unknown mode {job['mode']!r}; expected 'npz' or 'traverse'")


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
def build_embedder(job, device):
    """``(frames -> [B, 1024])`` plus a one-line description of what was built.

    ``ann`` and ``spike_threshold`` come from the checkpoint, never from this job:
    ``load_backbone`` is deliberately the authority, because several sweeps with different
    thresholds coexist and a mismatched one loads cleanly and returns a different
    descriptor.
    """
    from lens.eval_suite import chip_embedder, model_embedder
    from lens.models.chip_sim import build_chip_sim, load_backbone

    if not os.path.exists(job["checkpoint"]):
        raise FileNotFoundError(f"no LENS checkpoint at {job['checkpoint']}")

    if job["quantise"] == "chip":
        sim = build_chip_sim(job["checkpoint"], device=device, discretize=True,
                             keep_dcnn=False)
        return chip_embedder(sim), "int8 chip_sim (DynapcnnNetwork, discretized)"
    if job["quantise"] == "fp32":
        model, state = load_backbone(job["checkpoint"], device=device)
        model.eval()
        return (model_embedder(model, device),
                f"fp32 backbone (ann={state.get('ann', False)}, "
                f"spike_threshold={state.get('spike_threshold')})")
    raise ValueError(f"quantise must be 'fp32' or 'chip', got {job['quantise']!r}")


# ---------------------------------------------------------------------------
# 3. Extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract(job):
    from lens.src.i2e import eventcv_collate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder, what = build_embedder(job, device)
    dataset = build_dataset(job)
    # `eventcv_collate` because the leaf returns torch tensors already; it is the collate
    # every LENS loader uses, so a batch here is the batch training and eval both see.
    loader = DataLoader(dataset, batch_size=job["batch_size"], shuffle=False,
                        num_workers=job["workers"], pin_memory=(device.type == "cuda"),
                        drop_last=False, collate_fn=eventcv_collate)
    n = len(dataset)
    label = job.get("label", job["mode"])
    print(f"{label}: {n} frames, {what} on {device}, batch {job['batch_size']} "
          f"(pinned: sinabs makes the batch dimension visible to the network)", flush=True)

    # Written to `.tmp` and renamed on completion, so a reader never sees a partial bank.
    # The descriptor width comes from the first batch rather than from a constant, so a
    # differently shaped head cannot silently write into a mis-shaped file.
    pending = job["out"] + ".tmp"
    os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
    bank, at, start = None, 0, time.time()
    last, interval = start, float(job.get("progress_s", 15.0))
    zero_rows, spikes = 0, 0.0
    for frames, indices in loader:
        # Row order IS the contract: row i must keep file i's ground-truth coordinate.
        # shuffle=False makes this true by construction, so the check is free — and this
        # is a repo where four separate setup faults each produced a convincing 0.0%.
        if int(indices[0]) != at:
            raise RuntimeError(f"loader returned row {int(indices[0])} at position {at}; "
                               f"descriptors would not line up with their coordinates")
        desc = embedder(frames.to(device, non_blocking=True)).float().cpu().numpy()
        if bank is None:
            bank = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float32,
                                             shape=(n, desc.shape[1]))
        bank[at:at + len(desc)] = desc
        at += len(desc)
        # An all-zero descriptor is a *silent* failure: it survives normalisation (that is
        # what bounded_normalize is for), scores cosine 0 against everything, and turns a
        # working model into a convincing near-chance recall. Counted, not assumed.
        zero_rows += int((np.abs(desc).sum(axis=1) == 0).sum())
        spikes += float((desc != 0).sum())
        now = time.time()
        if now - last >= interval or at == n:
            rate = at / max(now - start, 1e-6)
            print(f"  {label}: {at}/{n}  {rate:.0f} frames/s  "
                  f"eta {(n - at) / rate / 60:.1f} min", flush=True)
            last = now
    if at != n:
        raise RuntimeError(f"wrote {at} of {n} descriptors — the loader lost rows")

    bank.flush()
    del bank
    os.replace(pending, job["out"])
    print(f"{label}: {n} descriptors -> {job['out']} "
          f"({os.path.getsize(job['out']) / 1e6:.0f} MB, {time.time() - start:.0f}s)",
          flush=True)
    print(f"{label}: health — dead descriptors {zero_rows}/{n} "
          f"({100.0 * zero_rows / max(n, 1):.2f}%), mean active dims "
          f"{spikes / max(n, 1):.1f}/1024", flush=True)
    if zero_rows == n:
        raise RuntimeError(
            f"every descriptor in {label} is all-zero. The network fired on nothing, so "
            f"any recall from this bank is meaningless. Check the event density against "
            f"the checkpoint's spike_threshold before rerunning.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", required=True, help="path to the JSON job description")
    args = parser.parse_args()

    with open(args.job) as handle:
        job = json.load(handle)
    _add_paths(job)
    extract(job)


if __name__ == "__main__":
    main()
