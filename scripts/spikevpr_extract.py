"""SpikeVPR descriptor extraction, run inside ``envs/spikevpr``.

Not invoked by hand. :mod:`src.spikevpr_bridge` writes a JSON job and launches this through
``pixi run --manifest-path envs/spikevpr/pixi.toml``; everything the child needs arrives as
plain values in that job, so nothing about megaevent's own environment has to be reachable
from here. See ``envs/spikevpr/pixi.toml`` for why the second environment exists at all.

    <job>.json --> [2, 260, 346] ON/OFF frames --> SEW-ResNet34 + spiking MixVPR
      --> [N, 4096] L2-normalised float32 bank, written atomically

Two input modes, one renderer:

===============  ===============================================  ==========================
``mode``         rows                                             source
===============  ===============================================  ==========================
``npz``          one per ``.npz`` path, in the order given         image sets (Tokyo 24/7,
                                                                   MSLS, Pitts, NYC)
``traverse``     one per ``dt_ms`` slice of one recording          pooled Brisbane / NSAVP
===============  ===============================================  ==========================

Both render through :func:`src.npzdata.onoff_from_stream`, which this script imports from
the parent repository rather than copying — the two environments then cannot disagree about
what a SpikeVPR input frame is. ``src/npzdata.py`` needs only numpy, torch and eventcv, all
of which are here.

The row order *is* the contract. ``npz`` mode consumes the path list the caller enumerated
(so row *i* keeps its ground-truth coordinate) and ``traverse`` mode walks slice indices in
order (so row *i* is slice *i*, which is what lets the pose grid index it). Neither re-sorts
anything.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)


def _add_paths(job):
    """Make eventcv, ``src.npzdata`` and the SpikeVPR package importable.

    The same ``sys.path`` injection :mod:`src.methods` uses to reach EventVLAD's networks
    and :mod:`src.model` to reach dinov2. eventcv is tried as a normal import first, so this
    keeps working if the environment ever gains it as a real dependency; the fallback is its
    checkout, whose compiled ``_rust.abi3.so`` loads under both numpy 1.x and 2.x.
    """
    try:
        import eventcv  # noqa: F401
    except ImportError:
        sys.path.insert(0, job["eventcv_path"])
    for path in (REPO_ROOT, os.path.join(job["spikevpr_repo"], "src")):
        if path not in sys.path:
            sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# 1. Datasets
# ---------------------------------------------------------------------------
class NpzOnOffDataset(Dataset):
    """One ``.npz`` event stream per row, rendered to ``[2, H, W]``."""

    def __init__(self, paths, size, max_events):
        from src.npzdata import load_onoff

        self.paths = list(paths)
        self.size = tuple(size)
        self.max_events = max_events
        self._render = load_onoff
        # Render one up front so a broken tree fails *here*, with a useful path in the
        # traceback, rather than inside a DataLoader worker. Same guard as
        # src.methods.NpzFrameDataset.
        probe = self._render(self.paths[0], size=self.size, max_events=self.max_events)
        if probe.shape != (2, *self.size):
            raise ValueError(f"{self.paths[0]} renders {tuple(probe.shape)}; SpikeVPR needs "
                             f"a 2-channel {self.size[0]}x{self.size[1]} ON/OFF frame")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(
            self._render(self.paths[i], size=self.size, max_events=self.max_events))


class SliceOnOffDataset(Dataset):
    """Fixed ``dt_ms`` slices of one recording, rendered to ``[2, H, W]``.

    The reader is opened lazily and never pickled (see ``__getstate__``): eventcv's is not
    fork-safe, so each DataLoader worker has to build its own. Same contract as
    :class:`src.inference.EventStreamDataset`, and the same filter settings — what differs
    is that no representation is set on the reader, because ``slice(i)`` then hands back the
    raw ``EventStream`` that :func:`onoff_from_stream` needs to split by polarity.
    """

    def __init__(self, recording, sensor, dt_ms, offset_ms, hot_pixel, filter_dt_us,
                 size, max_events):
        self.recording = recording
        self.sensor = tuple(sensor)
        self.dt_ms = int(dt_ms)
        self.offset_ms = int(offset_ms)
        self.hot_pixel = bool(hot_pixel)
        self.filter_dt_us = int(filter_dt_us) if filter_dt_us else None
        self.size = tuple(size)
        self.max_events = max_events
        self._reader = None

        from src.npzdata import onoff_from_stream

        self._render = onoff_from_stream
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
        return torch.from_numpy(
            self._render(self.reader.slice(int(i)), size=self.size,
                         max_events=self.max_events))

    def __getstate__(self):
        return {**self.__dict__, "_reader": None}


def build_dataset(job):
    if job["mode"] == "npz":
        return NpzOnOffDataset(job["paths"], job["size"], job["max_events"])
    if job["mode"] == "traverse":
        return SliceOnOffDataset(job["recording"], job["sensor"], job["dt_ms"],
                                 job["offset_ms"], job["hot_pixel"], job["filter_dt_us"],
                                 job["size"], job["max_events"])
    raise ValueError(f"unknown mode {job['mode']!r}; expected 'npz' or 'traverse'")


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
def build_model(job, device):
    """The checkpoint on ``device``, in eval mode.

    ``neuron`` is not recoverable from the checkpoint — ``IFNode`` and ``LIFNode`` are both
    parameter-free, so the wrong one loads ``state_dict`` cleanly and silently produces a
    different descriptor. It is therefore resolved once by the caller (see
    ``src/spikevpr_bridge.SPIKEVPR_CHECKPOINTS``) and only validated here.
    """
    from spikevpr.models import build_spikevpr

    if job["neuron"] not in ("IFNode", "LIFNode"):
        raise ValueError(f"neuron {job['neuron']!r} is neither IFNode nor LIFNode")
    if not os.path.exists(job["checkpoint"]):
        raise FileNotFoundError(f"no SpikeVPR checkpoint at {job['checkpoint']}")
    return build_spikevpr(job["encoder"], out_channels=job["out_channels"],
                          out_rows=job["out_rows"], neuron_type=job["neuron"],
                          checkpoint=job["checkpoint"], device=device, eval_mode=True)


# ---------------------------------------------------------------------------
# 3. Extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract(job):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(job, device)
    dataset = build_dataset(job)
    loader = DataLoader(dataset, batch_size=job["batch_size"], shuffle=False,
                        num_workers=job["workers"], pin_memory=(device.type == "cuda"),
                        drop_last=False)
    n = len(dataset)
    label = job.get("label", job["mode"])
    print(f"{label}: {n} frames, {job['encoder']}/{job['neuron']} on {device}, "
          f"{job['size'][0]}x{job['size'][1]}"
          + (f", first {job['max_events']} events" if job["max_events"] else ""),
          flush=True)

    # Written to `.tmp` and renamed on completion, so a reader never sees a partial bank —
    # the pattern scripts/eventvlad_pooled.py already uses. The descriptor width comes from
    # the first batch rather than out_channels * out_rows, so a differently configured head
    # cannot silently write into a mis-shaped file.
    pending = job["out"] + ".tmp"
    os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
    bank, at, start = None, 0, time.time()
    # Progress on a timer rather than per batch: a 76k-image split is 2400 batches, and the
    # parent relays every line into the run's log file.
    last, interval = start, float(job.get("progress_s", 15.0))
    for batch in loader:
        desc = model(batch.to(device, non_blocking=True)).float().cpu().numpy()
        if bank is None:
            bank = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float32,
                                             shape=(n, desc.shape[1]))
        bank[at:at + len(desc)] = desc
        at += len(desc)
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
          f"({os.path.getsize(job['out']) / 1e6:.0f} MB, "
          f"{time.time() - start:.0f}s)", flush=True)


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
