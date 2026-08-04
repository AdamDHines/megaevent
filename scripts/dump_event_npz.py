"""Materialise a traverse's real event slices as per-slice ``.npz``, in I2E's format.

Run from the repo root::

    pixi run python3 scripts/dump_event_npz.py --seq sunset1

This is the *real* arm of the real-vs-I2E ablation. The synthetic arm is already a tree of
``.npz`` files — that is what I2E writes — and every method in :mod:`src.methods` consumes
``.npz`` paths through :mod:`src.npzdata`. Writing the real slices in the same format means
both arms run through byte-identical code from loading onward, which is the whole point:
any difference in the recall is a difference in the *events*, not in the pipeline.

The reader is opened with exactly the call :meth:`src.inference.EventStreamDataset._open`
makes — same ``dt_ms``, same ``sensor_size``, same ``hot_pixel_filter``, same ``offset`` —
so slice ``i`` here is slice ``i`` of the run that produced this repo's published brisbane
numbers, and the ``real`` arm is a true regression test of them.

It also writes ``slice_times.npy`` — the ``[start_us, end_us]`` of every slice, taken from
the reader — because the slice grid is **not** ``offset + i * dt``. eventcv clamps ``offset``
up to the recording's first timestamp (``load.py``: *"clamped up to t_min, so an offset
before the recording is a no-op"*), and brisbane's sunset1 offset sits 1.22 s *before* its
first event, so its framing origin is the file's ``t_min`` and an analytic formula would put
every frame 24.5 slices out. :mod:`scripts.extract_aps` reads this file rather than
recomputing, which removes the whole class of off-by-one.

Two format details that are easy to get wrong:

* eventcv hands back ``t`` in **absolute** microseconds (~1.6e15), which does not fit in the
  ``int32`` I2E's format uses. Timestamps are rebased to the slice start; every
  representation in :mod:`src.npzdata` uses only relative time, so nothing is lost.
* eventcv encodes polarity ``{0, 1}``; I2E writes ``{-1, +1}``. ``src.npzdata`` tests
  ``p > 0``, so the mapping has to happen here.

Verified byte-identical: ``src.npzdata.load_countmask`` on the written file reproduces
``ecv.open(...).with_repr("countmask", ...)[i]` exactly (``scripts/verify_i2e_brisbane.py``).

Sharded across processes because each slice is an independent compress-and-write. Each
worker opens its own reader: the handle is not fork-safe (see ``src/inference.py``'s
``__getstate__``).
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eventcv as ecv                                            # noqa: E402

from src.inference import EVENTLAB_DATASETS, sensor_size         # noqa: E402


def stream_offset(dataset, seq):
    """``other.offset[<seq>]`` in milliseconds — eventcv's ``offset`` argument.

    Mirrors ``src/inference.py``'s brisbane branch rather than duplicating the number, so
    the slice grid cannot drift from the one descriptors are extracted on. Datasets without
    an offset table get 0, as they do there.
    """
    path = os.path.join(EVENTLAB_DATASETS, f"{dataset}.yaml")
    with open(path) as f:
        spec = yaml.safe_load(f)
    offsets = spec.get("other", {}).get("offset", {})
    if seq not in offsets:
        return 0.0
    return float(offsets[seq]) * 1000.0


_CFG = {}


def _init(cfg):
    """Per-worker reader. Built here, once, rather than per slice."""
    _CFG.update(cfg)
    _CFG["reader"] = ecv.open(cfg["path"], dt_ms=cfg["dt_ms"],
                              sensor_size=tuple(cfg["sensor"]),
                              hot_pixel_filter=cfg["hot_pixel"],
                              offset=cfg["offset"])


def framing_origin_ms(reader):
    """The timestamp slice 0 actually starts at, in milliseconds.

    ``reader.offset`` reports the offset as *given*, but eventcv clamps it up to the
    recording's first timestamp. sunset1's yaml offset precedes its first event by 1.22 s,
    so its real origin is ``t_min`` — hence the ``max``. Verified against the data:
    ``reader[0]`` on sunset1 spans 1587452583.5742-.6223 s, i.e. ``t_min`` + one window.
    """
    return max(float(reader.offset or 0.0), float(reader.time_span_ms[0]))


def _dump(bounds):
    lo, hi = bounds
    reader, out_dir = _CFG["reader"], _CFG["out_dir"]
    origin_us, dt_us = _CFG["origin_us"], _CFG["dt_us"]
    w, h = _CFG["sensor"]
    resolution = np.array([h, w], np.uint16)          # I2E stores [H, W]
    n_events = n_empty = 0
    for i in range(lo, hi):
        dst = os.path.join(out_dir, f"frame_{i:06d}.npz")
        a = reader[i].numpy()                          # [N,4] uint64, ('x','y','t','p')
        n_events += int(a.shape[0])
        n_empty += int(a.shape[0] == 0)
        if _CFG["resume"] and os.path.exists(dst):
            continue
        if a.shape[0]:
            # Rebased to the *nominal* window start, not the first event: absolute µs
            # overflows int32, and every renderer in src.npzdata is shift-invariant in t
            # (countmask and count ignore it, mcts uses differences, load_count_triplet
            # spans t.min()..t.max()), so the nominal origin is the better provenance.
            t = (a[:, 2].astype(np.int64) - (origin_us + i * dt_us)).astype(np.int32)
            p = np.where(a[:, 3] > 0, 1, -1).astype(np.int8)  # eventcv {0,1} -> I2E {-1,+1}
            x, y = a[:, 0].astype(np.uint16), a[:, 1].astype(np.uint16)
        else:
            x = y = np.empty(0, np.uint16)
            t = np.empty(0, np.int32)
            p = np.empty(0, np.int8)
        np.savez_compressed(dst, x=x, y=y, t=t, p=p, resolution=resolution)
    return hi - lo, n_events, n_empty


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="traverse name, e.g. sunset1")
    ap.add_argument("--dataset", default="brisbane_event")
    ap.add_argument("--eventlab-dir", default="/media/adam/vprdatasets/eventgem",
                    help="root holding <dataset>/<seq>/<seq>.hdf5")
    ap.add_argument("--npz-root", default="/media/adam/vprdatasets/megaevent/brisbane_npz",
                    help="slices land in <npz-root>/<dataset>/real/<seq>")
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--no-hot-pixel", action="store_true",
                    help="disable hot-pixel removal (the traverse evaluation leaves it on)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    path = os.path.join(args.eventlab_dir, args.dataset, args.seq, f"{args.seq}.hdf5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no recording for '{args.seq}' at {path}")
    out_dir = os.path.join(args.npz_root, args.dataset, "real", args.seq)
    os.makedirs(out_dir, exist_ok=True)

    sensor = sensor_size(args.dataset)
    offset = stream_offset(args.dataset, args.seq)
    cfg = {"path": path, "dt_ms": args.dt_ms, "sensor": sensor, "offset": offset,
           "hot_pixel": not args.no_hot_pixel, "out_dir": out_dir,
           "resume": not args.no_resume}

    probe = ecv.open(path, dt_ms=args.dt_ms, sensor_size=sensor,
                     hot_pixel_filter=cfg["hot_pixel"], offset=offset)
    n = int(probe.n_slices)
    origin_us = framing_origin_ms(probe) * 1000.0
    dt_us = float(args.dt_ms) * 1000.0
    del probe
    cfg["origin_us"], cfg["dt_us"] = origin_us, dt_us
    clamped = origin_us > offset * 1000.0 + 1.0
    print(f"{args.seq}: {n} slices @ {args.dt_ms} ms, hot-pixel {cfg['hot_pixel']}")
    print(f"  yaml offset {offset / 1000:.6f} -> framing origin {origin_us / 1e6:.6f}"
          + ("  (CLAMPED to the recording's first event)" if clamped else ""))
    print(f"  -> {out_dir}")

    # The grid every downstream stage aligns to. Written from the reader, never recomputed.
    starts = origin_us + np.arange(n, dtype=np.float64) * dt_us
    slice_times = np.stack([starts, starts + dt_us], axis=1)
    np.save(os.path.join(out_dir, "slice_times.npy"), slice_times)

    # Contiguous ranges, not a round-robin: eventcv reads the stream in order, so a worker
    # that walks a block sequentially stays on one part of the file.
    chunks = max(1, args.workers) * 4
    edges = np.linspace(0, n, chunks + 1).astype(int)
    bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    t0 = time.time()
    done = events = empty = 0
    with Pool(max(1, args.workers), initializer=_init, initargs=(cfg,)) as pool:
        for k, e, z in pool.imap_unordered(_dump, bounds):
            done += k
            events += e
            empty += z
            print(f"  {done}/{n} slices  ({time.time() - t0:.0f}s)", flush=True)

    total_bytes = sum(os.path.getsize(os.path.join(out_dir, f))
                      for f in os.listdir(out_dir) if f.endswith(".npz"))
    report = {"sequence": args.seq, "dataset": args.dataset, "source": path,
              "dt_ms": args.dt_ms, "yaml_offset_s": offset / 1000.0,
              "framing_origin_s": origin_us / 1e6, "offset_clamped": bool(clamped),
              "sensor_wh": list(sensor), "hot_pixel_filter": cfg["hot_pixel"],
              "background_activity_filter": False, "n_slices": n,
              "events": events, "empty_slices": empty, "bytes": total_bytes,
              "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "dump.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"  {events} events over {n} slices ({events / max(n, 1):.0f}/slice), "
          f"{empty} empty, {total_bytes / 1e9:.2f} GB in {report['seconds']:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
