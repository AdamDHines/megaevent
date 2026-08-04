"""Reading I2E's per-image event streams and rendering them into method inputs.

The ``.npz`` files are I2E's *raw event streams* (``x, y, t, p, resolution``), not
pre-rendered frames — the representation is chosen here, at load time, so the same data
can feed methods that were trained on completely different ones. Two things about them
are easy to get wrong:

* ``resolution`` is stored ``[H, W]``, while eventcv's ``sensor_size`` is ``(W, H)``;
* the streams are ~30 ms long, well under the ~1 s that eventcv's time-unit
  auto-detection assumes, so ``time_unit`` must be passed explicitly.

Ground truth is carried in the filenames, not a sidecar file: I2E's Tokyo 24/7 converter
writes ``@easting@northing@zone@band@lat@lon@...@.npz``, so the UTM coordinate is field 1
and 2 of an ``@``-split basename.

Split out of :mod:`src.imagevpr` so :mod:`src.methods` can render frames without
importing the evaluation loop that consumes them.
"""

import glob
import os

import eventcv as ecv
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Dataset layout
# ---------------------------------------------------------------------------
def split_dir(args, split):
    """``<data-dir>/<dataset>/numpy/<split>`` — I2E's output tree."""
    path = os.path.join(args.data_dir, args.dataset, "numpy", split)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"no '{split}' split at {path}")
    return path


def list_npz(directory):
    """Every ``.npz`` in ``directory``, sorted.

    The sort is what aligns row *i* of the descriptor bank with row *i* of the UTM array,
    so it has to be reproducible across runs and across a cache hit.
    """
    paths = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"no .npz files in {directory}")
    return paths


def utm_from_paths(paths):
    """``[N, 2]`` float64 easting/northing, parsed out of the ``@``-delimited basenames.

    I2E's ``tokyo247.get_dst_image_name`` writes
    ``@easting@northing@zone@band@lat@lon@pano@tile@...@.npz``, so an ``@``-split leaves
    the coordinate at index 1 and 2 (index 0 is the empty string before the first ``@``).
    """
    out = np.empty((len(paths), 2), dtype=np.float64)
    for i, path in enumerate(paths):
        fields = os.path.basename(path).split("@")
        try:
            out[i] = (float(fields[1]), float(fields[2]))
        except (IndexError, ValueError) as err:
            raise ValueError(
                f"{os.path.basename(path)} does not carry a UTM coordinate in the "
                f"@easting@northing@... form this evaluation derives ground truth from "
                f"({err})") from err
    return out


# ---------------------------------------------------------------------------
# 2. Event stream -> frames
# ---------------------------------------------------------------------------
def read_events(path):
    """``(x, y, t, p, height, width)`` straight out of one ``.npz``."""
    with np.load(path) as npz:
        x, y, t, p = npz["x"], npz["y"], npz["t"], npz["p"]
        height, width = (int(v) for v in npz["resolution"])      # stored [H, W]
    return x, y, t, p, height, width


def _stream(x, y, t, p, height, width):
    events = np.stack([x, y, t, p], axis=1).astype(np.int64)     # eventcv's 'xytp' layout
    return ecv.from_numpy(events, sensor_size=(width, height),   # sensor_size is (W, H)
                          time_unit="us", order="xytp")


def load_countmask(path):
    """One ``.npz`` -> ``[3, H, W]`` uint8 countmask frame, black background.

    Byte-for-byte identical to I2E's reference renderer
    (``i2e_infer.countmask_numpy(..., pct=99.0, white_frame=False)``, clipped to uint8) —
    eventcv stays the single source of truth for rendering, as it is in
    :mod:`src.inference`, so a frame here and a frame there cannot silently diverge.
    """
    x, y, t, p, height, width = read_events(path)
    if x.size == 0:                             # a saccade that produced no events at all
        return np.zeros((3, height, width), dtype=np.uint8)
    return _stream(x, y, t, p, height, width).countmask(white_frame=False).numpy()


def _area_resize(frame, size):
    """``[C, H, W]`` float32 -> ``[C, size, size]`` by area averaging.

    ``mode="area"`` is torch's equivalent of ``cv2.INTER_AREA``, which is what both
    baselines resize with. Area averaging is also the right choice for *counts*
    specifically: it preserves mean event density per pixel, so a downsampled frame
    carries the same event rate as the sensor saw.
    """
    x = torch.from_numpy(np.ascontiguousarray(frame)).float()[None]
    return F.interpolate(x, (size, size), mode="area")[0]


def load_count(path, size=None, clip=None):
    """One ``.npz`` -> ``[H, W]`` float32 event-count frame, both polarities summed.

    The input sparse_event works on. ``size`` resamples onto a common square grid —
    required whenever the splits differ in resolution, as Tokyo 24/7's do (database
    640x480, queries 480x854), because the method reads out *fixed pixel coordinates*.
    ``clip`` applies the method's ``remove_random_bursts`` saturation.
    """
    x, y, t, p, height, width = read_events(path)
    if x.size == 0:
        frame = np.zeros((1, height, width), dtype=np.float32)
    else:
        frame = _stream(x, y, t, p, height, width).count().numpy().astype(np.float32)
    out = _area_resize(frame, size)[0] if size else torch.from_numpy(frame[0])
    if clip is not None:
        out = out.clamp_(max=float(clip))
    return out.numpy()


def load_mcts(path, size=(240, 320), max_window_ms=30.0):
    """One ``.npz`` -> ``[10, H, W]`` float32 multi-channel time surface in [0, 1].

    SuperEvent's input representation, and so Event-GeM's: 5 exponential-decay time surfaces
    per polarity, ``exp(-dt / window)`` at each pixel's most recent event, zero where that
    pixel saw nothing inside the window. Already in [0, 1], so there is no mean/std
    normalisation to apply — see ``super_event.yaml`` (``input_representation: mcts``,
    ``input_channels: 10``).

    ``size`` is ``(H, W)`` and resamples onto a common grid, which Tokyo 24/7 needs for the
    same reason sparse_event does (database 640x480, queries 480x854). Unlike the count
    renderers this resizes in the **event domain** — ``EventStream.resize`` rebins the events
    themselves rather than interpolating a rendered frame, so the decay values stay exact
    rather than being averaged between pixels that fired at different times. The default
    240x320 is SuperEvent's own geometry (a cropped DAVIS346) and an exact multiple of the
    40-pixel input constraint its MaxViT backbone imposes (``patch_size`` 2 x 2^(stages-1)
    x ``partition_size`` 5), so nothing is cropped away.

    ``max_window_ms`` is eventcv's default, which is also what Event-GeM gets by never
    passing it (``eventgem/dataset.py:EventGeMMCTS``). It happens to bracket an I2E saccade's
    ~28.7 ms span almost exactly, so the longest window sees the whole stream.

    No hot-pixel filter, unlike upstream's ``ecv.open(..., hot_pixel_filter=True)``: that is
    for real DAVIS recordings, and these are synthetic I2E events with no hot pixels. None of
    the other renderers here apply one either.
    """
    height, width = size if size else (None, None)
    x, y, t, p, h, w = read_events(path)
    if x.size == 0:
        return np.zeros((10, height or h, width or w), dtype=np.float32)
    stream = _stream(x, y, t, p, h, w)
    if size:
        stream = stream.resize(width=width, height=height)
    return stream.mcts(max_window_ms=max_window_ms).numpy()


def load_count_triplet(path, size=256, bins=3, percentile=99.0):
    """One ``.npz`` -> ``[bins, size, size]`` float32 in [0, 1], one plane per sub-window.

    EventVLAD's denoiser expects ``bins`` *consecutive time slices* of a stream, which a
    single image does not obviously have — but an I2E saccade does: it is a short
    trajectory sampled at 23 discrete time steps over ~30 ms, so splitting it into equal
    sub-windows recovers exactly the kind of short temporal sequence the network was
    trained on, rather than repeating one frame ``bins`` times.

    Each plane is normalised by the ``percentile``-th percentile of its own non-zero
    counts and clipped to [0, 1], matching ``normalize_frame`` in Event-LAB's
    ``utils/eventvlad_denoiser.py``.
    """
    x, y, t, p, height, width = read_events(path)
    planes = np.zeros((bins, height, width), dtype=np.float32)
    if x.size:
        edges = np.linspace(int(t.min()), int(t.max()) + 1, bins + 1)
        for i in range(bins):
            m = (t >= edges[i]) & (t < edges[i + 1])
            if not m.any():
                continue
            planes[i] = _stream(x[m], y[m], t[m], p[m], height,
                                width).count().numpy()[0].astype(np.float32)
    for i in range(bins):
        nz = planes[i][planes[i] > 0]
        scale = float(np.percentile(nz.astype(np.float64), percentile)) if nz.size else 0.0
        if scale > 0:
            planes[i] /= scale
    return _area_resize(np.clip(planes, 0.0, 1.0), size).numpy()
