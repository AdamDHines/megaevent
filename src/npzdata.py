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


def _cap_events(events, max_events):
    """The first ``max_events`` events in time order, from an ``(N, 4)`` ``xytp`` array.

    SpikeVPR's own Brisbane pipeline frames a slice with tonic's
    ``ToFrame(event_count=15000)`` and keeps frame 0, which is exactly this: the leading
    fixed-size chunk of the stream, not a time window. It exists here as a contingency,
    not as the default — see :func:`onoff_from_stream`.
    """
    if max_events is None or len(events) <= max_events:
        return events
    order = np.argsort(events[:, 2], kind="stable")[:max_events]
    return events[np.sort(order)]


def onoff_from_stream(stream, size=(260, 346), max_events=None):
    """An ``EventStream`` -> ``[2, H, W]`` float32 ON/OFF event-count frame.

    SpikeVPR's input representation, and the only one it has: a SEW-ResNet with
    ``in_channels=2`` fed a per-pixel positive count and a per-pixel negative count. The
    counts are raw — the network has no input normalisation, only a ``BatchNorm2d`` running
    on frozen training statistics, so the *magnitude* here is part of the model's contract
    and must not be rescaled.

    ``size`` is ``(H, W)`` and defaults to the 260x346 that
    ``spikevpr.models.factory`` hardcodes MixVPR for (a (2, 260, 346) frame produces the
    (512, 9, 11) map its ``_FEATURE_H/_FEATURE_W`` describe), so it is not really optional.
    Every dataset here has to meet there: Brisbane is already a DAVIS346, NSAVP is 640x480,
    NYC 1280x720, and Tokyo 24/7's two splits are 640x480 and 480x854.

    Like :func:`load_mcts` this resizes in the **event domain** — ``EventStream.resize``
    rebins the events themselves rather than interpolating a rendered frame, which for
    counts specifically means the frame holds exactly as many events as the sensor saw. An
    area-resized count frame would only hold the same *mean*.

    ``max_events`` truncates to the leading events of the stream (see :func:`_cap_events`).
    Left ``None`` for every headline run: the protocol is that SpikeVPR sees the same events
    every other method sees. Whether that is *in distribution* varies a lot by dataset,
    which is what the cap is for — measured medians on this grid, against the 0.167
    events/px SpikeVPR's Brisbane checkpoint trained on:

    ==================  ========  ================
    source              med ev/px  p10-p90
    ==================  ========  ================
    brisbane_event      0.278     0.077 - 0.771
    nycevent            0.621     0.020 - 1.776
    nsavp               1.878     0.044 - 4.191
    msls                2.109     1.042 - 8.304
    pitts               2.3 - 4.2 0.324 - 6.369
    tokyo247            7.2 - 8.4 3.784 - 11.133
    ==================  ========  ================

    The two traverse datasets sit within ~1.7x of the window their checkpoint saw. Tokyo
    24/7 is ~43x, because an I2E saccade packs ~600k events into 29 ms, and the network
    normalises its input with nothing but a frozen BatchNorm.
    """
    height, width = size
    if max_events is not None:
        events = _cap_events(np.asarray(stream.numpy()), max_events)
        stream = ecv.from_numpy(events.astype(np.int64), sensor_size=stream.sensor_size,
                                time_unit="us", order="xytp")
    if size:
        stream = stream.resize(width=width, height=height)
    on = stream.filter_polarity(True).count().numpy()
    off = stream.filter_polarity(False).count().numpy()
    return np.concatenate([on, off], axis=0).astype(np.float32)


def load_onoff(path, size=(260, 346), max_events=None):
    """One ``.npz`` -> ``[2, H, W]`` float32 ON/OFF event-count frame.

    The ``.npz`` wrapper over :func:`onoff_from_stream`; the traverse path calls that
    directly on a reader slice, so the two share one renderer and cannot diverge.
    """
    height, width = size if size else (None, None)
    x, y, t, p, h, w = read_events(path)
    if x.size == 0:                             # a saccade that produced no events at all
        return np.zeros((2, height or h, width or w), dtype=np.float32)
    return onoff_from_stream(_stream(x, y, t, p, h, w), size=size, max_events=max_events)


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
