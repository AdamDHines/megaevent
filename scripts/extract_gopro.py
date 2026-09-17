"""Pull the co-recorded windscreen-camera frames out of a Brisbane-Event-VPR video, one per
event slice.

Run from the repo root::

    pixi run python3 scripts/extract_gopro.py --seq daytime --verify   # check the clock first
    pixi run python3 scripts/extract_gopro.py --seq daytime

``scripts/dump_event_npz.py`` must have run first: it writes ``slice_times.npy`` from the
eventcv reader, and that file -- not any formula -- defines the grid. This is the same
contract ``scripts/extract_aps.py`` works to, and for the same reason: eventcv clamps its
``offset`` up to the recording's first timestamp, so ``offset + i * dt`` is 24.5 slices out
on sunset1.

This is the video counterpart of ``scripts/extract_aps.py``. Where that selects a DAVIS APS
frame per slice from a rosbag, this selects a video frame per slice from the ``.mp4``
Brisbane-Event-VPR ships alongside each recording. The APS frames share the event sensor but
are 346x260, dusk-exposed and vignetted (``aps/sunset1/select.json``: 15.6% of pixels
saturated, 11.8% black); the video is 1920x1080 of the same scenes through a camera that can
actually see them. Converting *those* frames with I2E gives the ``i2e_gopro`` arm, and the
difference from ``i2e`` is what says whether the sim2real gap is a property of I2E or a
property of the DAVIS's intensity channel.

**The time base is MEASURED, not assumed** (2026-09-11). The wide-window Sobel-NCC
measurement (``scripts/gopro_align.py``, 128 windows/traverse against the APS tree)
showed two things about the shipped ``_concat.mp4`` files:

* the video-vs-event lag sits within ~1 s of Event-LAB's yaml ``other.offset`` — the
  2026-09-10 ``VIDEO_BEGINNING`` table (yaml +11 s / +6 s), which had been chosen by
  maximising R@1 on cached banks, was simply wrong;
* the lag is **not constant**: morning holds +0.43 s for its first 300 s (bin IQRs of
  0.1 s) and then steps down by ~1.0 s in discrete ~0.3-0.5 s drops, sunset1 by ~1.1 s
  — the signature of frames lost at the chapter-concat joins. No single ``(v0, fps)``
  places every frame; NTSC 29.97 was tried and does not fit either.

So the mapping is the measured **time warp**: ``scripts/gopro_timewarp.py`` fits
running-median knots through the align measurements (held-out residual p90 is the
accuracy claim, asserted <= 0.35 s ~= 2-5 m at route speed), and ``--timewarp`` hands
the warp to this script, which selects frame ``round(((centre - v0_base) + delta) *
fps)``. The camera's own burnt-in clock is **not** usable -- it reads 0-5 s off,
differently per session -- and ``--verify`` re-measures the lag against whichever
mapping would frame the tree.

The videos are container-CFR (``nb_frames == round(duration * fps)`` for all four), so
within any span with no join the analytic index holds and ``|dt|`` is bounded by half a
frame interval plus the warp's local accuracy.

Alongside the frames:

* ``frame_times.npy`` -- ``[n_slices]`` float64 unix seconds, the stamp actually selected;
* ``select.json`` -- ``scripts/extract_aps.py``'s schema (plus a ``timewarp``
  provenance block), so ``src.traversenpz.frame_validity_mask`` reads both trees the
  same way. Its ``out_of_tolerance`` list is the slices the video does not cover — the
  event streams start before and run past the video, and every clamped slice is a
  duplicate of the video's first or last frame. Those rows are dropped from *every* arm
  (``--align-sources aps gopro`` in the pooled scripts, the aps∪gopro union in
  ``scripts/gopro_pairwise.py``) so the comparison never scores a real frame against a
  duplicated one.

Note the footage carries a burnt-in date stamp and GPS/speed readout, measured at rows
**1018-1045** of the 1080 (nothing above 1018, nothing below 1045, in all four traverses).
I2E would fire on those glyph edges at fixed pixel locations in every frame of every
traverse. This script writes the frame exactly as the camera recorded it -- the tree is the
raw record -- and the overlay is removed one stage later by ``i2e_infer.py --crop-bottom 70``
(see ``scripts/gopro_i2e.sh``), which is the only place the crop happens.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_aps import aps_quality, slice_centres                # noqa: E402

DEFAULT_VIDEO_ROOT = "/media/adam/vprdatasets/megaevent/brisbane_event/gopro"

# Upstream ships one concatenated video per traverse and the traverse name is in the filename.
# Only these four exist -- there is no sunset2 or night video -- which is exactly the set the
# pooled Sim2Real protocol uses (query sunset1, database daytime+morning+sunrise).
VIDEOS = {
    "sunset1": "20200421_170039-sunset1_concat.mp4",
    "daytime": "20200424_151015-daytime_concat.mp4",
    "morning": "20200428_091154-morning_concat.mp4",
    "sunrise": "20200429_061912-sunrise_concat.mp4",
}

# Absolute time of video frame 0, in the event clock — for a traverse WITHOUT a measured
# time warp. Empty on purpose: the 2026-09-10 table that used to live here (yaml offset
# +11 s sunset1 / +6 s morning / +6 s daytime / +8 s sunrise) was chosen by maximising
# R@1 on cached descriptor banks — fitted to the test metric — and the 2026-09-11
# wide-window Sobel-NCC measurement (scripts/gopro_align.py, 128 windows/traverse)
# refuted it: both measured traverses sit within ~1 s of the yaml offset, not 6-11 s
# above it. The same measurement showed the video-vs-event lag is not even constant
# (flat then ~0.3-0.5 s steps down, the signature of frames lost at the _concat chapter
# joins), so no single v0 places every frame: use ``--timewarp`` with the warp fitted by
# scripts/gopro_timewarp.py from those measurements instead. A seq listed here would be
# framed as v0 + i/fps with no warp — only ever add one on the strength of a PASSing
# gopro_align.py run showing drift_ok on a flat lag profile.
VIDEO_BEGINNING = {}


def resolve_video(seq, video, video_root):
    """The video for ``seq``. Explicit path wins, then the upstream name in ``video_root``."""
    if video:
        if not os.path.exists(video):
            raise FileNotFoundError(f"no video at {video}")
        return video
    if seq not in VIDEOS:
        raise FileNotFoundError(
            f"{seq} has no video: Brisbane-Event-VPR ships one only for "
            f"{', '.join(sorted(VIDEOS))}")
    path = os.path.join(video_root, VIDEOS[seq])
    if not os.path.exists(path):
        raise FileNotFoundError(f"no video for {seq} at {path}")
    return path


def probe(path):
    """``(width, height, fps, n_frames, duration)`` from ffprobe, with the CFR assertion.

    ``round((centre - v0) * fps)`` is only the nearest frame if the stream really is constant
    rate. All four videos satisfy ``nb_frames == round(duration * fps)`` exactly; a variable
    rate one would silently drift, so it is refused rather than approximated.
    """
    keys = ["width", "height", "r_frame_rate", "nb_frames", "duration"]
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=" + ",".join(keys), "-of", "default=noprint_wrappers=1:nokey=0", path],
        capture_output=True, text=True, check=True).stdout
    got = dict(line.split("=", 1) for line in out.strip().splitlines())
    num, _, den = got["r_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    width, height = int(got["width"]), int(got["height"])
    n_frames, duration = int(got["nb_frames"]), float(got["duration"])
    expected = round(duration * fps)
    if abs(n_frames - expected) > 1:
        raise SystemExit(
            f"{path}: {n_frames} frames but {duration:.3f} s at {fps:g} fps implies "
            f"{expected}. The stream is not constant-rate, so a frame index cannot be "
            f"derived from a timestamp; this script has no other way to place a frame.")
    return width, height, fps, n_frames, duration


def frame_index(centres, v0, fps, n_frames):
    """``(idx[n], stamp[n], dt[n])`` -- the nearest video frame to every slice centre.

    Analytic rather than searched: frame ``n`` is at ``v0 + n / fps`` exactly (CFR, asserted
    in :func:`probe`). Clamped at both ends, which is where ``dt`` can exceed half a frame
    interval -- the event stream starts before and runs past the video, and those slices are
    what ``select.json``'s ``out_of_tolerance`` records.
    """
    raw = np.rint((centres - v0) * fps).astype(np.int64)
    idx = np.clip(raw, 0, n_frames - 1)
    stamp = v0 + idx / fps
    return idx, stamp, stamp - centres


def load_timewarp(path, seq):
    """The measured video->event time warp from ``scripts/gopro_timewarp.py``."""
    with open(path) as handle:
        warp = json.load(handle)
    if warp["sequence"] != seq:
        raise SystemExit(f"{path} is the warp for {warp['sequence']}, not {seq}")
    if not warp.get("pass"):
        raise SystemExit(f"{path}: the warp FAILED its holdout acceptance "
                         f"({warp.get('checks')}) — re-measure, do not extract on it")
    return warp


def warp_container_time(centres, warp):
    """``[n]`` predicted CONTAINER time of each slice's matching frame.

    ``(centre - v0_base) + delta`` with ``delta`` interpolated through the measured
    knots; beyond the measured span the edge value holds (the affected slices are the
    unmeasurable first/last seconds, which mostly sit outside the video anyway and are
    masked by ``out_of_tolerance``).
    """
    t = centres - warp["v0_base_s"]
    return t + np.interp(t, np.asarray(warp["knots_t_s"]),
                         np.asarray(warp["knots_delta_s"]))


def frame_index_warp(centres, warp, fps, n_frames):
    """``(idx[n], stamp[n], dt[n])`` under the measured time warp.

    ``dt`` is the residual between the chosen frame's container time and the warped
    prediction (container and wall seconds differ by well under the 25 ms tolerance over
    a frame interval), and ``stamp`` is the slice centre plus that residual — the event-
    clock instant the chosen frame actually shows, to the warp's holdout accuracy.
    """
    pred = warp_container_time(centres, warp)
    raw = np.rint(pred * fps).astype(np.int64)
    idx = np.clip(raw, 0, n_frames - 1)
    dt = idx / fps - pred
    return idx, centres + dt, dt


def stream_frames(path, width, height, wanted, on_frame):
    """Decode ``path`` once, calling ``on_frame(n, bgr)`` for every ``n`` in ``wanted``.

    ``wanted`` is a sorted unique index array. Decoding is sequential because that is the only
    way to read a long H.264 stream at speed -- seeking per frame would re-decode from the
    preceding keyframe every time -- and the pipe is read in whole frames so a short read at
    the end is an error rather than a torn frame.
    """
    stride = width * height * 3
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, bufsize=stride)
    want = set(int(v) for v in wanted)
    last = int(wanted[-1])
    n = 0
    try:
        while n <= last:
            buf = proc.stdout.read(stride)
            if len(buf) < stride:
                break
            if n in want:
                on_frame(n, np.frombuffer(buf, np.uint8).reshape(height, width, 3).copy())
            n += 1
    finally:
        proc.stdout.close()
        proc.wait()
    return n


def sobel_ncc_prep(img, tw, th):
    """Zero-mean unit-norm Sobel magnitude at ``tw x th``.

    Sobel rather than intensity because that is the only statistic that survives the
    comparison: the video is a bright wide-angle colour frame and the APS is a dark,
    vignetted 346x260 one, so their intensities share almost nothing, but their *edges* are
    the same street furniture. Measured NCC on a true pair is ~0.28-0.44; high-passed
    intensity gives no usable peak at all.
    """
    import cv2

    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    m = np.sqrt(cv2.Sobel(f, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(f, cv2.CV_32F, 0, 1) ** 2)
    m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_AREA)
    m -= m.mean()
    return m / (np.linalg.norm(m) + 1e-6)


def decode_window(path, t0, n_frames, width, height):
    """``[n]`` grayscale frames of ``path`` starting at video time ``t0``, at ``width x height``."""
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t0:.4f}", "-i", path,
         "-frames:v", str(n_frames), "-vf", f"scale={width}:{height}:flags=area",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, bufsize=width * height)
    out = []
    try:
        while True:
            buf = proc.stdout.read(width * height)
            if len(buf) < width * height:
                break
            out.append(np.frombuffer(buf, np.uint8).reshape(height, width))
    finally:
        proc.stdout.close()
        proc.wait()
    return out


# Candidate crops of the 480x270 proxy, narrowest to full frame. The video sees roughly twice
# the DAVIS's linear field, so a crop is needed before the two can be compared at all; which
# one wins is not used for anything, but its consistency across slices is a free check that
# the field-of-view assumption is sane.
VERIFY_CROPS = ((150, 40, 240, 180), (135, 25, 270, 202), (120, 10, 300, 225),
                (60, 0, 360, 270), (0, 0, 480, 270))


def verify_lag(video_path, aps_dir, pred, fps, duration, speed, covered,
               n_sample=80, win_s=1.5, proxy=(480, 270), thumb=(87, 65)):
    """Per-slice video-vs-APS lag: ``(summary dict, per-slice lags, per-slice NCC)``.

    ``pred`` is the predicted CONTAINER time of each slice's matching frame — the same
    mapping the extraction uses (``(centre - v0) + warp delta`` under ``--timewarp``, or
    ``centre - v0`` without) — so the reported lag is the residual of exactly the model
    that framed the tree.

    For each sampled slice, decode a +-``win_s`` window and score every (frame, crop) against
    that slice's APS frame; the best frame's offset from the predicted one is one independent
    estimate of the lag. The **spread** of those estimates is the error bar, which is the
    whole point: a single global cross-correlation of the two streams returns a peak with a
    ~1.8 s FWHM and cannot say whether its own answer is meaningful.

    Only slices where the car is moving above 5 m/s are sampled -- a stationary slice matches
    every lag equally well and would dilute the median with pure noise.
    """
    import cv2

    dw, dh = proxy
    tw, th = thumb
    usable = np.flatnonzero(covered & (speed > 5.0)
                            & (pred > win_s + 1.0)
                            & (pred < duration - win_s - 1.0))
    if len(usable) < 8:
        raise SystemExit(f"only {len(usable)} usable moving slices inside the video span — "
                         f"cannot measure a lag")
    pick = usable[np.linspace(0, len(usable) - 1, min(n_sample, len(usable))).astype(int)]

    n_frames = int(round(2 * win_s * fps)) + 1
    lags, peaks, crops = [], [], []
    for at, i in enumerate(pick):
        aps = cv2.imread(os.path.join(aps_dir, f"frame_{int(i):06d}.png"),
                         cv2.IMREAD_GRAYSCALE)
        if aps is None:
            raise FileNotFoundError(f"{aps_dir}: no frame_{int(i):06d}.png — run "
                                    f"scripts/extract_aps.py first")
        ref = sobel_ncc_prep(aps, tw, th)
        t0 = pred[i] - win_s
        frames = decode_window(video_path, t0, n_frames, dw, dh)
        if len(frames) < n_frames // 2:
            continue
        best = (-9.0, 0.0, -1)
        for k, frame in enumerate(frames):
            for ci, (x, y, cw, ch) in enumerate(VERIFY_CROPS):
                score = float((ref * sobel_ncc_prep(frame[y:y + ch, x:x + cw],
                                                    tw, th)).sum())
                if score > best[0]:
                    best = (score, t0 + k / fps - pred[i], ci)
        peaks.append(best[0])
        lags.append(best[1])
        crops.append(best[2])
        if (at + 1) % 20 == 0:
            print(f"    {at + 1}/{len(pick)} windows", flush=True)

    lags, peaks = np.asarray(lags), np.asarray(peaks)
    med = float(np.median(lags))
    # Bootstrap the median rather than quoting sd/sqrt(n): the per-slice distribution is
    # heavy-tailed (a window that matched the wrong building is off by a second), which is
    # exactly the case where a normal-theory standard error understates the uncertainty.
    rng = np.random.default_rng(0)
    boot = np.median(rng.choice(lags, size=(2000, len(lags)), replace=True), axis=1)
    summary = {
        "n_sampled": int(len(lags)),
        "median_lag_s": med,
        "bootstrap_se_s": float(boot.std()),
        "bootstrap_ci95_s": [float(np.percentile(boot, 2.5)),
                             float(np.percentile(boot, 97.5))],
        "sd_s": float(lags.std()),
        "iqr_s": [float(np.percentile(lags, 25)), float(np.percentile(lags, 75))],
        "ncc_median": float(np.median(peaks)),
        "ncc_max": float(peaks.max()),
        "crop_histogram": np.bincount(crops, minlength=len(VERIFY_CROPS)).tolist(),
        "median_speed_ms": float(np.median(speed[covered & (speed > 5.0)])),
    }
    summary["median_lag_m"] = abs(med) * summary["median_speed_ms"]
    return summary, lags, peaks


def write_frames(video_path, out_dir, width, height, idx, img_format,
                 workers=6, resume=True):
    """Stream the video once, writing every selected frame under each slice index using it."""
    import cv2

    ext = ".png" if img_format == "png" else ".jpg"
    params = ([cv2.IMWRITE_PNG_COMPRESSION, 1] if img_format == "png"
              else [cv2.IMWRITE_JPEG_QUALITY, 98, cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                    cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    # A frame chosen for two adjacent centres is written twice, under both indices, so the
    # tree stays dense and index i means the same instant in every arm.
    slices_for = {}
    for i, n in enumerate(idx):
        slices_for.setdefault(int(n), []).append(i)
    wanted = np.array(sorted(slices_for), dtype=np.int64)

    done = [0]
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = []

    def on_frame(n, frame):
        for i in slices_for[n]:
            dst = os.path.join(out_dir, f"frame_{i:06d}{ext}")
            if resume and os.path.exists(dst):
                continue
            futures.append(pool.submit(cv2.imwrite, dst, frame, params))
        done[0] += 1
        if done[0] % 2000 == 0:
            print(f"    {done[0]}/{len(wanted)} source frames", flush=True)
        # Keep the queue bounded: a 1080p BGR frame is 6.2 MB and the encoder is slower than
        # the decoder, so an unbounded queue would hold the whole video in RAM.
        while len(futures) > 4 * workers:
            futures.pop(0).result()

    decoded = stream_frames(video_path, width, height, wanted, on_frame)
    for fut in futures:
        fut.result()
    pool.shutdown()
    if decoded <= int(wanted[-1]):
        raise RuntimeError(
            f"{video_path}: decoder stopped at frame {decoded} but frame {int(wanted[-1])} "
            f"was needed — the file is short or truncated")
    return len(wanted), ext


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help=f"traverse name, one of {sorted(VIDEOS)}")
    ap.add_argument("--video", default=None,
                    help="default: <video-root>/<upstream concat name> (see VIDEOS)")
    ap.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--npz-root", default="/media/adam/vprdatasets/megaevent/brisbane_npz",
                    help="frames land in <npz-root>/<dataset>/gopro/<seq>")
    ap.add_argument("--dataset", default="brisbane_event")
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--slice-times", default=None,
                    help="slice_times.npy written by scripts/dump_event_npz.py; default "
                         "<npz-root>/<dataset>/real/<seq>/slice_times.npy. This defines the "
                         "grid — never recompute it as offset + i*dt.")
    ap.add_argument("--fps", type=float, default=None,
                    help="override the probed frame rate; the probe asserts CFR and is "
                         "what every published number should use")
    ap.add_argument("--img-format", default="png", choices=["png", "jpg"],
                    help="png (default) is lossless over what the decoder produced; jpg is "
                         "q98 4:4:4 and cuts the intermediate tree ~4x")
    ap.add_argument("--workers", type=int, default=6, help="image-encoder threads")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--eventlab-dir", default="/media/adam/vprdatasets/eventgem",
                    help="holds <dataset>/<seq>/<seq>_ground_truth.nmea, for --verify's "
                         "speed gate")
    ap.add_argument("--verify-samples", type=int, default=80)
    ap.add_argument("--verify", action="store_true",
                    help="write no frames: measure the video-vs-APS lag per slice "
                         "against the same mapping extraction would use, and report it "
                         "with a bootstrap CI into <out_dir>/verify.json. Provenance "
                         "for the time base, not a correction.")
    ap.add_argument("--timewarp", default=None,
                    help="warp JSON from scripts/gopro_timewarp.py — the measured "
                         "video->event-clock mapping. Required for any traverse whose "
                         "lag profile is not flat (all four measured so far), and the "
                         "only supported time base for sunset1/morning.")
    args = ap.parse_args()

    video_path = resolve_video(args.seq, args.video, args.video_root)
    slice_times = args.slice_times or os.path.join(
        args.npz_root, args.dataset, "real", args.seq, "slice_times.npy")
    if not os.path.exists(slice_times):
        raise FileNotFoundError(
            f"no slice grid at {slice_times} — run scripts/dump_event_npz.py --seq "
            f"{args.seq} first; it writes the grid this alignment is built on")

    centres = slice_centres(slice_times)
    n_slices = len(centres)
    warp = load_timewarp(args.timewarp, args.seq) if args.timewarp else None
    if warp is None:
        if args.seq not in VIDEO_BEGINNING:
            raise SystemExit(
                f"no time base for {args.seq}: pass --timewarp with the warp from "
                f"scripts/gopro_timewarp.py (measured via scripts/gopro_align.py). "
                f"VIDEO_BEGINNING is only for a traverse whose measured lag profile "
                f"is flat, and none is.")
        v0 = VIDEO_BEGINNING[args.seq]
    else:
        v0 = warp["v0_base_s"]
    width, height, fps, n_frames, duration = probe(video_path)
    fps = args.fps or fps

    print(f"{args.seq}: {n_slices} slices @ {args.dt_ms} ms, grid from {slice_times}")
    print(f"  video {os.path.basename(video_path)}  {width}x{height} @ {fps:g} fps  "
          f"{n_frames} frames  {duration:.2f} s")
    if warp is None:
        print(f"  video begins {v0:.6f} (event clock, VIDEO_BEGINNING)")
    else:
        print(f"  time warp {args.timewarp}  (base {v0:.6f}, "
              f"{len(warp['knots_t_s'])} knots, delta "
              f"[{min(warp['knots_delta_s']):+.2f}, {max(warp['knots_delta_s']):+.2f}] s, "
              f"holdout p90 {warp['holdout']['p90_abs_s']:.3f} s, "
              f"fingerprint {warp['fingerprint']})")
    print(f"  centres [{centres[0]:.6f}, {centres[-1]:.6f}]  "
          f"lead {centres[0] - v0:+.3f} s  tail {centres[-1] - (v0 + duration):+.3f} s")

    out_dir = os.path.join(args.npz_root, args.dataset, "gopro", args.seq)

    if args.verify:
        from src import traversegps as tg

        aps_dir = os.path.join(args.npz_root, args.dataset, "aps", args.seq)
        lat0, lon0 = tg.track_origin(args.eventlab_dir, args.dataset, [args.seq])
        _, covered, speed, _ = tg.frame_coords(args.eventlab_dir, args.dataset, args.seq,
                                               lat0, lon0, args.dt_ms,
                                               npz_root=args.npz_root)
        print(f"  verifying against {aps_dir}")
        pred = (warp_container_time(centres, warp) if warp is not None
                else centres - v0)
        summary, _, _ = verify_lag(video_path, aps_dir, pred, fps, duration,
                                   speed, covered, n_sample=args.verify_samples)
        if warp is not None:
            summary["timewarp"] = {"source": args.timewarp,
                                   "fingerprint": warp["fingerprint"]}
        lo, hi = summary["bootstrap_ci95_s"]
        print(f"  lag {summary['median_lag_s']:+.3f} s  (95% CI [{lo:+.3f}, {hi:+.3f}], "
              f"per-slice sd {summary['sd_s']:.3f}) over {summary['n_sampled']} slices")
        print(f"  = {summary['median_lag_m']:.1f} m at the median speed of "
              f"{summary['median_speed_ms']:.1f} m/s")
        print(f"  NCC median {summary['ncc_median']:.3f} max {summary['ncc_max']:.3f}; "
              f"crop histogram {summary['crop_histogram']}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "verify.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return 0

    os.makedirs(out_dir, exist_ok=True)
    print(f"  -> {out_dir}")

    idx, times, dts = (frame_index_warp(centres, warp, fps, n_frames)
                       if warp is not None
                       else frame_index(centres, v0, fps, n_frames))
    uniq, ext = write_frames(video_path, out_dir, width, height, idx,
                             args.img_format, args.workers, not args.no_resume)

    tol = args.dt_ms / 2000.0                       # dt/2, in seconds — extract_aps's rule
    bad = np.flatnonzero(np.abs(dts) > tol)
    report = {
        "sequence": args.seq,
        "video": video_path,
        "slice_times": slice_times,
        "dt_ms": args.dt_ms,
        "fps": float(fps),
        "video_beginning_s": float(v0),
        "timewarp": (None if warp is None else
                     {"source": args.timewarp, "fingerprint": warp["fingerprint"],
                      "holdout_p90_s": warp["holdout"]["p90_abs_s"],
                      "n_knots": len(warp["knots_t_s"])}),
        "video_frames": int(n_frames),
        "video_duration_s": float(duration),
        "frame_wh": [int(width), int(height)],
        "img_format": args.img_format,
        "first_centre": float(centres[0]),
        "last_centre": float(centres[-1]),
        "n_slices": int(n_slices),
        "frames_written": int(n_slices),
        "unique_source_frames": int(uniq),
        "abs_dt_ms": {"mean": float(np.abs(dts).mean() * 1e3),
                      "p99": float(np.percentile(np.abs(dts), 99) * 1e3),
                      "max": float(np.abs(dts).max() * 1e3)},
        "tolerance_ms": args.dt_ms / 2.0,
        "out_of_tolerance": bad.tolist(),
        "aps_quality": aps_quality(out_dir, n_slices, ext=ext),
    }
    with open(os.path.join(out_dir, "select.json"), "w") as f:
        json.dump(report, f, indent=2)
    np.save(os.path.join(out_dir, "frame_times.npy"), times)

    print(f"  wrote {n_slices} frames from {uniq} unique video frames")
    print(f"  |dt|: mean {report['abs_dt_ms']['mean']:.2f} ms  "
          f"p99 {report['abs_dt_ms']['p99']:.2f} ms  max {report['abs_dt_ms']['max']:.2f} ms")
    print(f"  outside +-{args.dt_ms / 2:.0f} ms: {len(bad)} slice(s)"
          + (f" (indices {bad[0]}..{bad[-1]})" if len(bad) else ""))
    q = report["aps_quality"]
    if q:
        print(f"  exposure: mean {q['intensity_mean']:.1f}  black {q['black_fraction_mean']:.3f}"
              f"  saturated {q['saturated_fraction_mean']:.3f}  "
              f"laplacian-var {q['laplacian_var_mean']:.0f}")


if __name__ == "__main__":
    sys.exit(main())
