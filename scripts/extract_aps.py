"""Pull the DAVIS APS frames out of a Brisbane-Event-VPR rosbag, one per event slice.

Run with **Event-LAB's** interpreter — it is the only environment here carrying ``rosbag``
and ``cv_bridge``::

    /home/adam/repo/Event-LAB/.pixi/envs/default/bin/python scripts/extract_aps.py \
        --seq sunset1 \
        --slice-times <npz-root>/brisbane_event/real/sunset1/slice_times.npy

``scripts/dump_event_npz.py`` must have run first: it writes ``slice_times.npy`` from the
eventcv reader, and that file — not any formula — defines the grid. eventcv clamps its
``offset`` up to the recording's first timestamp, and sunset1's configured offset is 1.22 s
*before* its first event, so ``offset + i * dt`` would place every frame 24.5 slices off.

The DAVIS346 recorded intensity frames and events through the same sensor on the same
clock, so the ``/dvs/image_raw`` stream is a frame-camera view of exactly what the event
stream saw. Converting those frames with I2E gives synthetic events of the same scenes,
which is what makes the real-vs-synthetic ablation possible at all.

The frames arrive faster than the event slices (sunset1 ~40 Hz, sunset2 ~31 Hz, against a
20 Hz slicing grid), so this **selects** rather than exports: for each event slice ``i``,
centred at ``offset + (i + 0.5) * dt``, it writes the APS frame whose timestamp is nearest.
That is what makes index ``i`` mean the same instant in both arms of the ablation, and it
is why both arms can then share one ground-truth matrix and one resample.

Selection is a single streaming pass. Messages arrive in time order, so it is enough to
hold the previous frame and, as each centre is passed, keep whichever of previous/current
is nearer — a two-pass design (collect timestamps, then re-read) would mean two full scans
of a 15 GB bag.

``msg.header.stamp`` is the sensor clock and is what the event timestamps share; the bag's
own record time runs ~10 ms behind it and is not used.

Alongside the frames:

* ``frame_times.npy`` -- ``[n_slices]`` float64 unix seconds, the stamp actually selected;
* ``select.json`` -- provenance and, importantly, the indices whose nearest frame was
  further than ``dt/2`` away. Both traverses have a handful at the ends (sunset1's event
  stream starts 1.24 s before its first APS frame; sunset2's last slice runs 68 ms past its
  last one). :mod:`src.traversevpr` drops those from *both* arms so the comparison never
  scores a duplicated frame against a real one.
"""

import argparse
import json
import os
import sys

import numpy as np

SENSOR_H, SENSOR_W = 260, 346       # DAVIS346, the only geometry we accept
TOPIC = "/dvs/image_raw"

# Upstream ships the bags under their recording timestamps, and nothing in the traverse names
# recovers which is which. Transcribed from ensemble-event-vpr's
# ``correspondence_event_camera_frame_camera.py`` (``traverse_to_name``, minus its
# ``run_tobi3_`` prefix), which is the authority for this dataset.
BAG_NAMES = {
    "sunset1": "dvs_vpr_2020-04-21-17-03-03",
    "sunset2": "dvs_vpr_2020-04-22-17-24-21",
    "daytime": "dvs_vpr_2020-04-24-15-12-03",
    "night":   "dvs_vpr_2020-04-27-18-13-29",
    "morning": "dvs_vpr_2020-04-28-09-14-11",
    "sunrise": "dvs_vpr_2020-04-29-06-20-23",
}


def resolve_bag(seq, bag, bag_root):
    """The bag for ``seq``. Explicit path wins, then ``<root>/<seq>/<seq>.bag``, then the
    upstream recording name flat in ``<root>``.

    Two layouts exist locally and both are legitimate: sunset1/sunset2 were renamed into
    per-traverse directories long ago, while a fresh ``hf download`` lands every bag flat
    under its upstream name. Trying the renamed form first keeps the older tree working.
    """
    if bag:
        if not os.path.exists(bag):
            raise FileNotFoundError(f"no bag at {bag}")
        return bag
    renamed = os.path.join(bag_root, seq, f"{seq}.bag")
    if os.path.exists(renamed):
        return renamed
    if seq in BAG_NAMES:
        upstream = os.path.join(bag_root, f"{BAG_NAMES[seq]}.bag")
        if os.path.exists(upstream):
            return upstream
    raise FileNotFoundError(
        f"no bag for {seq} under {bag_root}: tried {renamed}"
        + (f" and {os.path.join(bag_root, BAG_NAMES[seq] + '.bag')}" if seq in BAG_NAMES
           else f" (and {seq} is not a known traverse, so there is no upstream name to try)"))


def slice_centres(slice_times_path):
    """Mid-bin time of every event slice, in unix seconds.

    ``slice_times.npy`` is ``[n_slices, 2]`` float64 microseconds ``(start, end)``, written
    by ``scripts/dump_event_npz.py`` straight from the eventcv reader. Reading it rather
    than recomputing is the whole reason the two arms cannot drift apart.
    """
    st = np.load(slice_times_path)
    if st.ndim != 2 or st.shape[1] != 2:
        raise ValueError(f"{slice_times_path}: expected [n_slices, 2], got {st.shape}")
    return st.mean(axis=1) / 1e6


def select_and_write(bag_path, out_dir, centres, dt_ms):
    """Stream the bag once, writing the nearest APS frame to each centre.

    Returns ``(times, dts, written)``: the selected stamp per slice, its signed offset from
    the slice centre, and how many frames were actually written (a frame chosen for two
    adjacent centres is written twice, under both indices, so the tree stays dense).
    """
    import cv2
    import rosbag
    from cv_bridge import CvBridge
    from tqdm import tqdm

    bridge = CvBridge()
    n = len(centres)
    times = np.full(n, np.nan, dtype=np.float64)
    written = 0

    # `at` is the next slice still waiting for a frame; `prev` the last frame seen. Both
    # advance monotonically, which is what keeps this O(messages + slices).
    at = 0
    prev_t, prev_img = None, None

    def emit(idx, stamp, img):
        nonlocal written
        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:06d}.png"), img)
        times[idx] = stamp
        written += 1

    with rosbag.Bag(bag_path, "r") as bag:
        total = bag.get_type_and_topic_info().topics[TOPIC].message_count
        with tqdm(total=total, desc=os.path.basename(bag_path), unit="msg",
                  disable=None) as bar:
            for _, msg, _ in bag.read_messages(topics=[TOPIC]):
                bar.update(1)
                # A handful of frames carry an uninitialised header; upstream's own
                # exporter drops them the same way.
                if msg.header.stamp.secs < 100:
                    continue
                if msg.height != SENSOR_H or msg.width != SENSOR_W:
                    continue
                stamp = msg.header.stamp.to_sec()
                img = bridge.imgmsg_to_cv2(msg, "bgr8")     # rgb8 on the wire -> BGR for cv2

                # Every centre now behind `stamp` can be decided: its nearest frame is
                # either `prev` or this one, and nothing later can be closer.
                while at < n and centres[at] <= stamp:
                    if prev_t is None or abs(stamp - centres[at]) < abs(prev_t - centres[at]):
                        emit(at, stamp, img)
                    else:
                        emit(at, prev_t, prev_img)
                    at += 1
                if at >= n:
                    break
                prev_t, prev_img = stamp, img

    # Centres past the last frame in the bag: the last frame is the nearest there is.
    while at < n and prev_t is not None:
        emit(at, prev_t, prev_img)
        at += 1
    if at < n:
        raise RuntimeError(f"{bag_path}: no usable {TOPIC} frames at all")

    return times, times - centres, written


def aps_quality(out_dir, n_slices, n_sample=200, ext=".png"):
    """Exposure and sharpness statistics over a sample of the written frames.

    Shared with ``scripts/extract_gopro.py`` -- hence ``ext`` -- so the video arm's exposure
    lands in the same fields as the APS arm's and the two are directly comparable.

    Both available bags are dusk traverses, the worst case for a DAVIS346's APS. If the
    baselines drop on the synthetic arm, "the frames were under-exposed at sunset" is a
    competing explanation to "I2E is a domain shift", and it is the first thing a reviewer
    will raise. Recording the numbers here means the question is answerable from the run
    itself rather than needing a second pass over the data.
    """
    import cv2

    idx = np.linspace(0, n_slices - 1, min(n_sample, n_slices)).astype(int)
    means, dark, bright, sharp = [], [], [], []
    for i in idx:
        img = cv2.imread(os.path.join(out_dir, f"frame_{i:06d}{ext}"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        means.append(img.mean())
        dark.append(float((img <= 5).mean()))
        bright.append(float((img >= 250).mean()))
        sharp.append(float(cv2.Laplacian(img, cv2.CV_64F).var()))
    if not means:
        return {}
    return {"n_sampled": len(means),
            "intensity_mean": float(np.mean(means)),
            "intensity_p5": float(np.percentile(means, 5)),
            "intensity_p95": float(np.percentile(means, 95)),
            "black_fraction_mean": float(np.mean(dark)),
            "saturated_fraction_mean": float(np.mean(bright)),
            "laplacian_var_mean": float(np.mean(sharp)),
            "laplacian_var_p5": float(np.percentile(sharp, 5))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="traverse name, e.g. sunset1")
    ap.add_argument("--bag", default=None,
                    help="default: <bag-root>/<seq>/<seq>.bag, else <bag-root>/"
                         "<upstream recording name>.bag (see BAG_NAMES)")
    ap.add_argument("--bag-root", default="/media/adam/vprdatasets/eventlab/brisbane_event")
    ap.add_argument("--npz-root", default="/media/adam/vprdatasets/megaevent/brisbane_npz",
                    help="frames land in <npz-root>/brisbane_event/aps/<seq>")
    ap.add_argument("--dataset", default="brisbane_event")
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--slice-times", default=None,
                    help="slice_times.npy written by scripts/dump_event_npz.py; default "
                         "<npz-root>/<dataset>/real/<seq>/slice_times.npy. This defines "
                         "the grid — never recompute it as offset + i*dt (eventcv clamps "
                         "the offset up to the recording's first event).")
    args = ap.parse_args()

    bag_path = resolve_bag(args.seq, args.bag, args.bag_root)

    slice_times = args.slice_times or os.path.join(
        args.npz_root, args.dataset, "real", args.seq, "slice_times.npy")
    if not os.path.exists(slice_times):
        raise FileNotFoundError(
            f"no slice grid at {slice_times} — run scripts/dump_event_npz.py --seq "
            f"{args.seq} first; it writes the grid this alignment is built on")

    out_dir = os.path.join(args.npz_root, args.dataset, "aps", args.seq)
    os.makedirs(out_dir, exist_ok=True)

    centres = slice_centres(slice_times)
    n_slices = len(centres)
    print(f"{args.seq}: {n_slices} slices @ {args.dt_ms} ms, grid from {slice_times}")
    print(f"  centres [{centres[0]:.6f}, {centres[-1]:.6f}]  -> {out_dir}")

    times, dts, written = select_and_write(bag_path, out_dir, centres, args.dt_ms)

    tol = args.dt_ms / 2000.0                       # dt/2, in seconds
    bad = np.flatnonzero(np.abs(dts) > tol)
    uniq = len(np.unique(times))
    report = {
        "sequence": args.seq,
        "bag": bag_path,
        "slice_times": slice_times,
        "dt_ms": args.dt_ms,
        "first_centre": float(centres[0]),
        "last_centre": float(centres[-1]),
        "n_slices": int(n_slices),
        "frames_written": int(written),
        "unique_source_frames": int(uniq),
        "abs_dt_ms": {"mean": float(np.abs(dts).mean() * 1e3),
                      "p99": float(np.percentile(np.abs(dts), 99) * 1e3),
                      "max": float(np.abs(dts).max() * 1e3)},
        "tolerance_ms": args.dt_ms / 2.0,
        "out_of_tolerance": bad.tolist(),
        "aps_quality": aps_quality(out_dir, n_slices),
    }
    with open(os.path.join(out_dir, "select.json"), "w") as f:
        json.dump(report, f, indent=2)
    np.save(os.path.join(out_dir, "frame_times.npy"), times)

    print(f"  wrote {written} frames from {uniq} unique APS frames")
    print(f"  |dt|: mean {report['abs_dt_ms']['mean']:.2f} ms  "
          f"p99 {report['abs_dt_ms']['p99']:.2f} ms  max {report['abs_dt_ms']['max']:.2f} ms")
    print(f"  outside +-{args.dt_ms / 2:.0f} ms: {len(bad)} slice(s)"
          + (f" (indices {bad[0]}..{bad[-1]})" if len(bad) else ""))


if __name__ == "__main__":
    sys.exit(main())
