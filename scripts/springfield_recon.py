"""Reconstruct intensity video from the Springfield event streams with V2V-E2VID.

    pixi run python3 scripts/springfield_recon.py --sessions query_night/20260829T090907Z
    pixi run python3 scripts/springfield_recon.py            # everything, resume-safe

The event-to-video model is V2V-E2VID — the recommended reconstruction model of "V2V:
Scaling Event-Based Vision through Efficient Video-to-Voxel Simulation" (NeurIPS 2025,
https://github.com/HYLZ-2019/V2V), checkpoint ``v2v_e2vid_10k/epoch_0077.pth``
(E2VIDRecurrent: 5-bin voxels, convlstm, 3 encoders). Loaded from the local clone at
``/home/adam/repo/V2V``; inference conventions copied from its ``test_e2vid.py`` +
``model/train_utils.py:forward_sequence``: raw count voxels (``normalize_voxels: false``),
input padded to a multiple of 16 (1280x720 already is), ``model(voxel)['image']`` in
[0, 1], clamped and written as 8-bit grayscale.

Framing matches the benchmark exactly: one frame per 50 ms eventcv slice per partition
(``ecv.open(path, dt_ms=50, sensor_size, hot_pixel_filter=True, offset=0)`` — the same
call ``src.inference.EventStreamDataset._open`` makes, hot-pixel on, BA off), so frame
``i`` of a partition is slice ``i`` of every other representation. Voxels replicate
V2V's ``TestH5Dataset.make_voxel`` bit-for-bit: polarities to ±1, 5 bins over the
window's own [first event, last event] span, plain counts (our timestamps are already
µs, so only their seconds→µs scaling drops out). An empty slice is a zero voxel.

Recurrent state carries across partitions inside a session (partition rotation is
gapless in camera time) and resets between sessions. Resume is per partition: a
completed partition has ``recon.json`` next to its frames; resuming mid-session
restarts the ConvLSTM cold at that partition boundary, which the model re-warms in
~1 s of stream — recorded as ``"cold_start"`` in that partition's json.

Output tree:  <out-root>/<sid>/<partition_stem>/frame_%06d.png  (1280x720 gray)
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/adam/repo/V2V")

import eventcv as ecv  # noqa: E402

from springfield_eval import list_partitions, sensor_from_h5  # noqa: E402
from springfield_full import DEFAULT_ROOT  # noqa: E402

V2V_ROOT = "/home/adam/repo/V2V"
CKPT = os.path.join(V2V_ROOT, "checkpoints", "v2v_e2vid_10k", "epoch_0077.pth")
MODEL_ID = "v2v_e2vid_10k/epoch_0077"
NUM_BINS = 5
SWEEP_DIRS = ("database", "query_day", "query_dawn", "query_night")


def load_model(device):
    from model.model import E2VIDRecurrent  # noqa: E402  (V2V repo import)

    kwargs = dict(num_bins=NUM_BINS, skip_type="sum", recurrent_block_type="convlstm",
                  num_encoders=3, base_num_channels=32, num_residual_blocks=2,
                  use_upsample_conv=True, final_activation="", norm="none")
    model = E2VIDRecurrent(kwargs)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = {}
    for k, v in ck["state_dict"].items():  # train.convert_to_compiled, minus its imports
        parts = k.split(".")
        if parts[0] == "_orig_mod":
            parts.pop(0)
        if parts[0] == "module":
            parts.pop(0)
        state[".".join(parts)] = v
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing {missing[:3]} "
                         f"unexpected {unexpected[:3]}")
    return model.to(device).eval()


def make_voxel(ev, height, width):
    """V2V TestH5Dataset.make_voxel, verbatim semantics, on eventcv's (N,4) slices."""
    if ev.shape[0] == 0:
        return np.zeros((NUM_BINS, height, width), np.float32)
    xs = ev[:, 0].astype(np.int64)
    ys = ev[:, 1].astype(np.int64)
    ts = ev[:, 2].astype(np.int64)
    ps = ev[:, 3].astype(np.int8) * 2 - 1     # {0,1} -> {-1,+1}
    ts = ts - ts[0]                           # already µs
    t_per_bin = (ts[-1] + 0.001) / NUM_BINS
    bin_idx = np.floor(ts / t_per_bin).astype(np.int64)
    flat = (bin_idx * height + ys) * width + xs
    vox = np.bincount(flat, weights=ps, minlength=NUM_BINS * height * width)
    return vox.reshape(NUM_BINS, height, width).astype(np.float32)


def producer(path, sensor, out_q, stop):
    """Reads slices and builds voxels on a thread; the GPU loop consumes."""
    width, height = sensor
    reader = ecv.open(path, dt_ms=50, sensor_size=sensor, hot_pixel_filter=True,
                      offset=0)
    n = reader.n_slices
    try:
        for i in range(n):
            if stop.is_set():
                return
            ev = ecv.numpy(reader.slice(i))
            out_q.put((i, make_voxel(ev, height, width)))
    finally:
        out_q.put(None)


def partition_done(pdir):
    meta_path = os.path.join(pdir, "recon.json")
    if not os.path.exists(meta_path):
        return False
    with open(meta_path) as handle:
        meta = json.load(handle)
    n_png = len([f for f in os.listdir(pdir) if f.endswith(".png")])
    return n_png == meta.get("n_slices") and meta.get("model") == MODEL_ID


def run_partition(model, device, path, pdir, cold_start, use_fp16, writers):
    sensor = sensor_from_h5(path)
    width, height = sensor
    probe = ecv.open(path, dt_ms=50, sensor_size=sensor, hot_pixel_filter=True, offset=0)
    n = probe.n_slices
    del probe

    os.makedirs(pdir, exist_ok=True)
    out_q = queue.Queue(maxsize=6)
    stop = threading.Event()
    thread = threading.Thread(target=producer, args=(path, sensor, out_q, stop),
                              daemon=True)
    thread.start()

    written, futures, t0 = 0, [], time.time()
    try:
        with torch.no_grad():
            while True:
                item = out_q.get()
                if item is None:
                    break
                i, vox = item
                x = torch.from_numpy(vox).unsqueeze(0).to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=use_fp16):
                    img = model(x)["image"]
                img = (img.float().clamp(0, 1)[0, 0] * 255).to(torch.uint8)
                arr = img.cpu().numpy()
                futures.append(writers.submit(
                    cv2.imwrite, os.path.join(pdir, f"frame_{i:06d}.png"), arr))
                written += 1
                while len(futures) > 16:
                    futures.pop(0).result()
                if written % 2000 == 0:
                    rate = written / (time.time() - t0)
                    print(f"      {written}/{n}  {rate:.1f} fr/s  "
                          f"eta {(n - written) / rate / 60:.1f} min", flush=True)
        for fut in futures:
            fut.result()
    finally:
        stop.set()
        thread.join(timeout=10)

    if written != n:
        raise SystemExit(f"{path}: wrote {written} frames for {n} slices")
    meta = {"partition": os.path.basename(path), "n_slices": int(n),
            "model": MODEL_ID, "ckpt": CKPT, "num_bins": NUM_BINS, "dt_ms": 50,
            "hot_pixel": True, "filter_dt_us": None, "fp16": bool(use_fp16),
            "cold_start": bool(cold_start), "sensor_wh": [width, height],
            "seconds": round(time.time() - t0, 1)}
    tmp = os.path.join(pdir, "recon.json.tmp")
    with open(tmp, "w") as handle:
        json.dump(meta, handle, indent=1)
    os.replace(tmp, os.path.join(pdir, "recon.json"))
    return n, time.time() - t0


def discover_sessions(root, only):
    out = []
    for sweep in SWEEP_DIRS:
        base = os.path.join(root, sweep)
        if not os.path.isdir(base):
            continue
        for sid in sorted(os.listdir(base)):
            sdir = os.path.join(base, sid)
            if os.path.isdir(sdir):
                rel = f"{sweep}/{sid}"
                if not only or rel in only or sid in only:
                    out.append((rel, sdir))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out-root",
                    default="/media/adam/vprdatasets/megaevent/springfield/recon_v2v_e2vid")
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="restrict to these (sweep/sid or bare sid); default all")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--write-workers", type=int, default=3)
    args = ap.parse_args()

    sessions = discover_sessions(args.root, args.sessions)
    if not sessions:
        raise SystemExit("no sessions matched")
    device = torch.device("cuda")
    model = load_model(device)
    print(f"{MODEL_ID} on {len(sessions)} session(s), fp16={args.fp16} -> {args.out_root}")

    writers = ThreadPoolExecutor(max_workers=args.write_workers)
    total, t_all = 0, time.time()
    for rel, sdir in sessions:
        sid = os.path.basename(sdir)
        parts = list_partitions(sdir)
        model.reset_states()
        cold = False        # a fresh session start is a defined reset, not a cold resume
        print(f"  {rel}: {len(parts)} partition(s)")
        for path in parts:
            stem = os.path.splitext(os.path.basename(path))[0]
            pdir = os.path.join(args.out_root, sid, stem)
            if partition_done(pdir):
                print(f"    {stem}: done, skipping (state restarts cold next partition)")
                cold = True
                model.reset_states()
                continue
            n, dt = run_partition(model, device, path, pdir, cold, args.fp16, writers)
            cold = False
            total += n
            print(f"    {stem}: {n} frames in {dt / 60:.1f} min "
                  f"({n / max(dt, 1e-9):.1f} fr/s)", flush=True)
    writers.shutdown()
    print(f"done: {total} new frames in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
