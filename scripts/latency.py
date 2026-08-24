"""End-to-end single-query latency, from event slice to retrieved match, stage by stage.

Run from the repository root::

    pixi run python3 -u scripts/latency.py
    pixi run python3 -u scripts/latency.py --models megaevent_vitb_salad eventgem --repeats 100

What is timed, per query, at batch 1, as the median over ``--repeats`` *distinct* slices
(a repeated index would time a cache rather than the pipeline):

``read``
    Pulling one 50 ms slice out of the recording. **Storage, not method** — and measured as
    such: it is timed once per representation and the same figure is added to every model
    that shares it, because a per-model read would charge each network a different draw from
    this drive's latency distribution rather than a different amount of work. Reported
    page-cache-warm, with the cold random-seek figure printed beside it. A camera-fed
    deployment pays neither: events arrive over USB as they are generated.
``render``
    Events to the network's input frame — background-activity filter at 50,000 us and then
    whichever representation the method was trained on. Also per-representation, so also
    timed once. Timed over slices already resident in RAM, which is what separates the
    representation build from the read above, and the frames it produces are checked
    byte-identical against the chained reader the evaluation scripts actually use, so this
    cannot be timing a cheaper computation than the one that ships.
``tensor``
    Scale, normalise and resize to the network's input — the rest of what
    ``src/inference.py``'s ``EventStreamDataset.__getitem__`` does before the forward pass.
``encode``
    Host-to-device copy plus the forward pass under autocast, synchronised.
``match``
    Whitening if the method uses it, then cosine against a resident gallery and an argmax,
    synchronised. Quoted against pitts250k's 83,952-frame gallery: it scales with both
    gallery size and descriptor width, so the number means nothing without that pair.
``rerank``
    Second-stage geometric verification. Zero for every method except Event-GeM.

**Four representations, not one.** countmask ``[3,346,260]`` for megaevent and the two RGB
controls, SuperEvent MCTS ``[10,240,320]`` for Event-GeM, normalised count triplets
``[3,256,256]`` for EventVLAD, ON/OFF counts ``[2,260,346]`` for SpikeVPR. ``read`` and
``render`` are therefore not shared across method families, and the differences between them
are real differences in what each method has to build before it can look at anything.

**Event-GeM is quoted as deployed** — whitened at its own power 1.0 and re-ranked over the
top-50 by homography, because both are the method rather than options on it. Its global stage
alone is printed underneath for anyone comparing single-stage numbers. The 50 verifications
are single-threaded cv2, matching how ``src/eventgemlocal.py`` runs them per query; the
production path parallelises *across* queries, which does nothing for one query's latency.

**SpikeVPR runs in ``envs/spikevpr``** (megaevent's environment has no spikingjelly), so it is
timed by ``scripts/latency_spikevpr.py`` in that environment and merged in here. Skipped with
``--no-spikevpr``.

**A busy GPU invalidates the GPU columns.** ``encode`` and ``match`` are the only two that
share the device, so a fixed reference matmul is timed up front and the script refuses to
report at all unless the device is idle — ``--allow-contended`` overrides, and then those
columns are upper bounds rather than measurements.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import eventcv as ecv  # noqa: E402
from src import inference as inf  # noqa: E402
from src.methods import (  # noqa: E402
    EVENTVLAD_DENOISE_SIZE, MEGALOC_DESC_DIM, MEGALOC_HUB, SALAD_DESC_DIM, SALAD_HUB,
    build_dino_salad, megaloc_transform,
)

# Brisbane sunset1: the DAVIS346 recording every pooled result in this repo is queried from.
RECORDING = "/media/adam/vprdatasets/eventgem/brisbane_event/sunset1/sunset1.hdf5"
SENSOR = (346, 260)                 # (W, H), eventcv's axis order
DT_MS = 50
FILTER_DT_US = 50000
RESOLUTION = 322
GALLERY_N = 83952                   # pitts250k database
CKPTS = os.path.join(REPO, "ckpts")
EVENTLAB_REPO = "/home/adam/repo/Event-LAB"   # main.py --eventlab-repo default

MCTS_SIZE = (240, 320)              # eventgem_pooled.py --eventgem-size
MCTS_MAX_WINDOW_MS = 30.0
EVENTGEM_P = 5.0
EVENTGEM_TOP_K = 50
EVENTGEM_RANSAC = 5.0
EVENTGEM_MATCH_FILTER = "mutual"
EVENTGEM_MATCH_RATIO = 0.8
EVENTGEM_PCA_POWER = 1.0            # Event-GeM's own whitening power, not the shared 0.5

ALL = ["megaevent_vitb_mloc", "megaevent_vitb_salad", "megaevent_vits_mloc",
       "megaevent_vits_salad", "salad", "megaloc", "eventgem", "eventvlad"]


# ---------------------------------------------------------------------------
# 1. Representations: (fused reader, compute-only render) per input format
# ---------------------------------------------------------------------------
def _open(recording, sensor):
    return ecv.open(recording, dt_ms=DT_MS, sensor_size=tuple(sensor),
                    hot_pixel_filter=True, offset=0)


def rep_countmask(recording, sensor):
    fused = _open(recording, sensor).background_activity_filter(
        FILTER_DT_US).with_repr("countmask", window_ms=DT_MS, white_frame=False)

    def render(stream):
        # No window_ms here: the reader already sliced at dt_ms, and the free function takes
        # the window from the stream it is handed. The byte-identity check is what proves it.
        return np.asarray(ecv.countmask(
            ecv.background_activity_filter(stream, FILTER_DT_US),
            white_frame=False).numpy())
    return fused, render


def rep_mcts(recording, sensor):
    fused = _open(recording, sensor).background_activity_filter(FILTER_DT_US).resize(
        width=MCTS_SIZE[1], height=MCTS_SIZE[0]).with_repr(
            "mcts", max_window_ms=MCTS_MAX_WINDOW_MS)

    def render(stream):
        s = ecv.background_activity_filter(stream, FILTER_DT_US)
        s = ecv.resize(s, width=MCTS_SIZE[1], height=MCTS_SIZE[0])
        return np.asarray(ecv.mcts(s, max_window_ms=MCTS_MAX_WINDOW_MS).numpy())
    return fused, render


def rep_count(recording, sensor):
    fused = _open(recording, sensor).background_activity_filter(
        FILTER_DT_US).with_repr("count")

    def render(stream):
        return np.asarray(ecv.count(
            ecv.background_activity_filter(stream, FILTER_DT_US)).numpy())
    return fused, render


REPS = {"countmask": rep_countmask, "mcts": rep_mcts, "count": rep_count}


def measure_rep(name, recording, sensor, meas, cold, raw):
    """``(read_ms, render_ms, {i: frame})`` for one representation, model-independent."""
    fused, render = REPS[name](recording, sensor)
    for i in meas:                                  # first touch, into the page cache
        fused[int(i)]
    disk, frames = [], {}
    for i in meas:
        i = int(i)
        t = time.perf_counter()
        frame = np.asarray(fused[i])
        disk.append((time.perf_counter() - t) * 1e3)
        frames[i] = frame
    cold_ms = []
    for i in cold:
        t = time.perf_counter()
        fused[int(i)]
        cold_ms.append((time.perf_counter() - t) * 1e3)

    streams = {int(i): raw[int(i)] for i in meas}
    comp, bad = [], 0
    for i, stream in streams.items():
        t = time.perf_counter()
        out = render(stream)
        comp.append((time.perf_counter() - t) * 1e3)
        bad += int(not np.allclose(np.asarray(out), frames[i], atol=0, rtol=0))
    if bad:
        raise SystemExit(
            f"{name}: {bad}/{len(streams)} in-RAM renders differ from the reader's own "
            f"output — timing one and quoting the other would be wrong")
    return (statistics.median(disk), statistics.median(comp), frames,
            statistics.median(cold_ms))


# ---------------------------------------------------------------------------
# 2. Methods
# ---------------------------------------------------------------------------
def build(name, device):
    """``(rep, to_tensor, encode, desc_dim, params, label)``.

    ``to_tensor`` takes the rendered frame and returns the batch-1 CPU tensor; ``encode``
    takes that and returns an L2-normalised ``[1, D]`` descriptor on the device.
    """
    if name in ("salad", "megaloc"):
        if name == "megaloc":
            model = torch.hub.load(MEGALOC_HUB, "get_trained_model", source="github",
                                   trust_repo=True).eval().to(device)
            dim, label = MEGALOC_DESC_DIM, "MegaLoc"
        else:
            model = build_dino_salad().eval().to(device)
            dim, label = SALAD_DESC_DIM, "SALAD (RGB control)"
        tf = megaloc_transform(RESOLUTION)

        def to_tensor(frame):
            return tf(torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0))

        def encode(x):
            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                q = model(x.unsqueeze(0).to(device, non_blocking=True))
            return torch.nn.functional.normalize(q.float(), dim=1)
        return "countmask", to_tensor, encode, dim, model, label

    if name.startswith("megaevent"):
        model, cfg, step = inf.load_model(os.path.join(CKPTS, f"{name}.pt"), device)
        cfg.H = cfg.W = RESOLUTION
        tf = inf.eval_transform(cfg)

        def to_tensor(frame):
            return tf(torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0))

        def encode(x):
            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                q = model(x.unsqueeze(0).to(device, non_blocking=True))
            return torch.nn.functional.normalize(q.float(), dim=1)
        return ("countmask", to_tensor, encode, cfg.desc_dim, model,
                f"{name} (step {step})")

    if name == "eventvlad":
        import cv2
        from types import SimpleNamespace
        from src.methods import EventVLADMethod
        sys.path.insert(0, HERE)
        from brisbane_resolution import _Args
        args = _Args("/media/adam/vprdatasets/eventgem", "brisbane_event", DT_MS, False, False)
        args.eventlab_repo = EVENTLAB_REPO       # where the EventVLAD checkout and weights live
        method = EventVLADMethod(args, device)

        def plane(frame):
            """One count slice -> the normalised 256x256 plane the denoiser is fed."""
            f = np.asarray(frame)
            if f.ndim == 3:
                f = f[0] if f.shape[0] == 1 else (f.sum(0) if f.shape[0] == 2 else f.mean(0))
            f = f.astype(np.float32, copy=False)
            vmax = float(np.percentile(np.abs(f), 99.0))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = max(float(np.max(np.abs(f))), 1.0)
            f = np.clip(f / vmax, 0.0, 1.0)
            h, w = f.shape
            f = f[:h - h % 32, :w - w % 32]
            return cv2.resize(f, (EVENTVLAD_DENOISE_SIZE, EVENTVLAD_DENOISE_SIZE),
                              interpolation=cv2.INTER_AREA)

        prev = []

        def to_tensor(frame):
            # Steady state: the two previous planes are already built, so a new query costs
            # one plane plus the stack. A cold start would pay three; a live pipeline does
            # not, and charging it here would make EventVLAD look 3x worse than it runs.
            p = plane(frame)
            while len(prev) < 2:
                prev.append(p)
            out = torch.from_numpy(np.ascontiguousarray(np.stack([prev[-2], prev[-1], p])))
            prev.append(p)
            del prev[:-2]
            return out

        def encode(x):
            return torch.nn.functional.normalize(method.encode(x.unsqueeze(0)).float(), dim=1)
        modules = torch.nn.ModuleList([method.denoiser, method.encoder])
        return "count", to_tensor, encode, 1000, modules, "EventVLAD"

    if name == "eventgem":
        from types import SimpleNamespace
        from src import eventgemlocal as egl
        args = SimpleNamespace(eventgem_repo=os.path.join(REPO, "external", "eventgem"))
        egl.add_to_path(args)           # `eventgem.utils.rerank_utils` for the second stage
        root = egl.superevent_root(args)
        # SuperEvent's `models/` has no __init__.py, so it is a *namespace* package. SALAD's
        # torch.hub checkout ships a real `models/__init__.py`, and Python's path finder
        # treats a namespace portion as provisional: it records SuperEvent's directory, keeps
        # searching, finds SALAD's regular package further down sys.path and returns that
        # instead. Putting SuperEvent's root first therefore does not help, which is why
        # `build_superevent`'s own sys.path guard is not enough once a hub model has loaded —
        # and why this only breaks when --models puts salad or megaloc before eventgem.
        # Both the cached module and the hub directories have to go for the duration.
        hub = os.path.abspath(torch.hub.get_dir())
        saved_path = list(sys.path)
        sys.path[:] = [p for p in sys.path if not os.path.abspath(p).startswith(hub)]
        stash = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k == "models" or k.startswith("models.")}
        try:
            model, config, fast_nms = egl.build_superevent(root, device)
        finally:
            sys.path[:] = saved_path
            for k, v in stash.items():      # put the hub repo's package back untouched
                sys.modules.setdefault(k, v)
        off_top, off_left, hc, wc = egl.crop_offsets(*MCTS_SIZE, egl.input_multiple(config))

        def to_tensor(frame):
            return torch.from_numpy(np.ascontiguousarray(frame))

        def encode(x):
            with torch.no_grad():
                y = x.unsqueeze(0).to(device, non_blocking=True)
                if (hc, wc) != MCTS_SIZE:
                    y = y[:, :, off_top:off_top + hc, off_left:off_left + wc]
                f = model.fpn(model.backbone(y)).float()
                pooled = torch.nn.functional.avg_pool2d(
                    f.clamp(min=1e-6).pow(EVENTGEM_P),
                    (f.shape[-2], f.shape[-1])).pow(1.0 / EVENTGEM_P)
            return torch.nn.functional.normalize(
                pooled.squeeze(-1).squeeze(-1), p=2, dim=1)
        extra = {"model": model, "config": config, "fast_nms": fast_nms,
                 "crop": (off_top, off_left, hc, wc)}
        return "mcts", to_tensor, encode, 128, model, "Event-GeM (deployed)", extra

    raise SystemExit(f"unknown method {name!r}; choose from {ALL}")


# ---------------------------------------------------------------------------
# 3. Event-GeM's second stage
# ---------------------------------------------------------------------------
def eventgem_keypoints(extra, x, device):
    """Detector + descriptor heads off the trunk already run for GeM -> one frame's bank."""
    model, config, fast_nms = extra["model"], extra["config"], extra["fast_nms"]
    off_top, off_left, hc, wc = extra["crop"]
    from src.eventgemlocal import sample_descriptors_at_kpts
    top_k = max(hc, wc) // 2
    with torch.inference_mode():
        y = x.unsqueeze(0).to(device, non_blocking=True)
        if (hc, wc) != MCTS_SIZE:
            y = y[:, :, off_top:off_top + hc, off_left:off_left + wc]
        features = model.fpn(model.backbone(y))
        _, prob = model.detector(features)
        _, desc_map = model.descriptor(features)
        kpts_all, _ = fast_nms(prob, config, top_k=top_k)
        k = kpts_all[0]
        n = min(len(k), top_k)
        if n == 0:
            return None
        kpts = k[:n].float()
        desc = sample_descriptors_at_kpts(kpts.to(device), desc_map[:1], hc, wc)
        out = np.zeros((n, 2), dtype=np.float32)
        out[:, 0] = (kpts[:, 1] + off_left).cpu().numpy()
        out[:, 1] = (kpts[:, 0] + off_top).cpu().numpy()
    return {"desc": desc.float().cpu().numpy(), "kpts": out}


def eventgem_rerank(q_data, db_bank, shortlist, base):
    """One query's top-50, verified and re-ordered. Mirrors ``LocalReranker._rerank_column``."""
    import cv2
    from eventgem.utils.rerank_utils import compute_inliers_2d
    cv2.setNumThreads(1)
    if q_data is None:
        return shortlist
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    scores = base.copy()
    for i, db_idx in enumerate(shortlist):
        inliers = compute_inliers_2d(q_data, db_bank[int(db_idx) % len(db_bank)], matcher,
                                     EVENTGEM_RANSAC, match_filter=EVENTGEM_MATCH_FILTER,
                                     match_ratio=EVENTGEM_MATCH_RATIO)
        if inliers > 0:
            scores[i] = base[i] - inliers * 0.05
    return shortlist[np.argsort(scores, kind="stable")]


def probe(device, n=4096, iters=20):
    """Median cost of a fixed matmul — a contention read on a GPU this script does not own."""
    a = torch.randn(n, n, device=device, dtype=torch.float16)
    for _ in range(5):
        a @ a
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        a @ a
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    del a
    torch.cuda.empty_cache()
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=ALL)
    ap.add_argument("--recording", default=RECORDING)
    ap.add_argument("--sensor", type=int, nargs=2, default=list(SENSOR), metavar=("W", "H"))
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--gallery", type=int, default=GALLERY_N)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-spikevpr", dest="spikevpr", action="store_false",
                    help="skip the child-environment SpikeVPR measurement")
    ap.add_argument("--spikevpr-model", default="brisbane")
    ap.add_argument("--allow-contended", action="store_true")
    ap.add_argument("--out-json", default=None)
    cli = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("no CUDA device — these numbers would not mean anything")
    p = probe(device)
    others = [ln for ln in os.popen(
        "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"
    ).read().splitlines() if ln.strip()]
    others = [ln for ln in others if int(ln.split(",")[0]) != os.getpid()]
    print(f"device {torch.cuda.get_device_name(0)}  probe {p:.2f} ms")
    if others:
        print(f"  !! {len(others)} other process(es) hold this GPU: {'; '.join(others)}")
        if not cli.allow_contended:
            raise SystemExit("refusing to report contended GPU timings; pass "
                             "--allow-contended to override")

    raw = _open(cli.recording, cli.sensor)
    n_slices = int(raw.n_slices)
    rng = np.random.default_rng(cli.seed)
    idx = rng.choice(n_slices, 2 * cli.repeats + cli.warmup, replace=False)
    warm = idx[:cli.warmup]
    meas = idx[cli.warmup:cli.warmup + cli.repeats]
    cold = idx[cli.warmup + cli.repeats:]
    print(f"recording {os.path.basename(cli.recording)}  {n_slices} slices at {DT_MS} ms, "
          f"sensor {cli.sensor[0]}x{cli.sensor[1]}, BA filter {FILTER_DT_US} us")
    print(f"{cli.repeats} timed slices (+{cli.warmup} warm-up), gallery {cli.gallery} x D, "
          f"batch 1\n")

    needed = sorted({"mcts" if m == "eventgem" else
                     "count" if m == "eventvlad" else "countmask"
                     for m in cli.models})
    reps = {}
    print("stage 1-2  read + render, per representation (model-independent):")
    for r in needed:
        rd, rn, frames, cd = measure_rep(r, cli.recording, cli.sensor, meas, cold, raw)
        reps[r] = (rd, rn, frames)
        shape = tuple(np.asarray(next(iter(frames.values()))).shape)
        print(f"  {r:10s} {str(shape):18s} read {rd:6.2f}  render {rn:6.2f}  "
              f"(cold read {cd:6.2f})  [renders verified byte-identical]")
    print()

    rows = {}
    for name in cli.models:
        built = build(name, device)
        rep, to_tensor, encode, dim, module, label = built[:6]
        extra = built[6] if len(built) > 6 else None
        params = sum(q.numel() for q in module.parameters())
        read_ms, render_ms, frames = reps[rep]
        gallery = torch.nn.functional.normalize(
            torch.randn(cli.gallery, dim, device=device), dim=1)

        pca = kp_bank = None
        if name == "eventgem":
            # Whitening is fit on the database offline; only the per-query apply is latency.
            pca = inf.pca_fit(gallery.cpu(), device, dim=dim, power=EVENTGEM_PCA_POWER)
            gallery = torch.nn.functional.normalize(
                inf.pca_apply(gallery.cpu(), pca, device).to(device), dim=1)
            # A shortlist of real database keypoints to verify against, from real frames.
            kp_bank = [eventgem_keypoints(extra, to_tensor(f), device)
                       for f in list(frames.values())[:EVENTGEM_TOP_K]]
            kp_bank = [b for b in kp_bank if b is not None] or [None]

        st = {k: [] for k in ("tensor", "encode", "match", "rerank")}
        order = list(warm[:5]) + list(meas)
        for j, i in enumerate(order):
            frame = frames[int(i)] if int(i) in frames else frames[int(meas[0])]
            timed = j >= 5
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            x = to_tensor(frame)
            t2 = time.perf_counter()
            q = encode(x)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            if pca is not None:
                q = torch.nn.functional.normalize(
                    inf.pca_apply(q.cpu(), pca, device).to(device), dim=1)
            sims = (gallery @ q.T).squeeze(1)
            if name == "eventgem":
                vals, short = torch.topk(sims, EVENTGEM_TOP_K)
                short = short.cpu().numpy()
                base = (1.0 - vals).cpu().numpy()
            else:
                _ = sims.argmax(0)
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            if name == "eventgem":
                q_data = eventgem_keypoints(extra, x, device)
                torch.cuda.synchronize()
                eventgem_rerank(q_data, kp_bank, short, base)
            t5 = time.perf_counter()
            if timed:
                st["tensor"].append((t2 - t1) * 1e3)
                st["encode"].append((t3 - t2) * 1e3)
                st["match"].append((t4 - t3) * 1e3)
                st["rerank"].append((t5 - t4) * 1e3)

        med = {k: statistics.median(v) for k, v in st.items()}
        med.update(read=read_ms, render=render_ms, params=params, desc_dim=dim, rep=rep)
        med["compute"] = med["render"] + med["tensor"] + med["encode"] + med["match"] \
            + med["rerank"]
        med["total"] = med["compute"] + med["read"]
        rows[label] = med
        print(f"{label:30s} params {params:>12,}  desc {dim}  ({rep})")
        print(f"    read {med['read']:5.2f} + render {med['render']:5.2f} + "
              f"tensor {med['tensor']:5.2f} + encode {med['encode']:5.2f} + "
              f"match {med['match']:5.2f} + rerank {med['rerank']:5.2f} "
              f"= {med['total']:6.2f} ms  (compute {med['compute']:.2f})")
        if name == "eventgem":
            g = med["compute"] - med["rerank"]
            print(f"    global stage alone (no rerank, whitening kept): {g:.2f} ms compute")
        del module, gallery
        if extra:
            del extra
        torch.cuda.empty_cache()

    if cli.spikevpr:
        print("\nSpikeVPR (envs/spikevpr):")
        out = os.path.join("/tmp", f"latency_spikevpr_{os.getpid()}.json")
        # The checkpoint/neuron pairing is resolved here, where SPIKEVPR_CHECKPOINTS lives:
        # the child environment has no loguru and so cannot import the bridge at all.
        from src.spikevpr_bridge import ENCODER, OUT_CHANNELS, OUT_ROWS, resolve_checkpoint
        ckpt, neuron = resolve_checkpoint(cli.spikevpr_model)
        cmd = ["pixi", "run", "--manifest-path", os.path.join(REPO, "envs", "spikevpr"),
               "python3", "-u", os.path.join(HERE, "latency_spikevpr.py"),
               "--recording", cli.recording, "--sensor", str(cli.sensor[0]),
               str(cli.sensor[1]), "--repeats", str(cli.repeats),
               "--warmup", str(cli.warmup), "--gallery", str(cli.gallery),
               "--checkpoint", ckpt, "--neuron", neuron, "--encoder", ENCODER,
               "--out-channels", str(OUT_CHANNELS), "--out-rows", str(OUT_ROWS),
               "--seed", str(cli.seed), "--out-json", out]
        rc = subprocess.run(cmd, cwd=REPO).returncode
        if rc == 0 and os.path.exists(out):
            m = json.load(open(out))
            rows[m["label"]] = m
            print(f"    read {m['read']:5.2f} + render {m['render']:5.2f} + "
                  f"tensor {m['tensor']:5.2f} + encode {m['encode']:5.2f} + "
                  f"match {m['match']:5.2f} + rerank {m['rerank']:5.2f} "
                  f"= {m['total']:6.2f} ms  (compute {m['compute']:.2f})")
            os.remove(out)
        else:
            print(f"    !! SpikeVPR measurement failed (rc={rc}); row omitted")

    print(f"\n{'method':30s} {'params':>12s} {'D':>6s} {'read':>6s} {'render':>7s} "
          f"{'tensor':>7s} {'encode':>7s} {'match':>7s} {'rerank':>7s} {'compute':>8s}")
    for label, m in sorted(rows.items(), key=lambda kv: kv[1]["compute"]):
        print(f"{label:30s} {m['params']:>12,} {m['desc_dim']:>6} {m['read']:6.2f} "
              f"{m['render']:7.2f} {m['tensor']:7.2f} {m['encode']:7.2f} {m['match']:7.2f} "
              f"{m['rerank']:7.2f} {m['compute']:8.2f}")

    if cli.out_json:
        with open(cli.out_json, "w") as h:
            json.dump({"device": torch.cuda.get_device_name(0), "probe_ms": p,
                       "contended": bool(others), "recording": cli.recording,
                       "gallery": cli.gallery, "repeats": cli.repeats, "rows": rows}, h,
                      indent=2)
        print(f"\n-> {cli.out_json}")


if __name__ == "__main__":
    main()
