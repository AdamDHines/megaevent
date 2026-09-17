"""Baselines on Springfield, under exactly the protocol megaevent is scored on.

Run from the repo root::

    CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python3 scripts/springfield_baselines.py \
        --method megaloc --allow-unrepacked --no-event-filter

The question this answers is "is the dataset hard, or is the model bad?", which a single
number cannot settle. Everything here — the slice grid, the clock fits, the 25 m ground
truth, the pooled seven-session gallery, the stride sweep, micro/macro/theta — is imported
from :mod:`scripts.springfield_full` and reused unchanged; the only thing that varies is
which model turns a slice into a descriptor. So a difference between two rows of the output
is a difference between two methods and nothing else.

Each method runs on **its own native representation**, which is how every other table in
this repo compares them: MegaLoc on countmask (the frames it is given everywhere else), and
Event-GeM on the 10-channel MCTS time surface SuperEvent was trained on. Representation is
part of a method, not a nuisance parameter — swapping one onto the other's frames measures
neither.

Two practical notes:

* countmask renders through eventcv at ~44 slices/s where accumulate falls back to a numpy
  port at ~6.8, so a MegaLoc pass over this capture is about an hour where an accumulate one
  would be six. That is a property of the renderer, not of the method.
* Banks cache per partition under the same scheme as the megaevent run, keyed by a tag that
  carries the method and representation, so the two never collide and a re-run is free.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src import inference as inf                      # noqa: E402
from src.traversegps import local_enu                 # noqa: E402
from springfield_eval import (                        # noqa: E402
    bank_for_partition, check_repacked, list_partitions, load_gps, make_dataset,
    partition_geometry, partition_manifest)
from springfield_full import (                        # noqa: E402
    DB_ARMS, DEFAULT_BANK_DIR, DEFAULT_CKPT, DEFAULT_OUT, DEFAULT_ROOT, SWEEPS,
    apply_curation, discover_queries, load_clock, load_curation,
    score_protocol, session_rate_mevs, stride_mask)


class _Cfg:
    """The three fields ``process_session`` and ``score_protocol`` read off a model config."""

    def __init__(self, representation, resolution, desc_dim):
        self.representation = representation
        self.H = self.W = resolution
        self.desc_dim = desc_dim
        self.vit = "-"
        self.aggregator = "-"


# ---------------------------------------------------------------------------
# 1. Methods — each supplies a representation, a transform and a callable model
# ---------------------------------------------------------------------------
def build_megaloc(cli, device):
    """MegaLoc (Berton & Masone 2025) from its authors' hub entry point, unmodified.

    The control that has never seen an event: trained on street-view, aerial and indoor
    photographs, and given the same countmask frames every event method here is given. What
    it scores is the floor that training on events has to clear.
    """
    from src.methods import MEGALOC_DESC_DIM, MEGALOC_HUB, megaloc_transform

    model = torch.hub.load(MEGALOC_HUB, "get_trained_model", source="github",
                           trust_repo=True).eval().to(device)
    params = sum(p.numel() for p in model.parameters())
    rep = cli.representation or "countmask"
    res = cli.eval_resolution or 322
    print(f"MegaLoc (torch.hub {MEGALOC_HUB}, {params / 1e6:.1f}M params): "
          f"desc={MEGALOC_DESC_DIM} rep={rep} in={res}x{res}")
    return {"model": model, "transform": megaloc_transform(res),
            "cfg": _Cfg(rep, res, MEGALOC_DESC_DIM),
            "tag": f"megaloc_{rep}_r{res}", "label": "megaloc",
            "meta": {"model": "megaloc", "hub": MEGALOC_HUB, "parameters": int(params),
                     "trained_on": "RGB images, no event data"}}


class _ReconFrames(torch.utils.data.Dataset):
    """One partition's slices as pre-reconstructed intensity frames from disk.

    The event-to-video arm: ``scripts/springfield_recon.py`` writes one grayscale PNG
    per 50 ms eventcv slice per partition (same reader, hot-pixel on, BA off), so frame
    ``i`` here is slice ``i`` of every event-domain representation and the whole
    geometry/keep/scoring machinery applies unchanged. Grayscale is replicated to three
    channels before the model transform — MegaLoc's ImageNet normalization expects RGB,
    and gray-in-RGB is the standard way reconstructed frames are fed to RGB VPR models.

    The per-partition ``recon.json`` is the completeness contract: its ``n_slices`` came
    from the same eventcv framing, and a frame-count or model mismatch is a hard error
    rather than a silently short bank.
    """

    def __init__(self, h5path, transform, recon_root,
                 model_id="v2v_e2vid_10k/epoch_0077"):
        sid = os.path.basename(os.path.dirname(h5path))
        stem = os.path.splitext(os.path.basename(h5path))[0]
        self.dir = os.path.join(recon_root, sid, stem)
        self.transform = transform
        meta_path = os.path.join(self.dir, "recon.json")
        if not os.path.exists(meta_path):
            raise SystemExit(f"{self.dir}: no recon.json — run scripts/"
                             f"springfield_recon.py for {sid} first")
        with open(meta_path) as handle:
            meta = json.load(handle)
        if meta.get("model") != model_id:
            raise SystemExit(f"{meta_path}: reconstructed by {meta.get('model')}, "
                             f"expected {model_id}")
        self._n = int(meta["n_slices"])
        n_png = len([f for f in os.listdir(self.dir) if f.endswith(".png")])
        if n_png != self._n:
            raise SystemExit(f"{self.dir}: {n_png} frames for {self._n} slices — "
                             f"incomplete reconstruction")

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        import cv2
        img = cv2.imread(os.path.join(self.dir, f"frame_{i:06d}.png"),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"{self.dir}/frame_{i:06d}.png")
        x = torch.from_numpy(img).float().div_(255.0).unsqueeze(0).repeat(3, 1, 1)
        return self.transform(x)


class _MCTSFromStream(torch.utils.data.Dataset):
    """One partition's slices as SuperEvent's 10-channel MCTS, straight from the events.

    :func:`src.npzdata.load_mcts` builds this from an ``.npz`` of raw events; the same
    construction works on a capture partition because eventcv's reader hands back the same
    ``(N, 4)`` xytp array per slice. Resizing happens in the *event* domain, as upstream
    does, so the decay values stay exact instead of being interpolated between pixels that
    fired at different times.
    """

    # Built through the *accumulate* reader on purpose. eventcv renders countmask itself, so
    # that reader's ``slice(i)`` hands back a rendered [3,H,W] frame; accumulate has no
    # eventcv renderer, so :class:`src.inference._RawRenderReader` wraps the raw slicing
    # reader and ``._reader.slice(i)`` is the (N,4) xytp array this needs. Nothing of the
    # accumulate rendering is used — only its reader's slicing, filters and frame indices,
    # which are the same ones every other representation sees.
    REP = "accumulate"

    def __init__(self, h5path, dt_ms, size, hot_pixel, filter_dt_us, max_window_ms=30.0):
        from torchvision import transforms
        self.h5path, self.size, self.max_window_ms = h5path, size, max_window_ms
        self._args = (dt_ms, hot_pixel, filter_dt_us)
        probe = make_dataset(h5path, transforms.Compose([]), self.REP, dt_ms,
                             hot_pixel, filter_dt_us)
        self._n = len(probe)
        self.sensor = probe.sensor
        del probe
        self._reader = None

    def __len__(self):
        return self._n

    def _open(self):
        import eventcv as ecv
        from torchvision import transforms
        dt_ms, hot_pixel, filter_dt_us = self._args
        ds = make_dataset(self.h5path, transforms.Compose([]), self.REP, dt_ms,
                          hot_pixel, filter_dt_us)
        inner = getattr(ds.reader, "_reader", None)
        if inner is None or not hasattr(inner, "slice"):
            raise SystemExit(f"{self.h5path}: no raw slicing reader behind the "
                             f"{self.REP} reader; MCTS needs events, not frames")
        probe = ecv.numpy(inner.slice(0))
        if probe.ndim != 2 or probe.shape[1] != 4:
            raise SystemExit(f"{self.h5path}: raw slice is {probe.shape}, expected (N, 4) "
                             f"xytp — this reader renders rather than slices")
        return inner, ds

    def __getitem__(self, i):
        import eventcv as ecv
        from src.npzdata import _stream
        if self._reader is None:
            self._reader, self._keep = self._open()
        ev = ecv.numpy(self._reader.slice(i))
        h, w = self.sensor[1], self.sensor[0]
        height, width = self.size
        if ev.shape[0] == 0:
            return torch.zeros((10, height, width), dtype=torch.float32)
        stream = _stream(ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3], h, w)
        stream = stream.resize(width=width, height=height)
        return torch.from_numpy(stream.mcts(max_window_ms=self.max_window_ms).numpy())


class _EventGeMTrunk(torch.nn.Module):
    """SuperEvent trunk + FPN + GeM — Event-GeM's global stage, exactly as it scores it.

    Wrapped as a module so the shared per-partition extractor can drive it like any other
    model. The detector and descriptor heads are skipped: only the shortlisted pairs need
    keypoints, and the descriptor head's ``[B, 256, H, W]`` map costs more than the rest put
    together. Re-ranking is Event-GeM's second stage and is **not** included here — this row
    is the global stage, directly comparable to every other single-pass method.
    """

    def __init__(self, model, gem_p, crop):
        super().__init__()
        self.model, self.gem_p, self.crop = model, gem_p, crop

    def forward(self, x):
        off_top, off_left, hc, wc = self.crop
        if (hc, wc) != tuple(x.shape[-2:]):
            x = x[:, :, off_top:off_top + hc, off_left:off_left + wc]
        f = self.model.fpn(self.model.backbone(x)).float()
        pooled = torch.nn.functional.avg_pool2d(
            f.clamp(min=1e-6).pow(self.gem_p), (f.shape[-2], f.shape[-1])
        ).pow(1.0 / self.gem_p)
        return torch.nn.functional.normalize(pooled.squeeze(-1).squeeze(-1), p=2, dim=1)


def build_eventgem(cli, device):
    """Event-GeM's global stage (GeM over SuperEvent's FPN map) on MCTS frames."""
    from src import eventgemlocal as egl
    from src.methods import EVENTGEM_P

    shim = argparse.Namespace(eventgem_repo=cli.eventgem_repo, dataset="springfield",
                              feature_dir=cli.feature_dir)
    egl.add_to_path(shim)
    model, cfg, _ = egl.build_superevent(egl.superevent_root(shim), device)
    size = tuple(cli.eventgem_size)
    crop = egl.crop_offsets(*size, egl.input_multiple(cfg))
    dim = int(cfg.get("fpn_size") or cfg.get("descriptor_size") or 128)
    print(f"Event-GeM global stage (SuperEvent trunk+FPN+GeM p={EVENTGEM_P}): "
          f"mcts {size[0]}x{size[1]}, crop {crop[2]}x{crop[3]}")
    return {"model": _EventGeMTrunk(model, EVENTGEM_P, crop).eval().to(device),
            "transform": None, "cfg": _Cfg("mcts", size[0], dim),
            "tag": f"eventgem_se_{size[0]}x{size[1]}", "label": "eventgem",
            "mcts_size": size,
            "meta": {"model": "eventgem-global", "backbone": "superevent",
                     "grid": list(size), "gem_p": EVENTGEM_P,
                     "note": "global stage only; the +rerank second pass is not included"}}


class _CountTripletFromStream(torch.utils.data.Dataset):
    """One partition's slices as EventVLAD's ``[3, 256, 256]`` count triplet.

    :func:`src.npzdata.load_count_triplet` splits a recording into ``bins`` consecutive
    sub-windows because EventVLAD's denoiser wants a short temporal sequence rather than
    one frame repeated. A 50 ms capture slice is exactly such a sequence, so the same split
    applies here — thirds of the slice's own timestamp span, each normalised by the 99th
    percentile of its own non-zero counts, as Event-LAB's ``normalize_frame`` does.
    """

    REP = "accumulate"          # see _MCTSFromStream.REP — needed for a raw slicing reader

    def __init__(self, h5path, dt_ms, size, bins, hot_pixel, filter_dt_us, percentile=99.0):
        from torchvision import transforms
        self.h5path, self.size, self.bins = h5path, size, bins
        self.percentile = percentile
        self._args = (dt_ms, hot_pixel, filter_dt_us)
        probe = make_dataset(h5path, transforms.Compose([]), self.REP, dt_ms,
                             hot_pixel, filter_dt_us)
        self._n, self.sensor = len(probe), probe.sensor
        del probe
        self._reader = None

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        import eventcv as ecv
        from torchvision import transforms
        from src.npzdata import _area_resize, _stream
        if self._reader is None:
            ds = make_dataset(self.h5path, transforms.Compose([]), self.REP, *self._args)
            inner = getattr(ds.reader, "_reader", None)
            if inner is None or not hasattr(inner, "slice"):
                raise SystemExit(f"{self.h5path}: no raw slicing reader for count triplets")
            self._reader, self._keep = inner, ds
        ev = ecv.numpy(self._reader.slice(i))
        w, h = self.sensor
        planes = np.zeros((self.bins, h, w), dtype=np.float32)
        if ev.shape[0]:
            t = ev[:, 2]
            edges = np.linspace(int(t.min()), int(t.max()) + 1, self.bins + 1)
            for b in range(self.bins):
                m = (t >= edges[b]) & (t < edges[b + 1])
                if not m.any():
                    continue
                planes[b] = _stream(ev[m, 0], ev[m, 1], t[m], ev[m, 3], h,
                                    w).count().numpy()[0].astype(np.float32)
        for b in range(self.bins):
            nz = planes[b][planes[b] > 0]
            scale = (float(np.percentile(nz.astype(np.float64), self.percentile))
                     if nz.size else 0.0)
            if scale > 0:
                planes[b] /= scale
        return _area_resize(np.clip(planes, 0.0, 1.0), self.size)


class _EventVLADEncoder(torch.nn.Module):
    """EventVLAD's denoiser + VGG16/NetVLAD as one callable, matching its own forward.

    ``src.methods.EventVLADMethod.encode`` op for op: the denoiser's channel 0 is the
    reconstruction (channel 1 is its error estimate and is discarded, as upstream does),
    scaled to 0-255, tiled to three channels, area-resized to 224 and passed through the
    encoder with the MatConvNet mean removed. The descriptors are deliberately **not**
    unit vectors, and ``topk_ranked`` scores with a plain dot product, which is exactly
    Event-LAB's own ``D = (1 - q @ r.T).T`` up to the sign convention.
    """

    def __init__(self, method):
        super().__init__()
        self.m = method

    def forward(self, x):
        # fp32, unconditionally. The shared extractor wraps every forward in fp16 autocast,
        # and this stack does not survive it: the denoiser rescales its reconstruction to
        # 0-255 and VGG's activations on that overflow half precision, giving a bank that is
        # 100% NaN — which scores R@1 0.0000, i.e. *below* the 0.0151 chance level, and so
        # reads as "EventVLAD is terrible" rather than "this row is broken". EventVLAD's own
        # path in src/methods.py calls encode() directly and never meets autocast; this
        # keeps that true whatever the caller does.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return self.m.encode(x.float())


def build_eventvlad(cli, device):
    """EventVLAD (Lee & Kim) — denoiser + VGG16/NetVLAD, from Event-LAB's checkout."""
    from src.methods import EVENTVLAD_DENOISE_SIZE, EventVLADMethod

    shim = argparse.Namespace(eventlab_repo=cli.eventlab_repo, dataset="springfield",
                              feature_dir=cli.feature_dir, limit=None)
    m = EventVLADMethod(shim, device)
    model = _EventVLADEncoder(m).eval().to(device)
    # Probed, not assumed: NetVLAD is declared K=64 x D=1000 but this checkout's VGG hands
    # it a 1000-d vector, so the descriptor is 1000-d — which is what every published
    # EventVLAD bank in this repo is (13451 x 1000). Guessing 64x1000 here would only have
    # shown up as a shape error on a partition with no selected rows.
    with torch.no_grad():
        dim = int(model(torch.zeros(1, 3, EVENTVLAD_DENOISE_SIZE,
                                    EVENTVLAD_DENOISE_SIZE, device=device)).shape[1])
    return {"model": model, "transform": None,
            "cfg": _Cfg("count_triplet", EVENTVLAD_DENOISE_SIZE, dim),
            "tag": "eventvlad_ct256_b3", "label": "eventvlad",
            "triplet": (EVENTVLAD_DENOISE_SIZE, 3),
            "meta": {"model": "eventvlad", **m.meta}}


def build_spikevpr(cli, device):
    """SpikeVPR's SEW-ResNet34 + MixVPR on ON/OFF counts, via its own conda env.

    Unlike the others this one does not hand back a torch module: the forward pass runs in
    ``envs/spikevpr`` through :mod:`src.spikevpr_bridge`, which already has a
    ``traverse_job`` mode producing one row per ``dt_ms`` slice of a recording — exactly a
    full partition bank. That mode has no row-subset support, which is why SpikeVPR is only
    run at full rate; at full rate it is the natural fit and is upstream's own path.
    """
    from src import spikevpr_bridge

    checkpoint, neuron = spikevpr_bridge.resolve_checkpoint(cli.spikevpr_model,
                                                            cli.spikevpr_repo)
    digest = spikevpr_bridge.checkpoint_sha256(checkpoint)
    dim = spikevpr_bridge.OUT_CHANNELS * spikevpr_bridge.OUT_ROWS
    print(f"SpikeVPR {spikevpr_bridge.ENCODER} trained on {cli.spikevpr_model} "
          f"(sha256 {digest[:10]}, {neuron}): {dim}-d, "
          f"in={spikevpr_bridge.GRID[0]}x{spikevpr_bridge.GRID[1]}")
    return {"model": None, "transform": None, "bridge": True,
            "cfg": _Cfg("onoff", spikevpr_bridge.GRID[0], dim),
            "tag": f"spikevpr_r34_{cli.spikevpr_model}_{digest[:10]}",
            "label": "spikevpr", "checkpoint": checkpoint, "neuron": neuron,
            "meta": {"model": "spikevpr", "checkpoint": checkpoint,
                     "checkpoint_sha256": digest, "trained_on": cli.spikevpr_model,
                     "neuron": neuron, "grid": list(spikevpr_bridge.GRID)}}

class _EG010Global(torch.nn.Module):
    """Event-GeM 0.1.0's released global stage: ECDPT ViT ``pr.pt`` + GeM(p=5).

    The published model, not the post-release ``main``-branch GeM-over-SuperEvent-FPN that
    :class:`src.methods.EventGeMMethod` implements. Weights, forward and loader math all come
    from the 0.1.0 worktree via :mod:`scripts.springfield_eg010`, which gates them against
    the released checkpoint's sha256 and the release's own ``EventGeMData`` before use.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        from springfield_eg010 import forward_gem
        return torch.nn.functional.normalize(forward_gem(self.backbone, x), p=2, dim=1)


def build_eg010(cli, device):
    """Event-GeM 0.1.0 global stage. Gated on import — see springfield_eg010."""
    import springfield_eg010 as eg

    eg.gate_checkpoint()
    eg.gate_loader_math()
    backbone = eg.build_backbone(device)
    model = _EG010Global(backbone).eval().to(device)
    with torch.no_grad():
        dim = int(model(torch.zeros(1, 2, 224, 224, device=device)).shape[1])
    print(f"Event-GeM 0.1.0 global (ECDPT ViT pr.pt + GeM p={eg.GEM_P:g}): desc={dim} "
          f"rep=polarity_counts in=224x224")
    return {"model": model, "transform": None,
            "cfg": _Cfg("polarity_counts", 224, dim),
            "tag": "eg010_gem224", "label": "eventgem010",
            "meta": {"model": "eventgem-0.1.0-global", "checkpoint": eg.CKPT,
                     "checkpoint_sha256_prefix": eg.CKPT_SHA, "gem_p": eg.GEM_P,
                     "note": "released 0.1.0 ECDPT global stage; the +rerank second pass "
                             "is added separately by springfield_eg010.py --stage rerank"}}

def _rgb_control(name):
    """Builder for any RGB control in ``scripts/rgb_pooled.py``'s registry.

    That registry already pins each model's loader, descriptor width, provenance and its
    *own* published evaluation resolution (322 for salad/boq/qaa/supervlad, 320 for
    mixvpr — structural, see the src.methods class docstrings). Duplicating those here
    would be a second place to get them wrong, so they are read from it. ``megaloc`` keeps
    the hand-written builder above: it predates the registry and its ``meta`` block is
    quoted in published tables.

    Why these exist at all: the Springfield table compared against MegaLoc alone, but
    MegaLoc is not the strongest RGB control on event frames — on pooled Brisbane
    (accumulate) mixvpr beats it by 0.116 R@1 and on NSAVP qaa beats it by 0.055. A margin
    measured against one control is not a margin against the state of the art.

    ``cricavpr`` is deliberately absent: its descriptors depend on the batch it is scored
    in (src/methods.py::CricaVPRMethod), which the per-partition banking here would make
    silently protocol-dependent.
    """
    def build(cli, device):
        import rgb_pooled as rp                                    # noqa: PLC0415
        from src.methods import megaloc_transform                  # noqa: PLC0415

        loader, desc_dim, source, trained_on, native_res, _batch = rp.MODELS[name]
        model = loader(device)
        params = sum(p.numel() for p in model.parameters())
        rep = cli.representation or "countmask"
        res = cli.eval_resolution or native_res
        stats = getattr(cli, "norm_stats", "imagenet")
        print(f"{name} ({source}, {params / 1e6:.1f}M params): desc={desc_dim} "
              f"rep={rep} in={res}x{res} norm={stats}")
        tag = f"{name}_{rep}_r{res}" + ("" if stats == "imagenet" else f"_{stats[:2]}stats")
        return {"model": model, "transform": megaloc_transform(res, stats=stats),
                "cfg": _Cfg(rep, res, desc_dim), "tag": tag, "label": name,
                "meta": {"model": name, "hub": source, "parameters": int(params),
                         "trained_on": trained_on, "norm_stats": stats}}
    return build


BUILDERS = {"megaloc": build_megaloc, "eventgem": build_eventgem,
            "eventvlad": build_eventvlad, "spikevpr": build_spikevpr,
            "eg010": build_eg010,
            **{n: _rgb_control(n) for n in ("salad", "mixvpr", "boq", "qaa", "supervlad")}}


# ---------------------------------------------------------------------------
# 2. Session processing — geometry first, then descriptors for selected rows only
# ---------------------------------------------------------------------------
def slice_count(cli, path, sid):
    """``n_slices`` for one partition without opening it, if any bank manifest knows.

    The count is a property of the recording and the 50 ms grid, not of the model, so any
    method's cached manifest answers for all of them — and the megaevent run already wrote
    one for all 199 partitions. That matters because the geometry pass has to run *before*
    extraction (the stride mask needs positions), and opening every partition just to learn
    a number would cost more than the extraction it is meant to shrink.
    """
    d = os.path.join(cli.bank_dir, sid)
    if os.path.isdir(d):
        stem = os.path.splitext(os.path.basename(path))[0]
        for name in os.listdir(d):
            if name.endswith(".json") and name.endswith(stem + ".json"):
                try:
                    with open(os.path.join(d, name)) as f:
                        n = json.load(f).get("n_slices")
                except (OSError, ValueError):
                    continue
                if n:
                    return int(n)
    return None


def session_geometry(cli, method, sdir, sid, limit=None):
    """Positions and the base keep mask for one session — no model, no extraction."""
    parts = list_partitions(sdir, limit)
    check_repacked(parts, cli.allow_unrepacked)
    clock, clock_flag = load_clock(sdir)
    gps = load_gps(sdir)
    s_latlon, s_keep, s_prov, counts = [], [], [], []
    for path in parts:
        n = slice_count(cli, path, sid)
        if n is None:
            n = len(build_dataset(cli, method, path))
        ll, kp, _ = partition_geometry(path, n, cli.dt_ms, clock, gps)
        s_latlon.append(ll); s_keep.append(kp); counts.append(n)
        s_prov.extend((sid, os.path.basename(path), i) for i in range(n))
    keep = np.concatenate(s_keep)
    return {"latlon": np.concatenate(s_latlon), "keep": keep, "prov": s_prov,
            "parts": parts, "counts": counts, "clock_flag": clock_flag,
            "mev_s": round(session_rate_mevs(sdir), 2), "n_partitions": len(parts),
            "n_slices": int(len(keep)), "n_kept": int(keep.sum()), "bank": None}


def build_dataset(cli, method, path):
    """The method's frame source for one partition."""
    rep = method["cfg"].representation
    filt = None if cli.no_event_filter else cli.event_filter_dt_us
    if rep == "v2ve2vid":
        return _ReconFrames(path, method["transform"], cli.recon_root)
    if rep == "mcts":
        return _MCTSFromStream(path, cli.dt_ms, method["mcts_size"],
                               not cli.no_hot_pixel, filt)
    if rep == "polarity_counts":
        from springfield_eg010 import PolarityCountsFromStream
        return PolarityCountsFromStream(path, cli.dt_ms, not cli.no_hot_pixel, filt)
    if rep == "count_triplet":
        size, bins = method["triplet"]
        return _CountTripletFromStream(path, cli.dt_ms, size, bins,
                                       not cli.no_hot_pixel, filt)
    return make_dataset(path, method["transform"], rep, cli.dt_ms,
                        not cli.no_hot_pixel, filt)


def bridge_bank(cli, method, path, n_slices):
    """SpikeVPR's whole-partition bank, computed in its own env by the vendored bridge.

    ``traverse_job`` renders one row per ``dt_ms`` slice of the recording, which is a full
    partition bank — so this is only ever correct at full rate, and the caller checks that.
    """
    import h5py
    from src import spikevpr_bridge

    with h5py.File(path, "r") as f:
        sensor = (int(f.attrs["width"]), int(f.attrs["height"]))
    out = os.path.join(cli.bank_dir, "_spikevpr_tmp",
                       f"{os.path.splitext(os.path.basename(path))[0]}_bank.npy")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    job = spikevpr_bridge.traverse_job(
        path, sensor, cli.dt_ms, 0, not cli.no_hot_pixel,
        None if cli.no_event_filter else cli.event_filter_dt_us,
        checkpoint=method["checkpoint"], neuron=method["neuron"], out=out,
        max_events=cli.spikevpr_max_events, spikevpr_repo=cli.spikevpr_repo,
        batch_size=cli.batch_size, label=os.path.basename(path))
    bank = spikevpr_bridge.run(job, cli.spikevpr_env)
    if bank.shape[0] != n_slices:
        raise SystemExit(f"{path}: spikevpr bridge returned {bank.shape[0]} rows for "
                         f"{n_slices} slices")
    return np.asarray(bank, dtype=np.float32)


def extract_session(cli, method, device, session, sid, select):
    """Fill ``session['bank']`` for the rows ``select`` marks, and nothing else.

    ``select`` is a boolean over the session's full slice grid. Only those rows are ever
    read downstream (``keep`` is set to the same mask), so extracting the rest is pure
    waste — and on a 1 m gallery stride that is 11 rows in 12. The bank is still stored on
    the full grid so every index in ``prov``, ``keep`` and ``latlon`` keeps meaning the
    same thing; only the *rows* are sparse, and the cache on disk holds just the filled
    ones plus their indices.
    """
    cfg, tag = method["cfg"], method["tag"]
    banks, off = [], 0
    for path, n in zip(session["parts"], session["counts"]):
        rows = np.flatnonzero(select[off:off + n]).astype(np.int64)
        off += n
        manifest = partition_manifest(cli, path, tag, cfg.representation, n)
        manifest["n_selected"] = int(len(rows))
        banks.append(bank_subset(cli, method, device, path, tag, sid, manifest, rows, n))
    session["bank"] = np.concatenate(banks)
    return session


def bank_subset(cli, method, device, path, tag, sid, manifest, rows, n_slices):
    """``[n_slices, D]`` with ``rows`` filled from the model and the rest left at zero.

    Cached as the compact ``[len(rows), D]`` block plus its row indices, so a 1 m stride
    costs a twelfth of the disk a full bank would. The manifest carries ``n_selected``, so
    a subset bank can never be mistaken for a full one.
    """
    out_dir = os.path.join(cli.bank_dir, sid)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{tag}_{os.path.splitext(os.path.basename(path))[0]}")
    arr_path, idx_path, man_path = stem + ".npy", stem + ".rows.npy", stem + ".json"

    if (os.path.exists(arr_path) and os.path.exists(man_path) and not cli.force_rebuild):
        with open(man_path) as f:
            cached = json.load(f)
        if cached == manifest:
            compact = np.load(arr_path)
            idx = np.load(idx_path) if os.path.exists(idx_path) else np.arange(len(compact))
            bank = np.zeros((n_slices, compact.shape[1]), np.float32)
            bank[idx] = compact
            print(f"    {os.path.basename(arr_path)}: cached ({len(idx)}/{n_slices} rows)")
            return bank

    if not len(rows):
        return np.zeros((n_slices, method["cfg"].desc_dim), np.float32)
    if method.get("bridge"):
        # The bridge has no row-subset mode: it always renders the whole partition. That is
        # not an error — the rows outside `select` are simply never read, since `keep` does
        # the selecting downstream. It only means a stride buys nothing here, so the full
        # bank is computed and stored whatever `rows` asks for.
        compact = bridge_bank(cli, method, path, n_slices)
        if not np.isfinite(compact).all():
            raise SystemExit(f"{os.path.basename(arr_path)}: bridge returned non-finite "
                             f"descriptors — refusing to cache a poisoned bank")
        rows = np.arange(n_slices, dtype=np.int64)
        bank = compact.copy()
        np.save(arr_path, compact)
        np.save(idx_path, rows)
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"    {os.path.basename(arr_path)}: bridge {compact.shape} -> cached")
        return bank
    ds = build_dataset(cli, method, path)
    if len(ds) != n_slices:
        raise SystemExit(f"{path}: dataset has {len(ds)} slices, geometry expected "
                         f"{n_slices}")
    sub = torch.utils.data.Subset(ds, rows.tolist())
    desc = inf.extract_descriptors(method["model"], sub, device,
                                   label=f"{sid}/{os.path.basename(path)}",
                                   batch_size=cli.batch_size, oom_backoff=True)
    compact = desc.numpy().astype(np.float32)
    # A bank that is silently non-finite scores *below* chance and reads as a weak method
    # rather than a broken one — EventVLAD's fp16 overflow produced exactly that, a 100%
    # NaN bank reported as R@1 0.0000 against a 0.0151 chance level. Fail at the source.
    if not np.isfinite(compact).all():
        raise SystemExit(f"{os.path.basename(arr_path)}: "
                         f"{int((~np.isfinite(compact)).any(axis=1).sum())} of "
                         f"{len(compact)} descriptors are non-finite — refusing to cache a "
                         f"poisoned bank")
    bank = np.zeros((n_slices, compact.shape[1]), np.float32)
    bank[rows] = compact
    np.save(arr_path, compact)
    np.save(idx_path, rows)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    {os.path.basename(arr_path)}: extracted {compact.shape} "
          f"({len(rows)}/{n_slices} rows) -> cached")
    return bank


def exclude_reversals(cli, q_sessions):
    """Mask out query slices whose camera is more than N degrees off the route direction.

    The reversed passes are a viewpoint condition no panoramic benchmark contains: Tokyo
    24/7's gallery is 6,332 panoramas cut into 12 views each, one every 30 degrees, so its
    worst possible query-to-reference offset is 15 degrees. Springfield's reference is three
    mounts fixed *relative to travel*, so a pass walked backwards is 180 degrees from the
    forward arm and at least 90 from every arm — unmatchable by construction rather than
    hard. Excluding it makes the remaining cell the one whose viewpoint condition a
    panorama-based benchmark would actually cover.

    Per-slice camera bearing comes from the diagnostic ``orient`` bundle, which derives it
    from the GPS travel bearing while moving (the query rig faces the walk) and fills the
    stationary slices from a per-session yaw fit. Its arrays are indexed over each session's
    kept slices in order, which is the same mask the queries carry here — queries are never
    strided — so the two line up index for index, and a length mismatch is a hard error
    rather than a silent misalignment.
    """
    path = os.path.join(cli.diag_dir, f"orient_{cli.orient_tag}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"no {path} — run scripts/springfield_diag.py --stage orient "
                         f"before excluding reversals")
    o = np.load(path, allow_pickle=False)
    gate, total = float(cli.exclude_reversals_deg), 0
    for sid, s in q_sessions.items():
        key = f"{sid}_psi"
        if key not in o:
            raise SystemExit(f"{path} has no psi for {sid}")
        psi = o[key]
        idx = np.flatnonzero(s["keep"])
        if len(psi) != len(idx):
            raise SystemExit(f"{sid}: orient holds {len(psi)} slices but {len(idx)} are "
                             f"kept — the two were built on different masks")
        drop = idx[np.abs(psi) >= gate]
        s["keep"][drop] = False
        total += len(drop)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="unused; kept so the shared "
                                                         "manifest helper has a field")
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--eval-resolution", type=int, default=None)
    ap.add_argument("--recon-root",
                    default="/media/adam/vprdatasets/megaevent/springfield/recon_v2v_e2vid",
                    help="reconstructed-frame tree from scripts/springfield_recon.py, "
                         "used when --representation v2ve2vid")
    ap.add_argument("--representation", default=None,
                    help="override the method's native representation (rarely right)")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--norm-stats", default="imagenet",
                    choices=["imagenet", "countmask", "accumulate"],
                    help="normalisation constants for an RGB control (ignored by the event "
                         "methods, which carry their own). Match it to --representation: "
                         "ImageNet stats sit ~2 sigma off a countmask frame and 2.0-4.5 "
                         "sigma off an accumulate one. Banks carry a _cmstats / _acstats "
                         "tag so the arms never collide.")
    ap.add_argument("--threshold-m", type=float, default=25.0)
    ap.add_argument("--event-filter-dt-us", type=int, default=50_000)
    ap.add_argument("--no-hot-pixel", action="store_true")
    ap.add_argument("--no-event-filter", action="store_true")
    ap.add_argument("--allow-unrepacked", action="store_true")
    ap.add_argument("--limit-partitions", type=int, default=None)
    ap.add_argument("--query-limit", type=int, default=None)
    ap.add_argument("--query-sessions", default=None)
    ap.add_argument("--sweeps", default=",".join(SWEEPS))
    ap.add_argument("--db-sessions", default=",".join(DB_ARMS))
    ap.add_argument("--db-stride-m", type=float, default=0.0,
                    help="gallery sampling stride in metres; 0 (the default) keeps the "
                         "native 50 ms stream, all 132,569 rows. Only selected rows are "
                         "ever extracted, so a stride is a real saving when one is wanted")
    ap.add_argument("--exclude-reversals-deg", type=float, default=0.0,
                    help="drop query slices whose camera is at least this far off the "
                         "route direction; 0 (the default) keeps them. This is a SCORING "
                         "mask, not an extraction one — the same banks serve both cells, "
                         "so the suite scores each method twice, at 0 and at 135")
    ap.add_argument("--diag-dir", default="output/springfield_diag")
    ap.add_argument("--orient-tag",
                    default="b_v8_accum_s750_s750_3f4eb54780_accumulate_r322_hp1_baoff",
                    help="which diag orient bundle supplies the per-slice camera bearing")
    ap.add_argument("--curation", default=None)
    ap.add_argument("--topk-per-session", type=int, default=0)
    ap.add_argument("--topk-fig", type=int, default=3)
    ap.add_argument("--db-chunk", type=int, default=40_000)
    ap.add_argument("--score-chunk", type=int, default=256)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--eventgem-repo", default="./external/eventgem")
    ap.add_argument("--eventgem-size", type=int, nargs=2, default=(240, 320))
    ap.add_argument("--feature-dir", default="features")
    ap.add_argument("--eventlab-repo", default="/home/adam/repo/Event-LAB")
    ap.add_argument("--spikevpr-model", default="brisbane",
                    help="SpikeVPR checkpoint stem; never defaulted upstream because the "
                         "three differ, so it is stated explicitly here")
    ap.add_argument("--spikevpr-repo", default="./SpikeVPR")
    ap.add_argument("--spikevpr-env", default="./envs/spikevpr")
    ap.add_argument("--spikevpr-max-events", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4,
                    help="DataLoader workers for extraction. Each opens its own reader on "
                         "the same multi-GB partition on a spinning disk, so more is not "
                         "better: measured 32 / 52 / 60 / 46 slices/s at 1 / 2 / 4 / 8 "
                         "workers on this box")
    cli = ap.parse_args()
    inf.NUM_WORKERS = cli.workers      # read by extract_descriptors when it builds its loader

    sweeps = [s for s in cli.sweeps.split(",") if s]
    db_sids = [s for s in cli.db_sessions.split(",") if s]
    unknown = [s for s in db_sids if s not in DB_ARMS]
    if unknown:
        raise SystemExit(f"unknown database session(s) {unknown}; known: {list(DB_ARMS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = BUILDERS[cli.method](cli, device)
    cfg, tag = method["cfg"], method["tag"]
    cli.eval_resolution = int(cfg.H)          # partition_manifest records it
    cli.ckpt = tag                            # no checkpoint file; the tag is the identity
    filter_tag = "baoff" if cli.no_event_filter else f"ba{cli.event_filter_dt_us // 1000}"
    cur_masks, cur_meta = load_curation(cli.curation)
    out_tag = f"{tag}_{filter_tag}" + (f"_cur{cur_meta['profile']}" if cur_meta else "")
    print(f"database: {len(db_sids)} sessions   sweeps: {sweeps}   "
          f"{cli.threshold_m:g} m radius, dt {cli.dt_ms} ms, filter {filter_tag}")

    # ---- pass 1: geometry only, no model ---------------------------------------------
    db_sessions, q_sessions = {}, {}
    for sid in db_sids:
        print(f"[db/{DB_ARMS[sid][0]}] {sid}")
        db_sessions[sid] = session_geometry(cli, method,
                                            os.path.join(cli.root, "database", sid), sid,
                                            limit=cli.limit_partitions)
        apply_curation(db_sessions[sid], sid, cur_masks)

    only = set(cli.query_sessions.split(",")) if cli.query_sessions else None
    queries = discover_queries(cli.root, cli.query_limit, sweeps, only)
    if not queries:
        raise SystemExit("no query sessions selected")
    for sweep, sid, sdir in queries:
        q_sessions[sid] = session_geometry(cli, method, sdir, sid)
        apply_curation(q_sessions[sid], sid, cur_masks)
        q_sessions[sid]["sweep"] = sweep

    anchor = db_sessions[db_sids[0]]
    fwd_kept = anchor["latlon"][anchor["keep"]]
    lat0, lon0 = float(fwd_kept[:, 0].mean()), float(fwd_kept[:, 1].mean())
    print(f"\nprojection origin ({lat0:.6f}, {lon0:.6f})")
    for s in list(db_sessions.values()) + list(q_sessions.values()):
        s["xy"] = local_enu(s["latlon"][:, 0], s["latlon"][:, 1], lat0, lon0)

    # ---- the selection every later stage is scored on --------------------------------
    db_stride = float(cli.db_stride_m)
    n_db = 0
    for sid in db_sids:
        s = db_sessions[sid]
        s["keep"] = stride_mask(s["xy"], s["keep"], db_stride) if db_stride > 0 \
            else s["keep"]
        s["keep_base"] = s["keep"].copy()
        s["n_kept"] = int(s["keep"].sum())
        n_db += s["n_kept"]
    n_cut = exclude_reversals(cli, q_sessions) if cli.exclude_reversals_deg else 0
    n_q = 0
    for s in q_sessions.values():
        s["keep_base"] = s["keep"].copy()
        s["n_kept"] = int(s["keep"].sum())
        n_q += s["n_kept"]
    print(f"gallery {n_db} rows at {db_stride:g} m stride; queries {n_q} slices"
          + (f" ({n_cut} reversal slices excluded past "
             f"{cli.exclude_reversals_deg:g} deg)" if n_cut else ""))

    # ---- pass 2: descriptors for exactly those rows ----------------------------------
    for n, sid in enumerate(db_sids):
        print(f"\n[db {n + 1}/{len(db_sids)}/{DB_ARMS[sid][0]}] {sid}")
        extract_session(cli, method, device, db_sessions[sid], sid,
                        db_sessions[sid]["keep"])
    for n, (sweep, sid, sdir) in enumerate(queries):
        print(f"\n[query_{sweep} {n + 1}/{len(queries)}] {sid}")
        extract_session(cli, method, device, q_sessions[sid], sid,
                        q_sessions[sid]["keep"])

    if method.get("model") is not None:
        del method["model"]
    torch.cuda.empty_cache()

    cfg_tag = out_tag + (f"_db{('%g' % db_stride).replace('.', 'p')}m" if db_stride > 0
                         else "")
    if cli.exclude_reversals_deg:
        cfg_tag += f"_norev{int(cli.exclude_reversals_deg)}"
    print(f"\n{'=' * 78}\n[{cli.method}] gallery {n_db} rows, {n_q} query slices, "
          f"{cli.threshold_m:g} m\n{'=' * 78}")
    results = score_protocol(cli, tag, cfg_tag, cfg, method["label"], 0, tag, None,
                             cur_meta, db_sessions, q_sessions, db_sids, list(sweeps),
                             device, db_stride, 0.0, free_banks=True)
    results["method"] = method["meta"]
    results["excluded_reversals_deg"] = cli.exclude_reversals_deg or None
    with open(os.path.join(cli.out_dir, f"results_{cfg_tag}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
