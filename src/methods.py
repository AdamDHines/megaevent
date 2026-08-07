"""The VPR methods this benchmark compares, behind one interface.

Every method turns a list of ``.npz`` paths into a descriptor bank and scores two banks
against each other. What differs is the representation each one wants, the network (or
absence of one) that encodes it, and the metric its descriptors live under:

===============  ====================  =========================  ==============
method           representation        encoder                    native metric
===============  ====================  =========================  ==============
``megaevent``    countmask [3,H,W]     DINOv2 ViT-S/14 + SALAD    cosine
``sparse_event`` event count [H,W]     none (150-pixel readout)   L1
``eventvlad``    3 count sub-windows   denoiser + VGG16/NetVLAD   dot product
``eventgem``     MCTS [10,H,W]         SuperEvent + GeM(p=5)      cosine
``spikevpr``     ON/OFF count [2,H,W]  SEW-ResNet34 + MixVPR      cosine
===============  ====================  =========================  ==============

``similarity`` always returns *higher is better*, because that is what
``recallAtK`` and :func:`src.imagevpr.evaluate` consume — sparse_event's L1 distance is
negated on the way out.

A method may also declare two optional extras, both consumed by
:func:`src.imagevpr._score_both` and both used only by eventgem so far: ``rerank`` for a
second scoring pass over each query's shortlist, and ``extra_pca_powers`` for descriptor
spaces beyond the shared whitening every method is reported under.

sparse_event and eventvlad are ports of Event-LAB
(https://github.com/EventLAB-Team/Event-LAB); eventgem is Event-GeM
(https://github.com/AdamDHines/Event-GeM); spikevpr runs the released SpikeVPR checkpoints
out of the vendored ``SpikeVPR/`` clone. See the per-class docstrings for the paper and the
exact source lines each one follows.
"""

import hashlib
import os
import sys

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from src.inference import (
    BATCH_SIZE, CKPT_DIR, NUM_WORKERS,
    eval_transform, extract_descriptors, load_model, sim_matrix,
)
from src.npzdata import load_count, load_count_triplet, load_countmask, load_mcts
from src import spikevpr_bridge

# sparse_event, "How Many Events Do You Need?" (Fischer & Milford, RA-L 2022)
SPARSE_PIXELS = 150             # num_target_pixels, baselines/sparse_event.yaml
SPARSE_RADIUS = 7               # local_suppression_radius
SPARSE_CLIP = 10                # remove_random_bursts threshold
SPARSE_SIZE = 224               # common grid; matches the geometry megaevent sees
# EventVLAD (Lee & Kim, IROS 2021)
EVENTVLAD_DENOISE_SIZE = 256
EVENTVLAD_VGG_SIZE = 224
EVENTVLAD_BATCH = 16            # 2.8 GB peak; 32 would be 4.8 GB and crowd the workers
# MatConvNet's ImageNet BGR/RGB means. The VGG here is a MatConvNet port, so inputs stay
# on the 0-255 scale and are only mean-centred — there is no /255 and no std division.
MATCONVNET_MEAN = (122.7449417, 114.9440994, 101.6417770)
# Event-GeM (Hines et al. 2026)
EVENTGEM_P = 5.0                # --gem-p, the GeM exponent upstream settled on
EVENTGEM_BATCH = 32             # global pass only: 1.8 GB peak, and it saturates the GPU


# ---------------------------------------------------------------------------
# Generic per-image dataset
# ---------------------------------------------------------------------------
class NpzFrameDataset(Dataset):
    """One rendered image per ``.npz``, in the order the paths were given."""

    def __init__(self, paths, render, check=None):
        self.paths = list(paths)
        self.render = render
        # Render one up front so a broken tree fails *here*, with a useful path in the
        # traceback, rather than inside a DataLoader worker.
        probe = self.render(self.paths[0])
        if check is not None:
            check(self.paths[0], probe)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.render(self.paths[i])


def _loader(ds, batch_size):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                      pin_memory=True, drop_last=False)


# ---------------------------------------------------------------------------
# 1. megaevent — this repo's own method
# ---------------------------------------------------------------------------
class MegaEventMethod:
    """DINOv2 ViT-S/14 + SALAD on countmask frames, L2-normalised, cosine.

    A thin wrapper over the path :mod:`src.inference` already implements, so the numbers
    it produces are unchanged by the existence of the baselines.
    """

    name = "megaevent"
    native_metric = "cosine"
    multi_seed = False

    def __init__(self, args, device):
        self.device = device
        explicit = getattr(args, "ckpt", None)
        ckpt = explicit or os.path.join(CKPT_DIR, f"{args.model}.pt")
        self.model, self.cfg, self.step = load_model(ckpt, device)
        label = os.path.splitext(os.path.basename(ckpt))[0]
        checkpoint_sha256 = None
        if explicit:
            with open(ckpt, "rb") as handle:
                checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        digest = checkpoint_sha256[:10] if checkpoint_sha256 else None
        fingerprint = f", sha256 {digest}" if digest else ""
        logger.info(f"{label} (step {self.step}{fingerprint}): vit={self.cfg.vit} "
                    f"agg={self.cfg.aggregator} desc={self.cfg.desc_dim} "
                    f"rep={self.cfg.representation} in={self.cfg.H}x{self.cfg.W}")
        self.tag = (f"{label}_s{self.step}_{digest}_{self.cfg.representation}" if explicit
                    else f"{args.model}_{self.cfg.representation}")
        self.meta = {"model": label, "step": self.step,
                     "representation": self.cfg.representation}
        if explicit:
            self.meta.update({"checkpoint": os.path.abspath(ckpt),
                              "checkpoint_sha256": checkpoint_sha256})

    def descriptors(self, paths, split):
        transform = eval_transform(self.cfg)

        def render(path):
            frame = load_countmask(path)
            x = torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0)
            return transform(x)

        def check(path, frame):
            if frame.ndim != 3 or frame.shape[0] != 3:
                raise ValueError(f"{path} renders {tuple(frame.shape)}; "
                                 f"this model needs a 3-channel frame")

        ds = NpzFrameDataset(paths, render, check)
        return extract_descriptors(self.model, ds, self.device, label=split)

    def similarity(self, ref, qry, device):
        return sim_matrix(ref, qry, device)          # descriptors are L2-normalised


# ---------------------------------------------------------------------------
# 2. sparse_event
# ---------------------------------------------------------------------------
def remove_random_bursts(frame, threshold=SPARSE_CLIP):
    """Saturating clip, upstream's ``utils.remove_random_bursts``.

    Named "burst removal" in the paper, but it is a clip: any pixel that saw more than
    ``threshold`` events in the window is pinned to ``threshold``.
    """
    out = np.array(frame, copy=True)
    out[out > threshold] = threshold
    return out


def adjust_and_normalize_probabilities(event_data, apply_outlier_correction=True):
    """Mean count frame -> sampling PMF, upstream's ``sparse_pixel_utils`` function.

    Pixels far above the mean are *suppressed* rather than favoured: a pixel that fires
    constantly is usually a hot pixel or a specular highlight, not a landmark.
    """
    adjusted = np.copy(event_data)
    if apply_outlier_correction:
        outlier_threshold = event_data.mean() + 2 * event_data.std()
        adjusted[adjusted > outlier_threshold] = 0.01
    return adjusted / adjusted.sum()


def get_random_pixels(num_pixels, im_width, im_height, local_suppression_radius,
                      prob_to_draw_from=None, rng=None):
    """``num_pixels`` (y, x) pairs at least ``local_suppression_radius`` apart.

    Rejection sampling against the saliency PMF, following upstream. Upstream calls
    ``np.random.choice`` with no seed at all, which makes its results irreproducible and
    is why Event-LAB records a recall std of 0.20 across repeat runs; ``rng`` is threaded
    through here so a run can be repeated exactly and so a seed sweep is meaningful.
    """
    rng = np.random.default_rng() if rng is None else rng
    flat = None if prob_to_draw_from is None else np.asarray(prob_to_draw_from).ravel()
    chosen, rejections = [], 0
    while len(chosen) < num_pixels:
        idx = rng.choice(im_height * im_width, p=flat)
        candidate = np.unravel_index(idx, (im_height, im_width))
        if chosen and np.min(np.linalg.norm(np.asarray(chosen) - np.asarray(candidate),
                                            axis=1)) <= local_suppression_radius:
            rejections += 1
            if rejections >= 100:
                raise RuntimeError(
                    f"could not place {num_pixels} pixels at least "
                    f"{local_suppression_radius} px apart on a {im_height}x{im_width} "
                    f"grid — lower --sparse-pixels or the suppression radius")
            continue
        chosen.append(candidate)
        rejections = 0
    return np.asarray(chosen)


class SparseEventMethod:
    """Event counts read out at 150 saliency-sampled pixels, compared with L1.

    "How Many Events Do You Need? Event-based Visual Place Recognition Using Sparse But
    Varying Pixels", Fischer & Milford, IEEE RA-L 7(4) 2022, doi 10.1109/LRA.2022.3216226.
    Ported from Event-LAB ``baselines/sparse_event.py`` and upstream
    ``sparse_event_vpr/{sparse_pixel_utils,utils}.py``; reimplemented rather than imported
    because the upstream package pulls in tonic, numba, cv2, pynmea2 and pandas at module
    scope for five functions of numpy.

    There are no weights — this is a handcrafted descriptor, not a network.

    Two departures from the published setting, both forced by the data:

    * **No sequence matching.** Upstream convolves the distance matrix with a
      ``seq_length``-long identity kernel, which needs temporally ordered frames. An
      image-retrieval split has no such order, so ``seq_length = 1`` and the convolution
      is the identity.
    * **A common pixel grid.** The method reads out fixed (y, x) coordinates, so frames
      are area-resized to ``SPARSE_SIZE`` square first. Tokyo 24/7's splits are 640x480
      and 480x854, and without this there is no shared coordinate system at all.
    """

    name = "sparse_event"
    native_metric = "l1"
    multi_seed = True           # pixel selection is stochastic; scored over several seeds

    def __init__(self, args, device):
        self.device = device
        self.size = SPARSE_SIZE
        self.num_pixels = SPARSE_PIXELS
        self.seed = 0                       # replaced per-seed by run()
        self.pixels = None                  # set by fit_pixels() on the database
        self.tag = f"sparse{self.num_pixels}_{self.size}"
        self.meta = {"num_pixels": self.num_pixels, "grid": self.size,
                     "clip": SPARSE_CLIP, "suppression_radius": SPARSE_RADIUS,
                     "seq_length": 1}

    def frames(self, paths, split):
        """``[N, size, size]`` float32 clipped count frames — the cacheable stage.

        Held as float16 by the caller: area resizing makes the counts fractional, so
        uint8 would quantise away detail the L1 distance actually uses.
        """
        ds = NpzFrameDataset(
            paths, lambda p: torch.from_numpy(
                load_count(p, size=self.size, clip=SPARSE_CLIP)))
        out = np.empty((len(paths), self.size, self.size), dtype=np.float16)
        at = 0
        from tqdm import tqdm
        with tqdm(total=len(ds), desc=f"{split} count frames", unit="frame",
                  disable=None, leave=False) as bar:
            for batch in _loader(ds, BATCH_SIZE):
                out[at:at + len(batch)] = batch.numpy().astype(np.float16)
                at += len(batch)
                bar.update(len(batch))
        return out

    def fit_pixels(self, db_frames, seed):
        """Choose the 150 readout pixels from the *database* saliency map."""
        self.seed = int(seed)
        mean_frame = db_frames.mean(axis=0, dtype=np.float64)
        pmf = adjust_and_normalize_probabilities(mean_frame)
        self.pixels = get_random_pixels(
            self.num_pixels, im_width=self.size, im_height=self.size,
            local_suppression_radius=SPARSE_RADIUS, prob_to_draw_from=pmf,
            rng=np.random.default_rng(seed))
        return self.pixels

    def encode(self, frames):
        """``[N, size, size]`` -> ``[N, 150]`` float32, unnormalised."""
        if self.pixels is None:
            raise RuntimeError("fit_pixels() must run on the database before encoding")
        y, x = self.pixels[:, 0], self.pixels[:, 1]
        return torch.from_numpy(frames[:, y, x].astype(np.float32))

    def similarity(self, ref, qry, device, chunk=4096):
        """Negated L1 distance, so higher is still better.

        ``torch.cdist(p=1)`` is what upstream's ``get_distance_matrix(metric="cityblock")``
        uses. Descriptors are deliberately *not* normalised — the event rate is part of
        the signal for this method.
        """
        out = np.empty((ref.size(0), qry.size(0)), dtype=np.float32)
        q = qry.to(device)[None]
        for s in range(0, ref.size(0), chunk):
            e = min(s + chunk, ref.size(0))
            d = torch.cdist(ref[s:e].to(device)[None], q, p=1)[0]
            out[s:e] = (-d).float().cpu().numpy()
        del q
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return out


# ---------------------------------------------------------------------------
# 3. EventVLAD
# ---------------------------------------------------------------------------
def _eventvlad_root(args):
    root = os.path.join(args.eventlab_repo, "baselines", "EventVLAD")
    if not os.path.isdir(os.path.join(root, "networks")):
        raise FileNotFoundError(
            f"no EventVLAD checkout at {root}. It is cloned by Event-LAB on first use "
            f"(git clone https://github.com/alexjunholee/EventVLAD.git); point "
            f"--eventlab-repo at an Event-LAB tree that has run the eventvlad baseline.")
    return root


class EventVLADMethod:
    """Denoiser-reconstructed edges through VGG16 + NetVLAD.

    "EventVLAD: Visual Place Recognition with Reconstructed Edges from Event Cameras",
    Lee & Kim, IROS 2021. The networks and both checkpoints are Event-LAB's own
    (``baselines/EventVLAD/{denoiser_brisbane,vgg16_eventvlad.tar}``), imported by
    ``sys.path`` injection the way :mod:`src.model` already reaches dinov2 — so the model
    is identical to Event-LAB's by construction and only the data path is ours.

    Worth knowing before reading the numbers: the descriptor is **not** L2-normalised
    (upstream's final ``F.normalize`` is commented out) and its norm is constant to ~10
    significant figures across unrelated inputs, because NetVLAD intra-normalises each
    cluster and ``lastfc`` collapses 64 of them with fixed weights. The descriptors
    occupy a very tight cone, and low recall is the expected behaviour rather than
    evidence of a wiring fault — Event-LAB's own brisbane runs score R@1 0.09-0.44.
    """

    name = "eventvlad"
    native_metric = "dot"
    multi_seed = False

    def __init__(self, args, device):
        self.device = device
        root = _eventvlad_root(args)
        if root not in sys.path:
            sys.path.insert(0, root)
        from networks.EventDenoiser import EventDenoiser
        from networks.netvlad import EmbedNet, NetVLAD
        from networks.vgg16 import Imagenet_vgg

        denoiser_path = os.path.join(root, "denoiser_brisbane")
        encoder_path = os.path.join(root, "vgg16_eventvlad.tar")
        for path in (denoiser_path, encoder_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"missing EventVLAD weights at {path} — Event-LAB downloads them "
                    f"from Google Drive on first use of the eventvlad baseline")

        self.denoiser = EventDenoiser(input_images=3, dep_S=5, dep_U=5, slope=0.2)
        state = torch.load(denoiser_path, map_location="cpu", weights_only=False)
        # the checkpoint was saved from a DataParallel wrapper
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = self.denoiser.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"denoiser load mismatch: {len(missing)} missing, "
                               f"{len(unexpected)} unexpected")
        self.denoiser = self.denoiser.eval().to(device).requires_grad_(False)

        ck = torch.load(encoder_path, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        net = EmbedNet(Imagenet_vgg(), NetVLAD(num_clusters=64, dim=1000))
        remapped = {}
        for k, v in sd.items():
            if k.startswith("encoder."):
                remapped["base_model." + k[len("encoder."):]] = v
            elif k.startswith("pool."):
                remapped["net_vlad." + k[len("pool."):]] = v
        # upstream trained the cluster assignment conv without a bias; NetVLAD declares
        # one, so it is synthesised as zeros to keep the load exact
        remapped.setdefault("net_vlad.conv.bias", torch.zeros(64))
        missing, unexpected = net.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"NetVLAD load mismatch: {missing} / {unexpected}")
        self.encoder = net.eval().to(device).requires_grad_(False)

        self.mean = torch.tensor(MATCONVNET_MEAN, device=device).view(1, 3, 1, 1)
        self.tag = "eventvlad"
        self.meta = {"denoiser": os.path.basename(denoiser_path),
                     "encoder": os.path.basename(encoder_path),
                     "epoch": int(ck.get("epoch", -1)), "bins": 3}
        logger.info(f"EventVLAD: denoiser + VGG16/NetVLAD(K=64, D=1000) from {root} "
                    f"(encoder epoch {self.meta['epoch']})")

    @torch.no_grad()
    def encode(self, x):
        x = x.to(self.device, non_blocking=True)
        # denoiser channel 0 is the reconstruction; channel 1 is its error
        # estimate and is discarded, as upstream does
        gray = self.denoiser(x)[:, 0:1].clamp(0.0, 1.0) * 255.0
        rgb = gray.repeat(1, 3, 1, 1)
        rgb = torch.nn.functional.interpolate(
            rgb, (EVENTVLAD_VGG_SIZE, EVENTVLAD_VGG_SIZE), mode="area")
        return self.encoder(rgb - self.mean).reshape(x.size(0), -1).float()

    @torch.no_grad()
    def descriptors(self, paths, split):
        from tqdm import tqdm

        ds = NpzFrameDataset(
            paths, lambda p: torch.from_numpy(
                load_count_triplet(p, size=EVENTVLAD_DENOISE_SIZE, bins=3)))
        out = []
        with tqdm(total=len(ds), desc=split, unit="frame", disable=None, leave=False) as bar:
            for batch in _loader(ds, EVENTVLAD_BATCH):
                out.append(self.encode(batch).cpu())
                bar.update(batch.size(0))
        return torch.cat(out)

    def similarity(self, ref, qry, device):
        """Raw inner product.

        Rank-identical to Event-LAB's ``D = (1 - q @ r.T).T`` (``eventvlad.py:208``),
        just kept in the higher-is-better convention the metric expects.
        """
        return sim_matrix(ref, qry, device)         # a plain dot product; see docstring


# ---------------------------------------------------------------------------
# 4. Event-GeM
# ---------------------------------------------------------------------------
class EventGeMMethod:
    """GeM-pooled SuperEvent features, re-ranked by homography on SuperEvent keypoints.

    "EventGeM: Global-to-Local Feature Matching for Event-Based Visual Place Recognition",
    Hines, Nair, Marticorena, Milford & Fischer, arXiv:2603.05807. This follows Event-GeM's
    ``main`` branch rather than the arXiv text: commit 98dce98 replaced the published ECDPT ViT
    global stage with a GeM pooling of SuperEvent's own pre-head feature map, which the repo
    measures as the stronger of the two and which needs one 6.3 MB checkpoint instead of two
    networks. Local features are SuperEvent either way ("SuperEvent: Cross-Modal Learning of
    Event-based Keypoint Detection", Gehrig et al., ICCV 2025).

    The only method here with a second scoring pass. ``descriptors``/``similarity`` are the
    global stage on its own — directly comparable to the other three — and ``rerank`` is the
    geometric verification that is the point of the paper, reported alongside as a ``+rerank``
    twin of every space.

    The model, the crop geometry, the keypoint budget and the re-ranking are all upstream's;
    see :mod:`src.eventgemlocal` for the two places where an image-retrieval split forced a
    departure. What is decided here is the input grid: Tokyo 24/7's splits are 640x480 and
    480x854, so as with sparse_event they have to meet on a common one. 240x320 is SuperEvent's
    own geometry (a cropped DAVIS346) and divides the 40-pixel constraint its backbone imposes,
    so nothing is cropped away. Meeting there squashes the two splits by different amounts,
    which costs the global stage something but not the second one: an anisotropic rescale is an
    affine map, and an affine map is a homography, so whatever homography related the two views
    before is still a homography afterwards.
    """

    name = "eventgem"
    native_metric = "cosine"
    multi_seed = False
    # Upstream whitens fully (`fit_whitening`, feature_extraction.py:71) and measures it as
    # worth ~10 points of R@50, where this repo's shared whitening uses PCA_POWER = 0.5.
    # Reporting only the shared one would understate the method, so its own is reported too.
    extra_pca_powers = (1.0,)

    def __init__(self, args, device):
        from src import eventgemlocal as egl

        self.device = device
        # Each extra whitening power costs a full re-rank pass, and on a large gallery that
        # means its own keypoint store: pca1's shortlists cover the whole pitts250k database,
        # so its store is 13.9 GB and thrashes a 31 GB box. --no-extra-pca drops those spaces
        # and reports only the shared PCA_POWER.
        if getattr(args, "no_extra_pca", False):
            self.extra_pca_powers = ()
        egl.add_to_path(args)
        self.model, self.cfg, self.fast_nms = egl.build_superevent(
            egl.superevent_root(args), device)
        self.size = tuple(args.eventgem_size)
        multiple = egl.input_multiple(self.cfg)
        if any(s % multiple for s in self.size):
            logger.warning(f"--eventgem-size {self.size[0]}x{self.size[1]} is not a multiple "
                           f"of {multiple}; it will be centre-cropped to "
                           f"{egl.crop_offsets(*self.size, multiple)[2:]} the way upstream "
                           f"crops a DAVIS346")
        self.gem_p = EVENTGEM_P
        self.tag = f"eventgem_se_{self.size[0]}x{self.size[1]}"
        self.meta = {"backbone": "superevent", "grid": list(self.size), "gem_p": self.gem_p,
                     "keypoints_per_frame": egl.keypoint_budget(*self.size, multiple),
                     "top_k": args.eventgem_top_k, "ransac_thresh": args.ransac_thresh,
                     "inlier_weight": args.inlier_weight, "match_filter": args.match_filter,
                     "local_dim": int(self.cfg["descriptor_size"])}
        self._local = egl.LocalReranker(
            self.model, self.cfg, self.fast_nms, device, self.size,
            cache_dir=os.path.join(args.feature_dir, args.dataset, "keypoints"),
            # The event source belongs in the tag, not just in the artifact names: the two
            # arms of the real-vs-I2E ablation hold identical basenames, so the store's
            # manifest check cannot tell them apart on its own.
            tag=self.tag + (f"_{args.source}" if getattr(args, "source", None) else "")
                + (f"_limit{args.limit}" if getattr(args, "limit", None) else ""),
            top_k=args.eventgem_top_k, ransac_thresh=args.ransac_thresh,
            inlier_weight=args.inlier_weight, match_filter=args.match_filter,
            match_ratio=args.match_ratio)

    @torch.no_grad()
    def descriptors(self, paths, split):
        """``[N, 128]`` L2-normalised GeM descriptors over SuperEvent's FPN map.

        Only the trunk runs here. ``SuperEvent.forward`` would also evaluate the detector and
        descriptor heads, and the descriptor head alone materialises a ``[B, 256, H, W]`` map
        that costs more than everything else put together — 76k images do not need it, only the
        few thousand that reach a shortlist do.

        :func:`src.inference.extract_descriptors` is not reused for the same reason: it calls
        ``model(x)``.
        """
        from tqdm import tqdm

        from src import eventgemlocal as egl

        off_top, off_left, hc, wc = egl.crop_offsets(*self.size, egl.input_multiple(self.cfg))

        def render(path):
            return torch.from_numpy(load_mcts(path, size=self.size))

        def check(path, frame):
            if frame.ndim != 3 or frame.shape[0] != int(self.cfg["input_channels"]):
                raise ValueError(f"{path} renders {tuple(frame.shape)}; SuperEvent needs a "
                                 f"{self.cfg['input_channels']}-channel MCTS frame")

        ds = NpzFrameDataset(paths, render, check)
        out = []
        with tqdm(total=len(ds), desc=split, unit="frame", disable=None, leave=False) as bar:
            for batch in _loader(ds, EVENTGEM_BATCH):
                x = batch.to(self.device, non_blocking=True)
                if (hc, wc) != self.size:
                    x = x[:, :, off_top:off_top + hc, off_left:off_left + wc]
                f = self.model.fpn(self.model.backbone(x)).float()
                # GeM, feature_extraction.py:68. p=5 is a near-max pool, so the clamp is what
                # keeps the backward-compatible zero features from collapsing the root.
                pooled = torch.nn.functional.avg_pool2d(
                    f.clamp(min=1e-6).pow(self.gem_p), (f.shape[-2], f.shape[-1])
                ).pow(1.0 / self.gem_p)
                out.append(torch.nn.functional.normalize(
                    pooled.squeeze(-1).squeeze(-1), p=2, dim=1).cpu())
                bar.update(x.size(0))
        return torch.cat(out)

    def similarity(self, ref, qry, device):
        return sim_matrix(ref, qry, device)          # descriptors are L2-normalised

    def rerank(self, sim, db_paths, q_paths, label=""):
        return self._local.rerank(sim, db_paths, q_paths, label=label)


# ---------------------------------------------------------------------------
# 5. SpikeVPR
# ---------------------------------------------------------------------------
class SpikeVPRMethod:
    """A spiking SEW-ResNet34 + MixVPR head on ON/OFF count frames, L2-normalised, cosine.

    "Event-Driven Neuromorphic Vision Enables Energy-Efficient Visual Place Recognition",
    Keime, Cuperlier & Cottereau, arXiv:2604.03277. The model and its three released
    checkpoints come from the vendored ``SpikeVPR/`` clone unmodified; what is decided here
    is the input, and there are two things to know about it.

    **The frame is raw counts.** The network has no input normalisation — a ``BatchNorm2d``
    on frozen training statistics is the first thing the events meet — so the event *rate*
    is part of the model's contract in a way it is not for the other four methods. Against
    the 0.167 events/px SpikeVPR's Brisbane checkpoint trained on, the measured medians on
    this grid are 0.278 for a Brisbane 50 ms slice and 1.878 for an NSAVP one (both within
    ~1.7x of their own checkpoint's window), 0.621 for NYC, ~2.1-4.2 for MSLS and Pitts, and
    7.2-8.4 for Tokyo 24/7, whose I2E saccades pack ~600k events into 29 ms. The protocol is
    unchanged either way — SpikeVPR sees the same events every other method sees — but that
    is what ``--spikevpr-max-events`` exists for, and why ``scripts/spikevpr_health.py``
    should confirm a bank has not collapsed to a near-constant descriptor before its recall
    is believed.

    **The grid is not a choice.** ``spikevpr.models.factory`` hardcodes MixVPR for the
    (512, 9, 11) feature map a (2, 260, 346) input produces, so every dataset is resized to
    260x346 — in the event domain, as :func:`src.npzdata.onoff_from_stream` explains. Tokyo
    24/7's 640x480 and 480x854 splits therefore meet on it having been squashed by different
    amounts, the same compromise sparse_event and eventgem already make.

    The forward pass itself runs in ``envs/spikevpr`` rather than here; see
    :mod:`src.spikevpr_bridge` for why and how.
    """

    name = "spikevpr"
    native_metric = "cosine"
    multi_seed = False

    def __init__(self, args, device):
        self.device = device
        model = args.spikevpr_model
        repo = args.spikevpr_repo
        checkpoint, neuron = spikevpr_bridge.resolve_checkpoint(model, repo)
        digest = spikevpr_bridge.checkpoint_sha256(checkpoint)

        self.checkpoint, self.neuron, self.model = checkpoint, neuron, model
        self.repo, self.env_dir = repo, args.spikevpr_env
        self.max_events = args.spikevpr_max_events
        self.out_dir = os.path.join(args.feature_dir, args.dataset)
        self.suffix = f"_limit{args.limit}" if getattr(args, "limit", None) else ""

        # The checkpoint stem alone would be enough to tell the three apart, but the digest
        # follows MegaEventMethod's precedent: it is what makes a bank traceable to the
        # bytes that produced it rather than to a filename that could be replaced.
        cap = f"_e{self.max_events}" if self.max_events else ""
        self.tag = f"spikevpr_r34_{model}_{digest[:10]}{cap}"
        self.meta = {"checkpoint": checkpoint, "checkpoint_sha256": digest,
                     "trained_on": model, "neuron": neuron,
                     "encoder": spikevpr_bridge.ENCODER,
                     "descriptor_dim": spikevpr_bridge.OUT_CHANNELS * spikevpr_bridge.OUT_ROWS,
                     "grid": list(spikevpr_bridge.GRID),
                     "event_window": ("full stream" if not self.max_events
                                      else f"first {self.max_events} events")}
        logger.info(f"SpikeVPR {spikevpr_bridge.ENCODER} trained on {model} "
                    f"(sha256 {digest[:10]}, MixVPR {neuron}): "
                    f"{self.meta['descriptor_dim']}-d, in={spikevpr_bridge.GRID[0]}x"
                    f"{spikevpr_bridge.GRID[1]}, {self.meta['event_window']}")

    def descriptors(self, paths, split):
        out = os.path.join(self.out_dir, f"{self.tag}_{split}{self.suffix}_bank.npy")
        job = spikevpr_bridge.npz_job(
            paths, checkpoint=self.checkpoint, neuron=self.neuron, out=out,
            max_events=self.max_events, spikevpr_repo=self.repo, label=split)
        return torch.from_numpy(spikevpr_bridge.run(job, self.env_dir))

    def similarity(self, ref, qry, device):
        return sim_matrix(ref, qry, device)          # the MixVPR head L2-normalises


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
METHODS = {
    "megaevent": MegaEventMethod,
    "sparse_event": SparseEventMethod,
    "eventvlad": EventVLADMethod,
    "eventgem": EventGeMMethod,
    "spikevpr": SpikeVPRMethod,
}


def get_method(name, args, device):
    if name not in METHODS:
        raise ValueError(f"unknown method '{name}'; choose from {sorted(METHODS)}")
    return METHODS[name](args, device)
