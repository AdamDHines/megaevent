"""The VPR methods this benchmark compares, behind one interface.

Every method turns a list of ``.npz`` paths into a descriptor bank and scores two banks
against each other. What differs is the representation each one wants, the network (or
absence of one) that encodes it, and the metric its descriptors live under:

===============  ====================  =========================  ==============
method           representation        encoder                    native metric
===============  ====================  =========================  ==============
``megaevent``    countmask [3,H,W]     DINOv2 ViT-S/14 + SALAD    cosine
``megaloc``      countmask [3,H,W]     MegaLoc, RGB-trained       cosine
``salad``        countmask [3,H,W]     DINOv2-SALAD, RGB-trained  cosine
``mixvpr``       countmask [3,H,W]     ResNet50 + MixVPR, RGB     cosine
``cricavpr``     countmask [3,H,W]     CricaVPR, RGB-trained      cosine
``sparse_event`` event count [H,W]     none (150-pixel readout)   L1
``eventvlad``    3 count sub-windows   denoiser + VGG16/NetVLAD   dot product
``eventgem``     MCTS [10,H,W]         SuperEvent + GeM(p=5)      cosine
``spikevpr``     ON/OFF count [2,H,W]  SEW-ResNet34 + MixVPR      cosine
``lens``         ON/OFF count [2,128²] LENSv2 spiking conv net    cosine
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
out of the vendored ``SpikeVPR/`` clone; lens runs LENS v2
(https://github.com/AdamDHines/LENSV2) out of its own checkout. See the per-class
docstrings for the paper and the exact source lines each one follows.
"""

import hashlib
import os
import sys

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.inference import (
    BATCH_SIZE, CKPT_DIR, NUM_WORKERS,
    eval_transform, extract_descriptors, load_model, sim_matrix,
)
from src import vprbench
from src.npzdata import (load_accumulate, load_count, load_count_triplet, load_countmask,
                         load_mcts)

# npz -> frame renderers for the image-set path, keyed by the representation the
# checkpoint was trained on (cfg.representation). The traverse path has its own dispatch
# (src.inference._LOCAL_RENDERERS); both must cover any representation a shipped or
# candidate checkpoint carries.
_NPZ_RENDERERS = {"countmask": load_countmask, "accumulate": load_accumulate}
from src import lens_bridge, spikevpr_bridge

# MegaLoc (Berton & Masone 2025), loaded from its authors' torch.hub entry point
MEGALOC_HUB = "gmberton/MegaLoc"
MEGALOC_DESC_DIM = 8448
MEGALOC_RESOLUTION = 322        # MegaLoc's own evaluation size, and SALAD's
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# The countmask training statistics carried by every shipped megaevent checkpoint
# (tencode_mean/std in ckpts/*.pt, computed by gept's --compute-stats over the I2E training
# tree). ImageNet stats centre a countmask frame ~2 sigma off; the fairness arm re-runs an
# RGB control under these instead, so "was the ImageNet normalisation handicapping the
# controls" is a measurement rather than a debate. See megaloc_transform(stats=...).
COUNTMASK_MEAN = (0.0975, 0.2804, 0.0977)
COUNTMASK_STD = (0.1866, 0.4155, 0.1867)
# The same quantity for `accumulate`, read from the v8 checkpoint that trains on it
# (v8_bench/ckpts/b_v8_accum_s750.pt, config.tencode_mean/std — same gept --compute-stats
# provenance as the countmask pair above). accumulate is a WHITE-background render, so
# ImageNet stats sit 2.0-4.5 sigma off here (R 4.47, G 2.00, B 3.71) and overscale by up to
# 2.2x — a worse mismatch than the ~2 sigma countmask case that cost MegaLoc 0.031 R@1.
ACCUMULATE_MEAN = (0.952720206155105, 0.8816854731414574, 0.9289652490639286)
ACCUMULATE_STD = (0.10467739838289758, 0.2129141097718445, 0.14082843440174486)
# DINOv2-SALAD (Izquierdo & Civera, CVPR 2024) — megaevent's own architecture, RGB weights
SALAD_HUB = "serizba/salad"
SALAD_DIRNAME = "serizba_salad_main"        # torch.hub's checkout name for SALAD_HUB
SALAD_CKPT = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
SALAD_ARCH = "dinov2_vitb14"
SALAD_DESC_DIM = 64 * 128 + 256             # clusters*cluster_dim + token = 8448
# MixVPR (Ali-bey et al., WACV 2023). The architecture is VPR-methods-evaluation's copy of
# upstream's; the checkpoint is the authors' released 4096-d one, fetched once from their
# Google Drive and parked beside the datasets rather than under the repo (which lives on a
# root filesystem with ~29 GB free).
MIXVPR_SRC = "/home/adam/repo/VPR-methods-evaluation/vpr_models/mixvpr.py"
MIXVPR_CKPT = ("/media/adam/vprdatasets/megaevent/baseline_weights/mixvpr/"
               "resnet50_MixVPR_4096_channels(1024)_rows(4)")
MIXVPR_DESC_DIM = 4096          # out_channels 1024 x out_rows 4, the released configuration
MIXVPR_RESOLUTION = 320         # MixVPRModel.forward resizes to 320x320 itself
# CricaVPR (Lu et al., CVPR 2024)
CRICAVPR_HUB = "Lu-Feng/CricaVPR"
CRICAVPR_DESC_DIM = 14 * 768    # 14 region tokens x DINOv2 ViT-B width
CRICAVPR_RESOLUTION = 224       # forced by the hardcoded 16x16 patch-grid slicing; see below
CRICAVPR_BATCH = 16             # upstream's --infer_batch_size, and part of the method
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
        # DINOv2 interpolates its position embeddings, so a checkpoint trained at 224 can be
        # evaluated at 322 — which is what every traverse result here uses (see
        # scripts/tokyo_trajectory.py and brisbane_pooled.py). Setting it on the image path
        # too is what lets one number be compared against another across all six datasets.
        resolution = getattr(args, "eval_resolution", None)
        if resolution:
            self.cfg.H = self.cfg.W = resolution
        # Memory only: the bank is the same at any batch size, so this stays out of the tag.
        self.batch_size = getattr(args, "batch_size", None) or BATCH_SIZE
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
        # The resolution is part of the tag only when it was asked for, so the banks and
        # results already on disk at the checkpoint's own 224 keep their names.
        stamp = f"_r{resolution}" if resolution else ""
        self.tag = (f"{label}_s{self.step}_{digest}_{self.cfg.representation}{stamp}" if explicit
                    else f"{args.model}_{self.cfg.representation}{stamp}")
        self.meta = {"model": label, "step": self.step,
                     "representation": self.cfg.representation,
                     "resolution": int(self.cfg.H)}
        if explicit:
            self.meta.update({"checkpoint": os.path.abspath(ckpt),
                              "checkpoint_sha256": checkpoint_sha256})

    def descriptors(self, paths, split):
        transform = eval_transform(self.cfg)
        try:
            load = _NPZ_RENDERERS[self.cfg.representation]
        except KeyError:
            raise ValueError(
                f"no npz renderer for representation {self.cfg.representation!r} on the "
                f"image-set path — add it to methods._NPZ_RENDERERS") from None

        def render(path):
            frame = load(path)
            x = torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0)
            return transform(x)

        def check(path, frame):
            if frame.ndim != 3 or frame.shape[0] != 3:
                raise ValueError(f"{path} renders {tuple(frame.shape)}; "
                                 f"this model needs a 3-channel frame")

        ds = NpzFrameDataset(paths, render, check)
        return extract_descriptors(self.model, ds, self.device, label=split,
                                   batch_size=self.batch_size)

    def similarity(self, ref, qry, device):
        return sim_matrix(ref, qry, device)          # descriptors are L2-normalised


# ---------------------------------------------------------------------------
# 1b. megaloc — the RGB state of the art, unretrained, on the same frames
# ---------------------------------------------------------------------------
_NORM_STATS = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "countmask": (COUNTMASK_MEAN, COUNTMASK_STD),
    "accumulate": (ACCUMULATE_MEAN, ACCUMULATE_STD),
}


def megaloc_transform(resolution, stats="imagenet"):
    """countmask ``[3,H,W]`` already scaled to [0,1] -> the tensor MegaLoc expects.

    ImageNet statistics, then resize — the constants and the order
    ``VPR-methods-evaluation``'s own event path uses, so a bank built here reproduces one
    built there. The resize is bilinear+antialias rather than megaevent's bicubic for the
    same reason: each model keeps the preprocessing its own harness ships with, and the
    interpolation kernel is not what this comparison is about.

    ``stats="countmask"`` / ``stats="accumulate"`` swap in the event-training statistics
    megaevent itself uses for that representation — the fairness arm: banks built under
    either must carry a distinct tag, never the default one. Pick the pair that matches the
    representation being rendered; countmask stats on an accumulate frame are worse than
    ImageNet, not better (black-bg constants on a white-bg render).
    """
    try:
        mean, std = _NORM_STATS[stats]
    except KeyError:
        raise ValueError(f"unknown norm stats {stats!r}; "
                         f"expected one of {sorted(_NORM_STATS)}") from None
    return transforms.Compose([
        transforms.Normalize(mean, std),
        transforms.Resize((resolution, resolution), antialias=True),
    ])


class MegaLocMethod:
    """MegaLoc on countmask frames: an RGB-trained retrieval model, not retrained.

    "MegaLoc: One Retrieval to Place Them All" (Berton & Masone, CVPR workshops 2025),
    loaded unmodified from the authors' ``torch.hub`` entry point. It is the only method
    here that has never seen an event: it was trained on street-view, aerial and indoor
    *photographs*, and the frames it is given are the same countmask renders megaevent is
    given, at the same size and under the same ground truth.

    That makes it the control this benchmark otherwise lacks. Every other baseline is an
    event method, so a win over them says megaevent is the better event method; it does
    not say that training on events was worth doing at all. A countmask frame is still an
    image — edges on a black background — and a large RGB model may simply read it. What
    MegaLoc scores here is therefore the floor that fine-tuning has to clear.

    Two things to keep in front of any comparison against it. It is **228.6M parameters
    against megaevent ViT-B's 88.0M** — not on the backbone, which is the same DINOv2 ViT-B
    at 86.6M in both, but entirely in the head: SALAD widened to cluster_dim 256 and then
    compressed by a 140.6M-parameter ``Linear(16640 -> 8448)``. For the comparison that
    holds capacity fixed see :class:`SaladMethod`, which is megaevent's architecture exactly.
    And the datasets differ in how much of a photograph survives in their events: Tokyo
    24/7, Pitts250k and MSLS are I2E simulations of RGB frames, where Brisbane-Event, NSAVP
    and NYC-Event are real DAVIS/Prophesee recordings.
    """

    name = "megaloc"
    native_metric = "cosine"
    multi_seed = False

    def __init__(self, args, device):
        self.device = device
        self.resolution = getattr(args, "eval_resolution", None) or MEGALOC_RESOLUTION
        # a memory knob only for these three: descriptors do not depend on the
        # batch, unlike cricavpr. Honours --batch-size so a shared card can be
        # survived without changing the numbers.
        self.batch_size = getattr(args, "batch_size", None) or BATCH_SIZE
        # trust_repo: the checkout is already in the hub cache, and a prompt would hang a
        # backgrounded run. source="github" keeps it resolving the same way upstream does.
        model = torch.hub.load(MEGALOC_HUB, "get_trained_model", source="github",
                               trust_repo=True)
        self.model = model.eval().to(device)
        params = sum(p.numel() for p in self.model.parameters())
        # countmask unless the caller asks otherwise; the tag carries it so the
        # accumulate banks never collide with the published countmask ones.
        self.representation = getattr(args, "representation", None) or "countmask"
        self.tag = f"megaloc_{self.representation}_r{self.resolution}"
        self.meta = {"model": "megaloc", "hub": MEGALOC_HUB, "representation": self.representation,
                     "resolution": self.resolution, "descriptor_dim": MEGALOC_DESC_DIM,
                     "parameters": int(params), "trained_on": "RGB images, no event data"}
        logger.info(f"MegaLoc (torch.hub {MEGALOC_HUB}, {params / 1e6:.1f}M params): "
                    f"desc={MEGALOC_DESC_DIM} rep={self.representation} "
                    f"in={self.resolution}x{self.resolution}")

    def descriptors(self, paths, split):
        transform = megaloc_transform(self.resolution)

        load = _NPZ_RENDERERS[self.representation]

        def render(path):
            frame = load(path)
            x = torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0)
            return transform(x)

        def check(path, frame):
            if frame.ndim != 3 or frame.shape[0] != 3:
                raise ValueError(f"{path} renders {tuple(frame.shape)}; "
                                 f"this model needs a 3-channel frame")

        ds = NpzFrameDataset(paths, render, check)
        return extract_descriptors(self.model, ds, self.device, label=split,
                                   batch_size=self.batch_size, oom_backoff=True)

    def similarity(self, ref, qry, device):
        return sim_matrix(ref, qry, device)          # MegaLoc L2-normalises its output


# ---------------------------------------------------------------------------
# 1c. salad — megaevent's own architecture, RGB-trained
# ---------------------------------------------------------------------------
def build_dino_salad(arch=SALAD_ARCH):
    """The released DINOv2-SALAD, built without pytorch_lightning.

    ``serizba/salad``'s hubconf routes through ``vpr_model.VPRModel``, a LightningModule
    whose ``__init__`` calls ``save_hyperparameters()`` and builds a loss and a miner —
    pytorch_lightning plus pytorch_metric_learning, neither in this environment and neither
    needed for a forward pass. The backbone and the aggregator are plain ``nn.Module``s and
    the released checkpoint is a raw state_dict keyed ``backbone.*`` / ``aggregator.*``, so
    a two-attribute wrapper takes it ``strict=True``. Everything that computes is upstream's.
    """
    salad_dir = os.path.join(torch.hub.get_dir(), SALAD_DIRNAME)
    if not os.path.isdir(salad_dir):
        raise FileNotFoundError(
            f"{salad_dir} not found. Fetch the checkout once with "
            f"torch.hub.load('{SALAD_HUB}', 'dinov2_salad') on a machine with network "
            f"access, or clone {SALAD_HUB} there.")
    if salad_dir not in sys.path:
        sys.path.insert(0, salad_dir)
    from models.aggregators.salad import SALAD
    from models.backbones.dinov2 import DINOv2, DINOV2_ARCHS

    class DinoSalad(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # hubconf's own defaults. num_trainable_blocks only gates gradients, which
            # never flow here, but it is kept so the module tree matches the checkpoint.
            self.backbone = DINOv2(model_name=arch, num_trainable_blocks=4,
                                   return_token=True, norm_layer=True)
            self.aggregator = SALAD(num_channels=DINOV2_ARCHS[arch], num_clusters=64,
                                    cluster_dim=128, token_dim=256)

        def forward(self, x):
            return self.aggregator(self.backbone(x))

    model = DinoSalad()
    state = torch.hub.load_state_dict_from_url(SALAD_CKPT, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=True)
    return model


class SaladMethod:
    """DINOv2-SALAD on countmask frames: *megaevent's architecture*, RGB weights.

    "Optimal Transport Aggregation for Visual Place Recognition" (Izquierdo & Civera,
    CVPR 2024), the released GSV-Cities checkpoint, run unmodified on the same countmask
    frames under the same ground truth.

    This is the control MegaLoc cannot be. megaevent ViT-B *is* this model — DINOv2 ViT-B/14,
    SALAD with 64 clusters x 128 + a 256-d token, an 8448-d descriptor, evaluated at 322 —
    down to 88.0M parameters against SALAD's 87,991,489. Two things differ, and only two:
    the backbone's starting point (GEPT's event pretraining, not raw DINOv2) and what the
    weights were then trained on (I2E events, not GSV-Cities photographs). So the gap to
    this column is what the event fine-tuning bought, with architecture, capacity,
    descriptor width and input geometry all held fixed.

    MegaLoc answers a different question and both are needed. It is 228.6M parameters —
    the same ViT-B backbone, then SALAD widened to cluster_dim 256 and learnedly compressed
    by a 140.6M-parameter ``Linear(16640 -> 8448)`` — trained on street-view, aerial and
    indoor imagery far beyond GSV-Cities. Beating SALAD says the fine-tuning worked;
    beating MegaLoc says the result stands against the best available RGB model. A gap that
    opens against MegaLoc but not against SALAD is a statement about head capacity and
    training scale, not about events.
    """

    name = "salad"
    native_metric = "cosine"
    multi_seed = False

    def __init__(self, args, device):
        self.device = device
        self.resolution = getattr(args, "eval_resolution", None) or MEGALOC_RESOLUTION
        # a memory knob only for these three: descriptors do not depend on the
        # batch, unlike cricavpr. Honours --batch-size so a shared card can be
        # survived without changing the numbers.
        self.batch_size = getattr(args, "batch_size", None) or BATCH_SIZE
        self.model = build_dino_salad().eval().to(device)
        params = sum(p.numel() for p in self.model.parameters())
        # countmask unless the caller asks otherwise; the tag carries it so the
        # accumulate banks never collide with the published countmask ones.
        self.representation = getattr(args, "representation", None) or "countmask"
        self.tag = f"salad_{self.representation}_r{self.resolution}"
        self.meta = {"model": "salad", "hub": SALAD_HUB, "representation": self.representation,
                     "resolution": self.resolution, "descriptor_dim": SALAD_DESC_DIM,
                     "parameters": int(params), "trained_on": "GSV-Cities RGB, no event data"}
        logger.info(f"DINOv2-SALAD ({SALAD_HUB}, {params / 1e6:.1f}M params): "
                    f"desc={SALAD_DESC_DIM} rep={self.representation} "
                    f"in={self.resolution}x{self.resolution}")

    def descriptors(self, paths, split):
        # SALAD ships the same ImageNet normalisation and 322 square resize MegaLoc does
        # (dataloaders/GSVCitiesDataset.py, eval.py's --image_size 322), so the two RGB
        # controls see byte-identical inputs and differ only in the network.
        transform = megaloc_transform(self.resolution)

        load = _NPZ_RENDERERS[self.representation]

        def render(path):
            frame = load(path)
            x = torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0)
            return transform(x)

        def check(path, frame):
            if frame.ndim != 3 or frame.shape[0] != 3:
                raise ValueError(f"{path} renders {tuple(frame.shape)}; "
                                 f"this model needs a 3-channel frame")

        ds = NpzFrameDataset(paths, render, check)
        return extract_descriptors(self.model, ds, self.device, label=split,
                                   batch_size=self.batch_size, oom_backoff=True)

    def similarity(self, ref, qry, device):
        return sim_matrix(ref, qry, device)          # SALAD's last op is an L2 normalise


# ---------------------------------------------------------------------------
# 1d. mixvpr — a CNN RGB baseline, no transformer anywhere
# ---------------------------------------------------------------------------
def build_mixvpr(desc_dim=MIXVPR_DESC_DIM, ckpt=MIXVPR_CKPT):
    """The released 4096-d MixVPR, built from VPR-methods-evaluation's copy of upstream.

    Loaded by *file path* rather than as ``from vpr_models import mixvpr``:
    ``vpr_models/__init__.py`` eagerly imports apgem, boq, qaa, supervlad and friends, none
    of which are installed here, so the package import fails before reaching the one module
    that is needed. ``mixvpr.py`` itself only wants torch, torchvision and gdown.

    Upstream's ``get_mixvpr`` downloads the checkpoint to a **CWD-relative**
    ``trained_models/mixvpr/``, which would put 44 MB somewhere different for every caller
    and needs gdown (absent from this environment). The architecture is taken from that
    module and the state_dict is loaded here from a fixed path instead. Everything that
    computes is upstream's.
    """
    import importlib.util
    import types

    spec = importlib.util.spec_from_file_location("_vpr_mixvpr", MIXVPR_SRC)
    module = importlib.util.module_from_spec(spec)
    # `import gdown` sits at the top of that file purely for get_mixvpr's downloader, and
    # gdown is not in this environment. Standing in a module whose only attribute raises
    # keeps the import working while making any *actual* download attempt loud rather than
    # silent — the checkpoint has to come from MIXVPR_CKPT.
    if "gdown" not in sys.modules:
        stub = types.ModuleType("gdown")

        def _no_download(*args, **kwargs):
            raise RuntimeError(
                f"gdown is not installed here; MixVPR's checkpoint must already be at "
                f"{MIXVPR_CKPT}")

        stub.download = _no_download
        sys.modules["gdown"] = stub
    spec.loader.exec_module(module)

    url, filename, out_channels, out_rows = module.MODELS_INFO[desc_dim]
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"{ckpt} not found. Fetch it once with "
            f"`gdown {url.split('/d/')[1].split('/')[0]} -O '{filename}'`.")
    # upstream's get_mixvpr() config, verbatim
    model = module.MixVPRModel(agg_config={
        "in_channels": 1024, "in_h": 20, "in_w": 20, "out_channels": out_channels,
        "mix_depth": 4, "mlp_ratio": 1, "out_rows": out_rows,
    })
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model


class MixVPRMethod:
    """MixVPR on countmask frames: the CNN RGB baseline, never retrained on events.

    "MixVPR: Feature Mixing for Visual Place Recognition" (Ali-bey, Chaib-draa & Giguère,
    WACV 2023), the authors' released 4096-d GSV-Cities checkpoint, run unmodified on the
    same countmask frames under the same ground truth as every other column.

    It is the only RGB control here with no transformer in it: a ResNet50 truncated after
    ``layer3`` (``layer4`` and the fc are ``nn.Identity``) feeding a stack of four
    feature-mixer MLPs, 10.9M parameters against SALAD's 88.0M and MegaLoc's 228.6M. So it
    reads the third capacity point, and reads it with a fundamentally different inductive
    bias — where :class:`SaladMethod` and :class:`MegaLocMethod` both sit on DINOv2 ViT-B/14
    and therefore share whatever a patch-token backbone does or does not see in a countmask
    frame, MixVPR shares none of that.

    Geometry is upstream's, not this benchmark's 322: ``MixVPRModel.forward`` begins with
    ``Resize([320, 320])`` because the aggregator's ``in_h``/``in_w`` are fixed at 20, which
    is 320/16 after the ResNet's stride. Passing 322 would silently be resized to 320
    anyway, so 320 is passed explicitly and recorded. Normalisation is ImageNet, shared with
    the other two RGB controls.
    """

    name = "mixvpr"
    native_metric = "cosine"
    multi_seed = False

    def __init__(self, args, device):
        self.device = device
        # Not `args.eval_resolution`: 320 is structural here, not a tuning choice.
        self.resolution = MIXVPR_RESOLUTION
        self.batch_size = getattr(args, "batch_size", None) or BATCH_SIZE
        self.model = build_mixvpr().eval().to(device)
        params = sum(p.numel() for p in self.model.parameters())
        # countmask unless the caller asks otherwise; the tag carries it so the
        # accumulate banks never collide with the published countmask ones.
        self.representation = getattr(args, "representation", None) or "countmask"
        self.tag = f"mixvpr_{self.representation}_r{self.resolution}"
        self.meta = {"model": "mixvpr", "hub": MIXVPR_SRC, "representation": self.representation,
                     "resolution": self.resolution, "descriptor_dim": MIXVPR_DESC_DIM,
                     "parameters": int(params), "checkpoint": MIXVPR_CKPT,
                     "trained_on": "GSV-Cities RGB, no event data"}
        logger.info(f"MixVPR ({os.path.basename(MIXVPR_CKPT)}, {params / 1e6:.1f}M params): "
                    f"desc={MIXVPR_DESC_DIM} rep={self.representation} "
                    f"in={self.resolution}x{self.resolution}")

    def descriptors(self, paths, split):
        transform = megaloc_transform(self.resolution)

        load = _NPZ_RENDERERS[self.representation]

        def render(path):
            frame = load(path)
            x = torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0)
            return transform(x)

        def check(path, frame):
            if frame.ndim != 3 or frame.shape[0] != 3:
                raise ValueError(f"{path} renders {tuple(frame.shape)}; "
                                 f"this model needs a 3-channel frame")

        ds = NpzFrameDataset(paths, render, check)
        return extract_descriptors(self.model, ds, self.device, label=split,
                                   batch_size=self.batch_size, oom_backoff=True)

    def similarity(self, ref, qry, device):
        return sim_matrix(ref, qry, device)          # MixVPR's aggregator ends in F.normalize


# ---------------------------------------------------------------------------
# 1e. cricavpr — the RGB control whose descriptors depend on the batch
# ---------------------------------------------------------------------------
def build_cricavpr(hub=CRICAVPR_HUB):
    """The released CricaVPR, unwrapped from the ``DataParallel`` its hubconf returns.

    ``hubconf.trained_model()`` wraps ``CricaVPRNet`` in ``torch.nn.DataParallel`` because
    that is how the checkpoint's keys are named. On one GPU the wrapper is a no-op that
    still costs a scatter/gather, and — more to the point — DataParallel splits along dim 0,
    which for this network is the *cross-image attention sequence* (see
    :class:`CricaVPRMethod`). Unwrapping to ``.module`` keeps a multi-GPU box from silently
    changing what the method computes.
    """
    model = torch.hub.load(hub, "trained_model", source="github", trust_repo=True)
    return getattr(model, "module", model)


class CricaVPRMethod:
    """CricaVPR on countmask frames: an RGB baseline whose descriptors are batch-dependent.

    "CricaVPR: Cross-image Correlation-aware Representation Learning for Visual Place
    Recognition" (Lu, Lan, Zhang, Dong, Wang & Yuan, CVPR 2024), the authors' released
    checkpoint, run unmodified on the same countmask frames under the same ground truth.
    DINOv2 ViT-B/14 backbone, 14 multi-scale GeM region tokens (1 class + a 2x2 grid + a 3x3
    grid), a 2-layer transformer encoder over them, 10752-d.

    **Two things about it are configuration, not detail.**

    *The input must be exactly 224x224.* ``network.py:57-64`` derives the patch grid as
    ``W = H = int(sqrt(P - 1))`` and then slices fixed regions out of it — ``[0:8, 0:8]``,
    ``[5:11, 5:11]``, ``[11:, 11:]`` and so on. Those indices only partition a 16x16 grid,
    which is 224/14. At this benchmark's usual 322 the grid is 23x23 and the "3x3" regions
    would silently cover 11 of 23 rows twice and the last 12 not at all. So CricaVPR keeps
    its own published geometry, exactly as Event-GeM keeps its 240x320 — what is held fixed
    across the table is the event stream, the countmask render and the ground truth, not the
    resize each published method specifies.

    *Descriptors depend on which other images shared the batch.* The encoder is built with
    ``batch_first=False``, i.e. it reads ``(seq, batch, feature)``, and is then handed a
    ``(B, 14, D)`` tensor — so the attention sequence is **B**, the images. Upstream says so
    in as many words at ``network.py:48-49``: *"Our input tensor is provided as (batch, seq,
    feature), which performs encoding on the 'batch' dimension."* That is the "cross-image"
    in the name and it is the whole method, but it means a descriptor is not a function of
    its image alone.

    Measured here rather than assumed, on one image encoded three ways: against 15
    neighbours vs alone, cosine **0.9905**; against 15 neighbours vs 14 *different*
    neighbours, cosine **0.99998**. So it is the batch *size* that moves a descriptor, and
    which particular frames fill the batch barely does. Consequences:

    * the batch size is pinned to upstream's ``--infer_batch_size`` default of 16, for every
      split and both the image-set and pooled paths, and recorded in ``meta``. Changing it
      changes the numbers, so it is not a memory knob here;
    * the file order fixes the batching, and every path list in this repo is a sorted
      listing or a slice index, so a rerun reproduces a run exactly;
    * ``drop_last=False`` leaves one short final batch per split. Those frames see fewer
      neighbours and shift by roughly the 0.99 cosine above — at most 15 frames of ~450k,
      and it is upstream's behaviour too.
    """

    name = "cricavpr"
    native_metric = "cosine"
    multi_seed = False

    def __init__(self, args, device):
        self.device = device
        # Not `args.eval_resolution` — 224 is structural, see the class docstring.
        self.resolution = CRICAVPR_RESOLUTION
        self.batch_size = CRICAVPR_BATCH
        self.model = build_cricavpr().eval().to(device)
        params = sum(p.numel() for p in self.model.parameters())
        # countmask unless the caller asks otherwise; the tag carries it so the
        # accumulate banks never collide with the published countmask ones.
        self.representation = getattr(args, "representation", None) or "countmask"
        self.tag = f"cricavpr_{self.representation}_r{self.resolution}"
        self.meta = {"model": "cricavpr", "hub": CRICAVPR_HUB, "representation": self.representation,
                     "resolution": self.resolution, "descriptor_dim": CRICAVPR_DESC_DIM,
                     "parameters": int(params), "batch_size": self.batch_size,
                     "cross_image_batch": True,
                     "trained_on": "GSV-Cities RGB, no event data"}
        logger.info(f"CricaVPR (torch.hub {CRICAVPR_HUB}, {params / 1e6:.1f}M params): "
                    f"desc={CRICAVPR_DESC_DIM} rep={self.representation} "
                    f"in={self.resolution}x{self.resolution} batch={self.batch_size} "
                    f"(cross-image attention runs over the batch)")

    def descriptors(self, paths, split):
        transform = megaloc_transform(self.resolution)

        load = _NPZ_RENDERERS[self.representation]

        def render(path):
            frame = load(path)
            x = torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0)
            return transform(x)

        def check(path, frame):
            if frame.ndim != 3 or frame.shape[0] != 3:
                raise ValueError(f"{path} renders {tuple(frame.shape)}; "
                                 f"this model needs a 3-channel frame")

        ds = NpzFrameDataset(paths, render, check)
        # batch_size is part of the method here, not a memory knob — see the class docstring.
        return extract_descriptors(self.model, ds, self.device, label=split,
                                   batch_size=self.batch_size)

    def similarity(self, ref, qry, device):
        return sim_matrix(ref, qry, device)          # forward() ends in F.normalize


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
        # allow_unnormalized: EventVLAD's descriptors are deliberately not unit vectors,
        # so this is the one similarity here that is a dot product rather than a cosine.
        return sim_matrix(ref, qry, device, allow_unnormalized=True)


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


class LensMethod:
    """LENS v2: a 111,694-parameter spiking conv net for a Speck2f, cosine.

    "LENS: Locational Encoding with Neuromorphic Systems" (Hines, Milford & Fischer,
    Science Robotics 2025) in its v2 form — a descriptor network rather than v1's one-hot
    place classifier. Six conv layers, 454 KiB of weights, and a 1024-D descriptor that is
    literally the spike count of a 16x8x8 output layer. It is two to four orders of
    magnitude smaller than everything else in this table, and the comparison is worth
    reading with that in front of it: megaevent's ViT-B is ~86M parameters at 322x322,
    LENS is 0.11M at 128x128.

    Three things about its input, all of which are model contract rather than preference.

    **Raw counts, polarity never merged.** The frame is ``(2, 128, 128)`` ON/OFF counts and
    the first spiking layer is an IAF firing on their absolute magnitude, so the event
    *rate* is part of the contract exactly as it is for SpikeVPR. Unlike SpikeVPR, LENS
    trains on I2E output: ~48 events/px on its own 128x128 grid. Tokyo 24/7 renders 37-45
    there and is therefore **in distribution**, where for SpikeVPR it is ~43x out of it.
    Brisbane is LENS's out-of-distribution end at 3.3 — 15x sparser than training.

    **The rebin is LENS's own.** Reaching 128 through :func:`src.npzdata.load_onoff`'s
    (260, 346) would resample twice and rescale every bin, moving the IAF operating point.
    So this is the one method that does not share ``src/npzdata.py``; see
    :mod:`src.lens_bridge`.

    **int8 is not a tax.** ``--lens-quantise chip`` runs the discretised
    ``DynapcnnNetwork`` that actually deploys, and on Brisbane it *beats* the fp32 weights
    (sunset1 R@1 60.8 -> 67.1) while nearly doubling descriptor spike count. The two are
    different models; whichever produced a bank is recorded in the tag and in ``meta``.

    The forward pass runs in LENS's own pixi environment rather than here, because it needs
    sinabs to build the chip network at all; see :mod:`src.lens_bridge` for the seam.
    """

    name = "lens"
    native_metric = "cosine"
    multi_seed = False

    def __init__(self, args, device):
        self.device = device
        model = args.lens_model
        self.repo = args.lens_repo
        checkpoint = lens_bridge.resolve_checkpoint(model, self.repo)
        digest = lens_bridge.checkpoint_sha256(checkpoint)
        provenance = lens_bridge.describe(checkpoint)

        self.checkpoint = checkpoint
        self.quantise = args.lens_quantise
        self.batch_size = args.lens_batch_size
        self.out_dir = os.path.join(args.feature_dir, args.dataset)
        self.suffix = f"_limit{args.limit}" if getattr(args, "limit", None) else ""

        # The tag has to separate every input that changes a descriptor, because
        # src.scoring's cache is keyed on it: the checkpoint bytes, fp32-vs-int8 (different
        # models), and the batch size (sinabs makes the batch dimension visible, so 64 and
        # 128 are not the same network).
        label = os.path.splitext(os.path.basename(checkpoint))[0]
        self.tag = f"lens_{label}_{digest[:10]}_{self.quantise}_b{self.batch_size}"
        # 16 readout channels -> 1024-D, 64 -> 4096-D; both deploy on the same seven cores.
        self.descriptor_dim = int(provenance.get("descriptor_channels") or 16) * 8 * 8
        self.meta = {"checkpoint": checkpoint, "checkpoint_sha256": digest,
                     "quantise": self.quantise, "batch_size": self.batch_size,
                     "descriptor_dim": self.descriptor_dim,
                     "grid": list(lens_bridge.INPUT_SHAPE[1:]),
                     "parameters": 111694, **provenance}
        logger.info(f"LENS v2 {label} (sha256 {digest[:10]}, step {provenance['step']}, "
                    f"ann={provenance['ann']}, "
                    f"spike_threshold={provenance['spike_threshold']}): "
                    f"{lens_bridge.DESC_DIM}-d {self.quantise}, "
                    f"in={lens_bridge.INPUT_SHAPE[1]}x{lens_bridge.INPUT_SHAPE[2]}, "
                    f"batch {self.batch_size}")

    def descriptors(self, paths, split):
        out = os.path.join(self.out_dir, f"{self.tag}_{split}{self.suffix}_bank.npy")
        job = lens_bridge.npz_job(
            paths, checkpoint=self.checkpoint, out=out, quantise=self.quantise,
            batch_size=self.batch_size, workers=lens_bridge.WORKERS,
            lens_repo=self.repo, label=split, descriptor_dim=self.descriptor_dim)
        return torch.from_numpy(lens_bridge.run(job, self.repo))

    def similarity(self, ref, qry, device):
        return sim_matrix(ref, qry, device)          # descriptor() L2-normalises


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 1f. boq / qaa / supervlad — further RGB controls, borrowed from VPR-methods-evaluation
# ---------------------------------------------------------------------------
class _VPRBenchMethod:
    """Shared body for the controls whose networks live in VPR-methods-evaluation.

    MegaLoc, SALAD, MixVPR and CricaVPR each get their own class because each carries a
    different structural constraint (CricaVPR's batch, MixVPR's 320). These three do not:
    all are DINOv2 backbones evaluated at 322 under ImageNet normalisation, producing an
    already-L2-normalised descriptor, so they differ only in a builder, a width and a name.
    One base with three thin subclasses says that, where three copies would not.

    The networks themselves are loaded out of the VPR-methods-evaluation checkout — see
    :mod:`src.vprbench` for why by path rather than by importing ``vpr_models``.
    """

    native_metric = "cosine"
    multi_seed = False
    desc_dim = None
    resolution = None
    source = None

    def _build(self):
        raise NotImplementedError

    def __init__(self, args, device):
        self.device = device
        self.model = self._build().eval().to(device)
        params = sum(p.numel() for p in self.model.parameters())
        self.batch_size = getattr(args, "batch_size", None) or BATCH_SIZE
        self.representation = getattr(args, "representation", None) or "countmask"
        self.tag = f"{self.name}_{self.representation}_r{self.resolution}"
        self.meta = {"model": self.name, "hub": self.source,
                     "representation": self.representation,
                     "resolution": self.resolution, "descriptor_dim": self.desc_dim,
                     "parameters": int(params),
                     "trained_on": "GSV-Cities RGB, no event data"}
        logger.info(f"{self.name} ({self.source}, {params / 1e6:.1f}M params): "
                    f"desc={self.desc_dim} rep={self.representation} "
                    f"in={self.resolution}x{self.resolution}")

    def descriptors(self, paths, split):
        transform = megaloc_transform(self.resolution)
        load = _NPZ_RENDERERS[self.representation]

        def render(path):
            frame = load(path)
            x = torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0)
            return transform(x)

        def check(path, frame):
            if frame.ndim != 3 or frame.shape[0] != 3:
                raise ValueError(f"{path} renders {tuple(frame.shape)}; "
                                 f"this model needs a 3-channel frame")

        ds = NpzFrameDataset(paths, render, check)
        return extract_descriptors(self.model, ds, self.device, label=split,
                                   batch_size=self.batch_size, oom_backoff=True)

    def similarity(self, ref, qry, device):
        return sim_matrix(ref, qry, device)          # all three end in an L2 normalise


class BoQMethod(_VPRBenchMethod):
    """BoQ on DINOv2 ViT-B — learnable "bag of queries" cross-attention, 12288-d at 322.

    The strongest RGB control here after MegaLoc, and a different aggregation family from
    every other column: instead of pooling patch tokens (SALAD's optimal transport, CricaVPR's
    GeM regions, MixVPR's feature mixing) it cross-attends a fixed set of 64 learned queries
    against them. BoQ also ships a ResNet50 variant at the same aggregator, so if the CNN /
    ViT split on these frames is a backbone story rather than a method story, that pair is
    what would show it.
    """

    name = "boq"
    desc_dim = vprbench.BOQ_DESC_DIM
    resolution = vprbench.BOQ_RESOLUTION
    source = vprbench.BOQ_SRC

    def _build(self):
        return vprbench.build_boq()


class QAAMethod(_VPRBenchMethod):
    """QAA on DINOv2 — 8192-d at 322, the most recent method in the comparison set."""

    name = "qaa"
    desc_dim = vprbench.QAA_DESC_DIM
    resolution = vprbench.QAA_RESOLUTION
    source = vprbench.QAA_SRC

    def _build(self):
        return vprbench.build_qaa(self.desc_dim)


class SuperVLADMethod(_VPRBenchMethod):
    """SuperVLAD on DINOv2 ViT-B — 3072-d at 322.

    The compact point of the ViT ladder: a tenth of MegaLoc's width and a quarter of BoQ's,
    which is what makes it worth reading beside them on frames none of the three trained on.
    This is the plain arm, not ``SuperVLAD-CrossImage`` — the latter's encoder attends across
    the batch the way CricaVPR's does, so it could not share this class's OOM backoff.
    """

    name = "supervlad"
    desc_dim = vprbench.SUPERVLAD_DESC_DIM
    resolution = vprbench.SUPERVLAD_RESOLUTION
    source = vprbench.SUPERVLAD_SRC

    def _build(self):
        return vprbench.build_supervlad("SuperVLAD")


METHODS = {
    "megaevent": MegaEventMethod,
    "megaloc": MegaLocMethod,
    "salad": SaladMethod,
    "mixvpr": MixVPRMethod,
    "cricavpr": CricaVPRMethod,
    "boq": BoQMethod,
    "qaa": QAAMethod,
    "supervlad": SuperVLADMethod,
    "sparse_event": SparseEventMethod,
    "eventvlad": EventVLADMethod,
    "eventgem": EventGeMMethod,
    "spikevpr": SpikeVPRMethod,
    "lens": LensMethod,
}


def get_method(name, args, device):
    if name not in METHODS:
        raise ValueError(f"unknown method '{name}'; choose from {sorted(METHODS)}")
    return METHODS[name](args, device)
