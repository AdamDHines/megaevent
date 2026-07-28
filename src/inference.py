"""Standalone event-VPR inference: one reference traverse vs one query traverse.

    <seq>.hdf5 --eventcv--> dt_ms slices --repr--> [3,H,W] uint8 frames
      --normalise + resize--> ViT-S/14 + SALAD --> L2-normalised descriptor
    cosine similarity against the reference bank --> R@k over the Event-LAB GT band

Driven entirely by ``main.py``'s argparse; this module has no CLI of its own. Frame
rendering is eventcv's job — we only name the representation the checkpoint was trained
on (``cfg.representation``) and hand eventcv any options it needs (see ``_repr_kwargs``).

Two properties of the shipped checkpoints that are easy to get wrong:

* the training step lives at ``ck["meta"]["step"]``, not ``ck["step"]``;
* the saved config carries ``white_frame: true``, but the model was trained on
  *black-background* frames — its normalisation mean [0.14, 0.35, 0.14] is a dark frame
  with a bright green activity mask. The key is ignored here; eventcv's renderer is the
  single source of truth and must produce black-background frames.
"""

import os
import time

import matplotlib
matplotlib.use("Agg")                       # headless: we only ever write PNGs

import eventcv as ecv
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from skimage.transform import resize as sk_resize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.config import VPRConfig
from src.model import VPRModel

# ---------------------------------------------------------------------------
# Tunables that main.py does not expose
# ---------------------------------------------------------------------------
KS = (1, 5, 10, 20)                 # recall cutoffs
BATCH_SIZE = 64
NUM_WORKERS = 8
# PCA-whitening of the descriptors, fit on the reference bank only (so it stays valid
# transductive post-processing — no query ever leaks into the transform). power=0.5 with
# dim ~2048-4096 is the SALAD optimum; 0.25 is the *GeM* optimum and does not transfer.
PCA_DIM, PCA_POWER, PCA_EPS = 2048, 0.5, 1e-4

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "ckpts"))
# Event-LAB's per-dataset spec, for the sensor resolution. Mirrors src/eventlab.py's
# EVENTLAB_ROOT, but resolved locally so this module stays importable on its own.
EVENTLAB_DATASETS = os.path.normpath(os.path.join(THIS_DIR, "..", "external", "eventlab", "datasets"))

def _load_recall_at_k():
    """VPR_Tutorial's ``recallAtK``, loaded straight from its file.

    Deliberately *not* by putting ``external/vprtutorial`` on sys.path, the way
    src/model.py reaches dinov2: vprtutorial ships a ``datasets`` package with an
    ``__init__.py``, and Event-LAB's ``datasets`` has none, so it is a PEP 420 namespace
    package. Python records a namespace portion and keeps scanning the path, then takes a
    regular package if it finds one — so vprtutorial would shadow Event-LAB's ``datasets``
    from *anywhere* on sys.path, breaking ``src/eventlab.py``'s imports. Loading the one
    module by path sidesteps the package machinery entirely; ``metrics.py`` only needs numpy.
    """
    import importlib.util

    path = os.path.normpath(os.path.join(THIS_DIR, "..", "external", "vprtutorial",
                                         "evaluation", "metrics.py"))
    spec = importlib.util.spec_from_file_location("vprtutorial_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.recallAtK


recallAtK = _load_recall_at_k()


# ---------------------------------------------------------------------------
# 1. Event-LAB dataset layout
# ---------------------------------------------------------------------------
def sequence_path(args, seq):
    """The raw recording for one traverse: ``<eventlab-dir>/<dataset>/<seq>/<seq>.hdf5``."""
    path = os.path.join(args.eventlab_dir, args.dataset, seq, f"{seq}.hdf5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no recording for '{seq}' at {path}")
    return path


def ground_truth_path(args):
    """Event-LAB writes ``<dataset>/ground_truth/<ref>_<query>_GT.npy``, stored [ref, query]."""
    path = os.path.join(args.eventlab_dir, args.dataset, "ground_truth",
                        f"{args.ref}_{args.query}_GT.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no ground truth at {path} — rerun the Event-LAB step")
    return path


def sensor_size(dataset):
    """``(W, H)`` from Event-LAB's dataset spec — eventcv's own axis order.

    Read from the spec rather than hardcoded so DAVIS346 (346x260) and DVXplorer
    (nsavp, 640x480) both work without a per-dataset branch here.
    """
    with open(os.path.join(EVENTLAB_DATASETS, f"{dataset}.yaml")) as f:
        spec = yaml.safe_load(f)
    w, h = spec["dataset"]["resolution"]
    return int(w), int(h)


# ---------------------------------------------------------------------------
# 2. Event stream -> model input
# ---------------------------------------------------------------------------
def eval_transform(cfg):
    """uint8 ``[3,H,W]`` frame (already scaled to [0,1]) -> the tensor the model expects.

    ``Normalize`` **then** ``Resize``, matching the order the training pipeline used. The
    scale-by-255 is done by the caller because the input is already a CHW tensor rather
    than a PIL image. A mismatch here silently shifts the whole descriptor space.
    """
    return transforms.Compose([
        transforms.Normalize(cfg.tencode_mean, cfg.tencode_std),
        transforms.Resize((cfg.H, cfg.W), interpolation=transforms.InterpolationMode.BICUBIC),
    ])


def _repr_kwargs(name, dt_ms):
    """Per-representation options for :meth:`eventcv.EventReader.with_repr`.

    The one place representation knowledge lives. eventcv does the rendering; this only
    passes through the options a given representation needs.
    """
    if name == "tencode":
        return {"window_ms": dt_ms}     # must equal dt_ms or the time channel is wrong
    if name == "countmask":
        return {"window_ms": dt_ms, "white_frame": False}  # black background                   
    return {}


class EventStreamDataset(Dataset):
    """Fixed ``dt_ms`` slices of one recording, rendered by eventcv into model inputs.

    The reader is opened lazily and never pickled (see ``__getstate__``): it is not
    fork-safe, so each DataLoader worker has to build its own.
    """

    def __init__(self, dataset, traverse, path, transform, representation, dt_ms, sensor,
                 hot_pixel=True, filter_dt_us=None, no_event_filter=True, event_filter_dt_ms=None):
        self.dataset = dataset
        self.traverse = traverse
        self.path = path
        self.transform = transform
        self.representation = representation
        self.dt_ms = dt_ms
        self.sensor = tuple(sensor)
        self.hot_pixel = bool(hot_pixel)
        self.filter_dt_us = int(filter_dt_us) if filter_dt_us else None
        self.no_event_filter = bool(no_event_filter)
        self.event_filter_dt_ms = int(event_filter_dt_ms) if event_filter_dt_ms else None
        self._reader = None

        # Open once up front to learn the length and to fail loudly *here* rather than
        # inside a worker, where the traceback would be far less useful.
        reader = self._open()
        self._n = int(reader.n_slices)
        frame = np.asarray(reader[0])
        if frame.ndim != 3 or frame.shape[0] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"representation '{representation}' renders {frame.shape} {frame.dtype}; "
                f"this model needs a 3-channel uint8 [3,H,W] frame")
        self._reader = None                 # don't carry the handle across a fork

    def _open(self):
        # Get offsets for datasets with known offsets
        if self.dataset == "brisbane_event":
            # Load the brisbane_event.yaml from eventlab
            with open(os.path.join(EVENTLAB_DATASETS, "brisbane_event.yaml"), 'r') as f:
                spec = yaml.safe_load(f)
                offset = spec["other"]["offset"][self.traverse] * 1000 # for conversion to ms

        else:
            offset = 0

        try:
            reader = ecv.open(self.path, dt_ms=self.dt_ms, sensor_size=self.sensor,
                                  hot_pixel_filter=self.hot_pixel, offset=offset)

            # Run filtering on the event stream
            if not self.no_event_filter:
                if self.event_filter_dt_ms is not None:
                    reader = reader.background_activity_filter(self.event_filter_dt_ms)
                else:
                    reader = reader.background_activity_filter(self.dt_ms)

            return reader.with_repr(self.representation,
                                    **_repr_kwargs(self.representation, self.dt_ms))
        except ValueError as err:
            if "representation" in str(err):
                raise ValueError(
                    f"the installed eventcv cannot render '{self.representation}', which is "
                    f"what this checkpoint was trained on ({err}). Upgrade eventcv to a "
                    f"version providing it — no other change is needed here.") from err
            raise

    @property
    def reader(self):
        if self._reader is None:            # first touch in this process/worker
            self._reader = self._open()
        return self._reader

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        frame = np.asarray(self.reader[i])              # [3,H,W] uint8, rendered by eventcv
        x = torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0)
        return self.transform(x)

    def __getstate__(self):
        return {**self.__dict__, "_reader": None}


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------
# Architecture *and* normalisation stats come from the checkpoint — the stats were baked
# in at training time, so they are part of the model. Paths are not restored.
_RESTORE = ("n_embed", "n_head", "num_register_tokens", "n_layer", "P", "backbone_img_size",
            "H", "W", "n_tokens_per_image", "tencode_mean", "tencode_std", "aggregator",
            "desc_dim", "gem_p_init", "gem_eps", "representation",
            "salad_clusters", "salad_cluster_dim", "salad_token_dim", "salad_mlp_dim",
            "salad_dropout")


def cfg_from_ckpt(ck):
    """A local :class:`VPRConfig` matching the trained weights."""
    cfg = VPRConfig()
    saved = ck.get("config", {})
    if "vit" in saved:
        # Must come first: it re-derives the register-token count (0 for small, 4 for
        # base), and a wrong count fails the strict state_dict load.
        cfg.apply_vit(saved["vit"])
    for k in _RESTORE:
        if k in saved:
            setattr(cfg, k, saved[k])
    cfg.load_pretrained = False     # the weights come from this checkpoint, not small.pt
    cfg.transfer = "linear"         # inference: freeze the whole encoder
    # NB: no cfg.finalize() — the saved desc_dim/salad_* are authoritative, and finalize()
    # would overwrite them for a checkpoint trained with salad_init="megaloc".
    return cfg


def load_model(ckpt_path, device):
    """``(model, cfg, step)`` — the checkpoint loaded strict and completely frozen."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = cfg_from_ckpt(ck)
    model = VPRModel(cfg).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().requires_grad_(False)
    # this repo's checkpoints keep the step under "meta"; older ones had it at top level
    step = int(ck.get("meta", {}).get("step", ck.get("step", -1)))
    return model, cfg, step


# ---------------------------------------------------------------------------
# 4. Descriptors
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_descriptors(model, ds, device, label=""):
    """Descriptors for every frame of a traverse, in order. ``[N, D]`` float32 on CPU."""
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True, drop_last=False)
    autocast = torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda"))
    out = []
    # disable=None: no bar unless stderr is a terminal, so piped and redirected runs stay
    # clean. leave=False: the finished bar clears itself, leaving the caller's logged
    # summary as the record — terminal output and the log file then say the same thing.
    # (main.py routes loguru through tqdm.write so log lines never land on top of the bar.)
    with tqdm(total=len(ds), desc=label, unit="frame", disable=None, leave=False) as bar:
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast:
                out.append(model(imgs).float().cpu())
            bar.update(imgs.size(0))
    return torch.cat(out)


def _cache_path(args, cfg, seq):
    """Cache key. The filters change the descriptors, so they belong in the name."""
    tag = f"hp{int(not args.no_hot_pixel)}"
    tag += "_baoff" if args.no_event_filter else f"_ba{args.event_filter_dt_ms or args.dt_ms}"
    name = f"{args.model}_{seq}_dt{args.dt_ms}_{cfg.representation}_{tag}.npy"
    return os.path.join(args.feature_dir, args.dataset, name)


def descriptors_for(model, traverse, cfg, args, seq, device):
    """Descriptors for one traverse, reusing ``--feature-dir`` when the cache hits."""
    path = _cache_path(args, cfg, seq)
    if os.path.exists(path):
        desc = torch.from_numpy(np.load(path))
        logger.info(f"{seq}: loaded {tuple(desc.shape)} descriptors from {path}")
        return desc

    ds = EventStreamDataset(args.dataset, traverse,
        sequence_path(args, seq), eval_transform(cfg), cfg.representation, args.dt_ms,
        sensor_size(args.dataset),
        hot_pixel=not args.no_hot_pixel,
        # eventcv's filter window is in the stream's native time unit (microseconds)
        filter_dt_us=None if args.no_event_filter
        else (args.event_filter_dt_ms or args.dt_ms) * 1000,
    )
    logger.info(f"{seq}: {len(ds)} slices @ {args.dt_ms} ms, extracting descriptors...")
    t0 = time.time()
    desc = extract_descriptors(model, ds, device, label=seq)
    logger.info(f"{seq}: {tuple(desc.shape)} descriptors in {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, desc.numpy())
    logger.info(f"{seq}: cached -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)")
    return desc


# ---------------------------------------------------------------------------
# 5. PCA-whitening (fit on the reference traverse, applied to both)
# ---------------------------------------------------------------------------
def pca_fit(ref_desc, device, dim=PCA_DIM, power=PCA_POWER, eps=PCA_EPS):
    """Fit whitening on the reference descriptors -> a transform dict.

    Centre on the reference mean, rotate onto the top-``dim`` principal axes of the
    reference set, then scale each axis by ``(var + eps) ** -power``.

    Uses a *randomized* SVD rather than an exact one: at 8448-d the exact decomposition of
    a full traverse takes minutes on CPU, against ~1.5s here, and ``dim`` is well inside
    the range where the whitening was measured to plateau anyway. Its sketch is seeded
    (in a forked RNG, so the caller's stream is untouched) because otherwise the whitened
    recall wobbles by ~0.001 between runs on identical descriptors.
    """
    X = ref_desc.to(device)
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    d = min(int(dim), Xc.shape[1])
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(0)
        _, S, V = torch.svd_lowrank(Xc, q=d, niter=2)   # V: [D, d] principal axes
    var = (S ** 2) / max(Xc.shape[0] - 1, 1)
    return {"mu": mu.cpu(), "V": V.cpu(), "scale": ((var + eps) ** -power).cpu(),
            "dim": d, "power": float(power), "eps": float(eps)}


def pca_apply(X, pca, device, chunk=4096):
    """Apply a fitted :func:`pca_fit` transform -> L2-normalised ``[N, dim]`` float32."""
    mu, V, scale = pca["mu"].to(device), pca["V"].to(device), pca["scale"].to(device)
    out = []
    for s in range(0, X.size(0), chunk):
        Y = (X[s:s + chunk].to(device) - mu) @ V
        out.append(F.normalize(Y * scale, p=2, dim=1).cpu())
    return torch.cat(out)


# ---------------------------------------------------------------------------
# 6. Matching + metrics
# ---------------------------------------------------------------------------
def load_gt(gt_path, nr, nq):
    """``<ref>_<query>_GT.npy`` -> bool ``[nr, nq]``: rows = reference, columns = query.

    Event-LAB already stores it ``[ref, query]``, which is the orientation the metrics want
    (the best match for a query is the argmax down its column), so there is no transpose.
    The band is built on a frame grid derived from the pose track, which need not match the
    number of event slices, so it is resampled onto the descriptor grid with bilinear
    interpolation thresholded at ``>= 0.5`` — that threshold is what preserves the band's
    density under a large resample.
    """
    gt = np.load(gt_path)
    g = (gt != 0).astype(np.float32)                    # already [ref, query]
    if g.shape == (nr, nq):
        return g >= 0.5, gt.shape
    rs = sk_resize(g, (nr, nq), order=1, mode="edge", anti_aliasing=False, preserve_range=True)
    return rs >= 0.5, gt.shape


@torch.no_grad()
def sim_matrix(ref_desc, q_desc, device, chunk=512):
    """Cosine **similarity** ``[nr, nq]`` float32 on CPU — rows = reference, columns = query.

    Similarity, not distance: higher means a better match, which is what
    :func:`recall_at_k` expects. Descriptors are already L2-normalised, so the dot product
    *is* the cosine.
    """
    out = np.empty((ref_desc.size(0), q_desc.size(0)), dtype=np.float32)
    q = q_desc.to(device)
    for s in range(0, ref_desc.size(0), chunk):
        e = min(s + chunk, ref_desc.size(0))
        out[s:e] = (ref_desc[s:e].to(device) @ q.t()).float().cpu().numpy()
    del q
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def recall_at_k(sim, gt, ks=KS):
    """R@k for each k, via VPR_Tutorial's reference implementation.

    ``recallAtK`` (``external/vprtutorial/evaluation/metrics.py``) takes a **similarity**
    matrix oriented ``[reference, query]`` and ranks references down each query column. It
    is single-best-match by construction and consumes the hard ground truth only — it has
    no GTsoft parameter, which is why the band must arrive already dilated (Event-LAB
    applies its ±tolerance when it writes the GT).

    Note its denominator: queries with no ground-truth reference at all are **discarded**,
    so this is recall over scorable queries, not over every query.
    """
    return {k: float(recallAtK(sim, gt, K=k)) for k in ks}


# ---------------------------------------------------------------------------
# 7. Figure
# ---------------------------------------------------------------------------
# Sequential single-hue blue ramp, low->high, so the *most similar* pairs are the darkest
# ink and the match band reads as a dark diagonal. TP/FP is blue vs orange (the green/red
# status pair is invisible to a colourblind reader), with marker shape as a second channel.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_SIM = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
C_TP, C_FP = "#2a78d6", "#eb6834"
C_GT, C_TEXT, C_MUTED = "#c9c8c3", "#0b0b0b", "#52514e"


def _block_reduce(a, out_hw, how):
    """Downsample ``[H,W]`` for display by block min/max — never by striding, which would
    drop the one-frame-wide match band."""
    H, W = a.shape
    fh, fw = max(1, H // out_hw[0]), max(1, W // out_hw[1])
    H2, W2 = (H // fh) * fh, (W // fw) * fw
    b = a[:H2, :W2].reshape(H2 // fh, fh, W2 // fw, fw)
    return b.min(axis=(1, 3)) if how == "min" else b.max(axis=(1, 3))


def make_figure(sim, gt, top1, tp, rec, ref, query, subtitle, out_png, gt_shape):
    """Reference on the y axis, query on the x axis — the ``[ref, query]`` orientation."""
    nr, nq = sim.shape
    s_small = _block_reduce(sim, (1250, 1400), "max")             # max: keep the dark band
    g_small = _block_reduce(gt.astype(np.uint8), (1250, 1400), "max").astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    fig.suptitle(f"{ref} (reference) vs {query} (query) — all {nr} x {nq} pairs, {subtitle}",
                 fontsize=12, color=C_TEXT, y=1.04)

    ax = axes[0]
    im = ax.imshow(s_small, cmap=CMAP_SIM, aspect="auto", interpolation="nearest",
                   extent=[0, nq, nr, 0])
    ax.contour(np.linspace(0, nq, g_small.shape[1]), np.linspace(0, nr, g_small.shape[0]),
               g_small.astype(float), levels=[0.5], colors=[C_FP], linewidths=0.7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("cosine similarity  (dark = better match)", color=C_MUTED, fontsize=9)
    cb.ax.tick_params(colors=C_MUTED, labelsize=8)
    ax.set_title("(a) similarity matrix + ground-truth band", fontsize=10, color=C_TEXT, loc="left")
    ax.set_xlabel(f"{query} query frame", color=C_MUTED, fontsize=9)
    ax.set_ylabel(f"{ref} reference frame", color=C_MUTED, fontsize=9)
    ax.legend(handles=[Line2D([], [], color=C_FP, lw=0.9, label="ground-truth band")],
              loc="lower right", fontsize=8, framealpha=0.9)

    ax = axes[1]
    ax.imshow(g_small, cmap=LinearSegmentedColormap.from_list("gt", ["#ffffff", C_GT]),
              aspect="auto", interpolation="nearest", extent=[0, nq, nr, 0])
    qi = np.arange(nq)
    ax.scatter(qi[tp], top1[tp], s=1.6, c=C_TP, marker=".", linewidths=0,
               label=f"true positive (R@1 = {rec[KS[0]]:.3f})")
    ax.scatter(qi[~tp], top1[~tp], s=6.0, c=C_FP, marker="x", linewidths=0.5,
               label=f"false positive ({(~tp).sum()} of {nq})")
    ax.set_xlim(0, nq)
    ax.set_ylim(nr, 0)
    ax.set_title("(b) top-1 retrieval vs ground truth", fontsize=10, color=C_TEXT, loc="left")
    ax.set_xlabel(f"{query} query frame", color=C_MUTED, fontsize=9)
    ax.set_ylabel(f"{ref} reference frame (retrieved)", color=C_MUTED, fontsize=9)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9,
              markerscale=4).set_title("shaded = GT band", prop={"size": 8})

    for ax in axes:
        ax.tick_params(colors=C_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#dcdbd6")

    fig.text(0.5, -0.03, "   ".join(f"R@{k} {rec[k]:.3f}" for k in KS)
             + f"   |   GT {gt_shape[0]}x{gt_shape[1]} -> {nr}x{nq} (bilinear, thr>=0.5)"
             + f"   |   GT band density {gt.mean():.4f}",
             ha="center", fontsize=8.5, color=C_MUTED)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------
def evaluate(tag, subtitle, ref_desc, q_desc, gt, gt_shape, args, device, out_dir):
    """Score one descriptor space: similarity -> R@k -> saved matrix + figure."""
    t0 = time.time()
    suffix = "" if tag == "raw" else f"_{tag}"
    sim = sim_matrix(ref_desc, q_desc, device)      # [nr, nq]
    rec = recall_at_k(sim, gt)
    top1 = sim.argmax(axis=0)                       # best reference for each query
    tp = gt[top1, np.arange(gt.shape[1])]

    np.save(os.path.join(out_dir, f"sim_{args.ref}_{args.query}{suffix}.npy"), sim)
    make_figure(sim, gt, top1, tp, rec, args.ref, args.query, subtitle,
                os.path.join(out_dir, f"fig_{args.ref}_{args.query}{suffix}.png"), gt_shape)
    del sim

    logger.info(f"[{tag}] " + "  ".join(f"R@{k}={rec[k]:.3f}" for k in KS)
                + f"   ({len(ref_desc)}r x {len(q_desc)}q, {ref_desc.size(1)}-d, "
                  f"GT density {gt.mean():.4f}, {time.time() - t0:.0f}s)")
    return rec


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model, cfg, step = load_model(os.path.join(CKPT_DIR, f"{args.model}.pt"), device)
    logger.info(f"{args.model} (step {step}): vit={cfg.vit} agg={cfg.aggregator} "
                f"desc={cfg.desc_dim} rep={cfg.representation} in={cfg.H}x{cfg.W}")
    logger.info(f"hot-pixel filter: {not args.no_hot_pixel}, event filter: "
                + ("off" if args.no_event_filter
                   else f"{args.event_filter_dt_ms or args.dt_ms} ms"))

    ref_desc = descriptors_for(model, args.ref, cfg, args, args.ref, device)
    q_desc = descriptors_for(model, args.query, cfg, args, args.query, device)

    gt, gt_shape = load_gt(ground_truth_path(args), len(ref_desc), len(q_desc))
    # recall_at_k scores only the queries that have a reference at all; say how many.
    scorable = int((gt.sum(0) > 0).sum())
    logger.info(f"ground truth {gt_shape[0]}x{gt_shape[1]} -> {gt.shape[0]}x{gt.shape[1]} "
                f"[ref, query], density {gt.mean():.4f}, "
                f"{scorable}/{gt.shape[1]} queries scorable")

    out_dir = os.path.join(args.feature_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    evaluate("raw", "no subsampling", ref_desc, q_desc, gt, gt_shape, args, device, out_dir)

    pca = pca_fit(ref_desc, device)
    logger.info(f"PCA-whitening (fit on the reference bank): dim={pca['dim']} "
                f"power={pca['power']} eps={pca['eps']}")
    evaluate("pca", f"PCA-whitened to {pca['dim']}-d",
             pca_apply(ref_desc, pca, device), pca_apply(q_desc, pca, device),
             gt, gt_shape, args, device, out_dir)

    logger.info(f"artifacts -> {out_dir}")
