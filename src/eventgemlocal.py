"""Event-GeM's local-feature stage: SuperEvent keypoints and homography re-ranking.

The global stage (GeM over SuperEvent's FPN map) lives with the other encoders in
:mod:`src.methods`; this is the second pass no other method in the benchmark has. Given a
similarity matrix it re-scores each query's top-K shortlist by geometric verification —
mutual-nearest-neighbour matching of 256-D local descriptors followed by a RANSAC homography —
and subtracts ``inlier_weight`` per inlier from the candidate's distance.

Both halves come from one forward of the shared trunk, exactly as upstream does
(``Event-GeM/eventgem/feature_extraction.py:163``): ``model.fpn(model.backbone(x))`` is the
dominant cost, and the detector and descriptor heads branch off it rather than re-running it.

Two departures from upstream, both forced by the scale of an image-retrieval split:

* **The keypoint store is compacted onto the shortlist.** Upstream extracts keypoints for
  every frame of a traverse; here that would be 75984 x 160 x 256 x 4 = 12.4 GB of padded
  descriptors. Re-ranking only ever *subtracts* from a shortlisted candidate's distance, so a
  database image in no query's top-K can never improve its rank — extracting only the union of
  the shortlists is exactly equivalent, and is a small fraction of the work.
* **Our own shortlist loop**, because upstream's ``process_single_query`` assumes store row ==
  database index, which compaction breaks. The matching and RANSAC inside it are upstream's
  ``compute_inliers_2d`` called unmodified, so the algorithm is theirs and only the bookkeeping
  is ours.
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from src.imagevpr import _check_manifest
from src.inference import NUM_WORKERS

try:
    from threadpoolctl import threadpool_limits
except ImportError:                             # optional; without it BLAS keeps its own count
    from contextlib import contextmanager

    @contextmanager
    def threadpool_limits(limits=None, user_api=None):
        yield


# The descriptor head upsamples its map to [B, 256, H, W], 630 MB per 8 frames at 240x320 — the
# batch size here is set by that tensor, not by the trunk. Measured 2.0 GB peak at 8 and an OOM
# at 16 on an 8 GB card already hosting another job.
KP_BATCH = 8


# ---------------------------------------------------------------------------
# 1. Locating and building the model
# ---------------------------------------------------------------------------
def add_to_path(args):
    """Put the Event-GeM checkout on ``sys.path`` so ``eventgem.utils.*`` imports resolve.

    The same ``sys.path`` injection :class:`src.methods.EventVLADMethod` uses to reach
    Event-LAB's networks, and for the same reason: the re-ranker is upstream's code, so
    importing it is the only way to be sure the algorithm is upstream's too. Only the
    ``eventgem`` package is exposed, which is namespaced and safe to leave on the path —
    unlike the SuperEvent root, see :func:`build_superevent`.
    """
    repo = args.eventgem_repo
    if not os.path.isdir(os.path.join(repo, "eventgem", "utils")):
        raise FileNotFoundError(
            f"no Event-GeM checkout at {repo} (expected eventgem/utils/rerank_utils.py). "
            f"Clone it with --recurse-submodules and point --eventgem-repo at it.")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo


def superevent_root(args):
    root = os.path.join(args.eventgem_repo, "eventgem", "external", "superevent")
    if not os.path.isdir(os.path.join(root, "models")):
        raise FileNotFoundError(
            f"no SuperEvent checkout at {root}. It is a submodule of Event-GeM; clone with "
            f"--recurse-submodules and point --eventgem-repo at the result.")
    return root


def load_config(root):
    """``config/super_event.yaml`` merged with its backbone config, as upstream loads it.

    Follows ``Event-GeM/eventgem/feature_extraction.py:282``. The backbone file carries the
    geometry the input has to satisfy, so unlike upstream a missing merge is an error here
    rather than a warning — it would otherwise silently give the wrong crop.
    """
    with open(os.path.join(root, "config", "super_event.yaml")) as f:
        config = yaml.safe_load(f)
    backbone = os.path.join(root, "config", "backbones", f"{config['backbone']}.yaml")
    if not os.path.exists(backbone):
        raise FileNotFoundError(f"no backbone config at {backbone}")
    with open(backbone) as f:
        config.update(yaml.safe_load(f))
    config["backbone_config"]["input_channels"] = config["input_channels"]
    return config


def build_superevent(root, device):
    """``(model, config, fast_nms)`` — SuperEvent in eval mode with its checkpoint loaded.

    Follows ``Event-GeM/eventgem/feature_extraction.py:305``. The load is strict (the
    checkpoint does match exactly), so a mismatch means the vendored submodule and the weights
    have drifted apart rather than something to paper over.

    The SuperEvent root only sits on ``sys.path`` for the duration of the import: that
    directory exposes ``models``, ``util``, ``config`` and ``data`` as top-level packages, and
    :mod:`src.inference` already documents (line 60) what one stray namespace package costs
    this repo. The modules stay in ``sys.modules`` afterwards, so nothing is lost by removing
    it again.
    """
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        from models.super_event import SuperEvent, SuperEventFullRes
        from models.util import fast_nms
    finally:
        if added and root in sys.path:
            sys.path.remove(root)

    config = load_config(root)
    cls = SuperEventFullRes if config.get("pixel_wise_predictions", False) else SuperEvent
    model = cls(config, tracing=False)
    weights = os.path.join(root, "saved_models", "super_event_weights.pth")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and any(k in state for k in ("model", "state_dict")):
        state = state.get("model", state.get("state_dict", state))
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    logger.info(f"SuperEvent ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params): "
                f"{config['input_channels']}ch {config['input_representation']} -> "
                f"{config['backbone_output_channels']}-D map + "
                f"{config['descriptor_size']}-D local, weights {weights}")
    return model, config, fast_nms


# ---------------------------------------------------------------------------
# 2. Input geometry
# ---------------------------------------------------------------------------
def input_multiple(config):
    """The factor the input height and width must both be divisible by.

    ``EventGeMMCTS`` (``Event-GeM/eventgem/dataset.py``) derives it the same way: the stem's
    patch size, times one halving per stage after the first, times the windowed attention's
    partition size. 2 x 2^2 x 5 = 40 for the shipped MaxViT config.
    """
    bcfg = config.get("backbone_config")
    if not bcfg:
        return int(config["grid_size"])
    factor = int(bcfg["stem"]["patch_size"]) * 2 ** (len(bcfg["num_blocks"]) - 1)
    attention = bcfg.get("stage", {}).get("attention")
    if attention:
        factor *= int(np.max(attention["partition_size"]))
    return factor


def crop_offsets(height, width, multiple):
    """``(top, left, Hc, Wc)`` for a centre crop onto a multiple of ``multiple``.

    Upstream crops because a DAVIS346 is 260x346 and neither dimension divides 40. The default
    240x320 grid divides exactly, so this is a no-op there — it exists so a swept
    ``--eventgem-size`` degrades the way upstream does instead of failing inside the backbone.
    """
    dy, dx = height % multiple, width % multiple
    return -(-dy // 2), -(-dx // 2), height - dy, width - dx


def keypoint_budget(height, width, multiple):
    """Keypoints kept per frame — ``EventGeMMCTS.get_topk()``, i.e. half the longer side."""
    _, _, hc, wc = crop_offsets(height, width, multiple)
    return max(hc, wc) // 2


# ---------------------------------------------------------------------------
# 3. Keypoint extraction
# ---------------------------------------------------------------------------
def sample_descriptors_at_kpts(keypoints, descriptors, Hc, Wc):
    """Bilinear lookup into the dense descriptor map at ``(y, x)`` keypoints -> ``[N, D]``.

    Verbatim from ``Event-GeM/eventgem/feature_extraction.py:325``. The L2 normalisation at the
    end is load-bearing downstream: mutual-nearest-neighbour matching ranks by inner product,
    which only agrees with Euclidean distance on the unit sphere.
    """
    kpts_xy = keypoints.float()[:, [1, 0]]
    grid = torch.zeros((1, 1, kpts_xy.shape[0], 2), dtype=kpts_xy.dtype, device=kpts_xy.device)
    grid[0, 0, :, 0] = 2.0 * kpts_xy[:, 0] / (Wc - 1) - 1.0
    grid[0, 0, :, 1] = 2.0 * kpts_xy[:, 1] / (Hc - 1) - 1.0
    sampled = F.grid_sample(descriptors, grid, mode="bilinear", align_corners=True)
    return F.normalize(sampled[0, :, 0, :].t(), p=2, dim=1)


def extract_keypoints(model, config, fast_nms, paths, device, size, out_path, label=""):
    """Run SuperEvent over ``paths`` and stream the keypoints into one ``.pt`` store.

    The store is upstream's own format (``eventgem/utils/kp_store.py``), so
    ``rerank_utils.open_keypoint_bank`` reads it directly: padded ``(N, K, D)`` descriptors,
    ``(N, K, 2)`` keypoints as ``(x, y)`` in uncropped image coordinates, and a per-frame valid
    count. It is memory-mapped while being filled, so resident memory is one batch however long
    the shortlist is.
    """
    from eventgem.utils.kp_store import KeypointStoreWriter

    from src.methods import NpzFrameDataset, _loader
    from src.npzdata import load_mcts

    height, width = size
    multiple = input_multiple(config)
    off_top, off_left, hc, wc = crop_offsets(height, width, multiple)
    top_k = max(hc, wc) // 2
    desc_dim = int(config["descriptor_size"])

    ds = NpzFrameDataset(paths, lambda p: torch.from_numpy(load_mcts(p, size=size)))
    writer = KeypointStoreWriter(out_path=out_path, n_frames=len(paths), k_max=top_k,
                                 desc_dim=desc_dim, image_shape=(height, width))
    try:
        # fast_nms zeroes its input in place, so it has to run inside inference_mode with the
        # tensor it was given -- calling it outside raises on the inplace update.
        with torch.inference_mode(), tqdm(total=len(ds), desc=label or "keypoints",
                                          unit="frame", disable=None, leave=False) as bar:
            for batch in _loader(ds, KP_BATCH):
                x = batch.to(device, non_blocking=True)
                if (hc, wc) != (height, width):
                    x = x[:, :, off_top:off_top + hc, off_left:off_left + wc]
                # One trunk pass, both heads branched off it: calling model(x) here would pay
                # for the backbone and FPN a second time.
                features = model.fpn(model.backbone(x))
                _, prob = model.detector(features)
                _, desc_map = model.descriptor(features)
                kpts_all, scores_all = fast_nms(prob, config, top_k=top_k)

                b = x.shape[0]
                # One padded block per batch, filled on the GPU and copied down in a single
                # transfer rather than three small copies per frame.
                blk_kpts = torch.zeros(b, top_k, 2, device=device)
                blk_desc = torch.zeros(b, top_k, desc_dim, device=device)
                blk_scores = torch.zeros(b, top_k, device=device)
                blk_counts = torch.zeros(b, dtype=torch.int32)
                for i in range(b):
                    n = min(len(kpts_all[i]), top_k)
                    if n == 0:
                        continue
                    kpts = kpts_all[i][:n].float()              # (n, 2) as (y, x), cropped
                    blk_desc[i, :n] = sample_descriptors_at_kpts(
                        kpts.to(device), desc_map[i:i + 1], hc, wc)
                    blk_kpts[i, :n, 0] = kpts[:, 1] + off_left          # x, uncropped
                    blk_kpts[i, :n, 1] = kpts[:, 0] + off_top           # y, uncropped
                    blk_scores[i, :n] = scores_all[i][:n]
                    blk_counts[i] = n
                writer.add_batch(keypoints=blk_kpts.cpu(), descriptors=blk_desc.float().cpu(),
                                 scores=blk_scores.float().cpu(), counts=blk_counts)
                bar.update(b)
        writer.save()
    finally:
        writer.cleanup()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out_path


# ---------------------------------------------------------------------------
# 4. Re-ranking
# ---------------------------------------------------------------------------
class LocalReranker:
    """SuperEvent keypoints over a shortlist, plus Event-GeM's homography re-scoring.

    One instance per run, holding the model and the cache directory. :meth:`rerank` is called
    once per descriptor space; each call caches its own database store, keyed on the shortlist
    it was built from, so the spaces do share the (single) query store but not their shortlists.
    Re-extracting the overlap between spaces costs a few minutes and buys a cache that cannot
    go stale, which is the better trade at this scale.
    """

    def __init__(self, model, config, fast_nms, device, size, cache_dir, tag, top_k,
                 ransac_thresh, inlier_weight, match_filter, match_ratio):
        self.model, self.config, self.fast_nms = model, config, fast_nms
        self.device, self.size = device, tuple(size)
        self.cache_dir, self.tag = cache_dir, tag
        self.top_k = int(top_k)
        self.ransac_thresh = float(ransac_thresh)
        self.inlier_weight = float(inlier_weight)
        self.match_filter, self.match_ratio = match_filter, float(match_ratio)
        self._query_store = None

    def _store(self, paths, kind):
        """Path to a keypoint store for ``paths``, extracting it if the cache is cold.

        The manifest is the list of basenames the store was built from, checked by
        :func:`src.imagevpr._check_manifest` — the same guard the descriptor caches use, and
        the reason a shortlist that changed underneath a store cannot silently mis-index it.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        store = os.path.join(self.cache_dir, f"{self.tag}_{kind}_kps.pt")
        manifest = os.path.join(self.cache_dir, f"{self.tag}_{kind}_kps_paths.txt")
        if os.path.exists(store) and _check_manifest(manifest, paths):
            logger.info(f"{kind}: loaded {len(paths)} keypoint frames from {store}")
            return store
        logger.info(f"{kind}: extracting keypoints for {len(paths)} frames "
                    f"({keypoint_budget(*self.size, input_multiple(self.config))} per frame)...")
        extract_keypoints(self.model, self.config, self.fast_nms, paths, self.device,
                          self.size, store, label=f"{kind} keypoints")
        with open(manifest, "w") as f:
            f.write("\n".join(os.path.basename(p) for p in paths))
        return store

    def rerank(self, sim, db_paths, q_paths, label=""):
        """``[n_db, n_q]`` similarities -> the same matrix with every shortlist re-scored.

        ``sim`` is higher-is-better, the convention :func:`src.imagevpr.evaluate` consumes;
        upstream works in distances, so this converts with ``1 - sim`` on the way in and back
        on the way out. That offset is constant across candidates and therefore
        rank-irrelevant, which is what lets upstream's ``inlier_weight`` calibration carry over
        unchanged.
        """
        from eventgem.utils.rerank_utils import open_keypoint_bank

        n_db, n_q = sim.shape
        k = min(self.top_k, n_db)
        dist = 1.0 - np.asarray(sim, dtype=np.float32)

        # Shortlists first: their union is what has to be extracted, and it is far smaller than
        # the database because Tokyo 24/7's queries cluster into ~100 distinct locations.
        part = np.argpartition(dist, k - 1, axis=0)[:k]                 # [k, n_q], unordered
        order = np.argsort(np.take_along_axis(dist, part, axis=0), axis=0)
        shortlists = np.take_along_axis(part, order, axis=0)            # [k, n_q], nearest first
        union = np.unique(shortlists)
        # Compaction pays for itself only while the shortlists cover a small slice of the
        # database, as Tokyo 24/7's do. A traverse inverts that: 14478 queries x top-50 over
        # 12825 database frames covers essentially all of it, and the store is then keyed on
        # `label`, so each descriptor space would extract and keep its own near-identical
        # 2.1 GB copy. Above the threshold, fall back to the whole database under a
        # space-independent tag: widening the extracted set can only add rows no shortlist
        # reads, so the re-ranking is unchanged, and the three spaces then share one store.
        if len(union) >= 0.9 * n_db:
            union = np.arange(n_db)
            store_tag = "db"
        else:
            store_tag = f"{label}_db"
        row_of = np.full(n_db, -1, dtype=np.int64)
        row_of[union] = np.arange(len(union))
        logger.info(f"[{label} rerank] top-{k} shortlists cover {len(union)} of {n_db} "
                    f"database images ({100 * len(union) / n_db:.1f}%)"
                    + ("" if store_tag != "db" else " — sharing one full-database store"))

        db_bank = open_keypoint_bank(self._store([db_paths[i] for i in union], store_tag))
        if self._query_store is None:
            self._query_store = self._store(q_paths, "queries")
        q_bank = open_keypoint_bank(self._query_store)

        out = dist.copy()
        # Threads, not processes: the inner loop is BLAS and cv2, both of which release the
        # GIL, and a process pool would mmap the whole store once per worker. BLAS is pinned to
        # one thread per worker because the per-pair gemm is only 160x256x160, where its own
        # threading buys nothing and oversubscribes. Both follow upstream
        # (Event-GeM/eventgem/feature_extraction.py:566).
        cv2.setNumThreads(1)
        with threadpool_limits(limits=1):
            results = Parallel(n_jobs=NUM_WORKERS, backend="threading",
                               return_as="generator_unordered")(
                delayed(self._rerank_query)(q, shortlists[:, q], row_of, dist[:, q],
                                            db_bank, q_bank)
                for q in tqdm(range(n_q), desc=f"{label} rerank", unit="query", disable=None,
                              leave=False))
            verified = 0
            for q, column, matched in results:
                out[:, q] = column
                verified += matched
        logger.info(f"[{label} rerank] {verified} of {k * n_q} candidate pairs found a "
                    f"homography ({100 * verified / (k * n_q):.1f}%)")
        return 1.0 - out

    def _rerank_query(self, q, shortlist, row_of, base, db_bank, q_bank):
        """One query's shortlist, re-scored. Worker for the parallel loop above.

        Matching and RANSAC are upstream's ``compute_inliers_2d``, called unmodified; only the
        mapping from database index to compacted store row is ours.
        """
        from eventgem.utils.rerank_utils import bank_lookup, compute_inliers_2d

        column = base.copy()
        q_data = bank_lookup(q_bank, q)
        if q_data is None:                      # fewer than 4 keypoints: nothing to verify
            return q, column, 0
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matched = 0
        for db_idx in shortlist:
            inliers = compute_inliers_2d(
                q_data, bank_lookup(db_bank, int(row_of[db_idx])), matcher, self.ransac_thresh,
                match_filter=self.match_filter, match_ratio=self.match_ratio)
            if inliers > 0:
                column[db_idx] = base[db_idx] - inliers * self.inlier_weight
                matched += 1
        return q, column, matched
