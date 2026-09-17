"""Event-GeM **0.1.0 as released** on Springfield: ECDPT global stage + its own rerank.

    pixi run python3 scripts/springfield_eg010.py --stage gate      # parity gates only
    pixi run python3 scripts/springfield_eg010.py --stage all

This is the version a reader downloads (tag 0.1.0, worktree
``/media/adam/vprdatasets/megaevent/eventgem_010``), **not** the post-release ``main``
branch that replaced the published ECDPT ViT global stage with a GeM over SuperEvent's own
FPN map. ``src/methods.py:EventGeMMethod`` implements the latter; this file exists because
they are different models and the paper's baseline is the released one.

Why it runs here rather than in the 0.1.0 env: that env has no eventcv (verified — h5py,
torch and numpy are present, eventcv is not), so it cannot slice Springfield's continuous
h5 recordings at all; every 0.1.0 driver in that worktree reads pre-sliced npz. Rather than
materialise ~450 GB of intermediate frames, the release's *code and weights* are imported
here and driven over streamed slices. Nothing about the computation is ours:

* global — ``vit_contrastive_patch16_small(mask_ratio=0, in_chans=2, num_classes=512)`` with
  the release's checkpoint remap, its forward (patch_embed -> +pos_embed -> cls tokens ->
  blocks -> norm -> patch tokens -> GeM(p=5)) and its ``EventGeMData.__getitem__`` loader
  math, all copied from the worktree's own ``nyc_extract.py`` driver;
* keypoints — the release's SuperEvent weights through :mod:`src.eventgemlocal`, written in
  the release's own per-frame ``.feat.npz`` format so its rerank reads them unchanged;
* rerank — the release's ``rerank.process_single_query``, unmodified.

Three gates run before anything is extracted, and all three must pass:

1. the loader math must equal ``EventGeMData`` on a real Brisbane frame;
2. ``pr.pt`` must be sha256 d4c0bf6c… — the byte-identical copy the paper used;
3. re-running the SuperEvent stage on the existing Brisbane ``mcts_sunset1_50`` store must
   reproduce the campaign's ``kps_sunset1`` files.

0.1.1 errata this guards: (5) a crashed keypoint run poisons the cache and the rerank then
silently reports base == reranked, so stores are written to ``.partial`` and renamed only
when complete; (6) MCTS is streamed, never materialised per recording; (8) the release
globs ``*.npy`` and cannot read h5, which is why the frames are produced here.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

EG010 = "/media/adam/vprdatasets/megaevent/eventgem_010"
CKPT = os.path.join(EG010, "eventgem", "ckpt", "pr.pt")
CKPT_SHA = "d4c0bf6c"
BRIS_FRAME = ("/media/adam/vprdatasets/eventgem/brisbane_event/sunset1/"
              "sunset1-frames-50/frame_002438.npy")
BRIS_MCTS = "/media/adam/vprdatasets/eventgem/brisbane_event/sunset1/mcts_sunset1_50"
BRIS_KPS = os.path.join(EG010, "eventgem", "keypoints", "brisbane_event", "kps_sunset1")
GEM_P = 5.0


def _release_on_path():
    for p in (EG010, os.path.join(EG010, "eventgem", "external", "backbone")):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# 1. The release's own maths, copied verbatim from its nyc_extract.py driver
# ---------------------------------------------------------------------------
def loader_math(data):
    """``EventGeMData.__getitem__``, verbatim, minus the file read."""
    tensor = torch.from_numpy(data).float()
    if tensor.shape[-1] in [2, 3] and tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    tensor = tensor.squeeze(0)
    flat = tensor.view(-1)
    if flat.numel() > 0:
        k = int(0.98 * flat.numel())
        robust_max, _ = torch.kthvalue(flat, k)
        if robust_max < 1e-6:
            robust_max = 1.0
        tensor = torch.clamp(tensor, max=robust_max)
        tensor = tensor / robust_max
        tensor = tensor * 2 - 1
    return tensor


def build_backbone(device):
    """The release's ECDPT ViT with its checkpoint remap (nyc_extract.py:91-107)."""
    _release_on_path()
    from eventgem.external.backbone.model.ours_model.ours_model_pretrain import (
        vit_contrastive_patch16_small)

    backbone = vit_contrastive_patch16_small(mask_ratio=0.0, in_chans=2, num_classes=512)
    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
    for key in ("checkpoint", "model", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            checkpoint = checkpoint[key]
            break
    remapped = {k.replace("encoder_q.", "").replace("module.", ""): v
                for k, v in checkpoint.items()}
    msg = backbone.load_state_dict(remapped, strict=False)
    if "patch_embed.proj.weight" in msg.missing_keys:
        raise SystemExit("input layer (patch_embed) did not load from pr.pt")
    return backbone.to(device).eval()


def gem(feats, p=GEM_P):
    """``EventGeM.GeM``, verbatim semantics (feature_extraction.py:74)."""
    return F.avg_pool2d(feats.clamp(min=1e-6).pow(p),
                        (feats.size(-2), feats.size(-1))).pow(1.0 / p)


@torch.no_grad()
def forward_gem(backbone, x):
    """The release's global forward (nyc_extract.py:116-135) for one batch."""
    x = backbone.patch_embed(x)
    x = x + backbone.pos_embed
    cls = backbone.tokens.expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1)
    for blk in backbone.blocks:
        x = blk(x)
    x = backbone.norm(x)
    patch = x[:, 2:, :]
    b, n, c = patch.shape
    h = w = int(n ** 0.5)
    patch = patch.transpose(1, 2).reshape(b, c, h, w)
    return gem(patch).squeeze(-1).squeeze(-1).float()


# ---------------------------------------------------------------------------
# 2. Springfield slices -> the two inputs the release wants
# ---------------------------------------------------------------------------
class PolarityCountsFromStream(torch.utils.data.Dataset):
    """Per-slice ``(H, W, 2)`` [pos, neg] counts, through the release's loader math.

    The same array the release's ``<trav>-frames-50/frame_NNNNNN.npy`` files hold — channel
    0 positive counts, channel 1 negative — rendered from the capture's own events instead
    of read from disk, and then passed through ``loader_math`` so what the backbone sees is
    byte-for-byte what it would have seen from a materialised frame.
    """

    REP = "accumulate"       # the only representation whose reader exposes raw slices

    def __init__(self, h5path, dt_ms, hot_pixel, filter_dt_us):
        from torchvision import transforms
        from springfield_eval import make_dataset
        self.h5path, self._args = h5path, (dt_ms, hot_pixel, filter_dt_us)
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
        from springfield_eval import make_dataset
        if self._reader is None:
            ds = make_dataset(self.h5path, transforms.Compose([]), self.REP, *self._args)
            inner = getattr(ds.reader, "_reader", None)
            if inner is None or not hasattr(inner, "slice"):
                raise SystemExit(f"{self.h5path}: no raw slicing reader")
            self._reader, self._keep = inner, ds
        ev = ecv.numpy(self._reader.slice(i))
        w, h = self.sensor
        frame = np.zeros((h, w, 2), dtype=np.float32)
        if ev.shape[0]:
            m = (ev[:, 0] < w) & (ev[:, 1] < h) & (ev[:, 0] >= 0) & (ev[:, 1] >= 0)
            idx = (ev[m, 1].astype(np.int64) * w + ev[m, 0].astype(np.int64))
            pol = ev[m, 3] > 0
            frame[:, :, 0] = np.bincount(idx[pol], minlength=h * w).reshape(h, w)
            frame[:, :, 1] = np.bincount(idx[~pol], minlength=h * w).reshape(h, w)
        return loader_math(frame)


# ---------------------------------------------------------------------------
# 3. Gates
# ---------------------------------------------------------------------------
def gate_checkpoint():
    with open(CKPT, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if not digest.startswith(CKPT_SHA):
        raise SystemExit(f"pr.pt sha256 {digest[:10]} != {CKPT_SHA}… — this is not the "
                         f"released checkpoint the paper used")
    print(f"[gate 1/3] pr.pt sha256 {digest[:10]}… matches the released checkpoint")


def gate_loader_math():
    _release_on_path()
    from eventgem.dataset import EventGeMData
    ref = EventGeMData(os.path.dirname(BRIS_FRAME))
    idx = ref.files_sort.index(BRIS_FRAME)
    if not torch.equal(ref[idx], loader_math(np.load(BRIS_FRAME))):
        raise SystemExit("loader-math parity FAILED vs the release's EventGeMData")
    print("[gate 2/3] loader math identical to the release's EventGeMData")


def gate_keypoints(device, n=3):
    """Re-extract 3 Brisbane frames and match the campaign's stored keypoints."""
    import argparse as _a
    from src import eventgemlocal as egl
    shim = _a.Namespace(eventgem_repo=EG010, dataset="springfield", feature_dir="features")
    model, cfg, fast_nms = egl.build_superevent(egl.superevent_root(shim), device)
    ref0 = np.load(os.path.join(BRIS_KPS, "mcts_00000.feat.npz"))
    size = tuple(int(v) for v in ref0["image_shape"])
    ok = 0
    for i in range(n):
        mcts = np.load(os.path.join(BRIS_MCTS, f"mcts_{i:05d}.npz"))["mcts"]
        got = superevent_frame(model, cfg, fast_nms, torch.from_numpy(mcts), size, device)
        ref = np.load(os.path.join(BRIS_KPS, f"mcts_{i:05d}.feat.npz"))
        _assert_same_keypoints(got, ref, i)
        ok += 1
    print(f"[gate 3/3] SuperEvent reproduces the campaign's kps_sunset1 on {ok} frames "
          f"(as sets — see _assert_same_keypoints on why order is not part of it)")
    return model, cfg, fast_nms


def _assert_same_keypoints(got, ref, i):
    """The same keypoints, scores and descriptors — compared as a SET, not a sequence.

    Two points with identical detector scores can come out of the NMS top-k in either
    order, and they do: frame 1 of Brisbane's store differs from this run only by one such
    swap (160 of 160 coordinates present in both, every score equal, max coordinate delta
    exactly 4.0 — one pair exchanged). Re-ranking consumes keypoints as an unordered set
    for RANSAC, so that is not a difference in the method; comparing positionally would
    make the gate fail on a tie-break while missing a real change in the points themselves.
    Descriptors are compared after aligning both sides on keypoint identity, so a genuine
    change in the features still fails.
    """
    gk, rk = got["keypoints"], ref["keypoints"]
    if gk.shape != rk.shape:
        raise SystemExit(f"[gate 3/3] frame {i}: {gk.shape} keypoints vs {rk.shape}")
    order_g = np.lexsort((gk[:, 1], gk[:, 0]))
    order_r = np.lexsort((rk[:, 1], rk[:, 0]))
    for name, atol in (("keypoints", 1e-3), ("scores", 1e-3), ("descriptors", 1e-3)):
        a, b = got[name][order_g], ref[name][order_r]
        if not np.allclose(a, b, atol=atol):
            raise SystemExit(f"[gate 3/3] frame {i}: {name} differ beyond {atol} once "
                             f"both are ordered by keypoint position — max delta "
                             f"{np.abs(a - b).max():.4g}")


@torch.no_grad()
def superevent_frame(model, cfg, fast_nms, mcts, size, device):
    """One MCTS frame -> the release's per-frame keypoint record."""
    from src import eventgemlocal as egl
    height, width = size
    mult = egl.input_multiple(cfg)
    off_top, off_left, hc, wc = egl.crop_offsets(height, width, mult)
    top_k = max(hc, wc) // 2
    x = mcts.unsqueeze(0).to(device).float()
    if (hc, wc) != (height, width):
        x = x[:, :, off_top:off_top + hc, off_left:off_left + wc]
    feats = model.fpn(model.backbone(x))
    _, prob = model.detector(feats)
    _, desc_map = model.descriptor(feats)
    kpts_all, scores_all = fast_nms(prob, cfg, top_k=top_k)
    kpts = kpts_all[0][:top_k].float()
    desc = egl.sample_descriptors_at_kpts(kpts.to(device), desc_map[0:1], hc, wc)
    out_k = torch.zeros(len(kpts), 2)
    out_k[:, 0] = kpts[:, 1] + off_left          # x, uncropped
    out_k[:, 1] = kpts[:, 0] + off_top           # y, uncropped
    return {"keypoints": out_k.cpu().numpy().astype(np.float32),
            "scores": scores_all[0][:top_k].float().cpu().numpy().astype(np.float32),
            "descriptors": desc.float().cpu().numpy().astype(np.float32),
            "image_shape": np.array(size, dtype=np.int32)}


# ---------------------------------------------------------------------------
# 4. Keypoints and the released rerank
# ---------------------------------------------------------------------------
TOP_K = 50              # main.py --top-k default
RANSAC_THRESH = 5.0     # main.py --ransac-thresh default
INLIER_WEIGHT = 0.05    # main.py --inlier-weight default
KP_PATTERN = "kp_{:05d}.feat.npz"
MCTS_SIZE = (240, 320)  # SuperEvent's own geometry, a multiple of its 40-px constraint


def _load_protocol(cli, device):
    """The pooled gallery and queries with eg010 banks attached — all from cache."""
    import springfield_baselines as sb
    from src.traversegps import local_enu
    from springfield_full import DB_ARMS, discover_queries, SWEEPS

    method = sb.build_eg010(cli, device)
    cli.ckpt = method["tag"]      # partition_manifest records it; no checkpoint file here
    db_sids = [s for s in (cli.db_sessions.split(",") if cli.db_sessions else DB_ARMS) if s]
    only = set(cli.query_sessions.split(",")) if cli.query_sessions else None
    sweeps = [s for s in cli.sweeps.split(",") if s]
    db, q = {}, {}
    for sid in db_sids:
        db[sid] = sb.session_geometry(cli, method,
                                      os.path.join(cli.root, "database", sid), sid,
                                      limit=cli.limit_partitions)
    queries = discover_queries(cli.root, None, sweeps, only)
    for sweep, sid, sdir in queries:
        q[sid] = sb.session_geometry(cli, method, sdir, sid)
        q[sid]["sweep"] = sweep
    anchor = db[db_sids[0]]
    kept = anchor["latlon"][anchor["keep"]]
    lat0, lon0 = float(kept[:, 0].mean()), float(kept[:, 1].mean())
    for sess in list(db.values()) + list(q.values()):
        sess["xy"] = local_enu(sess["latlon"][:, 0], sess["latlon"][:, 1], lat0, lon0)
    for sid in db_sids:
        sb.extract_session(cli, method, device, db[sid], sid, db[sid]["keep"])
    for sweep, sid, sdir in queries:
        sb.extract_session(cli, method, device, q[sid], sid, q[sid]["keep"])
    return method, db, q, queries, db_sids


def _pool(sessions, order):
    """Pooled descriptors, positions and per-row (session, local slice index)."""
    desc = np.concatenate([sessions[s]["bank"][sessions[s]["keep"]] for s in order])
    xy = np.concatenate([sessions[s]["xy"][sessions[s]["keep"]] for s in order])
    prov = [(s, int(i)) for s in order
            for i in np.flatnonzero(sessions[s]["keep"])]
    return desc.astype(np.float32), xy, prov


@torch.no_grad()
def _shortlists(db_desc, q_desc, device, top_k, chunk=256):
    """``[n_q, top_k]`` gallery indices by ascending release distance (1 - cosine)."""
    dbt = torch.from_numpy(db_desc).to(device)
    out = np.empty((len(q_desc), top_k), np.int64)
    for a in range(0, len(q_desc), chunk):
        blk = torch.from_numpy(q_desc[a:a + chunk]).to(device)
        sim = blk @ dbt.T                                  # [b, n_db], both L2-normalised
        idx = torch.topk(sim, top_k, dim=1).indices
        out[a:a + chunk] = idx.cpu().numpy()
    del dbt
    torch.cuda.empty_cache()
    return out


def stage_kps(cli, device, rows_needed, sessions, label):
    """Write the release's per-frame ``.feat.npz`` for exactly the rows asked for.

    Compacted onto the shortlist union, which is what upstream's own re-ranker does and the
    only reason this is affordable: the alternative is SuperEvent over all 148,455 slices.
    The store is built under ``.partial`` and renamed only once every file is present, so a
    crashed run cannot leave a half-filled directory that the rerank would silently read as
    complete (0.1.1 erratum 5).
    """
    import argparse as _a
    import shutil
    from src import eventgemlocal as egl
    from springfield_baselines import _MCTSFromStream
    from springfield_eval import list_partitions

    shim = _a.Namespace(eventgem_repo=EG010, dataset="springfield",
                        feature_dir=cli.feature_dir)
    model, cfg, fast_nms = egl.build_superevent(egl.superevent_root(shim), device)
    root = os.path.join(cli.kp_dir, label)
    partial = root + ".partial"
    if os.path.isdir(root) and len(os.listdir(root)) >= len(rows_needed):
        print(f"  {label}: {len(os.listdir(root))} keypoint files cached")
        return root
    os.makedirs(partial, exist_ok=True)

    by_session = {}
    for pooled_idx, (sid, local) in rows_needed:
        by_session.setdefault(sid, []).append((pooled_idx, local))
    done = 0
    for sid, items in by_session.items():
        sess = sessions[sid]
        sdir = os.path.join(cli.root, "database" if sid in _DB else
                            f"query_{sess['sweep']}", sid)
        parts = list_partitions(sdir)
        edges = np.concatenate([[0], np.cumsum(sess["counts"])])
        cache = {}
        for pooled_idx, local in sorted(items, key=lambda t: t[1]):
            pi = int(np.searchsorted(edges, local, side="right") - 1)
            if pi not in cache:
                cache.clear()
                cache[pi] = _MCTSFromStream(parts[pi], cli.dt_ms, MCTS_SIZE,
                                            not cli.no_hot_pixel, None)
            mcts = cache[pi][local - int(edges[pi])]
            rec = superevent_frame(model, cfg, fast_nms, mcts, MCTS_SIZE, device)
            np.savez(os.path.join(partial, KP_PATTERN.format(pooled_idx)), **rec)
            done += 1
            if done % 500 == 0:
                print(f"  {label}: {done}/{len(rows_needed)} keypoint frames", flush=True)
        cache.clear()
    if len(os.listdir(partial)) != len(rows_needed):
        raise SystemExit(f"{label}: wrote {len(os.listdir(partial))} of "
                         f"{len(rows_needed)} keypoint files — refusing to publish a "
                         f"partial store (erratum 5)")
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.rename(partial, root)
    print(f"  {label}: {done} keypoint frames -> {root}")
    return root


def _orient_psi(cli, sid):
    """Per-slice camera-vs-route bearing for one session, from the diagnostic bundle.

    psi comes from GPS travel bearing and phone attitude, so it is model-independent — the
    bundle built for any checkpoint applies to every other one, provided the query keep
    masks match, which the length check enforces.
    """
    path = os.path.join(cli.diag_dir, f"orient_{cli.orient_tag}.npz")
    o = np.load(path, allow_pickle=False)
    key = f"{sid}_psi"
    if key not in o:
        raise SystemExit(f"{path} has no psi for {sid}")
    return o[key]


def stage_rerank(cli, device):
    """The released rerank over the pooled gallery, then scored on our ground truth.

    ``rerank.process_single_query`` is the release's own loop, called unmodified. It wants a
    full base-distance column per query and a directory it can index by gallery row, so a
    symlink farm maps pooled row -> that row's keypoint file — the same device the 2026-08
    campaign used to run the release over a pooled database. Columns are built in GPU
    batches and thrown away after each batch: the full matrix would be 132,569 x 15,886
    float32, or 8.4 GB.
    """
    import shutil
    from joblib import Parallel, delayed
    from pathlib import Path
    from src.imagesets import build_radius_gt
    from springfield_full import DB_ARMS
    _release_on_path()
    # rerank.py imports prettytable at module scope but only uses it at line 227, inside
    # the release's own results printing — process_single_query, the algorithm called here,
    # never touches it. Our env does not ship it, and adding it would re-solve the
    # environment underneath a run that is already in flight, so it is stubbed for the
    # import instead. Nothing in the re-ranking arithmetic is affected.
    if "prettytable" not in sys.modules:
        import types
        sys.modules["prettytable"] = types.ModuleType("prettytable")
    from eventgem.rerank import load_event_features, process_single_query

    global _DB
    _DB = set(DB_ARMS)
    method, db, q, queries, db_sids = _load_protocol(cli, device)
    db_desc, db_xy, db_prov = _pool(db, db_sids)
    qids = [sid for _, sid, _ in queries]
    q_desc, q_xy, q_prov = _pool(q, qids)
    sweeps = np.array([q[sid]["sweep"] for sid, _ in q_prov])
    print(f"gallery {len(db_xy)} rows, queries {len(q_xy)} slices")

    short = _shortlists(db_desc, q_desc, device, TOP_K)
    union = np.unique(short)
    print(f"shortlist union: {len(union)} of {len(db_xy)} gallery rows "
          f"({len(union) / len(db_xy):.1%}) — that is the keypoint bill")

    db_rows = [(int(i), db_prov[int(i)]) for i in union]
    q_rows = [(i, q_prov[i]) for i in range(len(q_prov))]
    db_kp = stage_kps(cli, device, db_rows, db, "kps_db")
    q_kp = stage_kps(cli, device, q_rows, q, "kps_queries")

    farm = os.path.join(cli.kp_dir, "farm_db")
    if os.path.isdir(farm):
        shutil.rmtree(farm)
    os.makedirs(farm)
    for i in union:                       # pooled row -> its keypoint file
        os.symlink(os.path.join(db_kp, KP_PATTERN.format(int(i))),
                   os.path.join(farm, KP_PATTERN.format(int(i))))

    gt = build_radius_gt(db_xy, q_xy, cli.threshold_m)
    dbt = torch.from_numpy(db_desc).to(device)
    base_top, rr_top = np.empty((len(q_xy), 20), np.int64), np.empty((len(q_xy), 20), np.int64)
    chunk = 128
    for a in range(0, len(q_xy), chunk):
        blk = torch.from_numpy(q_desc[a:a + chunk]).to(device)
        with torch.no_grad():
            dists = (1.0 - blk @ dbt.T).cpu().numpy().astype(np.float32)
        base_top[a:a + chunk] = np.argsort(dists, axis=1)[:, :20]
        # The release's own loader, not a raw np.load: it returns {"kpts", "desc"} —
        # the names compute_inliers_2d reads — and handles the empty-feature and 1-D
        # descriptor cases itself. Passing the npz straight through fails on 'desc'.
        qdata = [load_event_features(Path(q_kp), a + j, KP_PATTERN)
                 for j in range(dists.shape[0])]
        out = Parallel(n_jobs=cli.jobs, backend="threading")(
            delayed(process_single_query)(j, dists[j], TOP_K, qdata[j], Path(farm),
                                          KP_PATTERN, RANSAC_THRESH, INLIER_WEIGHT)
            for j in range(dists.shape[0]))
        for j, col in out:
            rr_top[a + j] = np.argsort(col)[:20]
        if a % (chunk * 10) == 0:
            print(f"  rerank {a}/{len(q_xy)}", flush=True)
    del dbt
    torch.cuda.empty_cache()

    # Erratum 5's failure mode is a rerank that silently does nothing: a poisoned or
    # incomplete keypoint store makes every candidate score zero inliers, and the reranked
    # numbers then come back exactly equal to the global ones and look merely disappointing
    # rather than broken. The store guard catches the incomplete case; this catches the rest.
    changed = float((base_top[:, 0] != rr_top[:, 0]).mean())
    print(f"rerank moved top-1 for {changed:.1%} of queries "
          f"(the campaign measured 83% on pooled Brisbane)")
    if changed == 0.0:
        raise SystemExit("the rerank changed nothing at all — that is erratum 5's "
                         "signature, not a result. Check the keypoint stores cover the "
                         "shortlist before believing any reranked number.")

    # Save the per-slice rankings, not just the aggregates. Scoring any other cell —
    # day+dawn, reversals excluded, a different radius — is then seconds of recomputation
    # instead of re-running a rerank whose keypoint pass costs four hours. The first
    # version of this kept only the recall blocks, and the paper's day+dawn cell could not
    # be produced from them at all.
    np.savez_compressed(
        os.path.join(cli.out_dir, "eg010_rerank_slices.npz"),
        base_top=base_top.astype(np.int32), rr_top=rr_top.astype(np.int32),
        q_sid=np.array([qids.index(sid) for sid, _ in q_prov], dtype=np.int32),
        sids=np.array(qids), sweeps=np.array([q[s]["sweep"] for s in qids]),
        q_xy=q_xy.astype(np.float32), db_xy=db_xy.astype(np.float32))

    psi = np.concatenate([_orient_psi(cli, s) for s in qids])
    cells = {"overall": np.ones(len(q_xy), bool),
             "day+dawn_norev": (sweeps != "night") & (np.abs(psi) < 135.0)}
    res = {"dataset": "springfield", "method": method["meta"],
           "rerank_changed_top1_frac": round(changed, 4),
           "threshold_m": cli.threshold_m, "n_database": int(len(db_xy)),
           "top_k": TOP_K, "ransac_thresh": RANSAC_THRESH,
           "inlier_weight": INLIER_WEIGHT,
           "shortlist_union": int(len(union)),
           "n_queries": {k: int(v.sum()) for k, v in cells.items()}}
    for name, top in (("global", base_top), ("rerank", rr_top)):
        hit = gt[top, np.arange(len(q_xy))[:, None]]
        blk = {}
        for scope in ("overall",) + tuple(dict.fromkeys(sweeps)):
            m = np.ones(len(q_xy), bool) if scope == "overall" else (sweeps == scope)
            blk[scope] = {str(k): float((hit[m, :k].any(axis=1)).mean())
                          for k in (1, 5, 10, 20)}
        for cname, m in cells.items():
            blk[cname] = {str(k): float((hit[m, :k].any(axis=1)).mean())
                          for k in (1, 5, 10, 20)}
        res[f"recall_{name}"] = blk
        print(f"{name:7s} overall R@1 {blk['overall']['1']:.4f} R@10 {blk['overall']['10']:.4f}"
              f"   |  day+dawn no-rev R@1 {blk['day+dawn_norev']['1']:.4f} "
              f"R@10 {blk['day+dawn_norev']['10']:.4f}")
    os.makedirs(cli.out_dir, exist_ok=True)
    out_path = os.path.join(cli.out_dir, "results_eg010_gem224_baoff_rerank.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="gate", choices=("gate", "rerank", "all"))
    ap.add_argument("--root", default="/media/adam/vprdatasets/megaevent/springfield/sessions")
    ap.add_argument("--bank-dir",
                    default="/media/adam/vprdatasets/megaevent/evaluations/springfield")
    ap.add_argument("--out-dir", default="output/springfield_full")
    ap.add_argument("--kp-dir",
                    default="/media/adam/vprdatasets/megaevent/evaluations/springfield/eg010_kps")
    ap.add_argument("--feature-dir", default="features")
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--threshold-m", type=float, default=25.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-hot-pixel", action="store_true")
    ap.add_argument("--no-event-filter", action="store_true", default=True)
    ap.add_argument("--event-filter-dt-us", type=int, default=50_000)
    ap.add_argument("--allow-unrepacked", action="store_true", default=True)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--limit-partitions", type=int, default=None)
    ap.add_argument("--eval-resolution", type=int, default=224)
    ap.add_argument("--representation", default=None)
    ap.add_argument("--db-sessions", default=None, help="smoke test: subset of the gallery")
    ap.add_argument("--query-sessions", default=None, help="smoke test: subset of queries")
    ap.add_argument("--sweeps", default="day,dawn,night")
    # Needed by _orient_psi for the day+dawn/no-reversal cell. Absent from this parser in
    # the first version, so the run wrote its per-slice rankings and then died before the
    # JSON — recoverable only because those rankings had just been saved.
    ap.add_argument("--diag-dir", default="output/springfield_diag")
    ap.add_argument("--orient-tag",
                    default="b_v8_accum_s750_s750_3f4eb54780_accumulate_r322_hp1_baoff")
    cli = ap.parse_args()
    os.makedirs(cli.kp_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate_checkpoint()
    gate_loader_math()
    gate_keypoints(device)
    print("\nall gates passed — the released global stage and keypoint stage both "
          "reproduce here")
    if cli.stage == "gate":
        return
    stage_rerank(cli, device)


if __name__ == "__main__":
    main()
