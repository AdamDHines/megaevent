"""Two-stage MegaEvent: global retrieval + RANSAC re-ranking over our own DINOv2 tokens.

    pixi run python3 scripts/megaevent_rerank.py --dataset brisbane_event
    pixi run python3 scripts/megaevent_rerank.py --dataset nsavp

Event-GeM's homography re-rank is worth ~+20 R@1 on these traverses and MegaEvent has no
second stage — yet a ViT encoder computes 529 patch tokens per 322² frame *anyway* and the
global head throws their geometry away. This script keeps them: the global stage's top-K
shortlist is re-scored by mutual-nearest-neighbour token matching + RANSAC homography,
`score += inliers * weight`, exactly Event-GeM's verification model (same K, threshold and
weight defaults) so the comparison is like-for-like.

Mechanics, mirroring src/eventgemlocal.py where it solved the same problems:
 * Shortlists come from the cached global banks under the pooled protocol (query traverse
   vs pooled gallery, 25 m) — the global ranking this re-ranks is bit-identical to the
   published row, and `--inlier-weight 0` must reproduce it exactly (the identity check).
 * Only the shortlist union + queries get token extraction. Tokens are PCA-whitened to
   `--token-dim` (default 128, power 0.5, fit on a frame sample) and L2-normalised, fp16,
   memmapped next to the banks: ~135 KB/frame instead of 812 KB raw.
 * Matching runs batched on the GPU (one [K, 529, 529] similarity block per query);
   RANSAC runs on CPU threads (cv2 releases the GIL). Token (row, col) centres in the 322²
   frame are the correspondences' coordinates.
 * A per-token activity fraction (mean of the countmask activity mask under the patch) is
   stored with the tokens; `--min-activity` masks near-empty background tokens out of the
   matching. 0 (default) reproduces the plain method.

Outputs `<out-dir>/rerank_<tag>.json` with global and re-ranked recall plus the knobs.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import brisbane_pooled as bp  # noqa: E402
import nsavp_pooled as npl  # noqa: E402
from brisbane_resolution import _Args  # noqa: E402
from src import inference as inf  # noqa: E402
from src.imagevpr import build_gt  # noqa: E402

ROOT = "/media/adam/vprdatasets/megaevent"

# dataset -> (eventlab_dir, query, database traverses, bank dir, bank label, ckpt)
CONFIGS = {
    "brisbane_event": ("/media/adam/vprdatasets/eventgem", "sunset1",
                       ("daytime", "morning", "night", "sunrise"),
                       f"{ROOT}/brisbane_ship", "s_mloc_s3500",
                       "ckpts/megaevent_vits_mloc.pt"),
    "nsavp": ("/media/adam/vprdatasets/eventlab", "R0_FA0",
              ("R0_FN0", "R0_FS0", "R0_RA0", "R0_RN0", "R0_RS0"),
              f"{ROOT}/nsavp_pooled", "pm_s3500",
              "ckpts/megaevent_vitb_mloc.pt"),
    # v8 accumulate (banks + ckpt from the 2026-08-21 benchmark pass)
    "brisbane_v8": ("/media/adam/vprdatasets/eventgem", "sunset1",
                    ("daytime", "morning", "night", "sunrise"),
                    f"{ROOT}/v8_bench/brisbane", "v8",
                    f"{ROOT}/v8_bench/ckpts/b_v8_accum_s750.pt"),
    "nsavp_v8": ("/media/adam/vprdatasets/eventlab", "R0_FA0",
                 ("R0_FN0", "R0_FS0", "R0_RA0", "R0_RN0", "R0_RS0"),
                 f"{ROOT}/v8_bench/nsavp", "v8",
                 f"{ROOT}/v8_bench/ckpts/b_v8_accum_s750.pt"),
}
# The dataset FAMILY (paths, sensor, geometry) behind each config key.
FAMILY = {"brisbane_event": "brisbane_event", "brisbane_v8": "brisbane_event",
          "nsavp": "nsavp", "nsavp_v8": "nsavp"}
RESOLUTION = 322
GRID = RESOLUTION // 14                         # 23x23 = 529 tokens
KS = (1, 5, 10, 20)


def geometry(dataset, eventlab_dir, sequences):
    if dataset == "nsavp":
        return npl.traverse_geometry(os.path.join(eventlab_dir, dataset), sequences, 50)
    args = _Args(eventlab_dir, dataset, 50, False, False)
    args.filter_dt_us = 50_000
    return bp.traverse_geometry(args, sequences, None)


def frame_dataset(dataset, eventlab_dir, seq, cfg):
    args = _Args(eventlab_dir, dataset, 50, False, False)
    return inf.EventStreamDataset(
        dataset, seq, inf.sequence_path(args, seq), inf.eval_transform(cfg),
        cfg.representation, 50, inf.sensor_size(dataset),
        hot_pixel=True, filter_dt_us=50_000)


class TokenStore:
    """fp16 memmap of [n, 529, dim] whitened tokens + [n, 529] activity fractions."""

    def __init__(self, path, n, dim, mode):
        self.tok = np.lib.format.open_memmap(path + ".tokens.npy", mode=mode,
                                             dtype=np.float16, shape=(n, GRID * GRID, dim))
        self.act = np.lib.format.open_memmap(path + ".activity.npy", mode=mode,
                                             dtype=np.float16, shape=(n, GRID * GRID))

    @classmethod
    def open(cls, path):
        self = cls.__new__(cls)
        self.tok = np.load(path + ".tokens.npy", mmap_mode="r")
        self.act = np.load(path + ".activity.npy", mmap_mode="r")
        return self


def fit_token_pca(model, cfg, device, plan, dim, n_frames=64):
    """Whitening basis for the raw patch tokens, fit on a spread of frames."""
    torch_gen = torch.Generator().manual_seed(0)
    sample = []
    flat = [(seq, ds, idx) for seq, (ds, idxs) in plan.items() for idx in idxs]
    pick = torch.randperm(len(flat), generator=torch_gen)[:n_frames]
    with torch.no_grad():
        for j in pick.tolist():
            seq, ds, idx = flat[j]
            x = ds[int(idx)][None].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                feat, _ = model.forward_encoder(x)          # [1, C, g, g]
            sample.append(feat.flatten(2)[0].T.float().cpu())
    tokens = torch.cat(sample)                              # [n*529, C]
    return inf.pca_fit(tokens, device, dim=dim, power=0.5)


def extract_tokens(model, cfg, device, plan, pca, store, batch):
    """Fill `store` rows following each traverse's (dataset, indices, row offsets)."""
    with torch.no_grad():
        for seq, (ds, idxs, rows) in plan.items():
            t0, done = time.time(), 0
            for s in range(0, len(idxs), batch):
                chunk = idxs[s:s + batch]
                x = torch.stack([ds[int(i)] for i in chunk]).to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    feat, _ = model.forward_encoder(x)      # [b, C, g, g]
                b, C = feat.shape[:2]
                tok = feat.flatten(2).transpose(1, 2).reshape(b * GRID * GRID, C).float()
                white = inf.pca_apply(tok.cpu(), pca, device).reshape(b, GRID * GRID, -1)
                # activity: fraction of active pixels (G channel above its normalised zero)
                act = _activity_from_input(x, cfg)
                for k, i in enumerate(chunk):
                    r = rows[s + k]
                    store.tok[r] = white[k].numpy().astype(np.float16)
                    store.act[r] = act[k].cpu().numpy().astype(np.float16)
                done += len(chunk)
                if done % (batch * 20) < batch:
                    rate = done / (time.time() - t0)
                    print(f"    tokens {seq}: {done}/{len(idxs)}  {rate:.1f} f/s", flush=True)


def _activity_from_input(x, cfg):
    """Per-token active-pixel fraction, recovered from the normalised G channel.

    countmask: G is a binary activity mask (1 = active) -> active means above 0.5.
    accumulate: G = 1 - pos - neg on a WHITE background -> any event darkens it, so
    active means below white (headroom for uint8 quantisation). The countmask test on
    accumulate frames would mark the *background* active and re-admit the static DAVIS
    bonnet the --min-activity mask exists to reject.
    """
    mean, std = cfg.tencode_mean[1], cfg.tencode_std[1]
    g = x[:, 1] * std + mean                                # [b, H, W] un-normalised G
    act = (g < 0.998) if cfg.representation == "accumulate" else (g > 0.5)
    b = act.shape[0]
    act = act.reshape(b, GRID, 14, GRID, 14).float().mean(dim=(2, 4))
    return act.reshape(b, GRID * GRID)


COORDS = np.stack(np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij"),
                  axis=-1).reshape(-1, 2)[:, ::-1] * 14.0 + 7.0   # [(x, y)] token centres


def rerank_query(q_tok, q_act, cand_tok, cand_act, scores, ransac_thresh, weight,
                 min_activity, device):
    """One query: mutual-NN matches on GPU, RANSAC per candidate on CPU. -> new scores."""
    K = cand_tok.shape[0]
    qt = torch.from_numpy(np.asarray(q_tok, dtype=np.float32)).to(device)
    ct = torch.from_numpy(np.asarray(cand_tok, dtype=np.float32)).to(device)
    sim = torch.einsum("nd,kmd->knm", qt, ct)               # [K, 529(q), 529(db)]
    nn12 = sim.argmax(dim=2)                                # best db token per q token
    nn21 = sim.argmax(dim=1)                                # best q token per db token
    arange = torch.arange(qt.shape[0], device=device)
    out = scores.copy()
    q_ok = np.asarray(q_act, dtype=np.float32) >= min_activity
    for k in range(K):
        mutual = (nn21[k][nn12[k]] == arange).cpu().numpy()
        keep = mutual & q_ok
        db_idx = nn12[k].cpu().numpy()
        keep &= np.asarray(cand_act[k], dtype=np.float32)[db_idx] >= min_activity
        if keep.sum() < 4:
            continue
        src = COORDS[keep].astype(np.float32)
        dst = COORDS[db_idx[keep]].astype(np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if H is not None and mask is not None:
            out[k] = scores[k] + float(mask.sum()) * weight
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="brisbane_event", choices=sorted(CONFIGS))
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--ransac-thresh", type=float, default=5.0)
    ap.add_argument("--inlier-weight", type=float, default=0.05,
                    help="added per inlier to the cosine (higher-is-better) score. "
                         "Event-GeM's 0.05 on a 1-cosine distance is the same scale.")
    ap.add_argument("--token-dim", type=int, default=128)
    ap.add_argument("--min-activity", type=float, default=0.0,
                    help="mask tokens whose patch has under this fraction of active "
                         "pixels out of the matching (0 = keep all). On Brisbane this is "
                         "load-bearing: the DAVIS bonnet/vignette is static, so with all "
                         "529 tokens ANY two frames find a big-inlier homography on the "
                         "static region and the rerank degrades (-0.08 measured at 0).")
    ap.add_argument("--whiten", action="store_true",
                    help="re-rank the pca4096p0.5 (best-space) global ranking instead of "
                         "the native one: whitening fit on the pooled database exactly as "
                         "brisbane_pooled does (seeded subsample), applied to both sides "
                         "before the shortlists. Token matching is unaffected.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default=None)
    cli = ap.parse_args()

    eventlab_dir, query, database, bank_dir, label, ckpt = CONFIGS[cli.dataset]
    out_dir = cli.out_dir or os.path.join(ROOT, "rerank", cli.dataset)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences = [query, *database]

    family = FAMILY[cli.dataset]
    geom = geometry(family, eventlab_dir, sequences)
    files = {s: {label: os.path.join(bank_dir, f"r322ba50_{s}_{label}.npy")}
             for s in sequences}
    db_desc, db_xy, source = bp.pool_database(files, geom, list(database), label)
    q_xy, q_cov, _, _ = geom[query]
    q_bank = np.load(files[query][label], mmap_mode="r")
    q_desc = torch.from_numpy(np.asarray(q_bank[q_cov]))
    q_xy = q_xy[q_cov]
    gt = build_gt(db_xy, q_xy, 25.0)
    n_db, n_q = db_desc.shape[0], q_desc.shape[0]
    space = "native"
    if cli.whiten:
        from src import scoring
        fit = db_desc
        if fit.size(0) > scoring.PCA_FIT_SAMPLES:
            gen = torch.Generator().manual_seed(scoring.PCA_FIT_SEED)
            fit = db_desc[torch.randperm(db_desc.size(0), generator=gen)
                          [:scoring.PCA_FIT_SAMPLES]]
        pca = inf.pca_fit(fit, device, dim=min(4096, fit.size(0) - 1), power=0.5)
        db_desc = inf.pca_apply(db_desc, pca, device)
        q_desc = inf.pca_apply(q_desc, pca, device)
        space = "pca4096p0.5"
        del pca, fit
        torch.cuda.empty_cache()
    print(f"{cli.dataset}: {n_db} db x {n_q} q, model {label}, global space {space}")

    # ---- global stage: ranked shortlists + their cosine scores --------------------
    db_chunk = npl.DB_CHUNK if family == "nsavp" else None
    ranked, scores = bp.topk_ranked(db_desc, q_desc, device, k=cli.top_k, chunk=256,
                                    db_chunk=db_chunk, return_scores=True)
    rec_g, _, _ = bp.recall_from_ranked(ranked, gt)
    print("  global   " + "  ".join(f"R@{k}={rec_g[k]:.4f}" for k in KS))

    # ---- token extraction over the shortlist union + queries ----------------------
    union = np.unique(ranked)
    print(f"  union: {len(union)} of {n_db} db frames "
          f"({100 * len(union) / n_db:.1f}%) + {n_q} queries")
    # pooled row -> (traverse, original frame index)
    row_seq = np.empty(n_db, dtype=np.int32)
    row_idx = np.empty(n_db, dtype=np.int64)
    for i, s in enumerate(database):
        m = source == i
        row_seq[m] = i
        row_idx[m] = np.flatnonzero(geom[s][1])
    model, cfg, step = inf.load_model(ckpt, device)
    cfg.H = cfg.W = RESOLUTION

    # The store's rows follow THIS ranking's shortlist union, so a different global space
    # (different shortlists) must never reuse another space's store. Reuse is gated on a
    # manifest written only after BOTH stores finish: the memmaps are preallocated at full
    # size, so a killed extraction leaves valid-looking files full of zero rows.
    store_path = os.path.join(out_dir, f"{label}_k{cli.top_k}_d{cli.token_dim}"
                              + ("_wht" if cli.whiten else ""))
    union_sha = hashlib.sha256(union.tobytes()).hexdigest()
    meta_path = store_path + ".meta.json"
    reuse = False
    if os.path.exists(meta_path):
        with open(meta_path) as h:
            meta = json.load(h)
        reuse = (meta.get("union_sha256") == union_sha
                 and meta.get("n_queries") == int(n_q)
                 and meta.get("token_dim") == cli.token_dim)
        if not reuse:
            print(f"  store manifest mismatch at {meta_path}: rebuilding", flush=True)
    elif os.path.exists(store_path + ".db.tokens.npy"):
        print("  store files present but no completion manifest: rebuilding", flush=True)
    if not reuse:
        datasets = {s: frame_dataset(family, eventlab_dir, s, cfg)
                    for s in sequences}
        # db store rows follow `union` order; query rows follow the query bank's rows
        plan = {}
        for i, s in enumerate(database):
            in_seq = row_seq[union] == i
            plan[s] = (datasets[s], row_idx[union[in_seq]], np.flatnonzero(in_seq))
        pca = fit_token_pca(model, cfg, device,
                            {s: (d, i) for s, (d, i, _) in plan.items()}, cli.token_dim)
        store = TokenStore(store_path + ".db", len(union), cli.token_dim, "w+")
        extract_tokens(model, cfg, device, plan, pca, store, cli.batch_size)
        qstore = TokenStore(store_path + ".q", n_q, cli.token_dim, "w+")
        qplan = {query: (datasets[query], np.flatnonzero(q_cov), np.arange(n_q))}
        extract_tokens(model, cfg, device, qplan, pca, qstore, cli.batch_size)
        del store, qstore
        with open(meta_path, "w") as h:
            json.dump({"union_sha256": union_sha, "union_frames": int(len(union)),
                       "n_queries": int(n_q), "token_dim": cli.token_dim,
                       "top_k": cli.top_k, "global_space": space, "ckpt": ckpt,
                       "built": time.strftime("%Y-%m-%dT%H:%M:%S")}, h, indent=1)
    db_store = TokenStore.open(store_path + ".db")
    q_store = TokenStore.open(store_path + ".q")
    row_of = np.full(n_db, -1, dtype=np.int64)
    row_of[union] = np.arange(len(union))

    # ---- verification pass --------------------------------------------------------
    out_ranked = ranked.copy()
    t0, verified_pairs = time.time(), 0
    for q in range(n_q):
        cand = ranked[:, q]
        rows = row_of[cand]
        new = rerank_query(q_store.tok[q], q_store.act[q], db_store.tok[rows],
                           db_store.act[rows], scores[:, q], cli.ransac_thresh,
                           cli.inlier_weight, cli.min_activity, device)
        order = np.argsort(-new, kind="stable")
        out_ranked[:, q] = cand[order]
        verified_pairs += int((new != scores[:, q]).sum())
        if (q + 1) % 1000 == 0:
            rate = (q + 1) / (time.time() - t0)
            print(f"    rerank {q + 1}/{n_q}  {rate:.1f} q/s "
                  f"eta {(n_q - q - 1) / rate / 60:.1f} min", flush=True)
    rec_r, _, _ = bp.recall_from_ranked(out_ranked, gt)
    print("  reranked " + "  ".join(f"R@{k}={rec_r[k]:.4f}" for k in KS))
    print(f"  {verified_pairs} of {cli.top_k * n_q} pairs verified "
          f"({100 * verified_pairs / (cli.top_k * n_q):.1f}%)")

    if cli.inlier_weight == 0.0 and not np.array_equal(out_ranked, ranked):
        raise SystemExit("identity check FAILED: weight 0 must reproduce the global "
                         "ranking exactly")

    tag = (f"{label}_{space}_k{cli.top_k}_d{cli.token_dim}_w{cli.inlier_weight}"
           f"_a{cli.min_activity}")
    out = {"dataset": cli.dataset, "model": label, "ckpt": ckpt, "step": step,
           "global_space": space,
           "protocol": "pooled, 25 m, r322ba50 banks",
           "top_k": cli.top_k, "ransac_thresh": cli.ransac_thresh,
           "inlier_weight": cli.inlier_weight, "token_dim": cli.token_dim,
           "min_activity": cli.min_activity,
           "n_database": int(n_db), "n_queries": int(n_q),
           "union_frames": int(len(union)),
           "verified_pair_fraction": verified_pairs / (cli.top_k * n_q),
           "recall_global": {str(k): rec_g[k] for k in KS},
           "recall_reranked": {str(k): rec_r[k] for k in KS}}
    path = os.path.join(out_dir, f"rerank_{tag}.json")
    with open(path, "w") as h:
        json.dump(out, h, indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
