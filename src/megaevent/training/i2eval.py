"""Held-out MSLS monitoring with independent per-city ground truth."""

import os

import numpy as np
import pandas as pd
import torch

from .eval import extract_paths, recall_at_k_cross

_INDEX_MEMO = {}


def _split_table(images_dir, meta_root, city, split, ext):
    """Join a converted MSLS split to its metric coordinates by image key."""
    csv_path = os.path.join(meta_root, city, split, "postprocessed.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    table = pd.read_csv(csv_path)
    missing = {"key", "easting", "northing"} - set(table.columns)
    if missing:
        raise ValueError(f"{csv_path} has no {sorted(missing)} columns")
    image_dir = os.path.join(images_dir, city, split, "images")
    have = set(os.listdir(image_dir))
    subset = table.iloc[[i for i, key in enumerate(table["key"]) if f"{key}{ext}" in have]]
    paths = [os.path.join(image_dir, f"{key}{ext}") for key in subset["key"]]
    coordinates = subset[["easting", "northing"]].to_numpy(dtype=np.float64)
    return paths, coordinates


def _within(query, reference, radius_m):
    return ((query[:, None, :] - reference[None, :, :]) ** 2).sum(-1) <= radius_m**2


def msls_paths(cfg):
    """``(image_root, ext)`` for the MSLS stream, resolved exactly as training resolves it.

    Reuses ``streams``' own root search and extension rules so the monitor cannot end up
    reading a different tree — or a different image format — than the training stream does.
    """
    from . import streams

    spec = streams.REGISTRY["msls"]
    root = streams._image_root(cfg, spec)
    if root is None:
        raise FileNotFoundError(
            f"MSLS images not found under {cfg.data_root} (tried {spec.subdirs}); the in-domain monitor needs the same converted tree the msls stream trains on"
        )
    return (root, streams._ext(cfg, spec))


def build_msls_val(
    images_dir,
    meta_root,
    cities,
    ext=".jpg",
    max_db=5000,
    max_q=1000,
    radius_m=25.0,
    seed=0,
    verbose=True,
):
    """Held-out-city retrieval set -> ``(query_paths, ref_paths, gt[nq, nr] bool, spans)``.

    ``spans`` is ``[(city, q_start, q_stop), ...]``, which is what lets the monitor report a
    per-city breakdown as well as the pooled number. That matters because **the cities differ
    in how held-out they really are.** MSLS's official validation pair is ``cph`` + ``sf``, but
    two of the six training streams are ``sf_xl_frontal`` / ``sf_xl_lateral``, which are
    entirely San Francisco — so for ``sf`` the *images* are unseen while the *city* is not.
    (``cph`` is clean: Copenhagen is in neither SF-XL nor GSV-Cities' 23 cities.) The
    contamination is constant across checkpoints, so it does not bias selection between them;
    it does mean the pooled number must not be reported as held-out geography. Logging each
    city separately keeps the clean sub-number available instead of burying the caveat.

    Per-city budgets (rather than one global subsample) so every held-out city is
    represented no matter how their sizes differ. The gallery is drawn first and the queries
    are then drawn **from those that have at least one GT match in it** — subsampling a
    gallery orphans queries, and a query set full of unanswerable queries would just add
    variance to the monitor.
    """
    cities = [c for c in cities if c]
    if not cities:
        raise ValueError("no cities given for the in-domain monitor")
    rng = np.random.default_rng(seed)
    per_db = max(1, max_db // len(cities))
    per_q = max(1, max_q // len(cities))
    q_paths, r_paths, blocks, report = ([], [], [], [])
    for city in cities:
        db_p, db_c = _split_table(images_dir, meta_root, city, "database", ext)
        q_p, q_c = _split_table(images_dir, meta_root, city, "query", ext)
        if not db_p or not q_p:
            raise RuntimeError(
                f"[i2eval] {city}: {len(db_p)} database / {len(q_p)} query images found — check the converted tree and {ext!r}"
            )
        sel_db = rng.permutation(len(db_p))[:per_db]
        sel_db.sort()
        db_p = [db_p[i] for i in sel_db]
        db_c = db_c[sel_db]
        answerable = np.flatnonzero(_within(q_c, db_c, radius_m).any(axis=1))
        if answerable.size == 0:
            raise RuntimeError(
                f"[i2eval] {city}: no query is within {radius_m} m of any sampled database image. Either the two splits do not overlap spatially or the coordinate columns are not in metres."
            )
        sel_q = answerable[rng.permutation(answerable.size)[:per_q]]
        sel_q.sort()
        q_p = [q_p[i] for i in sel_q]
        q_c = q_c[sel_q]
        qs, rs = (len(q_paths), len(r_paths))
        q_paths.extend(q_p)
        r_paths.extend(db_p)
        blocks.append((city, qs, len(q_paths), rs, len(r_paths), _within(q_c, db_c, radius_m)))
        report.append(f"{city}={len(q_p)}q/{len(db_p)}r")
    gt = np.zeros((len(q_paths), len(r_paths)), dtype=bool)
    spans = []
    for city, qs, qe, rs, re_, blk in blocks:
        gt[qs:qe, rs:re_] = blk
        spans.append((city, qs, qe))
    if verbose:
        print(
            f"[i2eval] held-out MSLS: {len(q_paths)} queries / {len(r_paths)} references ({', '.join(report)}), GT radius {radius_m:g} m, density {gt.mean():.4f}"
        )
    return (q_paths, r_paths, gt, spans)


def msls_val_index(cfg, verbose=True):
    """``build_msls_val`` driven by a config, memoised for the life of the process."""
    images_dir, ext = msls_paths(cfg)
    key = (
        images_dir,
        cfg.msls_meta_root,
        tuple(cfg.msls_val_cities),
        ext,
        cfg.msls_val_max_db,
        cfg.msls_val_max_q,
        cfg.msls_val_radius_m,
        cfg.msls_val_seed,
    )
    if key not in _INDEX_MEMO:
        _INDEX_MEMO[key] = build_msls_val(
            images_dir,
            cfg.msls_meta_root,
            cfg.msls_val_cities,
            ext=ext,
            max_db=cfg.msls_val_max_db,
            max_q=cfg.msls_val_max_q,
            radius_m=cfg.msls_val_radius_m,
            seed=cfg.msls_val_seed,
            verbose=verbose,
        )
    return _INDEX_MEMO[key]


@torch.no_grad()
def evaluate(model, cfg, device, ks=(1, 5, 10), num_workers=4, batch_size=None, verbose=True):
    """Score the current weights on the held-out MSLS cities -> ``{"R@1": ..., ...}``.

    Same transform as training's eval path (``dataset.build_transforms(train=False)``), so
    this is the training distribution at the training resolution — the axis Brisbane cannot
    report on.

    ``pca`` (``{"dim":, "power":, "eps":}``) additionally whitens on the reference bank and
    reports ``R@k_pca``, exactly as ``evalsuite.run_suite`` does. This exists so that
    ``--select-on whitened`` has a whitened number on *both* halves of the combined
    criterion: the real-event conditions already reported one, the in-domain monitor did
    not, and selecting on a mixture of whitened and native recall would be worse than
    selecting on either.
    """
    q_paths, r_paths, gt, spans = msls_val_index(cfg, verbose=verbose)
    bs = batch_size or cfg.eval_batch_size
    r_desc = extract_paths(model, r_paths, cfg, device, bs, num_workers)
    q_desc = extract_paths(model, q_paths, cfg, device, bs, num_workers)
    rec = recall_at_k_cross(q_desc, r_desc, gt, ks=ks, device=device)
    out = {f"R@{k}": v for k, v in rec.items()}
    out.update(nq=len(q_paths), nr=len(r_paths), gt_density=float(gt.mean()))
    if len(spans) > 1:
        for city, qs, qe in spans:
            city_rec = recall_at_k_cross(q_desc[qs:qe], r_desc, gt[qs:qe], ks=ks, device=device)
            out[f"R@1/{city}"] = city_rec[ks[0]]
    return out
