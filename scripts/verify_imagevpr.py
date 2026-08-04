"""Checks on the image-set evaluation that a smoke run cannot make.

Run from the repo root::

    pixi run python3 scripts/verify_imagevpr.py \
        --data-dir /media/adam/vprdatasets/megaevent -d tokyo247

Three things are asserted, in the order they would break:

1. **Render parity.** ``src.imagevpr.load_countmask`` must be byte-identical to I2E's
   reference renderer. eventcv and I2E implement countmask twice, in Rust and in NumPy,
   and only equality keeps a model trained on one from being evaluated on the other. Needs
   the I2E checkout; skipped with a warning if it is not there.
2. **Ground-truth statistics.** Recomputed from the filenames and printed, so a change in
   the converter or the threshold shows up as a number rather than as a shifted recall.
3. **Descriptor sanity.** SALAD descriptors are L2-normalised by construction, and two
   tiles of the same panorama must score higher against each other than against an
   unrelated query. A failure here means the frames are not reaching the model intact.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.imagevpr import (           # noqa: E402
    build_gt, list_npz, load_countmask, split_dir, utm_from_paths,
)
from src.inference import CKPT_DIR, eval_transform, load_model   # noqa: E402

I2E_ROOT = "/home/adam/repo/I2E"


def check_render_parity(paths, n=3):
    """eventcv's countmask vs ``i2e_infer.countmask_numpy``, byte for byte."""
    if not os.path.isdir(I2E_ROOT):
        print(f"  SKIP: no I2E checkout at {I2E_ROOT}")
        return True
    sys.path.insert(0, I2E_ROOT)
    from i2e_infer import countmask_numpy

    ok = True
    for path in paths[:n]:
        with np.load(path) as npz:
            x, y, t, p = npz["x"], npz["y"], npz["t"], npz["p"]
            height, width = (int(v) for v in npz["resolution"])
        ours = load_countmask(path)
        # The reference returns float32 in [0,255] CHW; render_countmask is what casts it.
        theirs = np.clip(countmask_numpy(x, y, t, p, height, width,
                                         pct=99.0, white_frame=False), 0, 255).astype(np.uint8)
        same = np.array_equal(ours, theirs)
        ok &= same
        print(f"  {'OK  ' if same else 'FAIL'} {ours.shape} {ours.dtype}  "
              f"maxdiff={int(np.abs(ours.astype(int) - theirs.astype(int)).max())}  "
              f"{os.path.basename(path)[:48]}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", "-d", default="tokyo247")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--ref", default="database")
    parser.add_argument("--query", default="queries")
    parser.add_argument("--positive-dist-threshold", type=float, default=25.0)
    parser.add_argument("--model", "-m", default="s_salad_ft4")
    args = parser.parse_args()

    db_paths = list_npz(split_dir(args, args.ref))
    q_paths = list_npz(split_dir(args, args.query))
    print(f"{args.dataset}: {len(db_paths)} {args.ref}, {len(q_paths)} {args.query}\n")

    print("1. render parity vs the I2E reference")
    ok = check_render_parity(db_paths)
    ok &= check_render_parity(q_paths)

    print("\n2. ground truth from filenames")
    db_utm, q_utm = utm_from_paths(db_paths), utm_from_paths(q_paths)
    gt = build_gt(db_utm, q_utm, args.positive_dist_threshold)
    per_query = gt.sum(0)
    print(f"  gt {gt.shape} [db, query], {args.positive_dist_threshold:g} m")
    print(f"  scorable {int((per_query > 0).sum())}/{len(q_paths)}   "
          f"positives per query: mean {per_query.mean():.1f} "
          f"min {per_query.min()} max {per_query.max()}")
    print(f"  chance R@1 for a uniform-random ranking: "
          f"{per_query.mean() / len(db_paths):.5f}")
    ok &= bool((per_query > 0).all())

    print("\n3. descriptors through the real model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, step = load_model(os.path.join(CKPT_DIR, f"{args.model}.pt"), device)
    transform = eval_transform(cfg)
    sample = db_paths[:4] + q_paths[:4]
    batch = torch.stack([
        transform(torch.from_numpy(np.ascontiguousarray(load_countmask(p))).float().div_(255.0))
        for p in sample]).to(device)
    with torch.no_grad():
        desc = model(batch).float().cpu()
    norms = desc.norm(dim=1)
    unit = bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))
    sim = (desc @ desc.t()).numpy()
    ordered = sim[0, 1] > sim[0, 4]
    print(f"  batch {tuple(batch.shape)} -> descriptors {tuple(desc.shape)}")
    print(f"  {'OK  ' if unit else 'FAIL'} L2 norms unit "
          f"(min {norms.min():.6f} max {norms.max():.6f})")
    print(f"  {'OK  ' if ordered else 'FAIL'} same-pano tiles {sim[0, 1]:.3f} > "
          f"database-vs-query {sim[0, 4]:.3f}")
    ok &= unit and ordered

    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
