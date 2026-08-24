"""H1: is countmask "an edge image with adaptive contrast", and is that why MegaLoc reads it?

    pixi run python3 scripts/mechanism_edges.py [--stride 1] [--model megaloc]

Five arms, all scored under the sim2real pooled protocol (query sunset1, database
daytime+morning+sunrise, APS-aligned rows, 25 m, native cosine), all through one
unretrained RGB model, so the only variable is what the frames show:

  aps_gray     the co-recorded APS intensity frame, grayscale -> 3ch. The RGB ceiling:
               how well the model does when given the actual photograph of the scene.
  aps_sobel    Sobel gradient magnitude of that photograph, per-frame 99th-percentile
               normalised, -> 3ch. "An edge image of the scene."
  aps_sobelpol signed Sobel rendered *as a countmask*: R = positive vertical-ish
               gradient, B = negative, G = activity mask, same per-frame alpha. The
               closest photographic analogue of what an event camera hands us.
  real_cm      countmask of the real event stream (eventcv render, the benchmark input).
  real_cm_ga   the same events with a *global* (per-traverse) alpha instead of the
               per-frame adaptive one — the ablation for "adaptive contrast is the bridge".

If MegaLoc(aps_sobelpol) ~ MegaLoc(real_cm), the mechanism "events arrive pre-converted
to the edge domain RGB models already know" is established with numbers. If real_cm_ga
falls well below real_cm, the per-frame alpha (adaptive contrast normalisation) is doing
real work in that bridge. aps_gray calibrates how much of the photograph's information
the event stream preserves at all.

Frames come from the index-aligned trees under brisbane_npz/brisbane_event/{aps,real}.
Geometry, pooling, ranking: scripts/brisbane_pooled.py's, unchanged.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import cv2  # noqa: E402

import brisbane_pooled as bp  # noqa: E402
from brisbane_resolution import _Args  # noqa: E402
from src import traversenpz as tnpz  # noqa: E402
from src.imagevpr import build_gt  # noqa: E402
from src.methods import build_dino_salad, megaloc_transform  # noqa: E402
from src.npzdata import read_events  # noqa: E402

ROOT = "/media/adam/vprdatasets/megaevent"
NPZ_ROOT = f"{ROOT}/brisbane_npz"
OUT = f"{ROOT}/mechanism"
QUERY = "sunset1"
DATABASE = ("daytime", "morning", "sunrise")     # the sim2real protocol: night excluded
ARMS = ("aps_gray", "aps_sobel", "aps_sobelpol", "real_cm", "real_cm_ga")


def aps_path(seq, i):
    return os.path.join(NPZ_ROOT, "brisbane_event", "aps", seq, f"frame_{i:06d}.png")


_REAL_PATHS = {}


def real_paths(seq):
    if seq not in _REAL_PATHS:
        _REAL_PATHS[seq] = tnpz.frame_paths(NPZ_ROOT, "brisbane_event", "real", seq)
    return _REAL_PATHS[seq]


def _percentile_alpha(counts):
    nz = counts[counts > 0]
    return float(np.percentile(nz, 99.0)) if nz.size else 1.0


def _as_countmask(pos, neg, alpha):
    r = np.clip(pos, 0, alpha) / alpha
    b = np.clip(neg, 0, alpha) / alpha
    g = ((pos + neg) > 0).astype(np.float32)
    return np.stack([r, g, b]).astype(np.float32)


def render(arm, seq, i, global_alpha=None):
    """One frame of one arm -> float32 [3,H,W] in [0,1]."""
    if arm.startswith("aps"):
        img = np.asarray(Image.open(aps_path(seq, i)).convert("L"), dtype=np.float32)
        if arm == "aps_gray":
            return np.repeat((img / 255.0)[None], 3, axis=0)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        if arm == "aps_sobel":
            mag = np.hypot(gx, gy)
            alpha = _percentile_alpha(mag)
            return np.repeat((np.clip(mag, 0, alpha) / alpha)[None], 3, axis=0)
        # aps_sobelpol: signed gradient as polarity, countmask channel layout
        s = gx + gy
        pos, neg = np.maximum(s, 0), np.maximum(-s, 0)
        alpha = _percentile_alpha(np.concatenate([pos[pos > 0], neg[neg > 0]])
                                  if (pos > 0).any() or (neg > 0).any()
                                  else np.zeros(1))
        return _as_countmask(pos, neg, alpha)
    # real event arms
    x, y, _, p, height, width = read_events(real_paths(seq)[i])
    pos = np.zeros((height, width), np.float32)
    neg = np.zeros((height, width), np.float32)
    if x.size:
        np.add.at(pos, (y[p == 1], x[p == 1]), 1.0)
        np.add.at(neg, (y[p != 1], x[p != 1]), 1.0)
    if arm == "real_cm":
        alpha = _percentile_alpha(np.concatenate([pos[pos > 0], neg[neg > 0]])
                                  if x.size else np.zeros(1))
    else:                                           # real_cm_ga
        alpha = global_alpha
    return _as_countmask(pos, neg, alpha)


def traverse_global_alpha(seq, idx):
    """Median per-frame alpha over a sample — the traverse-constant the ablation uses."""
    alphas = []
    for i in idx[:: max(1, len(idx) // 300)]:
        x, y, _, p, height, width = read_events(real_paths(seq)[i])
        if not x.size:
            continue
        pos = np.zeros((height, width), np.float32)
        neg = np.zeros((height, width), np.float32)
        np.add.at(pos, (y[p == 1], x[p == 1]), 1.0)
        np.add.at(neg, (y[p != 1], x[p != 1]), 1.0)
        alphas.append(_percentile_alpha(np.concatenate([pos[pos > 0], neg[neg > 0]])))
    return float(np.median(alphas))


def bank_for(model, transform, arm, seq, keep_idx, device, batch, tag):
    path = os.path.join(OUT, f"{tag}_{arm}_{seq}.npy")
    if os.path.exists(path):
        return np.load(path)
    ga = traverse_global_alpha(seq, keep_idx) if arm == "real_cm_ga" else None
    out, buf = [], []
    with torch.no_grad():
        for n, i in enumerate(keep_idx):
            buf.append(torch.from_numpy(render(arm, seq, i, global_alpha=ga)))
            if len(buf) == batch or n == len(keep_idx) - 1:
                x = transform(torch.stack(buf)).to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    out.append(model(x).float().cpu())
                buf = []
                if (n + 1) % (batch * 20) < batch:
                    print(f"    {arm}/{seq}: {n + 1}/{len(keep_idx)}", flush=True)
    bank = torch.cat(out).numpy()
    np.save(path, bank)
    return bank


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="megaloc", choices=["megaloc", "salad"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--stride", type=int, default=1,
                    help="frame stride; >1 thins BOTH query and database uniformly for a "
                         "faster (internally comparable, protocol-incomparable) pass")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eventlab-dir", default="/media/adam/vprdatasets/eventgem")
    cli = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = _Args(cli.eventlab_dir, "brisbane_event", 50, False, False, npz_root=NPZ_ROOT)
    seqs = [QUERY, *DATABASE]
    geom = bp.traverse_geometry(args, seqs, NPZ_ROOT, aps_align=True)
    keep = {s: np.flatnonzero(geom[s][1])[:: cli.stride] for s in seqs}
    xy = {s: geom[s][0][keep[s]] for s in seqs}
    gt = build_gt(np.concatenate([xy[s] for s in DATABASE]), xy[QUERY], 25.0)
    print(f"  {sum(len(keep[s]) for s in DATABASE)} db x {len(keep[QUERY])} q "
          f"(stride {cli.stride}), scorable {int((gt.sum(0) > 0).sum())}")

    if cli.model == "megaloc":
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model",
                               source="github", trust_repo=True).eval().to(device)
    else:
        model = build_dino_salad().eval().to(device)
    transform = megaloc_transform(322)
    tag = f"{cli.model}_s{cli.stride}"

    results = {}
    for arm in cli.arms:
        banks = {s: bank_for(model, transform, arm, s, keep[s], device,
                             cli.batch_size, tag) for s in seqs}
        db = torch.from_numpy(np.concatenate([banks[s] for s in DATABASE]))
        q = torch.from_numpy(banks[QUERY])
        ranked = bp.topk_ranked(db, q, device, chunk=256)
        rec, _, _ = bp.recall_from_ranked(ranked, gt)
        results[arm] = {str(k): v for k, v in rec.items()}
        print(f"  {arm:14s} " + "  ".join(f"R@{k}={rec[k]:.4f}" for k in bp.KS), flush=True)

    out_json = os.path.join(OUT, f"edges_{tag}.json")
    with open(out_json, "w") as h:
        json.dump({"model": cli.model, "stride": cli.stride, "query": QUERY,
                   "database": list(DATABASE), "threshold_m": 25.0,
                   "protocol": "sim2real pooled, aps-aligned", "recall": results}, h,
                  indent=1)
    print(f"-> {out_json}")


if __name__ == "__main__":
    main()
