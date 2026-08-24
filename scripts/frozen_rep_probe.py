"""Frozen GEPT encoder scored on pooled Brisbane under countmask vs accumulate rendering.

Sizes the representation-mismatch damage at init (see memory: gept-diet-is-accumulate-
not-countmask): the released GEPT checkpoints were pretrained on ``accumulate_to_rgb``
(white background, winner-take-all polarity, soft inverse mask), while every MegaEvent
model trains and evals on ``countmask`` (paper Eq. 2). This probe runs the UNTOUCHED
``gept/small.pt`` encoder — zero fine-tuning, zero trained head — over the pooled
Brisbane protocol with both renderings and compares retrieval directly.

Protocol notes (kept honest):
* Frames come from the raw-event npz tree (``brisbane_npz/brisbane_event/real``), which
  is hot-pixel filtered but NOT background-activity filtered — the published banks are
  ``ba50``. Absolute numbers therefore sit slightly off the published ones; the
  countmask-vs-accumulate comparison is exactly paired (same events, same frames).
* npz index == HDF5 frame index (dump.json: same source, origin, dt; n_slices matches
  ``traverse_geometry``; renderer parity pinned by tests/test_countmask_identity.py).
* Pooling is untrained: L2-normalised mean of patch tokens (primary) and CLS token
  (secondary). Whitened space = full-dim (384) PCA power 0.5 fit on the pooled database
  with the same seeded subsample brisbane_pooled uses.
* Each rendering is normalised with its own matched stats: countmask with the measured
  COUNTMASK stats, accumulate with GEP's own DSEC stats (the closest driving domain,
  ``gept/src/config.py`` DSEC_ME/DSEC_SE).

Output: ``/media/adam/vprdatasets/megaevent/frozen_rep_probe/results.json`` + a table.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "external", "dinov2"))

import brisbane_pooled as bp  # noqa: E402
from brisbane_resolution import _Args  # noqa: E402
from src import inference as inf  # noqa: E402
from src import scoring  # noqa: E402
from src.imagevpr import build_gt  # noqa: E402
from src.methods import COUNTMASK_MEAN, COUNTMASK_STD  # noqa: E402

# gept's byte-verified renderer ports; appended last so nothing shadows megaevent's src.
sys.path.append("/home/adam/repo/gept/src/vpr")
from representations import accumulate_numpy, countmask_numpy  # noqa: E402

GEPT_CKPT = "/home/adam/repo/gept/small.pt"        # the actual ViT-S init (0 registers)
NPZ_ROOT = "/media/adam/vprdatasets/megaevent/brisbane_npz/brisbane_event/real"
OUT_DIR = "/media/adam/vprdatasets/megaevent/frozen_rep_probe"
EVENTLAB = "/media/adam/vprdatasets/eventgem"
QUERY = "sunset1"
DATABASE = ("sunset2", "daytime", "morning", "night", "sunrise")
RESOLUTION = 322
KS = (1, 5, 10, 20)

# GEP's own normalisation for the accumulate diet: gept/src/config.py DSEC_ME / DSEC_SE.
ACCUMULATE_MEAN = (0.8993729784963826, 0.7969581014619264, 0.8928228776286392)
ACCUMULATE_STD = (0.22043360769748688, 0.2921656668186188, 0.22049927711486816)

REPS = {
    "countmask": (countmask_numpy, COUNTMASK_MEAN, COUNTMASK_STD),
    "accumulate": (accumulate_numpy, ACCUMULATE_MEAN, ACCUMULATE_STD),
}


class NpzRepDataset(Dataset):
    """Raw-event npz frames -> rendered, normalised, resized tensors."""

    def __init__(self, seq, indices, render, mean, std):
        self.dir = os.path.join(NPZ_ROOT, seq)
        self.indices = np.asarray(indices)
        self.render = render
        self.tf = transforms.Compose([          # Normalize THEN Resize: megaevent order
            transforms.Normalize(mean, std),
            transforms.Resize((RESOLUTION, RESOLUTION),
                              interpolation=transforms.InterpolationMode.BICUBIC),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        d = np.load(os.path.join(self.dir, f"frame_{self.indices[i]:06d}.npz"))
        W, H = int(d["resolution"][0]), int(d["resolution"][1])
        chw = self.render(d["x"], d["y"], d["t"], d["p"], H, W)     # uint8 CHW
        return self.tf(torch.from_numpy(chw).float() / 255.0)


def build_frozen_encoder(device):
    from dinov2.models.vision_transformer import vit_small
    enc = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6,
                    num_register_tokens=0)
    sd = torch.load(GEPT_CKPT, map_location="cpu", weights_only=False)["event_encoder"]
    enc.load_state_dict(sd, strict=True)
    print(f"[probe] loaded {GEPT_CKPT} strict into ViT-S/14 (0 registers)")
    return enc.to(device).eval()


@torch.no_grad()
def extract(enc, seq, indices, render, mean, std, device, batch, workers):
    ds = NpzRepDataset(seq, indices, render, mean, std)
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
    mean_pool, cls_pool = [], []
    t0, done = time.time(), 0
    for x in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            feats = enc.forward_features(x)
        mean_pool.append(feats["x_norm_patchtokens"].mean(dim=1).float().cpu())
        cls_pool.append(feats["x_norm_clstoken"].float().cpu())
        done += x.shape[0]
        if done % 3200 < batch:
            rate = done / (time.time() - t0)
            print(f"    {seq}: {done}/{len(ds)}  {rate:.1f} f/s", flush=True)
    out = {}
    for name, chunks in (("meanpatch", mean_pool), ("cls", cls_pool)):
        d = torch.cat(chunks)
        out[name] = torch.nn.functional.normalize(d, dim=1)
    return out


def score(db_desc, q_desc, gt, device):
    ranked = bp.topk_ranked(db_desc, q_desc, device, k=max(KS), chunk=256)
    rec, _, _ = bp.recall_from_ranked(ranked, gt)
    return {str(k): float(rec[k]) for k in KS}


def whiten(db_desc, q_desc, device):
    fit = db_desc
    if fit.size(0) > scoring.PCA_FIT_SAMPLES:
        gen = torch.Generator().manual_seed(scoring.PCA_FIT_SEED)
        fit = db_desc[torch.randperm(db_desc.size(0), generator=gen)
                      [:scoring.PCA_FIT_SAMPLES]]
    pca = inf.pca_fit(fit, device, dim=min(fit.size(1), fit.size(0) - 1), power=0.5)
    return inf.pca_apply(db_desc, pca, device), inf.pca_apply(q_desc, pca, device)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stride", type=int, default=1,
                    help="frame stride for smoke runs (1 = full pooled protocol)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reps", nargs="+", default=list(REPS), choices=list(REPS))
    cli = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    sequences = [QUERY, *DATABASE]

    args = _Args(EVENTLAB, "brisbane_event", 50, False, False)
    args.filter_dt_us = 50_000
    geom = bp.traverse_geometry(args, sequences, None)

    plan, xys = {}, {}
    for s in sequences:
        xy, cov = geom[s][0], geom[s][1]
        idxs = np.flatnonzero(cov)[::cli.stride]
        plan[s] = idxs
        xys[s] = xy[cov][::cli.stride]
    db_xy = np.concatenate([xys[s] for s in DATABASE])
    gt = build_gt(db_xy, xys[QUERY], 25.0)
    print(f"[probe] pooled {sum(len(plan[s]) for s in DATABASE)} db x "
          f"{len(plan[QUERY])} q (stride {cli.stride}), 25 m")

    enc = build_frozen_encoder(device)
    results = {"protocol": {"stride": cli.stride, "resolution": RESOLUTION,
                            "npz_filters": "hot_pixel only, NO ba50 (see docstring)",
                            "ckpt": GEPT_CKPT},
               "reps": {}}
    for rep in cli.reps:
        render, mean, std = REPS[rep]
        print(f"\n=== {rep} (stats mean {np.round(mean, 3).tolist()}) ===", flush=True)
        banks = {s: extract(enc, s, plan[s], render, mean, std, device,
                            cli.batch_size, cli.workers) for s in sequences}
        results["reps"][rep] = {}
        for pool in ("meanpatch", "cls"):
            db_desc = torch.cat([banks[s][pool] for s in DATABASE])
            q_desc = banks[QUERY][pool]
            native = score(db_desc, q_desc, gt, device)
            db_w, q_w = whiten(db_desc, q_desc, device)
            wht = score(db_w, q_w, gt, device)
            results["reps"][rep][pool] = {"native": native, "whitened": wht}
            print(f"  {pool:10s} native   " +
                  "  ".join(f"R@{k}={native[str(k)]:.4f}" for k in KS))
            print(f"  {pool:10s} whitened " +
                  "  ".join(f"R@{k}={wht[str(k)]:.4f}" for k in KS))
        with open(os.path.join(OUT_DIR, "results.json"), "w") as h:
            json.dump(results, h, indent=1)
        del banks

    if len(results["reps"]) == 2:
        a = results["reps"]["accumulate"]["meanpatch"]["whitened"]["1"]
        c = results["reps"]["countmask"]["meanpatch"]["whitened"]["1"]
        print(f"\n[probe] whitened meanpatch R@1: accumulate {a:.4f} vs countmask "
              f"{c:.4f}  (delta {a - c:+.4f})")
    print(f"-> {os.path.join(OUT_DIR, 'results.json')}")


if __name__ == "__main__":
    main()
