"""Whose patch features read an event frame best, with the aggregation head removed?

    pixi run python3 scripts/backbone_probe.py --stride 5
    pixi run python3 scripts/backbone_probe.py --backbones gept-s dinov2-s --stride 5 --gate

The mechanism question behind the whole investigation is "is our edge in the features or in
the descriptor". Every reported number confounds them: a checkpoint is a backbone *and* a
trained aggregator, and the aggregators differ between methods. This strips the aggregator
off every model and pools the patch tokens with a **parameter-free** L2-normalised mean (CLS
reported beside it), so all that varies is which encoder read the frame. Same frames, same
rendering, same pooling, same ranking, same 25 m ground truth.

**Protocol: single reference traverse**, matching `scripts/pairwise_sunset_ref.py` and the
`pairwise_sunset_ref_*.json` ledger — gallery = sunset1, one query traverse per cell
(daytime / morning / sunrise). This is deliberately NOT the pooled protocol: a pooled
gallery holds several traverses of the same route, so a query only needs one of them to be
an easy appearance match, and the cell stops measuring cross-condition retrieval. Each cell
here forces one illumination pair.

The roster spans the three things that could be responsible:

  dinov2-s / dinov2-b   Meta's RGB DINOv2 — no event exposure, no VPR training
  gept-s / gept-b       the same architecture after GEPT event pretraining, no VPR training
  megaloc / salad       RGB DINOv2 after large-scale *RGB VPR* training
  v8-b                  ours: GEPT init + VPR training on I2E event frames

``gept-s`` and ``dinov2-s`` are an exact architectural match (ViT-S/14, 0 registers), which
is why the ViT-S rung is the clean one; GEPT base carries 4 registers, so its RGB counterpart
must be the ``reg4`` checkpoint or the comparison is not like-for-like.

**Stride.** Every arm is scored on identically strided frames; the stride shrinks the
gallery, so absolute R@1 is not comparable to a full-protocol cell — only the arms are
comparable to each other, which is what this probe is for.

**Normalisation.** Each RGB backbone is run under both ImageNet stats (what it shipped with)
and accumulate-matched stats (src/methods.py::ACCUMULATE_MEAN — accumulate is a white
background render that ImageNet centres 2.0-4.5 sigma off), and the better arm is the one
that counts. Handicapping a control on input statistics is the defect this investigation
already found once; it is not repeated here.

The adapter layer below reaches into four different module layouts, so it is pinned by
``tests/test_backbone_probe.py`` rather than trusted: the ``[B, D, h, w] -> [B, h*w, D]``
reshape must preserve the token set exactly (its mean equals the spatial mean of the map),
and every backbone must return the same 529-token grid at 322.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "external", "dinov2"))

import brisbane_pooled as bp                                       # noqa: E402
import frozen_rep_probe as frp                                     # noqa: E402
from brisbane_resolution import _Args                              # noqa: E402
from src.imagevpr import build_gt                                  # noqa: E402
from src.methods import (ACCUMULATE_MEAN, ACCUMULATE_STD,          # noqa: E402
                         IMAGENET_MEAN, IMAGENET_STD, MEGALOC_HUB)

OUT_DIR = "/media/adam/vprdatasets/megaevent/backbone_probe"
HUB_CKPTS = os.path.expanduser("~/.cache/torch/hub/checkpoints")
# The reporting protocol: one reference traverse per cell (pairwise_sunset_ref_*.json).
REFERENCE = "sunset1"
QUERIES = ("daytime", "morning", "sunrise")
GEPT_S = "/home/adam/repo/gept/small.pt"
GEPT_B = "/home/adam/repo/gept/base.pt"
V8_B = "/media/adam/vprdatasets/megaevent/v8_bench/ckpts/b_v8_accum_s750.pt"


class _Tokens:
    """Adapter: every backbone below exposes ``forward_features`` like a DINOv2 ViT.

    ``frozen_rep_probe.extract`` reads ``x_norm_patchtokens`` / ``x_norm_clstoken`` off the
    result, so wrapping here means that function — and therefore the pooling, the ranking
    and the recall — is reused byte-for-byte rather than reimplemented per model.
    """

    def __init__(self, fn):
        self.fn = fn

    def forward_features(self, x):
        return self.fn(x)


def _from_vit(vit):
    """Meta's DinoVisionTransformer (GEPT, stock DINOv2, and SALAD's inner model)."""
    return _Tokens(vit.forward_features)


def _from_pair(module):
    """A backbone returning ``(patch_features [B, D, h, w], cls [B, D])`` — MegaLoc's and ours."""
    def fn(x):
        feat, cls = module(x)
        b, d = feat.shape[0], feat.shape[1]
        return {"x_norm_patchtokens": feat.reshape(b, d, -1).transpose(1, 2),
                "x_norm_clstoken": cls}
    return _Tokens(fn)


def _dinov2(size, registers, state_dict, device):
    from dinov2.models.vision_transformer import vit_base, vit_small
    ctor = vit_small if size == "small" else vit_base
    vit = ctor(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6,
               num_register_tokens=registers)
    vit.load_state_dict(state_dict, strict=True)
    return _from_vit(vit.to(device).eval())


def load_gept_s(device):
    sd = torch.load(GEPT_S, map_location="cpu", weights_only=False)["event_encoder"]
    return _dinov2("small", 0, sd, device)


def load_gept_b(device):
    sd = torch.load(GEPT_B, map_location="cpu", weights_only=False)["event_encoder"]
    return _dinov2("base", 4, sd, device)


def load_dinov2_s(device):
    sd = torch.load(os.path.join(HUB_CKPTS, "dinov2_vits14_pretrain.pth"),
                    map_location="cpu", weights_only=True)
    return _dinov2("small", 0, sd, device)


def load_dinov2_b(device):
    # GEPT base carries 4 registers, so the like-for-like RGB checkpoint is reg4.
    path = os.path.join(HUB_CKPTS, "dinov2_vitb14_reg4_pretrain.pth")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}\n  curl -sL -o {path} https://dl.fbaipublicfiles."
                         f"com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return _dinov2("base", 4, sd, device)


def load_megaloc(device):
    m = torch.hub.load(MEGALOC_HUB, "get_trained_model", source="github",
                       trust_repo=True).eval().to(device)
    return _from_pair(m.backbone)          # returns (patch [B,768,h,w], cls [B,768])


def load_salad(device):
    from src.methods import build_dino_salad
    m = build_dino_salad().eval().to(device)
    return _from_vit(m.backbone.model)     # a stock DinoVisionTransformer


def load_v8_b(device):
    from src import inference as inf
    model, _cfg, _step = inf.load_model(V8_B, device)   # (model, cfg, step)
    return _from_pair(model.forward_encoder)


# name -> (loader, event-pretrained?, which normalisation arms to try)
RGB, EVENT = ("imagenet", "accumulate"), ("accumulate",)
BACKBONES = {
    "dinov2-s": (load_dinov2_s, False, RGB),
    "gept-s":   (load_gept_s,   True,  EVENT),
    "dinov2-b": (load_dinov2_b, False, RGB),
    "gept-b":   (load_gept_b,   True,  EVENT),
    "megaloc":  (load_megaloc,  False, RGB),
    "salad":    (load_salad,    False, RGB),
    "v8-b":     (load_v8_b,     True,  EVENT),
}
STATS = {"imagenet": (IMAGENET_MEAN, IMAGENET_STD),
         "accumulate": (ACCUMULATE_MEAN, ACCUMULATE_STD)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES),
                    choices=list(BACKBONES))
    ap.add_argument("--stats", nargs="+", default=None, choices=list(STATS),
                    help="restrict which normalisation arms to run. Default runs both for "
                         "every RGB backbone and keeps the better one; once that is known, "
                         "'--stats imagenet' halves a full-stride run.")
    ap.add_argument("--stride", type=int, default=5,
                    help="frame stride. 1 is the full pooled protocol; 5 keeps every arm "
                         "on identical frames at a fifth of the cost, which is what a "
                         "relative comparison needs.")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--score-device", default="auto", choices=["auto", "cpu", "cuda"])
    cli = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_device = (device if cli.score_device == "auto" else torch.device(cli.score_device))
    os.makedirs(OUT_DIR, exist_ok=True)
    sequences = [REFERENCE, *QUERIES]

    args = _Args(frp.EVENTLAB, "brisbane_event", 50, False, False)
    args.filter_dt_us = 50_000
    geom = bp.traverse_geometry(args, sequences, None)
    plan, xys = {}, {}
    for s in sequences:
        xy, cov = geom[s][0], geom[s][1]
        plan[s] = np.flatnonzero(cov)[::cli.stride]
        xys[s] = xy[cov][::cli.stride]
    gts = {q: build_gt(xys[REFERENCE], xys[q], 25.0) for q in QUERIES}
    print(f"[probe] single reference {REFERENCE} ({len(plan[REFERENCE])} db) -> "
          + ", ".join(f"{q} ({len(plan[q])} q)" for q in QUERIES)
          + f"; stride {cli.stride}, 25 m, accumulate, parameter-free pooling", flush=True)

    render = frp.REPS["accumulate"][0]
    # Accumulate across invocations: adding an arm should not discard the ones already
    # measured on the same protocol (each is ~3 min of extraction).
    out_path = os.path.join(OUT_DIR, f"results_stride{cli.stride}.json")
    previous = {}
    if os.path.exists(out_path):
        with open(out_path) as handle:
            previous = json.load(handle).get("backbones", {})

    results = {"protocol": {"stride": cli.stride, "resolution": frp.RESOLUTION,
                            "kind": "single reference traverse",
                            "reference": REFERENCE, "queries": list(QUERIES),
                            "n_database": int(len(plan[REFERENCE])),
                            "n_queries": {q: int(len(plan[q])) for q in QUERIES},
                            "representation": "accumulate", "pooling": "L2(mean patch)",
                            "npz_filters": "hot_pixel only, NO ba50"},
               "backbones": dict(previous)}

    for name in cli.backbones:
        loader, is_event, arms = BACKBONES[name]
        if cli.stats:
            arms = tuple(a for a in arms if a in cli.stats) or arms[:1]
        enc = loader(device)
        results["backbones"][name] = {"event_pretrained": is_event, "arms": {}}
        for stats in arms:
            mean, std = STATS[stats]
            t0 = time.time()
            banks = {s: frp.extract(enc, s, plan[s], render, mean, std, device,
                                    cli.batch_size, cli.workers) for s in sequences}
            cell = {}
            for pool in ("meanpatch", "cls"):
                db = banks[REFERENCE][pool]
                cell[pool] = {}
                for q in QUERIES:
                    cell[pool][q] = frp.score(db, banks[q][pool], gts[q], score_device)
            results["backbones"][name]["arms"][stats] = cell
            per = cell["meanpatch"]
            mean_r1 = sum(per[q]["1"] for q in QUERIES) / len(QUERIES)
            print(f"[probe] {name:10s} {stats:10s} meanpatch R@1 "
                  + " ".join(f"{q[:4]}={per[q]['1']:.4f}" for q in QUERIES)
                  + f"  mean={mean_r1:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            del banks
        del enc
        torch.cuda.empty_cache() if device.type == "cuda" else None

        with open(out_path + ".tmp", "w") as h:
            json.dump(results, h, indent=2)
        os.replace(out_path + ".tmp", out_path)

    print(f"\n{'backbone':11s}{'pretrain':10s}{'stats':11s}"
          + "".join(f"{q:>10s}" for q in QUERIES) + f"{'mean':>9s}{'clsmean':>9s}")
    for name, blob in results["backbones"].items():
        tag = "event" if blob["event_pretrained"] else "rgb"
        for stats, cell in blob["arms"].items():
            mp, cl = cell["meanpatch"], cell["cls"]
            mean = sum(mp[q]["1"] for q in QUERIES) / len(QUERIES)
            clsm = sum(cl[q]["1"] for q in QUERIES) / len(QUERIES)
            print(f"{name:11s}{tag:10s}{stats:11s}"
                  + "".join(f"{mp[q]['1']:>10.4f}" for q in QUERIES)
                  + f"{mean:>9.4f}{clsm:>9.4f}")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
