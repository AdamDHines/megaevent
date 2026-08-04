"""Measure what JPEG chroma subsampling does to a countmask frame.

Run from the repo root::

    pixi run python3 scripts/verify_countmask_jpeg.py

This exists because ``I2E/submit_job.sh`` renders three of the training trees with
``--img-format jpg --quality 100`` and ``I2E/rerender_from_npz.py`` saves them with
``Image.fromarray(img).save(tmp, format="JPEG", quality=cfg["quality"])`` — **no
``subsampling=0``** — so PIL/libjpeg applies its default **4:2:0**. Since ``sf_xl``
contributes two sub-batches (frontal + lateral), that is ``gsv_cities``,
``sf_xl_frontal``, ``sf_xl_lateral`` and ``msls``: **4 of the 6 streams**, and the four
outdoor street-level ones. ``megascenes`` / ``scannet`` are PNG, and Tokyo eval renders
straight from ``.npz``, so the *test* side is exact.

Quality 100 sounds lossless, and for a photograph the damage would be cosmetic. It is not
cosmetic here, because of what countmask puts in each channel (GEPT Sec. 3.2 / Eq. 2)::

    R = clip(positive_count, 0, alpha) / alpha
    B = clip(negative_count, 0, alpha) / alpha
    G = (positive_count + negative_count) > 0        # binary activity mask

Under RGB->YCbCr the binary green mask dominates luma, so **the two polarity planes live
almost entirely in the chroma channels** — exactly what 4:2:0 halves in both axes before
re-upsampling. The event polarity signal is the thing being blurred.

The frames are rendered here from the local Tokyo ``.npz`` event streams rather than read
off the training tree (which is on HPC), using the same arithmetic as
``I2E/i2e_infer.py::countmask_numpy`` — verified byte-identical to eventcv's Rust
``countmask`` and to ``gept/src/vpr/representations.py::countmask_numpy``.

Reports, per channel and averaged over N frames: mean absolute delta, the fraction of
pixels changed, the fraction of exact-zero background pixels that became non-zero, and the
number of distinct values surviving in the binary green mask. The ``subsampling=0`` control
is the proposed fix.
"""

import argparse
import io
import os
import sys

import numpy as np
from PIL import Image, JpegImagePlugin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.npzdata import list_npz  # noqa: E402

DEFAULT_ROOT = "/media/adam/vprdatasets/megaevent/tokyo247/numpy/database"


def countmask(x, y, p, height, width, pct=99.0):
    """``[H, W, 3]`` uint8 countmask, matching ``I2E/i2e_infer.py::countmask_numpy``."""
    pos = np.zeros((height, width), np.float32)
    neg = np.zeros((height, width), np.float32)
    is_pos = p > 0
    np.add.at(pos, (y[is_pos], x[is_pos]), 1)
    np.add.at(neg, (y[~is_pos], x[~is_pos]), 1)
    nonzero = np.concatenate([pos.ravel(), neg.ravel()])
    nonzero = nonzero[nonzero > 0]
    alpha = float(np.percentile(nonzero, pct)) if nonzero.size else 1.0
    red = np.clip(pos, 0, alpha) / alpha
    blue = np.clip(neg, 0, alpha) / alpha
    green = ((pos + neg) > 0).astype(np.float32)
    return np.clip(np.stack([red, green, blue], axis=-1) * 255.0, 0, 255).astype(np.uint8)


def roundtrip(img, **save_kwargs):
    """``img`` through a JPEG encode/decode -> (decoded array, PIL sampling code)."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", **save_kwargs)
    buf.seek(0)
    decoded = Image.open(buf)
    decoded.load()
    return np.asarray(decoded).astype(np.int16), JpegImagePlugin.get_sampling(decoded)


def measure(img, **save_kwargs):
    """Damage statistics for one frame under one set of JPEG save options."""
    out, sampling = roundtrip(img, **save_kwargs)
    delta = np.abs(out - img.astype(np.int16))
    background = (img == 0).all(axis=-1)
    return {
        "sampling": sampling,
        "mean_abs": [float(delta[..., c].mean()) for c in range(3)],
        "changed_pct": float(delta.any(axis=-1).mean() * 100.0),
        "bg_polluted_pct": float((out[background] != 0).any(axis=-1).mean() * 100.0),
        "green_levels": int(len(np.unique(out[..., 1]))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz-dir", default=DEFAULT_ROOT,
                    help=f"directory of I2E .npz event streams (default {DEFAULT_ROOT})")
    ap.add_argument("--n", type=int, default=6, help="frames to average over")
    ap.add_argument("--quality", type=int, default=100, help="JPEG quality, as the trees used")
    args = ap.parse_args()

    paths = list_npz(args.npz_dir)[: args.n]
    print(f"{len(paths)} countmask frames rendered from {args.npz_dir}\n")

    arms = {f"quality={args.quality} (as the trees were written)": {"quality": args.quality},
            f"quality={args.quality} + subsampling=0 (the fix)": {"quality": args.quality,
                                                                 "subsampling": 0}}
    stats = {name: [] for name in arms}
    green_before = set()
    for path in paths:
        with np.load(path) as npz:
            height, width = (int(v) for v in npz["resolution"])
            img = countmask(npz["x"].astype(int), npz["y"].astype(int),
                            npz["p"].astype(int), height, width)
        green_before.add(len(np.unique(img[..., 1])))
        for name, kwargs in arms.items():
            stats[name].append(measure(img, **kwargs))

    for name, rows in stats.items():
        sampling = rows[0]["sampling"]
        note = "4:2:0 chroma subsampling" if sampling == 2 else "no chroma subsampling"
        print(f"{name}\n  PIL get_sampling() = {sampling}  ({note})")
        mean_abs = np.array([r["mean_abs"] for r in rows]).mean(axis=0)
        print(f"  mean |delta| R/G/B      : {mean_abs[0]:6.2f} /{mean_abs[1]:6.2f} /"
              f"{mean_abs[2]:6.2f}   of 255")
        print(f"  pixels changed          : {np.mean([r['changed_pct'] for r in rows]):6.1f} %")
        print(f"  zero background polluted: "
              f"{np.mean([r['bg_polluted_pct'] for r in rows]):6.1f} %")
        print(f"  green mask levels       : {sorted(green_before)} -> "
              f"{sorted({r['green_levels'] for r in rows})}\n")

    print("The green channel is a binary activity mask by construction, so any level count "
          "above 2 is\ndamage. Fix: pass subsampling=0 at I2E/rerender_from_npz.py:84, or "
          "render PNG.")


if __name__ == "__main__":
    main()
