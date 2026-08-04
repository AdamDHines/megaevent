"""Derive the DAVIS's dead vignette and write vignette-masked copies of both arms.

Run from the repo root::

    pixi run python3 scripts/mask_vignette.py --build-mask
    pixi run python3 scripts/mask_vignette.py --source real --seq sunset1
    pixi run python3 scripts/mask_vignette.py --source i2e  --seq sunset1

Brisbane's DAVIS frames have a heavily vignetted bottom quarter — the car bonnet plus lens
falloff — that sits near black for the whole traverse. That region is harmless to a real
event camera, which responds to *temporal* change and so produces almost nothing there
(measured: 8% of its events over 24% of the frame). It is not harmless to I2E, which
thresholds ``Δlog(luma + 1e-3)``: the sensitivity of that transform goes as
``1/(luma + 1e-3)``, so where the frame is near black, read noise clears the contrast
threshold easily. Measured on the same frames, 57-69% of the synthetic stream lands in the
vignette. Most of it is amplified noise rather than scene structure.

That confounds the real-vs-I2E comparison: a drop on the synthetic arm could be the domain
gap the ablation is about, or it could be this one fixable preprocessing artefact. Masking
separates them.

**The mask is applied to both arms, not just the synthetic one.** Removing the region from
I2E while leaving it in the real stream would hand the real arm 8% of its events for free
and change the field of view between them — reintroducing exactly the asymmetry the
index-aligned design exists to remove. Masked-against-masked is the only comparison that
isolates the artefact.

The mask is *static*, derived from how often each pixel is dark across the whole traverse
rather than per frame, because the vignette is a property of the optics and the mounting
and not of any particular scene — a per-frame threshold would also delete legitimately
dark scene content, such as shadow under a tree. It is taken as the union over both
traverses so that reference and query see identical geometry.

Masking happens on the event stream, not the image: the events keep their original sensor
coordinates and ``resolution``, so the masked trees stay drop-in compatible with everything
in :mod:`src.npzdata` and geometrically identical to the arms they came from.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MASK_SUFFIX = "_masked"


def build_mask(npz_root, dataset, seqs, dark=20, persist=0.90, n=300):
    """``[H, W]`` bool — True where the sensor is dead and should be dropped.

    A pixel qualifies when it is darker than ``dark`` in at least ``persist`` of the sampled
    frames, in *either* traverse. Sampling rather than reading every frame because the
    vignette does not move and 300 frames spread over 12 minutes estimate it to well within
    a pixel.
    """
    from PIL import Image

    union = None
    for seq in seqs:
        aps = os.path.join(npz_root, dataset, "aps", seq)
        total = len([f for f in os.listdir(aps) if f.endswith(".png")])
        idx = np.linspace(0, total - 1, min(n, total)).astype(int)
        acc = None
        for i in idx:
            grey = np.asarray(Image.open(os.path.join(aps, f"frame_{i:06d}.png")).convert("L"))
            hit = (grey < dark).astype(np.float32)
            acc = hit if acc is None else acc + hit
        mask = (acc / len(idx)) >= persist
        print(f"  {seq}: {mask.mean() * 100:5.2f}% of pixels are dark in >={persist:.0%} "
              f"of {len(idx)} sampled frames")
        union = mask if union is None else (union | mask)
    print(f"  union: {union.mean() * 100:5.2f}% of the sensor masked "
          f"({int(union.sum())} of {union.size} pixels)")
    return union


_CFG = {}


def _init(cfg):
    _CFG.update(cfg)
    _CFG["mask"] = np.load(cfg["mask_path"])


def _mask_one(names):
    mask = _CFG["mask"]
    src_dir, dst_dir = _CFG["src_dir"], _CFG["dst_dir"]
    kept = dropped = 0
    for name in names:
        dst = os.path.join(dst_dir, name)
        if _CFG["resume"] and os.path.exists(dst):
            continue
        with np.load(os.path.join(src_dir, name)) as z:
            x, y, t, p, res = z["x"], z["y"], z["t"], z["p"], z["resolution"]
        if x.size:
            h, w = int(res[0]), int(res[1])
            keep = ~mask[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
            dropped += int((~keep).sum())
            kept += int(keep.sum())
            x, y, t, p = x[keep], y[keep], t[keep], p[keep]
        np.savez_compressed(dst, x=x, y=y, t=t, p=p, resolution=res)
    return len(names), kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="brisbane_event")
    ap.add_argument("--seqs", nargs="+", default=["sunset2", "sunset1"])
    ap.add_argument("--seq", default=None, help="mask one traverse (default: all --seqs)")
    ap.add_argument("--source", default=None, choices=["real", "i2e"],
                    help="which arm to mask; omit with --build-mask to only derive the mask")
    ap.add_argument("--npz-root", default="/media/adam/vprdatasets/megaevent/brisbane_npz")
    ap.add_argument("--build-mask", action="store_true",
                    help="derive the mask from the APS frames and save it")
    ap.add_argument("--dark", type=int, default=20)
    ap.add_argument("--persist", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    mask_path = os.path.join(args.npz_root, args.dataset, "vignette_mask.npy")
    if args.build_mask or not os.path.exists(mask_path):
        print(f"deriving the vignette mask (dark < {args.dark}, persistent in "
              f">={args.persist:.0%} of frames)")
        mask = build_mask(args.npz_root, args.dataset, args.seqs, args.dark, args.persist)
        np.save(mask_path, mask)
        with open(mask_path.replace(".npy", ".json"), "w") as f:
            json.dump({"dark_threshold": args.dark, "persistence": args.persist,
                       "sequences": args.seqs, "masked_pixels": int(mask.sum()),
                       "sensor_hw": list(mask.shape),
                       "masked_fraction": float(mask.mean())}, f, indent=2)
        print(f"  -> {mask_path}")
    if args.source is None:
        return

    for seq in ([args.seq] if args.seq else args.seqs):
        src_dir = os.path.join(args.npz_root, args.dataset, args.source, seq)
        dst_dir = os.path.join(args.npz_root, args.dataset, args.source + MASK_SUFFIX, seq)
        os.makedirs(dst_dir, exist_ok=True)
        names = sorted(f for f in os.listdir(src_dir) if f.startswith("frame_"))
        cfg = {"mask_path": mask_path, "src_dir": src_dir, "dst_dir": dst_dir,
               "resume": not args.no_resume}
        chunks = np.array_split(names, max(1, args.workers) * 4)
        kept = dropped = done = 0
        with Pool(max(1, args.workers), initializer=_init, initargs=(cfg,)) as pool:
            for k, a, b in pool.imap_unordered(_mask_one, [list(c) for c in chunks if len(c)]):
                done += k
                kept += a
                dropped += b
        total = kept + dropped
        print(f"{args.source}{MASK_SUFFIX}/{seq}: {done} frames, dropped "
              f"{dropped / max(total, 1) * 100:.1f}% of events ({dropped} of {total}) "
              f"-> {dst_dir}")


if __name__ == "__main__":
    sys.exit(main())
