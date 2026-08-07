"""Is a descriptor bank actually discriminating, or has it collapsed?

Run from the repository root::

    pixi run python scripts/spikevpr_health.py <bank>.npy [<bank>.npy ...]

Written for SpikeVPR but bank-agnostic — it reads an ``[N, D]`` float32 ``.npy`` and knows
nothing else about it.

The failure this exists to catch is specific to SpikeVPR and specific to running it
cross-dataset. The network has no input normalisation: a ``BatchNorm2d`` on frozen training
statistics is the first thing the events meet, and the frame is raw per-pixel counts. Feed it
an input far denser than it was trained on — Tokyo 24/7 renders a median 7.2-8.4 events/px
against the 0.167 the Brisbane checkpoint saw, because an I2E saccade packs ~600k events into
29 ms — and the first spiking layer can saturate, after which every frame produces nearly the
same descriptor.

That does not raise. It produces a bank of the right shape and dtype, a similarity matrix
with no NaNs, and a low recall that reads exactly like "this method is weak on this dataset".
The three numbers below separate the two explanations:

* **zero-variance dimensions** — components identical across every frame. A healthy bank has
  essentially none; a saturated one has many.
* **off-diagonal cosine spread** — how far apart two unrelated frames land. A collapsed bank
  pins every pair near a single value, so the p1-p99 range closes up.
* **descriptor norm** — 1.0 for every SpikeVPR bank, since the MixVPR head L2-normalises.
  Anything else means the bank did not come from the head it should have.

None of this is a pass/fail oracle, and it deliberately does not print one: a genuinely hard
dataset also has high mean similarity. What it separates is "similar because the places look
alike" from "similar because the network stopped responding to the input".
"""

import argparse
import os

import numpy as np

SAMPLE = 2000                   # rows drawn for the pairwise statistics
SEED = 0


def health(bank, sample=SAMPLE, seed=SEED):
    """``{metric: value}`` for one ``[N, D]`` bank."""
    n, dim = bank.shape
    norms = np.linalg.norm(bank, axis=1)
    # Zero variance *across frames*, not zero value: a dimension that is always 0.4 carries
    # exactly as little information as one that is always 0.
    dead = float((bank.std(axis=0) == 0).mean())

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    sub = bank[np.sort(idx)].astype(np.float32)
    sub /= np.maximum(np.linalg.norm(sub, axis=1, keepdims=True), 1e-12)
    cos = sub @ sub.T
    off = cos[~np.eye(len(cos), dtype=bool)]

    p1, p50, p99 = (float(v) for v in np.percentile(off, [1, 50, 99]))
    return {"n": n, "dim": dim, "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()), "dead_dims": dead,
            "active_frac": float((bank != 0).mean()),
            "cos_mean": float(off.mean()), "cos_p1": p1, "cos_p50": p50, "cos_p99": p99,
            "cos_spread": p99 - p1, "sampled": len(sub)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("banks", nargs="+", help="[N, D] descriptor banks (.npy)")
    parser.add_argument("--sample", type=int, default=SAMPLE)
    cli = parser.parse_args()

    print(f"{'bank':52s} {'N':>7s} {'dim':>5s} {'|d|':>6s} {'dead':>6s} {'act':>6s} "
          f"{'cos':>7s} {'p1':>7s} {'p50':>7s} {'p99':>7s} {'spread':>7s}")
    for path in cli.banks:
        bank = np.load(path, mmap_mode="r")
        if bank.ndim != 2:
            print(f"{os.path.basename(path)[:52]:52s}  not an [N, D] bank ({bank.shape})")
            continue
        h = health(np.asarray(bank), sample=cli.sample)
        print(f"{os.path.basename(path)[:52]:52s} {h['n']:7d} {h['dim']:5d} "
              f"{h['norm_mean']:6.3f} {h['dead_dims']:6.3f} {h['active_frac']:6.3f} "
              f"{h['cos_mean']:7.4f} {h['cos_p1']:7.4f} {h['cos_p50']:7.4f} "
              f"{h['cos_p99']:7.4f} {h['cos_spread']:7.4f}")


if __name__ == "__main__":
    main()
