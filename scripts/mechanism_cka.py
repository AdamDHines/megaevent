"""H3: how similar are MegaLoc's representations of the photograph and of the events?

    pixi run python3 scripts/mechanism_cka.py [--model megaloc --stride 1]

Runs on the banks `scripts/mechanism_edges.py` saved (no extraction here): for each arm
pair, on the same index-aligned frames,

  * **linear CKA** between the descriptor matrices — do the two inputs land in similar
    subspaces at all?
  * **similarity-structure agreement** — Spearman rank correlation between the two arms'
    pairwise-similarity vectors over sampled frame pairs. Retrieval only sees this.
  * **top-1 agreement** — for a sample of queries against a sample gallery, how often the
    two arms retrieve the same frame.

High CKA + high agreement = direct representation transfer ("the model reads the events
as it reads the photo"). Low CKA + high agreement = different features, same ranking —
transfer at the metric level only. Both are answers; the pair (real_cm, aps_gray) is the
headline, (real_cm, aps_sobelpol) says how far "events = edge rendering of the photo"
carries, and (real_cm, real_cm_ga) isolates the adaptive-contrast term.
"""

import argparse
import itertools
import json
import os

import numpy as np
from scipy.stats import spearmanr

ROOT = "/media/adam/vprdatasets/megaevent/mechanism"
SEQS = ("sunset1", "daytime", "morning", "sunrise")
ARMS = ("aps_gray", "aps_sobel", "aps_sobelpol", "real_cm", "real_cm_ga")


def load_arm(tag, arm):
    banks = [np.load(os.path.join(ROOT, f"{tag}_{arm}_{s}.npy")) for s in SEQS]
    return np.concatenate(banks).astype(np.float64)


def linear_cka(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    cross = np.linalg.norm(Xc.T @ Yc) ** 2
    return float(cross / (np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="megaloc")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--sample", type=int, default=3000,
                    help="frames sampled for CKA/agreement (memory: D x D crossproducts)")
    cli = ap.parse_args()
    tag = f"{cli.model}_s{cli.stride}"
    rng = np.random.default_rng(0)

    banks = {}
    for arm in ARMS:
        try:
            banks[arm] = load_arm(tag, arm)
        except FileNotFoundError:
            print(f"  {arm}: banks missing (run mechanism_edges.py first) — skipped")
    n = min(len(b) for b in banks.values())
    idx = np.sort(rng.choice(n, min(cli.sample, n), replace=False))
    qi = rng.choice(len(idx), min(500, len(idx)), replace=False)

    out = {}
    print(f"{'pair':28s} {'CKA':>7s} {'simstruct':>10s} {'top1agree':>10s}")
    for a, b in itertools.combinations(banks, 2):
        X, Y = banks[a][idx], banks[b][idx]
        cka = linear_cka(X, Y)
        # pairwise-similarity structure over the same sampled pairs
        pi = rng.choice(len(idx), (4000, 2))
        sx = np.einsum("ij,ij->i", X[pi[:, 0]], X[pi[:, 1]])
        sy = np.einsum("ij,ij->i", Y[pi[:, 0]], Y[pi[:, 1]])
        rho = float(spearmanr(sx, sy).statistic)
        # top-1 agreement: sampled queries vs the sampled gallery (self excluded)
        simx, simy = X[qi] @ X.T, Y[qi] @ Y.T
        for m, q in ((simx, qi), (simy, qi)):
            m[np.arange(len(q)), q] = -np.inf
        agree = float((simx.argmax(1) == simy.argmax(1)).mean())
        out[f"{a}|{b}"] = {"cka": cka, "sim_spearman": rho, "top1_agreement": agree}
        print(f"{a} | {b:14s} {cka:>7.3f} {rho:>10.3f} {agree:>10.3f}")

    path = os.path.join(ROOT, f"cka_{tag}.json")
    with open(path, "w") as h:
        json.dump({"model": cli.model, "stride": cli.stride, "n_frames": int(n),
                   "sampled": int(len(idx)), "pairs": out}, h, indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
