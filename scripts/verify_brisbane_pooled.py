"""Checks on the pooled Brisbane evaluation, in the order they would catch a mistake.

    pixi run python3 scripts/verify_brisbane_pooled.py

The pooled protocol rests on two claims that no downstream number would expose if false:

1. that a descriptor row and a GPS coordinate refer to the same instant, and
2. that a streamed top-20 gives the same recall as the reference implementation.

A constant timing offset, for example, would move every query and every reference by the same
amount and leave R@1 looking entirely healthy — Brisbane's traverses run the same route, so a
uniformly shifted query still lands near a uniformly shifted reference. Check 2 below is the one
that would notice, by comparing against ground truth this repo did not generate.

Run against a small subsample where it can be; each check prints PASS/FAIL and the script exits
non-zero if any fails.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src import inference as inf  # noqa: E402
from src import traversegps as tg  # noqa: E402
from src.imagevpr import build_gt  # noqa: E402
from brisbane_pooled import recall_from_ranked, topk_ranked  # noqa: E402

EVENTLAB = "/media/adam/vprdatasets/eventgem"
NPZ_ROOT = "/media/adam/vprdatasets/megaevent/brisbane_npz"
BANKS = "/media/adam/vprdatasets/megaevent/brisbane_v4"
DATASET = "brisbane_event"
SEQUENCES = ("sunset1", "sunset2", "daytime", "morning", "night", "sunrise")
EVENTLAB_TOLERANCE_M = 70.0         # src/eventlab.py:50 — what the GT on disk was built at

results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------------------
print("\n1. projection and framing")
# ---------------------------------------------------------------------------
lat0, lon0 = tg.track_origin(EVENTLAB, DATASET, SEQUENCES)
geom = {s: tg.frame_coords(EVENTLAB, DATASET, s, lat0, lon0, 50, npz_root=NPZ_ROOT)
        for s in SEQUENCES}

lengths = np.array([geom[s][3]["route_len_m"] for s in SEQUENCES])
spread = (lengths.max() - lengths.min()) / lengths.mean()
check("route lengths agree", spread < 0.01,
      f"{lengths.min():.0f}-{lengths.max():.0f} m, spread {spread:.2%} (<1%)")

verified = [s for s in SEQUENCES if geom[s][3]["slice_times_checked"]]
check("start_tick matches the dumped slice_times", len(verified) == 2,
      f"verified against slice_times.npy for {verified}; the other four have no dump "
      f"and rest on metadata.json alone")

# ---------------------------------------------------------------------------
print("\n2. metric ground truth vs the Event-LAB band (the alignment check)")
# ---------------------------------------------------------------------------
# Rebuilt at 70 m, the tolerance the file on disk was generated at, so the only difference
# under test is *along-track arc* vs *Euclidean radius* — not the tolerance.
ref_xy, ref_cov, _, _ = geom["sunset2"]
qry_xy, qry_cov, _, _ = geom["sunset1"]


class _GTArgs:
    eventlab_dir, dataset, ref, query = EVENTLAB, DATASET, "sunset2", "sunset1"


band, band_shape = inf.load_gt(inf.ground_truth_path(_GTArgs()), len(ref_xy), len(qry_xy))
metric = build_gt(ref_xy, qry_xy, EVENTLAB_TOLERANCE_M)

both = band & metric
covered = ref_cov[:, None] & qry_cov[None, :]
band_cov = band & covered
overlap = float(both[covered].sum()) / float(band_cov.sum())
check("Event-LAB band positives are inside the 70 m metric radius", overlap > 0.80,
      f"{overlap:.1%} of band positives are within 70 m Euclidean "
      f"(band shape {band_shape}, resampled to {band.shape})")

# Sharper: the band's own centre line should sit on the nearest reference frame.
q_idx = np.flatnonzero(qry_cov)[::200]
d = np.linalg.norm(ref_xy[ref_cov][:, None, :] - qry_xy[q_idx][None, :, :], axis=2)
nearest = np.flatnonzero(ref_cov)[d.argmin(0)]
band_centre = np.array([np.flatnonzero(band[:, j]).mean() if band[:, j].any() else -1
                        for j in q_idx])
valid = band_centre >= 0
offset = np.abs(band_centre[valid] - nearest[valid])
check("band centre tracks the nearest reference frame", np.median(offset) < 60,
      f"median |band centre - argmin distance| = {np.median(offset):.1f} frames "
      f"({np.median(offset) * 0.05:.1f} s); a 10 s misalignment would read ~200")

# ---------------------------------------------------------------------------
print("\n3. streamed top-k vs the reference recall implementation")
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stride = 8                                  # small enough to materialise the full matrix
db_parts, xy_parts = [], []
for s in ("sunset2", "daytime"):
    bank = np.load(os.path.join(BANKS, f"r322_{s}_P64_s10000.npy"), mmap_mode="r")
    keep = np.flatnonzero(geom[s][1])[::stride]
    db_parts.append(np.asarray(bank[keep]))
    xy_parts.append(geom[s][0][keep])
    del bank
db = torch.from_numpy(np.concatenate(db_parts))
db_xy = np.concatenate(xy_parts)

qbank = np.load(os.path.join(BANKS, "r322_sunset1_P64_s10000.npy"), mmap_mode="r")
qkeep = np.flatnonzero(geom["sunset1"][1])[::stride]
q = torch.from_numpy(np.asarray(qbank[qkeep]))
q_xy = geom["sunset1"][0][qkeep]
del qbank

gt = build_gt(db_xy, q_xy, 25.0)
ranked = topk_ranked(db, q, device, chunk=256)
mine, _, _ = recall_from_ranked(ranked, gt)
theirs = inf.recall_at_k(inf.sim_matrix(db, q, device), gt, (1, 5, 10, 20))
agree = all(abs(mine[k] - theirs[k]) < 1e-12 for k in mine)
check("topk_ranked == inference.recall_at_k", agree,
      f"{db.shape[0]} db x {q.shape[0]} q — "
      + "  ".join(f"R@{k} {mine[k]:.6f}/{theirs[k]:.6f}" for k in sorted(mine)))

# ---------------------------------------------------------------------------
print("\n4. pooled ground-truth shape")
# ---------------------------------------------------------------------------
db_xy_full = np.concatenate([geom[s][0][geom[s][1]] for s in SEQUENCES if s != "sunset1"])
q_xy_full = geom["sunset1"][0][geom["sunset1"][1]]
gt_full = build_gt(db_xy_full, q_xy_full, 25.0)
per_query = gt_full.sum(0)
check("every query has a positive at 25 m", int((per_query == 0).sum()) == 0,
      f"{int((per_query > 0).sum())}/{len(per_query)} scorable, "
      f"positives/query mean {per_query.mean():.1f} median {np.median(per_query):.0f}")
check("the radius is stricter than the 70 m band it replaces", gt_full.mean() < 0.0181,
      f"pooled density {gt_full.mean():.4%} vs 1.81% for the along-track band")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
