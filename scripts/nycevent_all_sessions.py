"""NYC-Event leave-one-session-out: each recording session queries, all others are the gallery.

    pixi run python3 scripts/nycevent_all_sessions.py

**Why this exists.** NYC-Event's reported protocol is a random 10% query split (`protocol.json`:
`query_fraction 0.1, seed 0`) drawn across all 15 sessions, so a query's own session is still in
the gallery — sampled at 1 Hz, a few metres away. Measured on the manifest: **99.8% of queries
have a same-session database sample within 25 m, median 1.81 m**, and for 88.9% the single
nearest database sample is from its own session. The reported number is therefore dominated by
near-duplicate retrieval within one drive, not place recognition across conditions.

Holding a whole session out makes it the same measurement Brisbane and NSAVP already make:
one traverse queries, the rest are the gallery, 25 m Euclidean, native, no PCA. The `reported`
column is printed beside it so the size of the effect is visible.

Sessions span 2022-12-06 to 2023-04-20, so leaving one out is a genuine condition and seasonal
change rather than a re-sampling of the same drive.

Scored entirely from the cached banks — the database and queries banks are concatenated back
into the full 55,433 samples and re-partitioned by session, so nothing is re-extracted.
"""

import csv
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import brisbane_pooled as bp  # noqa: E402
from src.imagevpr import build_gt  # noqa: E402

ROOT = "/media/adam/vprdatasets/megaevent"
EVAL = f"{ROOT}/evaluations/nycevent"
GEM = f"{ROOT}/nycevent_eval_eventgem/nycevent"
# (pretty, bank dir, tag, reported R@1 under the random-split protocol)
MODELS = [
    ("megaevent B-MLoc  real s500",  EVAL, "v5b_mloc_s500_s500_cc9a4f7725_countmask_r322", 0.800),
    ("megaevent B-SALAD real s1000", EVAL, "v5b_salad_s1000_s999_1b254ca54e_countmask_r322", 0.817),
    ("megaevent S-MLoc  real s1000", EVAL, "v6s_mloc_s1000_s999_3b2e243800_countmask_r322", 0.791),
    ("megaevent S-SALAD real s1000", EVAL, "v6s_salad_s1000_s999_045cab36e1_countmask_r322", 0.773),
    ("megaevent B-MLoc  ship s3500", EVAL, "megaevent_vitb_mloc_s3500_b1c23c1bc1_countmask_r322", 0.813),
    ("megaevent B-SALAD ship s8000", EVAL, "megaevent_vitb_salad_s8000_dde11113fc_countmask_r322", 0.811),
    ("megaevent S-MLoc  ship s3500", EVAL, "megaevent_vits_mloc_s3500_1ed09057dd_countmask_r322", 0.803),
    ("megaevent S-SALAD ship s2000", EVAL, "megaevent_vits_salad_s2000_9dea56d731_countmask_r322", 0.788),
    ("MegaLoc",                      EVAL, "megaloc_countmask_r322", 0.767),
    ("Event-GeM (global)",           GEM,  "eventgem_se_240x320", 0.429),
    # v8: accumulate representation (main.py writes into the <feature-dir>/nycevent subdir)
    ("megaevent B-accum v8 s750",    f"{EVAL}/nycevent",
     "b_v8_accum_s750_s750_3f4eb54780_accumulate_r322", 0.9048),
    ("megaevent S-accum v8 s500",    f"{EVAL}/nycevent",
     "s_v8_accum_s500_s500_9aaa0c325c_accumulate_r322", 0.8877),
    # Event-GeM 0.1.0 as released (pr.pt backbone + GeM globals, extracted 2026-08-21
    # via the parity-checked nyc_extract.py driver in the 0.1.0 worktree)
    ("Event-GeM 0.1.0 (global)",     f"{EVAL}/nycevent", "eg010_gem224", 0.4354),
]

# v9 arch-ablation arms ({ViT-S,ViT-B} x {ft4,full} x {GeM,SALAD} plus the N sweep, all
# accumulate @ step2000). Discovered by glob rather than hardcoded: the tag embeds the
# checkpoint sha, which is not known until the wave is synced off the HPC. The step in the tag
# is left as a wildcard on purpose — it comes from ck["step"], which is 0-indexed, so the
# step2000.pt milestone tags itself `_s1999_`. Two matches means two shas for one arm, which
# is a stale bank rather than something to guess between, so it raises. A `None` in the
# reported slot means "no published random-split number to check against" — these arms are
# new, so the reproduction cross-check below is skipped for them (their random-split R@1 is
# still computed and printed, it just has nothing to be compared with).
V9_ARMS = ["s_gem_ft4", "s_gem_full", "b_gem_ft4", "b_gem_full",
           "s_salad_ft4", "s_salad_full", "b_salad_ft4", "b_salad_full",
           "b_salad_full_P32", "b_salad_full_P48"]
for _arm in V9_ARMS:
    _hits = sorted(glob.glob(
        f"{EVAL}/nycevent/{_arm}_s*_accumulate_r322_database.npy"))
    if len(_hits) > 1:
        raise SystemExit(f"v9 {_arm}: {len(_hits)} banks match, sha is ambiguous —\n  "
                         + "\n  ".join(os.path.basename(h) for h in _hits))
    if _hits:
        _tag = os.path.basename(_hits[0])[:-len("_database.npy")]
        MODELS.append((f"v9 {_arm}", f"{EVAL}/nycevent", _tag, None))


def load_full(bank_dir, tag):
    """The two cached splits concatenated back into all 55,433 samples, plus their basenames.

    Also returns the database-split row count, so the original random-split protocol can be
    reconstructed exactly (rows [:n_db] are the database, [n_db:] the queries).
    """
    parts, names = [], []
    for split in ("database", "queries"):
        parts.append(np.load(os.path.join(bank_dir, f"{tag}_{split}.npy"), mmap_mode="r"))
        with open(os.path.join(bank_dir, f"{tag}_{split}_paths.txt")) as h:
            names += h.read().split()
    bank = np.concatenate([np.asarray(p) for p in parts], axis=0)
    if len(names) != len(bank):
        raise SystemExit(f"{tag}: {len(bank)} rows but {len(names)} names")
    return bank, names, len(parts[0])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = {r["sample"]: (r["traverse"], float(r["easting"]), float(r["northing"]))
           for r in csv.DictReader(open(f"{ROOT}/nycevent/manifest.csv"))}

    # Every bank is built from the same dataset listing, so one row order serves them all —
    # asserted rather than assumed, because a cache built from a different listing would attach
    # the ground truth to the wrong descriptors.
    ref_names, n_db = None, None
    for name, d, tag, _ in MODELS:
        _, names, n = load_full(d, tag)
        if any(n_ not in man for n_ in names):
            raise SystemExit(f"{name}: basenames missing from the manifest")
        if ref_names is None:
            ref_names, n_db = names, n
        elif names != ref_names or n != n_db:
            raise SystemExit(f"{name}: bank row order differs from {MODELS[0][0]}")

    tr = np.array([man[n][0] for n in ref_names])
    xy = np.array([[man[n][1], man[n][2]] for n in ref_names])
    sessions = sorted(set(tr))

    # One GT per session, shared by every model (it depends only on the geometry).
    masks, gts, sizes = {}, {}, {}
    for s in sessions:
        q_mask = tr == s
        gts[s] = build_gt(xy[~q_mask], xy[q_mask], 25.0)
        masks[s] = q_mask
        scorable = int((np.asarray(gts[s]).sum(0) > 0).sum())
        sizes[s] = (int((~q_mask).sum()), int(q_mask.sum()), scorable)

    # Cross-check: the original random-split protocol, re-scored from the very same rows
    # (rows [:n_db] are the database split, [n_db:] the queries split). Each model's number
    # must reproduce the published one, or the banks are not what the published JSONs scored.
    split_gt = build_gt(xy[:n_db], xy[n_db:], 25.0)

    # A session whose route no other session covers within 25 m has zero scorable queries
    # under leave-one-session-out; it can only be reported as unscorable, never averaged.
    scored_sessions = [s for s in sessions if sizes[s][2] > 0]
    for s in sessions:
        if sizes[s][2] == 0:
            print(f"  NOTE {s}: no other session within 25 m of any of its "
                  f"{sizes[s][1]} frames — unscorable under LOSO, excluded from means")

    r1, r10, reproduced = {}, {}, {}
    for name, d, tag, rep in MODELS:
        bank, _, _ = load_full(d, tag)
        db = torch.from_numpy(np.asarray(bank[:n_db]))
        qd = torch.from_numpy(np.asarray(bank[n_db:]))
        ranked = bp.topk_ranked(db, qd, device, chunk=256, db_chunk=40000)
        rec, _, _ = bp.recall_from_ranked(ranked, split_gt)
        reproduced[name] = rec[1]
        if rep is not None and abs(rec[1] - rep) > 1e-3:  # rep is a 3 dp transcription
            print(f"  WARNING {name}: random-split R@1 {rec[1]:.4f} != reported {rep:.3f}",
                  flush=True)
        del db, qd, ranked
        for s in scored_sessions:
            q_mask = masks[s]
            db = torch.from_numpy(bank[~q_mask].copy())
            qd = torch.from_numpy(bank[q_mask].copy())
            ranked = bp.topk_ranked(db, qd, device, chunk=256, db_chunk=40000)
            rec, _, _ = bp.recall_from_ranked(ranked, gts[s])
            r1[(s, name)], r10[(s, name)] = rec[1], rec[10]
            del db, qd, ranked
        del bank
        print(f"  scored {name}  (random-split check {reproduced[name]:.4f})", flush=True)
    print("\n  session sizes (db x queries x scorable):")
    for s in sessions:
        print(f"    {s:22s} {sizes[s][0]:>7,} x {sizes[s][1]:>6,} x {sizes[s][2]:>6,}")

    for title, table in (("R@1", r1), ("R@10", r10)):
        print(f"\n\nNYC-Event {title} — leave-one-session-out "
              f"({len(scored_sessions)} scorable of {len(sessions)} sessions)\n")
        print(f"  {'model':31s}{'mean':>8s}{'min':>8s}{'max':>8s}"
              + ("   reported (random 10% split)" if title == "R@1" else ""))
        for name, _, _, rep in MODELS:
            vals = [table[(s, name)] for s in scored_sessions]
            m = sum(vals) / len(vals)
            if title != "R@1":
                tail = ""
            elif rep is None:                      # new arm: nothing published to compare to
                tail = f"      {reproduced[name]:.3f}   (this run)"
            else:
                tail = f"      {rep:.3f}   ({m - rep:+.3f})"
            print(f"  {name:31s}{m:>8.3f}{min(vals):>8.3f}{max(vals):>8.3f}{tail}")

    print("\n\nPer-session R@1\n")
    print(f"  {'session':22s}" + "".join(f"{n.split()[1][:7]:>8s}" for n, _, _, _ in MODELS[:8])
          + f"{'MegaLoc':>9s}{'E-GeM':>8s}")
    for s in scored_sessions:
        print(f"  {s:22s}" + "".join(f"{r1[(s, n)]:>8.3f}" for n, _, _, _ in MODELS[:8])
              + f"{r1[(s, MODELS[8][0])]:>9.3f}{r1[(s, MODELS[9][0])]:>8.3f}")

    # Persist beside the banks so table assemblers and the artifact can cite this run.
    out = {
        "protocol": "leave-one-session-out, 25 m Euclidean, native, no PCA",
        "sessions": {s: {"n_db": sizes[s][0], "n_queries": sizes[s][1],
                         "n_scorable": sizes[s][2]} for s in sessions},
        "unscorable_sessions": [s for s in sessions if s not in scored_sessions],
        "models": {},
    }
    for name, d, tag, rep in MODELS:
        vals = [r1[(s, name)] for s in scored_sessions]
        out["models"][name] = {
            "tag": tag,
            "reported_random_split_r1": rep,
            "reproduced_random_split_r1": reproduced[name],
            "loso_mean_r1": sum(vals) / len(vals),
            "loso_min_r1": min(vals),
            "loso_max_r1": max(vals),
            "per_session": {s: {"r1": r1[(s, name)], "r10": r10[(s, name)]}
                            for s in scored_sessions},
        }
    with open(f"{EVAL}/loso.json", "w") as h:
        json.dump(out, h, indent=1)
    with open(f"{EVAL}/loso.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["model", "session", "r1", "r10"])
        for name, _, _, _ in MODELS:
            for s in scored_sessions:
                w.writerow([name, s, f"{r1[(s, name)]:.4f}", f"{r10[(s, name)]:.4f}"])
    print(f"\n  wrote {EVAL}/loso.json and loso.csv", flush=True)


if __name__ == "__main__":
    main()
