"""The gopro sim2real cell: sunset1 (reference) -> morning (query), real vs i2e_gopro.

    pixi run python3 scripts/gopro_pairwise.py

Scores five methods — megaevent v8-B / v8-S, SpikeVPR (nsavp), EventVLAD, Event-GeM
0.1.0 global (+ rerank folded in if the 0.1.0 driver has run) — on ONE pairwise cell
at 25 m Euclidean, native, no PCA, in two arms:

  * ``real``       — the recorded DAVIS event stream (the published banks);
  * ``i2e_gopro``  — I2E micro-saccades over the co-recorded windscreen-camera frames.

Both arms are scored on the identical row set: GPS-covered AND NOT out_of_tolerance in
either the ``aps`` or the ``gopro`` frame tree (``src.traversenpz.frame_validity_mask``).
The union matters: the videos start after and end before the event streams, and every
clamped slice in the gopro tree is a duplicate of the video's first or last frame —
scoring those rows in any arm compares a real frame against a duplicated one.

**Gate before any gopro number**: the real arm re-scored on the FULL grid (no mask) must
reproduce the ``sunset1->morning`` cells of the pairwise ledger
(``v8_bench/pairwise_sunset_ref_brisbane_event.json``) to 5e-4, proving geometry, GT,
alignment and chunking are byte-compatible with the published protocol. The likely
culprits if it fails: the ENU origin list, EventVLAD's centre alignment, Event-GeM's
(64, 4096) chunking.

The masked real arm is also printed against the full-grid one (the mask-effect delta):
the union removes ~1-2% of rows, so a large delta means the mask itself is misaligned.

Time-base provenance is asserted, not assumed: the gopro tree's recorded
``video_beginning_s`` must equal the measured ``extract_gopro.VIDEO_BEGINNING`` (see
``scripts/gopro_align.py``), and one sampled ``i2e_gopro`` npz must carry the 494x260
short-side-260 resolution ``scripts/gopro_i2e.sh`` documents.
"""

import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from brisbane_resolution import _Args  # noqa: E402
import brisbane_pooled as bp  # noqa: E402
from src import traversenpz as tnpz  # noqa: E402
from src.imagevpr import build_gt  # noqa: E402

ROOT = "/media/adam/vprdatasets/megaevent"
V8 = f"{ROOT}/v8_bench"
S2R = f"{ROOT}/sim2real"
NPZ_ROOT = f"{ROOT}/brisbane_npz"
EVENTLAB = "/media/adam/vprdatasets/eventgem"
EG_FEATURES = f"{ROOT}/eventgem_010/eventgem/features"
NPZ_BANKS = f"{EG_FEATURES}/brisbane_npz"

DATASET = "brisbane_event"
REF, QRY = "sunset1", "morning"
# Full list for the ENU origin — must match pairwise_sunset_ref.CONFIGS geom_traverses,
# or the gate cannot reproduce the ledger.
GEOM_TRAVERSES = ["daytime", "morning", "night", "sunrise", "sunset1"]
MASK_SOURCES = ("aps", "gopro")
KS = (1, 5, 10, 20)
THRESHOLD_M = 25.0

LEDGER = f"{V8}/pairwise_sunset_ref_{DATASET}.json"
RERANK_JSON = f"{V8}/eg010_rerank_gopro_pairwise.json"
OUT_JSON = f"{V8}/gopro_pairwise_{DATASET}.json"

EG_NAME = "Event-GeM 0.1.0"
# (name, align, {arm: bank path template}, (chunk, db_chunk)). Chunking must match each
# row's published run: topk_ranked's db_chunk merge is exact for distinct similarities
# but not for ties, and EventVLAD's constant-norm descriptors tie often enough to move
# R@1 by ~1e-3.
MODELS = [
    ("megaevent v8-B", "grid", {
        "real": f"{V8}/brisbane/r322ba50_{{t}}_v8.npy",
        "i2e_gopro": f"{S2R}/r322_i2e_gopro_{{t}}_v8b.npy"}, (256, None)),
    ("megaevent v8-S", "grid", {
        "real": f"{V8}/brisbane_s/r322ba50_{{t}}_v8s.npy",
        "i2e_gopro": f"{S2R}/r322_i2e_gopro_{{t}}_v8s.npy"}, (256, None)),
    ("SpikeVPR", "grid", {
        "real": f"{ROOT}/spikevpr_pooled/spikevpr_nsavp_ba50_{{t}}.npy",
        "i2e_gopro": f"{S2R}/spikevpr_nsavp_i2e_gopro_{{t}}.npy"}, (256, None)),
    ("EventVLAD", "centre", {
        "real": f"{ROOT}/eventvlad_pooled/eventvlad_ba50_{{t}}_eventvlad.npy",
        "i2e_gopro": f"{S2R}/eventvlad_i2e_gopro_{{t}}_eventvlad.npy"}, (256, None)),
    # real = the released .pt banks (tail-aligned); i2e_gopro = the npz-extracted gem224
    # bank from eventgem_010/brisbane_npz_extract.py (full grid).
    (EG_NAME, "tail", {
        "real": None,
        "i2e_gopro": f"{NPZ_BANKS}/eg010_gem224_i2e_gopro_{{t}}.npy"}, (64, 4096)),
]


# ---------------------------------------------------------------------------
# Banks
# ---------------------------------------------------------------------------
def eg010_load_bank(traverse):
    """The released 0.1.0 global bank — pairwise_sunset_ref.eg010_load_bank verbatim."""
    paths = sorted(glob.glob(
        f"{EG_FEATURES}/{DATASET}/*/{DATASET}_{traverse}_features.pt"))
    if not paths:
        raise SystemExit(f"no released feature bank for {DATASET}/{traverse} under "
                         f"{EG_FEATURES}")
    banks = [torch.load(p, map_location="cpu") for p in paths]
    for b in banks[1:]:
        if b.shape != banks[0].shape or not torch.allclose(b, banks[0], atol=1e-4):
            raise SystemExit(f"{DATASET}/{traverse}: pair-dir copies differ ({paths})")
    return torch.nn.functional.normalize(banks[0].float(), dim=1)


def load_bank(name, align, templates, arm, traverse):
    if name == EG_NAME and arm == "real":
        return eg010_load_bank(traverse)
    path = templates[arm].format(t=traverse)
    if not os.path.exists(path):
        raise SystemExit(f"{name} [{arm}]: missing bank {path}")
    bank = torch.from_numpy(np.asarray(np.load(path, mmap_mode="r"),
                                       dtype=np.float32))
    # A zero row is impossible in a finished bank (every extractor L2-norms or at least
    # activates) — but brisbane_resolution.extract PREALLOCATES full-size memmaps, so a
    # killed run leaves a full-size file of zeros that resume logic mistakes for done.
    # That exact failure shipped a 75%-zeros morning bank on 2026-09-11.
    zero = int((bank.norm(dim=1) < 1e-6).sum())
    if zero:
        raise SystemExit(f"{name} [{arm}] {traverse}: {zero} zero rows in {path} — a "
                         f"killed extraction left a preallocated memmap; delete the "
                         f"file and re-extract")
    # Informational only: the protocol scores whatever the published extractor saved, so
    # a non-unit norm is a property of the row's method (the gate still has to pass),
    # but a norm DIFFERENCE between the arms of one method would be worth seeing.
    off = float((bank.norm(dim=1) - 1.0).abs().max())
    if off > 1e-3:
        print(f"      note: {name} [{arm}] {traverse} rows not unit-norm "
              f"(max |1-norm| {off:.2e})")
    return bank


def rows(name, align, bank, xy, keep):
    """(desc, xy) — the bank's rows mapped onto the kept slice set.

    ``grid``   — one row per slice, exact.
    ``centre`` — EventVLAD: descriptor *i* is the middle of a 3-slice window, so it maps
                 to slice *i+1* and the grid loses its first and last row. Getting this
                 wrong is a silent ~0.7 m shift, not an error.
    ``tail``   — Event-GeM's released banks can run a tail frame or two short.
    """
    if align == "centre":
        xy, keep = xy[1:-1], keep[1:-1]
        if len(bank) != len(keep):
            raise SystemExit(f"{name}: bank {len(bank)} rows != centre grid {len(keep)}")
    elif align == "grid":
        if len(bank) != len(keep):
            raise SystemExit(f"{name}: bank {len(bank)} rows != grid {len(keep)}")
    elif align == "tail":
        if abs(len(bank) - len(keep)) > 2:
            raise SystemExit(f"{name}: bank {len(bank)} vs grid {len(keep)} — more than "
                             f"a fencepost apart, alignment unproven")
    n = min(len(bank), len(keep))
    keep = keep[:n]
    return bank[:n][torch.as_tensor(keep)], xy[:n][keep]


# ---------------------------------------------------------------------------
# Scoring (pairwise_sunset_ref.score verbatim)
# ---------------------------------------------------------------------------
def score(db_desc, db_xy, q_desc, q_xy, device, db_chunk, chunk):
    gt = build_gt(db_xy, q_xy, THRESHOLD_M)
    ranked = bp.topk_ranked(db_desc, q_desc, device, k=max(KS), chunk=chunk,
                            db_chunk=db_chunk)
    rec, _, scorable = bp.recall_from_ranked(ranked, gt, ks=KS)
    del gt, ranked
    return rec, int(db_desc.shape[0]), int(q_desc.shape[0]), int(scorable.sum())


def score_arm(arm, geom, keep, device):
    """{name: (rec, n_db, n_q, scorable)} for one arm on the kept rows."""
    out = {}
    for name, align, templates, (chunk, db_chunk) in MODELS:
        db_desc, db_xy = rows(name, align,
                              load_bank(name, align, templates, arm, REF),
                              geom[REF][0], keep[REF])
        q_desc, q_xy = rows(name, align,
                            load_bank(name, align, templates, arm, QRY),
                            geom[QRY][0], keep[QRY])
        out[name] = score(db_desc, db_xy, q_desc, q_xy, device, db_chunk, chunk)
        print(f"    {name:<20s} R@1 {out[name][0][1]:.4f}  "
              f"({out[name][1]:,} db x {out[name][3]:,} scorable)", flush=True)
        del db_desc, q_desc
    return out


# ---------------------------------------------------------------------------
# Provenance and masks
# ---------------------------------------------------------------------------
def check_provenance(seq, n):
    """The gopro tree must have been framed by the CURRENT measured time warp."""
    path = os.path.join(NPZ_ROOT, DATASET, "gopro", seq, "select.json")
    with open(path) as handle:
        report = json.load(handle)
    if int(report["n_slices"]) != n:
        raise SystemExit(f"{path}: n_slices {report['n_slices']} != geometry {n}")
    warp_path = os.path.join(os.path.dirname(HERE), "logs", "gopro2",
                             f"warp_{seq}.json")
    with open(warp_path) as handle:
        warp = json.load(handle)
    tw = report.get("timewarp")
    if not tw or tw.get("fingerprint") != warp["fingerprint"]:
        raise SystemExit(
            f"{seq}: gopro tree framed by warp {tw and tw.get('fingerprint')} but "
            f"{warp_path} is {warp['fingerprint']} — stale tree, re-run "
            f"scripts/extract_gopro.py --seq {seq} --timewarp {warp_path}")
    probe = os.path.join(NPZ_ROOT, DATASET, "i2e_gopro", seq, "frame_007000.npz")
    with np.load(probe) as z:
        res = z["resolution"].tolist()
    if res != [260, 494]:
        raise SystemExit(f"{probe}: resolution {res} != [260, 494] — the i2e_gopro tree "
                         f"was not converted at --short-side 260")
    return {"video_beginning_s": float(report["video_beginning_s"]),
            "timewarp_fingerprint": tw["fingerprint"],
            "timewarp_holdout_p90_s": tw.get("holdout_p90_s")}


def gate(full_real):
    with open(LEDGER) as handle:
        cell = json.load(handle)["results"][f"{REF}->{QRY}"]["methods"]
    failures, max_delta = [], 0.0
    for name, _, _, _ in MODELS:
        rec, n_db, _, scorable = full_real[name]
        want = cell[name]
        w = want["recall"]["native"]
        d = max(abs(rec[1] - w["1"]), abs(rec[10] - w["10"]))
        max_delta = max(max_delta, d)
        ok = (d < 5e-4 and n_db == want["n_database"] and scorable == want["scorable"])
        print(f"    {name:<20s} R@1 {rec[1]:.4f} vs {w['1']:.4f}  "
              f"R@10 {rec[10]:.4f} vs {w['10']:.4f}  db {n_db:,}/{want['n_database']:,} "
              f"scorable {scorable:,}/{want['scorable']:,}  "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(name)
    if failures:
        raise SystemExit(f"\nledger gate FAILED for {failures} — geometry/alignment/"
                         f"chunking drifted; no gopro number is trustworthy until this "
                         f"reproduces {LEDGER}")
    print("    ledger gate PASSED — machinery reproduces the published cell")
    return max_delta


def fold_rerank(results):
    """Rerank rows from the 0.1.0 driver, cross-checked against our own base cells."""
    try:
        with open(RERANK_JSON) as handle:
            doc = json.load(handle)
    except FileNotFoundError:
        print(f"\n  (+rerank not folded — {RERANK_JSON} missing; run "
              f"eventgem_010/gopro_rerank.py)")
        return {}
    folded = {}
    for arm in ("real", "i2e_gopro"):
        got = doc["arms"][arm]
        own = results[arm][EG_NAME][0]
        for k in KS:
            d = abs(got["recall_base"][str(k)] - own[k])
            if d > 3e-4:
                raise SystemExit(
                    f"rerank fold: {arm} base R@{k} {got['recall_base'][str(k)]:.4f} "
                    f"disagrees with this scorer's {own[k]:.4f} by {d:.2e} — the two "
                    f"drivers are not scoring the same rows")
        folded[arm] = got["recall_reranked"]
    return folded


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{DATASET}: {REF} (reference) -> {QRY} (query), {THRESHOLD_M:g} m, native"
          f"  [{device}]")

    # Ledger-identical geometry: full traverse list for the origin, npz_root=None.
    args = _Args(EVENTLAB, DATASET, 50, False, False)
    args.filter_dt_us = 50_000
    geom = bp.traverse_geometry(args, GEOM_TRAVERSES, None)

    keep_full = {seq: geom[seq][1] for seq in (REF, QRY)}

    # Machinery first: the gate needs only the real banks and the ledger, so it can
    # fail (or pass) before the gopro trees even exist.
    print(f"\n  [gate] real arm on the FULL grid vs {os.path.basename(LEDGER)}")
    full_real = score_arm("real", geom, keep_full, device)
    max_delta = gate(full_real)

    keep_union, mask_info, v0 = {}, {}, {}
    for seq in (REF, QRY):
        covered = keep_full[seq]
        n = len(covered)
        v0[seq] = check_provenance(seq, n)
        union = covered & tnpz.frame_validity_mask(NPZ_ROOT, DATASET, seq, n,
                                                   MASK_SOURCES)
        keep_union[seq] = union
        mask_info[seq] = {"n_grid": n, "covered": int(covered.sum()),
                          "union_kept": int(union.sum())}
        print(f"  {seq}: grid {n}, covered {int(covered.sum())}, "
              f"aps∪gopro-masked {int(union.sum())}")

    results = {}
    for arm in ("real", "i2e_gopro"):
        print(f"\n  [{arm}] on the union-masked rows")
        results[arm] = score_arm(arm, geom, keep_union, device)

    print("\n  mask effect (real, full grid -> masked): ")
    for name, _, _, _ in MODELS:
        d = results["real"][name][0][1] - full_real[name][0][1]
        print(f"    {name:<20s} dR@1 {d:+.4f}")

    folded = fold_rerank(results)

    print(f"\n\n{DATASET} — {REF} -> {QRY}, {THRESHOLD_M:g} m, native, "
          f"rows = covered ∧ ¬(aps ∪ gopro out-of-tolerance)\n")
    print(f"  {'method':<22s}{'real':>28s}{'i2e_gopro':>28s}")
    print(f"  {'':<22s}" + f"{'R@1':>10s}{'R@5':>9s}{'R@10':>9s}" * 2)
    for name, _, _, _ in MODELS:
        line = ""
        for arm in ("real", "i2e_gopro"):
            rec = results[arm][name][0]
            line += f"{rec[1]:>10.3f}{rec[5]:>9.3f}{rec[10]:>9.3f}"
        print(f"  {name:<22s}{line}")
    if folded:
        line = ""
        for arm in ("real", "i2e_gopro"):
            r = folded[arm]
            line += f"{float(r['1']):>10.3f}{float(r['5']):>9.3f}{float(r['10']):>9.3f}"
        print(f"  {'  +rerank':<22s}{line}")

    out = {
        "dataset": DATASET,
        "protocol": f"{REF} reference -> {QRY} query, {THRESHOLD_M:g} m Euclidean, "
                    f"native cosine, no PCA; rows = GPS-covered AND NOT "
                    f"(aps ∪ gopro out_of_tolerance)",
        "time_base": {s: {**v0[s], "warp_json": f"logs/gopro2/warp_{s}.json",
                          "align_json": f"logs/gopro2/align_{s}.json"}
                      for s in (REF, QRY)},
        "mask": mask_info,
        "gate": {"reference": f"{LEDGER}#results.{REF}->{QRY}",
                 "max_delta": float(max_delta), "passed": True},
        "results": {},
    }
    for key, cells in (("real_fullgrid", full_real), ("real", results["real"]),
                       ("i2e_gopro", results["i2e_gopro"])):
        out["results"][key] = {
            name: {"n_database": n_db, "n_queries": n_q, "scorable": scorable,
                   "recall": {str(k): float(rec[k]) for k in KS}}
            for name, (rec, n_db, n_q, scorable) in cells.items()}
    if folded:
        out["rerank"] = {"source": RERANK_JSON, **folded}

    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(out, handle, indent=1)
    os.replace(tmp, OUT_JSON)
    print(f"\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
