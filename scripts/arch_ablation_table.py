"""Assemble the {ViT-S,ViT-B} x {ft4,full} x {GeM,SALAD} baseline table (v9, accumulate).

    pixi run python3 scripts/arch_ablation_table.py                  # markdown + detail
    pixi run python3 scripts/arch_ablation_table.py --latex          # the 8-row tabular
    pixi run python3 scripts/arch_ablation_table.py --ftagg          # tab:ftagg, markdown detail
    pixi run python3 scripts/arch_ablation_table.py --ftagg --latex  # tab:ftagg, the four rows

Reads the artefacts the v9 sweep writes: one `results_<arm>_s*_accumulate_r322.json` per arm
under `evaluations/nycevent/nycevent/` (from `main.py -d nycevent`) and the single
`arch_ablation_v9/brisbane_out/results_no_sunset2.json` (from `scripts/brisbane_pooled.py`).
Springfield adds one `output/springfield_full/results_<arm>_s*_accumulate_r322_hp1_baoff.json`
per arm (from `scripts/springfield_full.py`) plus that run's diag dump
`output/springfield_diag/dump_<tag>.npz` (from `scripts/springfield_diag.py --stage dump`).

**BOTH COLUMNS ARE RECORDED-EVENT DATASETS.** The previous version of this table paired
Brisbane with Tokyo 24/7, and Tokyo is I2E-simulated from RGB by the same generator family as
the ~4.77M-image training set — which is exactly the domain advantage a reviewer objects to
when the method under test was fine-tuned on I2E events. Measured on Brisbane, where the
DAVIS346 recorded intensity frames and events through one sensor on one clock, the real->I2E
drop is large and not equal across methods (v8-B -0.135 R@1, Event-GeM 0.1.0 -0.30), so an
I2E column cannot be read as a proxy for the recorded one. NYC-Event and Brisbane-Event are
both recorded, so this table is read entirely on real events. So is Springfield.

**NYC-Event uses its published protocol** — the random 10% query split the dataset ships
(`protocol.json`: query_fraction 0.1, seed 0), which is what every published number on this
dataset uses. Its known weakness belongs in the caption rather than in a silent protocol
substitution: 99.8% of queries have a same-session database sample within 25 m (median 1.81 m)
and for 88.9% the single nearest database sample is from the query's own session, so the
column rewards near-duplicate retrieval within one drive far more than condition invariance.
`scripts/nycevent_all_sessions.py` reports the leave-one-session-out alternative (everyone
falls ~0.80 -> ~0.10) and is the honest companion number if the caption needs one.

**Every arm comes from one sweep (gept `run_v9.sh`, wandb group `ftagg`) at one fixed step.**
That is the whole point of the v9 re-run. The previous version of this table drew rows 1-6
from the countmask `2x2x2` sweep and rows 7-8 from *shipped models of other waves*, which
differed additionally in `--P` (64 vs 16/32), domain randomisation (on vs off), step count and
head initialisation — so its two headline rows never sat on the axes the caption names. The
countmask sweep was also confounded in three further ways: no `--salad-proj` anywhere, ft4 on
a flat LR against full on `--llrd 0.75` (~7.5x the last-block encoder LR), and ViT-B at half
ViT-S's batch. `run_v9.sh`'s header documents all five with the evidence.

Reported at **step2000** for every arm, not `best.pt`: per-arm selection would land each arm at
its own peak on whichever metric the trainer watched (in-loop Brisbane pooled R@1) and make the
comparison a selection artefact. Cost of the fixed step, measured on the v8 anchor's own
in-loop curve (`runs/b_v8_accum_vpr` tfevents): step750, its shipped `best.pt`, scores .9219
and step2000 scores .9184 — 0.0035, with the I2E proxy still rising at 2000.

Brisbane pools `{daytime, morning, night, sunrise}` and **excludes `sunset2`**: it is the same
route recorded under sunset1's own illumination, so a pooled number that includes it is carried
by near-duplicate retrieval rather than by condition invariance.

**Every arm is reported in the native descriptor space — there is no PCA whitening anywhere in
this table, and none is computed.** These methods are not whitened anywhere else in this
project, and an ablation over fine-tuning depth and aggregator has to be read in the space the
models actually ship in; a whitened cell measures the aggregator *plus* how well a 4096-d basis
fits it, which is a second variable the table does not name. The choice is not neutral and is
worth a footnote: whitening is worth far more to GeM than to SALAD (in the countmask sweep,
`b_gem_full` went .5905 -> .7270 on Tokyo at 4096/0.5, +13.7), so a whitened table would
understate the aggregator gap this exists to measure. Brisbane runs `--pca none` and NYC runs
`main.py --no-pca`, so the whitening fit never happens — which also removes the step that
killed both pitts250k runs on RAM. (The two v8 shipping models' NYC files do carry a `pca`
block from their own benchmark run; only `native` is read here.)

**The projection is not ablated.** Every SALAD arm carries MegaLoc's post-SALAD
`Linear(16640 -> 8448)` — warm-started on ViT-B, randomly initialised on ViT-S, which is the
only option there (`salad.py::load_megaloc_aggregator` asserts a 768-d backbone). The shipped
system is that head, so this table ablates fine-tuning depth and aggregator *within* it.

**Springfield is read on the PAPER CELL, and computed here from the diag dump.** The cell is
the one the main Springfield baseline table quotes (megaevent v8-B 0.4925, MegaLoc 0.3838):
day + dawn query sessions, camera within 135 deg of the route direction (|psi| < 135 — the
reversed passes are 180 deg from the forward arm and at least 90 from every arm, a viewpoint
condition no panoramic benchmark contains), 5,557 of the 15,886 kept query slices, against the
full 132,569-row gallery at 25 m. Reversals are a scoring mask, not an extraction choice, so
the all-conditions number from the same banks is printed alongside in the detail view and is
NOT what goes in the LaTeX. The camera offset `psi` is a property of the capture (GPS travel
bearing while moving, a per-session yaw fit while standing), not of any model, so every row
reads the one orient bundle the ViT-B run produced — the same default
`scripts/springfield_baselines.py --orient-tag` scores every baseline against — and the slice
counts are asserted equal so a different mask fails loudly rather than misaligns. Springfield
rows are BA-off (the dataset's established protocol: the event filter is a Brisbane/NYC
convention), extracted at 322^2 like everything else. The ViT-B row recomputed this way
reproduces the published 0.4925 / 0.5715 / 0.6068 / 0.6392 to the digit.

**tab:ftagg mixes two waves, BY THE USER'S DECISION (2026-09-04).** The four rows the paper
prints are the two v9 `full` GeM arms (one sweep, step 2000) and the two v8 SHIPPING SALAD
models (`s_v8_accum_s500`, `b_v8_accum_s750`: best.pt-selected at step 500 / 750, a different
wave). That is the step/selection confound the v9 sweep was designed to remove, reintroduced
for the SALAD rows so the table shows the models the paper ships; the caption should say so.
The v9 SALAD arms remain in the 8-row table above for the clean comparison. Rounding note:
`b_v8_accum_s750` NYC R@1 is 0.9048 — 0.90 at 2 dp, not the 0.91 a second rounding of 0.905
gives.
"""


import argparse
import glob
import json
import os

import numpy as np

ROOT = "/media/adam/vprdatasets/megaevent"
# main.py appends the dataset name to --feature-dir, so the accumulate-era NYC artefacts sit
# one level deeper than the older ones. Per-arm JSONs, not one combined file.
NYC_DIR = f"{ROOT}/evaluations/nycevent/nycevent"
BRISBANE = f"{ROOT}/arch_ablation_v9/brisbane_out/results_no_sunset2.json"
# label -> (row name, aggregator)
ROWS = [
    ("s_gem_ft4",    "ViT-S ft4 + GeM",             "gem"),
    ("b_gem_ft4",    "ViT-B ft4 + GeM",             "gem"),
    ("s_gem_full",   "ViT-S full + GeM",            "gem"),
    ("b_gem_full",   "ViT-B full + GeM",            "gem"),
    ("s_salad_full", "ViT-S full + SALAD",          "salad"),
    ("b_salad_ft4",  "ViT-B ft4 + SALAD",           "salad"),
    ("s_salad_ft4",  "ViT-S ft4 + SALAD (Ours)",    "salad"),
    ("b_salad_full", "ViT-B full + SALAD (Ours)",   "salad"),
]
# The N sweep (tab:nval, scripts/table_nval.py) shares this wave: N is the trainer's --P and
# `b_salad_full` at --P 64 is both the factorial's ViT-B full + SALAD cell and the N=64 row.
# Those extra arms are not rows here.
LADDER_ROWS = []
SPACE = {"salad": "native", "gem": "native"}
# brisbane_pooled tags every arm with its run config; the sweep's arms are `r322ba50_<label>`.
BRIS_PREFIX = "r322ba50_"

# Springfield: the results json of each arm's springfield_full.py run, and the diag dump of the
# same run (top-20 gallery rows per query slice) that the paper cell is computed from.
SF_DIR = "output/springfield_full"
SF_DIAG = "output/springfield_diag"
# Per-slice camera-vs-route offset. Capture property, not model property — see the docstring.
SF_ORIENT = "b_v8_accum_s750_s750_3f4eb54780_accumulate_r322_hp1_baoff"
SF_SWEEPS = ("day", "dawn")
SF_REV_DEG = 135.0
SF_KS = (1, 5, 10, 20)
SF_PAPER_N = 5557            # the main table's cell; a different count is a different cell
# tab:ftagg — label -> (row name, group). Two waves by decision; see the docstring.
FTAGG_ROWS = [
    ("s_gem_full",      "ViT/S + GeM",   "gem"),
    ("b_gem_full",      "ViT/B + GeM",   "gem"),
    ("s_v8_accum_s500", "ViT/S + SALAD", "salad"),
    ("b_v8_accum_s750", "ViT/B + SALAD", "salad"),
]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def one_result(pattern, label, what):
    """The single artefact matching ``pattern``, or None if not written yet.

    The filename carries the checkpoint sha, and the step in it is `ck["step"]`, which is
    0-indexed — so the step2000.pt milestone writes `_s1999_`. Both are wildcards in the
    pattern, and two matches for one arm means two shas are on disk, which is a stale
    artefact rather than something to choose between.
    """
    hits = sorted(glob.glob(pattern))
    if len(hits) > 1:
        raise SystemExit(f"{label}: {len(hits)} {what} result files match, sha is ambiguous —"
                         + "".join("\n  " + os.path.basename(h) for h in hits))
    return load(hits[0]) if hits else None


def nyc_results(labels):
    """``{label: parsed json}`` for every arm that has been scored on NYC-Event."""
    out = {}
    for label in labels:
        r = one_result(f"{NYC_DIR}/results_{label}_s*_accumulate_r322.json", label, "NYC")
        if r is not None:
            out[label] = r
    return out


def springfield_results(labels):
    """``{label: parsed json}`` for every arm scored on Springfield under the megaevent
    protocol: BA-off, hot-pixel on, 322^2, accumulate, full gallery."""
    out = {}
    for label in labels:
        r = one_result(f"{SF_DIR}/results_{label}_s*_accumulate_r322_hp1_baoff.json",
                       label, "Springfield")
        if r is not None:
            out[label] = r
    return out


def nyc_row(results, key, space):
    """(R@1, R@10) from a main.py image-dataset result file."""
    row = results[key]["recall"][space]
    return float(row["1"]), float(row["10"])


def brisbane_row(results, key, space):
    """(R@1, R@10) from a brisbane_pooled result block at the headline tolerance."""
    row = results[key]["recall"][space]
    return float(row["1"]), float(row["10"])


_ORIENT = {}


def orient_psi(sids):
    """Per-slice camera offset psi over the query slices, concatenated in ``sids`` order."""
    path = f"{SF_DIAG}/orient_{SF_ORIENT}.npz"
    if path not in _ORIENT:
        if not os.path.exists(path):
            raise SystemExit(f"no {path} — run scripts/springfield_diag.py --stage orient")
        _ORIENT[path] = np.load(path, allow_pickle=False)
    o = _ORIENT[path]
    missing = [s for s in sids if f"{s}_psi" not in o]
    if missing:
        raise SystemExit(f"{path} has no psi for {missing[:3]}{'...' if len(missing) > 3 else ''}")
    return np.concatenate([o[f"{s}_psi"] for s in sids])


def springfield_paper_cell(res):
    """``({k: R@k}, n_slices)`` on the paper cell from the arm's diag dump, or None if the
    dump is not written yet.

    A hit at k is a gallery row within ``threshold_m`` among the dump's top k, which is how
    ``springfield_region_map.load_run`` scores hit1/hit10. The cell keeps a slice when its
    sweep is in SF_SWEEPS and |psi| < SF_REV_DEG — ``springfield_baselines.exclude_reversals``'s
    rule on the same psi.
    """
    tag = res["tag"]
    path = f"{SF_DIAG}/dump_{tag}.npz"
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    sids = [str(s) for s in d["sids"]]
    sweeps = np.array([str(s) for s in d["sweeps"]])
    psi = orient_psi(sids)
    if len(psi) != len(d["q_sid"]):
        raise SystemExit(f"{tag}: orient holds {len(psi)} query slices but the dump has "
                         f"{len(d['q_sid'])} — the two were built on different masks")
    q_xy, db_xy = d["q_xy"].astype(np.float64), d["db_xy"].astype(np.float64)
    top_d = np.linalg.norm(db_xy[d["top_i"]] - q_xy[None], axis=2)       # [20, n_q]
    hit = top_d <= float(res["threshold_m"])
    keep = np.isin(sweeps[d["q_sid"]], SF_SWEEPS) & (np.abs(psi) < SF_REV_DEG)
    return {k: float(hit[:k].any(axis=0)[keep].mean()) for k in SF_KS}, int(keep.sum())


def springfield_cells(sf, labels):
    """``{label: {"paper": {k: r} | None, "n_paper": int | None, "all": (R@1, R@10)}}``.

    An arm with a results json but no dump yet gets its all-conditions numbers and a None
    paper cell — the table prints it as pending rather than silently substituting the
    wrong cell.
    """
    out = {}
    for label in labels:
        if label not in sf:
            continue
        res = sf[label]
        micro = res["recall_micro"]["overall"]
        cell = springfield_paper_cell(res)
        if cell is not None and cell[1] != SF_PAPER_N:
            raise SystemExit(f"{label}: paper cell has {cell[1]} slices, the main table's has "
                             f"{SF_PAPER_N} — not the same cell")
        out[label] = {"paper": cell[0] if cell else None,
                      "n_paper": cell[1] if cell else None,
                      "all": (float(micro["1"]), float(micro["10"])),
                      "n_all": int(sum(res["scorable"].values())),
                      "n_db": int(res["n_database"])}
    return out


def spaces_of(block):
    return [k for k in block if k.startswith(("native", "pca")) and not k.endswith("_dim_eff")]


def emit_latex(cells):
    """The paper's tabular rows: NYC R@1/R@10 then Brisbane R@1/R@10, 2 dp.

    The column best among the non-(Ours) rows is underlined, which is the convention the
    existing table uses; the (Ours) rows are left plain for the caller to bold.
    """
    baseline = [l for l, n, _ in ROWS if l in cells and "(Ours)" not in n]
    best = [max(cells[l][c] for l in baseline) for c in range(4)] if baseline else [None] * 4
    print("% NYC-Event (published random 10% split) R@1/R@10, "
          "Brisbane-Event (pooled, no sunset2) R@1/R@10 — native, 25 m, step2000")
    for label, name, _ in ROWS + LADDER_ROWS:
        if label not in cells:
            continue
        ours = "(Ours)" in name
        cs = []
        for c, v in enumerate(cells[label]):
            t = f"{v:.2f}"
            if not ours and best[c] is not None and abs(v - best[c]) < 5e-9:
                t = f"\\underline{{{t}}}"
            cs.append(t)
        print(f"            {name.replace('ViT-', 'ViT/')} & " + " & ".join(cs) + r" \\")


def ftagg_cells(nyc, sf):
    """``{label: (nyc R@1, nyc R@10, springfield R@1, springfield R@10)}`` for tab:ftagg,
    plus the labels still missing an artefact (NYC json, Springfield json or dump)."""
    sfc = springfield_cells(sf, [l for l, _, _ in FTAGG_ROWS])
    cells, absent, n_paper = {}, [], None
    for label, _, _ in FTAGG_ROWS:
        if label not in nyc or label not in sfc or sfc[label]["paper"] is None:
            absent.append(label)
            continue
        n1, n10 = nyc_row(nyc, label, "native")
        paper = sfc[label]["paper"]
        cells[label] = (n1, n10, paper[1], paper[10])
        n_paper = sfc[label]["n_paper"]
    return cells, absent, sfc, n_paper


def emit_ftagg_latex(cells, absent, n_paper):
    """tab:ftagg's rows in the paper's layout: NYC R@1/R@10 then Springfield R@1/R@10, 2 dp.
    Best per column in bold, second best underlined — the caption's convention. Ties share
    the bold; second best is the largest value strictly below the best."""
    print("% tab:ftagg — NYC-Event (published random 10% split) R@1/R@10, native; Springfield "
          f"paper cell (day+dawn, |psi| < {SF_REV_DEG:g} deg, {n_paper or '?'} query slices, "
          "132,569-row gallery, BA-off) R@1/R@10, native; 25 m")
    print("% GeM rows: v9 sweep, step 2000.  SALAD rows: v8 shipping models "
          "(best.pt, step 500 / 750) — two waves, by decision; say so in the caption.")
    best, second = [], []
    for c in range(4):
        vals = sorted({round(cells[l][c], 9) for l in cells}, reverse=True)
        best.append(vals[0] if vals else None)
        second.append(vals[1] if len(vals) > 1 else None)
    prev_group = None
    for label, name, group in FTAGG_ROWS:
        if label not in cells:
            continue
        if prev_group is not None and group != prev_group:
            print("            \\midrule")
        prev_group = group
        cs = []
        for c, v in enumerate(cells[label]):
            t = f"{v:.2f}"
            if best[c] is not None and abs(v - best[c]) < 5e-9:
                t = f"\\textbf{{{t}}}"
            elif second[c] is not None and abs(v - second[c]) < 5e-9:
                t = f"\\underline{{{t}}}"
            cs.append(t)
        print(f"            {name} & " + " & ".join(cs) + r" \\")
    if absent:
        print(f"% not in the artefacts yet: {', '.join(absent)}")


def emit_ftagg_markdown(nyc, cells, absent, sfc, n_paper):
    any_sf = next(iter(sfc.values()), None)
    if any_sf:
        print(f"Springfield: {any_sf['n_db']} gallery rows; paper cell = {SF_SWEEPS[0]}+"
              f"{SF_SWEEPS[1]}, |psi| < {SF_REV_DEG:g} deg, {n_paper or '?'} slices; "
              f"all conditions = {any_sf['n_all']} slices; 25 m, native, BA-off")
    print("| Baseline Method | NYC R@1 | R@10 | Springfield paper R@1 | R@5 | R@10 | R@20 "
          "| all R@1 | R@10 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, name, _ in FTAGG_ROWS:
        n = f"{nyc_row(nyc, label, 'native')[0]:.3f} | {nyc_row(nyc, label, 'native')[1]:.3f}" \
            if label in nyc else "– | –"
        s = sfc.get(label)
        if s and s["paper"]:
            p = " | ".join(f"{s['paper'][k]:.4f}" for k in SF_KS)
        elif s:
            p = "pending dump | | |"
        else:
            p = "– | – | – | –"
        a = f"{s['all'][0]:.4f} | {s['all'][1]:.4f}" if s else "– | –"
        print(f"| {name} | {n} | {p} | {a} |")
    if absent:
        print(f"\nnot in the artefacts yet: {', '.join(absent)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="print the paper's tabular rows")
    ap.add_argument("--ftagg", action="store_true",
                    help="tab:ftagg — the four paper rows (v9 GeM + v8 ship SALAD) with the "
                         "Springfield paper cell instead of Brisbane")
    args = ap.parse_args()

    labels = [l for l, _, _ in ROWS + LADDER_ROWS]
    ftagg_labels = [l for l, _, _ in FTAGG_ROWS]
    nyc = nyc_results(sorted(set(labels + ftagg_labels)))
    sf = springfield_results(sorted(set(labels + ftagg_labels)))

    if args.ftagg:
        cells, absent, sfc, n_paper = ftagg_cells(nyc, sf)
        if not cells and not sfc:
            raise SystemExit("no tab:ftagg artefacts yet:\n"
                             f"  {NYC_DIR}/results_<arm>_s*_accumulate_r322.json\n"
                             f"  {SF_DIR}/results_<arm>_s*_accumulate_r322_hp1_baoff.json\n"
                             f"  {SF_DIAG}/dump_<tag>.npz")
        if args.latex:
            emit_ftagg_latex(cells, absent, n_paper)
        else:
            emit_ftagg_markdown(nyc, cells, absent, sfc, n_paper)
        return

    brisbane = load(BRISBANE)
    if not nyc or brisbane is None:
        raise SystemExit("sweep artefacts not written yet:\n"
                         f"  {NYC_DIR}/results_<arm>_s*_accumulate_r322.json ({len(nyc)} found)"
                         f"\n  {BRISBANE}")

    cells, absent = {}, []
    for label, _, agg in ROWS + LADDER_ROWS:
        space = SPACE[agg]
        # An arm still training, or one whose job failed, is skipped rather than crashing the
        # whole table — a partial wave is worth reading while the rest lands.
        try:
            cells[label] = (nyc_row(nyc, label, space)
                            + brisbane_row(brisbane["results"], BRIS_PREFIX + label, space))
        except KeyError:
            absent.append(label)
    sfc = springfield_cells(sf, labels)

    if args.latex:
        emit_latex(cells)
        if absent:
            print(f"% not in the artefacts yet: {', '.join(absent)}")
        return

    any_nyc = next(iter(nyc.values()))
    print(f"NYC-Event: {any_nyc['n_database']} database x {any_nyc['n_queries']} queries @ "
          f"{any_nyc['threshold_m']:g} m, published random 10% split")
    first = next(iter(brisbane["results"].values()))
    print(f"Brisbane-Event: {brisbane['query']} -> {'+'.join(brisbane['database'])}, "
          f"{first['n_database']} database x {first['n_queries']} queries @ "
          f"{brisbane['threshold_m']:g} m, {first['resolution']}^2, "
          f"BA filter {first['filter_arm']}")
    any_sf = next(iter(sfc.values()), None)
    if any_sf:
        n_paper = next((s["n_paper"] for s in sfc.values() if s["n_paper"]), None)
        print(f"Springfield: {any_sf['n_db']} gallery rows; paper cell = day+dawn, |psi| < "
              f"{SF_REV_DEG:g} deg, {n_paper or '?'} slices; all conditions = "
              f"{any_sf['n_all']} slices; BA-off")
    print()

    print("| Baseline Method | NYC-Event R@1 | R@10 | Brisbane-Event R@1 | R@10 "
          "| Springfield paper R@1 | R@10 | all R@1 | R@10 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, name, _ in ROWS + LADDER_ROWS:
        if label not in cells:
            continue
        n1, n10, b1, b10 = cells[label]
        s = sfc.get(label)
        if s and s["paper"]:
            sf_cols = (f"{s['paper'][1]:.3f} | {s['paper'][10]:.3f} | "
                       f"{s['all'][0]:.3f} | {s['all'][1]:.3f}")
        elif s:
            sf_cols = f"pending dump | | {s['all'][0]:.3f} | {s['all'][1]:.3f}"
        else:
            sf_cols = "– | – | – | –"
        print(f"| {name} | {n1:.3f} | {n10:.3f} | {b1:.3f} | {b10:.3f} | {sf_cols} |")

    # A whitened space would be a second variable this table does not name, so neither eval
    # computes one and there is nothing else to print per arm. If a space other than native
    # ever shows up in an artefact, say so loudly rather than silently reporting one column.
    for title, data, prefix in (("NYC-Event", nyc, ""),
                                ("Brisbane-Event", brisbane["results"], BRIS_PREFIX)):
        for label, _, _ in ROWS + LADDER_ROWS:
            if label not in cells:
                continue
            extra = [s for s in spaces_of(data[prefix + label]["recall"]) if s != "native"]
            if extra:
                print(f"\nWARNING {title} {label}: artefact carries non-native "
                      f"{', '.join(extra)} — this table is native-only by design")
    if absent:
        print(f"\nnot in the artefacts yet: {', '.join(absent)}")


if __name__ == "__main__":
    main()
