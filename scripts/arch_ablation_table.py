"""Assemble the {ViT-S,ViT-B} x {ft4,full} x {GeM,SALAD} baseline table (v9, accumulate).

    pixi run python3 scripts/arch_ablation_table.py

Reads the two artefacts the v9 sweep writes — `arch_ablation_v9/tokyo_out/results_r322.json`
from `scripts/tokyo_trajectory.py` and `arch_ablation_v9/brisbane_out/results_no_sunset2.json`
from `scripts/brisbane_pooled.py` — and prints the table in markdown.

**Every arm comes from one sweep (gept `run_v9.sh`, wandb group `ftagg`) at one fixed step.**
That is the whole point of the v9 re-run. The previous version of this table drew rows 1-6
from the countmask `2x2x2` sweep and rows 7-8 from *shipped models of other waves*, which
differed additionally in `--P` (64 vs 16/32), domain randomisation (on vs off), step count and
head initialisation — so its two headline rows never sat on the axes the caption names. The
countmask sweep was also confounded in three further ways: no `--salad-proj` anywhere, ft4 on
a flat LR against full on `--llrd 0.75` (~7.5x the last-block encoder LR), and ViT-B at half
ViT-S's batch. `run_v9.sh`'s header documents all five with the evidence.

Reported at **step2000** for every arm, not `best.pt`: Tokyo and Brisbane move in opposite
directions (audit 2026-07-30, +13.3 Tokyo cost 8.0 Brisbane 4-cond) and the in-loop selection
is Brisbane pooled R@1, so per-arm selection would land every arm at its Brisbane peak and
systematically penalise the Tokyo column.

Brisbane pools `{daytime, morning, night, sunrise}` and **excludes `sunset2`**: it is the same
route recorded under sunset1's own illumination, so a pooled number that includes it is carried
by near-duplicate retrieval rather than by condition invariance.

**Every arm is reported in the native descriptor space — no PCA whitening.** These methods are
not whitened anywhere else in this project, and an ablation over fine-tuning depth and
aggregator has to be read in the space the models actually ship in; a whitened cell measures
the aggregator *plus* how well a 4096-d basis fits it, which is a second variable the table
does not name. The choice is not neutral and is worth a footnote: whitening is worth far more
to GeM than to SALAD (in the countmask sweep, `b_gem_full` went .5905 -> .7270 on Tokyo at
4096/0.5, +13.7), so a whitened table would understate the aggregator gap this exists to
measure. The v9 eval runs `--pca none`; if the grid comes back, the per-space table underneath
prints it.

**The projection is not ablated.** Every SALAD arm carries MegaLoc's post-SALAD
`Linear(16640 -> 8448)` — warm-started on ViT-B, randomly initialised on ViT-S, which is the
only option there (`salad.py::load_megaloc_aggregator` asserts a 768-d backbone). The shipped
system is that head, so this table ablates fine-tuning depth and aggregator *within* it.
"""


import json
import os

ROOT = "/media/adam/vprdatasets/megaevent"
TOKYO = f"{ROOT}/arch_ablation_v9/tokyo_out/results_r322.json"
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


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def tokyo_row(results, key, space):
    """(R@1, R@10) from a tokyo_trajectory result block, keyed by cutoff-as-string."""
    row = results[key][space]
    return float(row["1"]), float(row["10"])


def brisbane_row(results, key, space):
    """(R@1, R@10) from a brisbane_pooled result block at the headline tolerance."""
    row = results[key]["recall"][space]
    return float(row["1"]), float(row["10"])


def spaces_of(block):
    return [k for k in block if k.startswith(("native", "pca")) and not k.endswith("_dim_eff")]


def main():
    tokyo, brisbane = load(TOKYO), load(BRISBANE)
    if tokyo is None or brisbane is None:
        raise SystemExit(f"sweep artefacts not written yet:\n  {TOKYO}\n  {BRISBANE}")

    cells, absent = {}, []
    for label, _, agg in ROWS + LADDER_ROWS:
        space = SPACE[agg]
        # An arm still training, or one whose job failed, is skipped rather than crashing the
        # whole table — a partial wave is worth reading while the rest lands.
        try:
            cells[label] = (tokyo_row(tokyo["results"], label, space)
                            + brisbane_row(brisbane["results"], BRIS_PREFIX + label, space))
        except KeyError:
            absent.append(label)

    print(f"Tokyo 24/7: {tokyo['n_database']} database x {tokyo['n_queries']} queries @ "
          f"{tokyo['threshold_m']:g} m, {tokyo['resolution']}^2")
    first = next(iter(brisbane["results"].values()))
    print(f"Brisbane-Event: {brisbane['query']} -> {'+'.join(brisbane['database'])}, "
          f"{first['n_database']} database x {first['n_queries']} queries @ "
          f"{brisbane['threshold_m']:g} m, {first['resolution']}^2, "
          f"BA filter {first['filter_arm']}\n")

    print("| Baseline Method | Tokyo 24/7 R@1 | R@10 | Brisbane-Event R@1 | R@10 |")
    print("|---|---|---|---|---|")
    for label, name, _ in ROWS + LADDER_ROWS:
        if label not in cells:
            continue
        t1, t10, b1, b10 = cells[label]
        print(f"| {name} | {t1:.3f} | {t10:.3f} | {b1:.3f} | {b10:.3f} |")

    for title, data, reader, prefix in (
            ("Tokyo 24/7", tokyo["results"], tokyo_row, ""),
            ("Brisbane-Event", brisbane["results"], brisbane_row, BRIS_PREFIX)):
        present = next(l for l, _, _ in ROWS + LADDER_ROWS if l in cells)
        block = data[prefix + present]
        spaces = spaces_of(block if prefix == "" else block["recall"])
        print(f"\n{title} — every descriptor space (R@1 / R@10); "
              f"the reported cell is starred")
        print("| arm | " + " | ".join(spaces) + " |")
        print("|---" * (len(spaces) + 1) + "|")
        for label, name, agg in ROWS + LADDER_ROWS:
            if label not in cells:
                continue
            cs = []
            for space in spaces:
                r1, r10 = reader(data, prefix + label, space)
                star = "*" if space == SPACE[agg] else ""
                cs.append(f"{star}{r1:.3f} / {r10:.3f}{star}")
            print(f"| {name} | " + " | ".join(cs) + " |")
    if absent:
        print(f"\nnot in the artefacts yet: {', '.join(absent)}")


if __name__ == "__main__":
    main()
