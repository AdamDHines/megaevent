"""Assemble the {ViT-S,ViT-B} x {ft4,full} x {GeM,SALAD} baseline table.

    pixi run python3 scripts/arch_ablation_table.py

Reads the two artefacts the sweep writes — `arch_ablation/tokyo_out/results_r322.json` from
`scripts/tokyo_trajectory.py` and `arch_ablation/brisbane_out/results_no_sunset2.json` from
`scripts/brisbane_pooled.py` — plus the already-scored shipping checkpoints, and prints the
table in markdown.

Brisbane pools `{daytime, morning, night, sunrise}` and **excludes `sunset2`**: it is the same
route recorded under sunset1's own illumination, so a pooled number that includes it is carried
by near-duplicate retrieval rather than by condition invariance.

**Every arm is reported in one descriptor space: dim 4096, power 0.5.** `pca_fit` clamps the dim
to the descriptor width, so that is the full-rank 2048-d basis for GeM and a 4096-d projection of
SALAD's 8448 — one setting, no per-aggregator or per-arm tuning, and the same space the two
shipping checkpoints were already scored in.

GeM was given its own power to compete in (`--pca ... 2048,0.25`, following the note at
`src/inference.py:52` that 0.25 is the GeM optimum) and the full grid says it does not want one:
0.25 beats 0.5 in 2 of the 8 GeM cells — the two ViT-S Tokyo arms — and loses the other 6,
including all four on Brisbane. So the extra power is a Tokyo/ViT-S accident, not an aggregator
property, and reporting it per arm would be selection on the test set. It changes no ordering
either way: even taking each GeM arm's *best* space per dataset, the strongest GeM number stays
below the weakest SALAD (Ours) number on both datasets. The full grid is printed underneath so
the choice is visible rather than hidden.
"""

import json
import os

ROOT = "/media/adam/vprdatasets/megaevent"
TOKYO = f"{ROOT}/arch_ablation/tokyo_out/results_r322.json"
BRISBANE = f"{ROOT}/arch_ablation/brisbane_out/results_no_sunset2.json"
# The two checkpoints the project ships, scored under the same protocol in earlier sessions —
# same 322^2, same filter arm, same clock calibration, and the same sunset2-free database.
# (tokyo file, tokyo key, brisbane file, brisbane key)
SHIPPED = {
    "s_ship": (f"{ROOT}/tokyo_small_v2/results_r322.json", "step10000",
               f"{ROOT}/brisbane_pooled/small_v2_no_sunset2.json", "r322ba50_S_step10000"),
    "b_ship": (f"{ROOT}/tokyo_v4_stage/A_out/results_r322.json", "P64_s10000",
               f"{ROOT}/brisbane_pooled/final_no_sunset2.json", "r322ba50"),
}
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
SHIP_ROWS = [
    ("s_ship", "ViT-S ft4 + SALAD (Ours, shipped)", "salad"),
    ("b_ship", "ViT-B full + SALAD (Ours, shipped)", "salad"),
]
SPACE = {"salad": "pca4096p0.5", "gem": "pca4096p0.5"}
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

    cells = {}
    for label, _, agg in ROWS:
        space = SPACE[agg]
        cells[label] = (tokyo_row(tokyo["results"], label, space)
                        + brisbane_row(brisbane["results"], BRIS_PREFIX + label, space))
    for label, _, agg in SHIP_ROWS:
        t_path, t_key, b_path, b_key = SHIPPED[label]
        t, b = load(t_path), load(b_path)
        if t is None or b is None:
            continue
        cells[label] = (tokyo_row(t["results"], t_key, SPACE[agg])
                        + brisbane_row(b["results"], b_key, SPACE[agg]))

    print(f"Tokyo 24/7: {tokyo['n_database']} database x {tokyo['n_queries']} queries @ "
          f"{tokyo['threshold_m']:g} m, {tokyo['resolution']}^2")
    first = next(iter(brisbane["results"].values()))
    print(f"Brisbane-Event: {brisbane['query']} -> {'+'.join(brisbane['database'])}, "
          f"{first['n_database']} database x {first['n_queries']} queries @ "
          f"{brisbane['threshold_m']:g} m, {first['resolution']}^2, "
          f"BA filter {first['filter_arm']}\n")

    print("| Baseline Method | Tokyo 24/7 R@1 | R@10 | Brisbane-Event R@1 | R@10 |")
    print("|---|---|---|---|---|")
    for label, name, _ in ROWS + SHIP_ROWS:
        if label not in cells:
            continue
        t1, t10, b1, b10 = cells[label]
        print(f"| {name} | {t1:.3f} | {t10:.3f} | {b1:.3f} | {b10:.3f} |")

    for title, data, reader, prefix in (
            ("Tokyo 24/7", tokyo["results"], tokyo_row, ""),
            ("Brisbane-Event", brisbane["results"], brisbane_row, BRIS_PREFIX)):
        block = data[prefix + ROWS[0][0]]
        spaces = spaces_of(block if prefix == "" else block["recall"])
        print(f"\n{title} — every descriptor space (R@1 / R@10); "
              f"the reported cell is starred")
        print("| arm | " + " | ".join(spaces) + " |")
        print("|---" * (len(spaces) + 1) + "|")
        for label, name, agg in ROWS:
            cs = []
            for space in spaces:
                r1, r10 = reader(data, prefix + label, space)
                star = "*" if space == SPACE[agg] else ""
                cs.append(f"{star}{r1:.3f} / {r10:.3f}{star}")
            print(f"| {name} | " + " | ".join(cs) + " |")


if __name__ == "__main__":
    main()
