"""Assemble `tab:nval` — the effect of the number of training places N on ViT-B recall (v9).

    pixi run python3 scripts/table_nval.py                 # markdown + LaTeX

N is the trainer's ``--P``: places per sub-batch, per stream. The multi-similarity loss is
computed *per stream*, so an anchor only ever sees ``P-1`` other places per call regardless of
global batch size — which is why this lever is about the loss, not the batch.

**Same wave as `tab:ftagg`.** All three rows come from gept `run_v9.sh` (wandb group `ftagg`),
are ViT-B + SALAD with MegaLoc's warm-started projection, ``--blocks 12 --steps 2000 --lr 5e-5
--encoder-lr-mult 0.1 --llrd none``, `accumulate` frames, seed 0, and are read at **step2000**
rather than at a selected checkpoint. They differ in exactly one token, ``--P``. N=64 is the
run `arch_ablation_table.py` reports as "ViT-B full + SALAD (Ours)" — one training run serves
both tables, so the N sweep costs only the two extra arms at 32 and 48.

**N=16 is not here.** The only P=16 ViT-B ever trained was a pre-countmask v1 run whose
checkpoint is gone. The table reports N in {32, 48, 64}.

Brisbane protocol
-----------------
The pooled one, and only that: `sunset1` as the query, `{daytime, morning, night, sunrise}`
pooled into one database, 25 m Euclidean radius, `sunset2` excluded because it is the same
route under sunset1's own illumination. That is what the paper's main Brisbane column reports.

The pairwise +-70 m along-track protocol this script used to offer has been removed. The v9
wave produces no pairwise scores, so the option could only have read the superseded v4
*countmask* sweep — a silent cross-representation, cross-protocol mix. The two protocols
differed by ~14 points at the same checkpoint, so that would not have been a small error.

**Read in the native descriptor space — no PCA whitening.** These models are not whitened
anywhere else in the project, and whitening is not neutral here: on the countmask sweep it
added ~7.7 points to every Brisbane cell while changing the N=32->64 Tokyo spread, so a
whitened `tab:nval` partly measures how well a 4096-d basis fits each bank rather than what N
did. The v9 eval runs `--pca none`.

**No seed repeat in this wave.** `gept/run_ablation.sh` fixes a decision rule that nothing
under **.03 R@1** is adoptable without a second seed reproducing it. The script prints the
N=32->64 spread against that bar; if a spread lands under it, report the row as unresolved
rather than as a trend, or add a `--seed 1` repeat of one N to the next wave.
"""

import json
import os

ROOT = "/media/adam/vprdatasets/megaevent"
V9 = f"{ROOT}/arch_ablation_v9"
TOKYO = f"{V9}/tokyo_out/results_r322.json"
BRISBANE = f"{V9}/brisbane_out/results_no_sunset2.json"

SPACE = "native"
DECISION_BAR = 0.03          # gept/run_ablation.sh's fixed adoption threshold, in R@1
# N -> the arm's --run-name in run_v9.sh. brisbane_pooled keys each arm `r322ba50_<label>`.
ROWS = [
    (32, "b_salad_full_P32"),
    (48, "b_salad_full_P48"),
    (64, "b_salad_full"),     # also tab:ftagg's "ViT-B full + SALAD (Ours)"
]
BRIS_PREFIX = "r322ba50_"


def tokyo(results, key):
    r = results[key][SPACE]
    return r["1"], r["10"]


def brisbane(doc, key):
    r = doc["results"][key]["recall"][SPACE]
    return r["1"], r["10"]


def main():
    for path in (TOKYO, BRISBANE):
        if not os.path.exists(path):
            raise SystemExit(
                f"v9 artefacts not written yet: {path}\n"
                f"Run {V9}/run.sh (Tokyo) and {V9}/brisbane.sh (Brisbane) once the wave "
                f"has returned its step2000.pt checkpoints.")

    tok = json.load(open(TOKYO))
    bris = json.load(open(BRISBANE))
    print(f"Tokyo 24/7   {TOKYO}")
    print(f"             {tok['resolution']}^2, {tok['n_database']:,} database x "
          f"{tok['n_queries']} queries, {tok['threshold_m']:.0f} m, space {SPACE}")
    print(f"Bris-Event   {BRISBANE}")
    print(f"             pooled {bris['threshold_m']:.0f} m, query {bris['query']}, "
          f"database {', '.join(bris['database'])}")

    rows, missing = [], []
    for n, key in ROWS:
        try:
            t1, t10 = tokyo(tok["results"], key)
            b1, b10 = brisbane(bris, BRIS_PREFIX + key)
        except KeyError:
            missing.append((n, key))
            continue
        rows.append((n, t1, t10, b1, b10))

    print(f"\n{'N':>4s} {'Tokyo R@1':>10s} {'Tokyo R@10':>11s} "
          f"{'Bris R@1':>9s} {'Bris R@10':>10s}")
    for n, t1, t10, b1, b10 in rows:
        print(f"{n:>4d} {t1:>10.3f} {t10:>11.3f} {b1:>9.3f} {b10:>10.3f}")

    if missing:
        print("\n  not in the artefacts yet: "
              + ", ".join(f"N={n} ({k})" for n, k in missing))

    # The whole point of quoting the spread: on the countmask wave the Brisbane spread was
    # under 2 points, i.e. inside the bar the repo's own decision rule sets.
    if len(rows) > 1:
        for name, idx in (("Tokyo", 1), ("Brisbane", 3)):
            spread = max(r[idx] for r in rows) - min(r[idx] for r in rows)
            over = spread >= DECISION_BAR
            print(f"\n  N spread, {name} R@1: {spread:.3f} — "
                  f"{'above' if over else 'BELOW'} the {DECISION_BAR:g} single-seed bar"
                  + ("" if over else "; report this row as unresolved, not as a trend"))

    print("\n% --- tab:nval body ---")
    for n, t1, t10, b1, b10 in rows:
        print(f"            {n} & {t1:.2f} & {t10:.2f} & {b1:.2f} & {b10:.2f} \\\\")


if __name__ == "__main__":
    main()
