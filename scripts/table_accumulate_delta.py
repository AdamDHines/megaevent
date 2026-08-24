"""megaevent v8 minus each RGB control, with both sides on ``accumulate`` frames.

    pixi run python3 scripts/table_accumulate_delta.py
    pixi run python3 scripts/table_accumulate_delta.py --latex

**Why this table exists.** The v8 megaevent models train on ``accumulate``, the GEPT-native
white-background render, while every published RGB-control number is still ``countmask``.
Differencing the two would confound the network with the frame it reads. This puts MixVPR,
CricaVPR, SALAD and MegaLoc on the *same* accumulate frames, so each delta is the network
alone.

Only the three **recorded** datasets are here — Brisbane-Event, NSAVP R0 and NYC-Event. The
I2E simulations are deliberately out of scope.

Everything is **native cosine at 25 m, no PCA whitening**, matching ``table_native.py``'s
policy and the v8 verdict table: a whitening basis is fit on one model's own database bank,
so reporting it is a per-model tuning step rather than a measurement.

Protocols, each identical on both sides of the delta:

* Brisbane-Event — pooled, query ``sunset1`` against a **4-traverse** database
  (``daytime + morning + night + sunrise``). ``sunset2`` is excluded throughout: it is the
  same route under the query's own illumination, so including it scores near-duplicate
  retrieval rather than condition invariance.
* NSAVP R0 — pooled, query ``R0_FA0`` against the five other ``R0_*`` traverses.
* NYC-Event — the dataset's own random 10% query split, seed 0.

The row-size guard from ``table_native`` is re-applied: a row whose cells disagree on the
database or scorable-query count is refused rather than printed, because a bank built from a
different file listing would difference against a different benchmark.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from table_native import EVAL, ROOT, read_cell  # noqa: E402

V8 = f"{ROOT}/v8_bench"
NYC = f"{EVAL}/nycevent"
# Each control at its own published input size — the same sizes table_native.py reports.
RES = {"mixvpr": 320, "cricavpr": 224, "salad": 322, "megaloc": 322,
       "boq": 322, "qaa": 322, "supervlad": 322}
# display name -> the key used in bank tags and output filenames
BASELINES = {"MixVPR": "mixvpr", "CricaVPR": "cricavpr", "SALAD": "salad",
             "MegaLoc": "megaloc", "BoQ": "boq", "QAA": "qaa", "SuperVLAD": "supervlad"}
OURS = ["Ours (ViT-S)", "Ours (ViT-B)"]
COLUMNS = list(BASELINES) + OURS


def _pooled_specs(pooled_json):
    """The accumulate control cells for one pooled (traverse) dataset."""
    return {label: ("pooled", f"{ROOT}/{model}_pooled/{pooled_json}",
                    f"r{RES[model]}ba50_accum", "native")
            for label, model in BASELINES.items()}


DATASETS = {
    "bris-event": {
        **_pooled_specs("brisbane_event_accumulate.json"),
        # 4-traverse rescore of the v8 banks (sunset2 excluded), the reported protocol
        "Ours (ViT-S)": ("pooled", f"{V8}/brisbane_s/brisbane_4trav.json",
                         "r322ba50", "native"),
        "Ours (ViT-B)": ("pooled", f"{V8}/brisbane/brisbane_4trav.json",
                         "r322ba50", "native"),
    },
    "nsavp": {
        **_pooled_specs("nsavp_accumulate.json"),
        "Ours (ViT-S)": ("pooled", f"{V8}/nsavp_s/results.json", "r322ba50", "native"),
        "Ours (ViT-B)": ("pooled", f"{V8}/nsavp/results.json", "r322ba50", "native"),
    },
    "nyc-event": {
        **{label: ("image", f"{NYC}/results_{model}_accumulate_r{RES[model]}.json",
                   None, "native")
           for label, model in BASELINES.items()},
        "Ours (ViT-S)": ("image",
                         f"{NYC}/nycevent/results_s_v8_accum_s500_s500_9aaa0c325c"
                         f"_accumulate_r322.json", None, "native"),
        "Ours (ViT-B)": ("image",
                         f"{NYC}/nycevent/results_b_v8_accum_s750_s750_3f4eb54780"
                         f"_accumulate_r322.json", None, "native"),
    },
}


def collect():
    """-> (rows, missing). rows = [(dataset, {column: (r1, r10)}, shape)]."""
    rows, missing = [], []
    for dataset, specs in DATASETS.items():
        cells, shapes = {}, {}
        for column in COLUMNS:
            value, shape, note = read_cell(specs[column])
            cells[column] = value
            if value is None:
                missing.append(f"{dataset}/{column}: {note}")
            elif shape:
                shapes[column] = shape
        # the guard: every cell in a row must have scored the same gallery and query set
        distinct = set(shapes.values())
        if len(distinct) > 1:
            missing.append(f"{dataset}: cells disagree on size {sorted(distinct)} — "
                           f"{ {c: s for c, s in shapes.items()} }")
            continue
        rows.append((dataset, cells, next(iter(distinct), None)))
    return rows, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--latex", action="store_true",
                    help="also print a tabular body of the delta rows")
    cli = ap.parse_args()

    rows, missing = collect()
    order = [d for d, _, _ in rows]
    cells = {d: c for d, c, _ in rows}
    shapes = {d: s for d, _, s in rows}

    def fmt(value, signed=False):
        if value is None:
            return f"{'--':>7s}{'--':>8s}"
        sign = "+" if signed else ""
        return f"{value[0]:>{sign}7.3f}{value[1]:>{sign}8.3f}"

    print("accumulate frames, native cosine, 25 m, no PCA whitening\n")
    print(f"{'method':14s}" + "".join(f"{d:>17s}" for d in order))
    print(f"{'':14s}" + "".join(f"{'R@1':>8s}{'R@10':>8s} " for _ in order))
    for column in COLUMNS:
        if column in OURS:
            continue
        print(f"{column:14s}" + "".join(fmt(cells[d][column]) + " " for d in order))
    print()
    for column in OURS:
        print(f"{column:14s}" + "".join(fmt(cells[d][column]) + " " for d in order))
    print(f"\n{'grid':14s}" + "".join(
        f"{(f'{shapes[d][0]:,}x{shapes[d][1]:,}' if shapes[d] else '-'):>17s}" for d in order))

    for ours in OURS:
        print(f"\ndelta = {ours} - baseline   (positive = megaevent ahead)\n")
        print(f"{'baseline':14s}" + "".join(f"{d:>17s}" for d in order))
        for column in BASELINES:
            line = f"{column:14s}"
            for d in order:
                mine, base = cells[d][ours], cells[d][column]
                line += (fmt(None) if mine is None or base is None else
                         fmt((mine[0] - base[0], mine[1] - base[1]), signed=True)) + " "
            print(line)

    if cli.latex:
        print("\n% deltas, ours - baseline; per dataset R@1 & R@10")
        for ours in OURS:
            print(f"% {ours}")
            for column in BASELINES:
                out = []
                for d in order:
                    mine, base = cells[d][ours], cells[d][column]
                    out += (["--", "--"] if mine is None or base is None else
                            [f"{mine[0] - base[0]:+.2f}", f"{mine[1] - base[1]:+.2f}"])
                print(f"{column} & " + " & ".join(out) + r" \\")

    if missing:
        print("\nnot yet available:")
        for note in missing:
            print(f"  {note}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
