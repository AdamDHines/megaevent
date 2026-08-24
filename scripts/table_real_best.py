"""The real-event-selected checkpoints on the three recorded datasets.

    pixi run python3 scripts/table_real_best.py
    pixi run python3 scripts/table_real_best.py --latex

`tab:native` reports four checkpoints picked by the trainer's **combined** in-loop metric
(each run's ``best.pt``). Every run also tracked a separate **real-event** metric and saved
``best_real.pt``, which lands early — step 500 or 1000, before the runs drift toward the I2E
half of the objective. This scores that second selection on the only three datasets it was
selecting for: NYC-Event, Brisbane-Event and NSAVP R0, all recorded off a real sensor.

Nothing else moves. Same 322x322 countmask frames, same background-activity filter, same 25 m
Euclidean radius, same pooled databases, same streamed top-k. **No PCA whitening** — a
whitening basis is fit on the model's own database bank, so reporting it is a per-model tuning
step rather than a measurement (``table_native.py`` says this at more length). The native
descriptor is what each model produces.

Two deltas per cell, both on R@1 and R@10:

* **vs shipped** — the same run's currently-reported checkpoint in ``tab:native``. Positive
  means selecting on real events beat selecting on the combined metric, with the architecture,
  the data and the protocol all held fixed. This is the only comparison here that isolates the
  selection metric.
* **vs +rerank** — Event-GeM's top-50 homography re-rank, the strongest event baseline on all
  three of these rows. Read through ``table_native.read_cell`` off the same JSON the table
  reads, so the two scripts cannot disagree about what the baseline scored.

The row-size guard from ``table_native`` is re-applied across the new cells *and* the ones they
are differenced against: a bank built from a different file listing would score silently and
produce a delta against a different benchmark.
"""

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from table_native import DATASETS, EVAL, ROOT, TAG, read_cell  # noqa: E402

# label -> (shipped column in tab:native it replaces, one-line architecture)
MODELS = {
    "v5b_mloc_s500": ("ViT-B MLoc", "ViT-B/14 + MegaLoc head, 228.6M", "b_noise_projmegaloc_P64_v5"),
    "v5b_salad_s1000": ("ViT-B SALAD", "ViT-B/14 + SALAD, 88.0M", "b_noise_P64_v5"),
    "v6s_mloc_s1000": ("ViT-S MLoc", "ViT-S/14 + MegaLoc head, 163.5M", "s_noise_projnaive_ft4_P64_v6"),
    "v6s_salad_s1000": ("ViT-S SALAD", "ViT-S/14 + SALAD, 22.9M", "s_noise_ft4_P64_v6"),
}
# Only the recorded datasets. The I2E simulations are not what best_real was selecting for and
# are deliberately not re-scored here.
ROWS = ["NYC-Event", "Brisbane-Event", "NSAVP R0"]
BRISBANE = f"{ROOT}/brisbane_realbest/realbest_no_sunset2.json"
NSAVP = f"{ROOT}/nsavp_pooled/results_route0_realbest.json"


def nyc_spec(label):
    """The image-set result JSON for one label, found rather than spelled out.

    ``src/methods.py`` builds the tag from ``basename + step + sha256[:10]``, and the step it
    records is the checkpoint's own (999 for a ``step1000.pt``, 500 for that one
    ``best_real.pt``). Globbing on the label and asserting a single match keeps this script
    from encoding a digest that only the extractor can know.
    """
    hits = sorted(glob.glob(f"{EVAL}/nycevent/results_{label}_s*_countmask_r322.json"))
    if len(hits) != 1:
        return ("image", f"{EVAL}/nycevent/results_{label}_s?_*_countmask_r322.json", None,
                "native")
    return ("image", hits[0], None, "native")


def specs_for(label):
    return {"NYC-Event": nyc_spec(label),
            "Brisbane-Event": ("pooled", BRISBANE, f"{TAG}_{label}", "native"),
            "NSAVP R0": ("pooled", NSAVP, f"{TAG}_{label}", "native")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--latex", action="store_true", help="also print the tabular body")
    cli = ap.parse_args()

    missing = []
    table = {}          # dataset -> {label: (recall, shape)}
    reference = {}      # dataset -> {"shipped"/"+rerank" per label: recall}
    for dataset in ROWS:
        specs = DATASETS[dataset][1]
        shapes, cells, refs = {}, {}, {}
        rerank, rerank_shape, note = read_cell(specs["+rerank"])
        if rerank is None:
            missing.append(f"{dataset} +rerank: {note}")
        elif rerank_shape:
            shapes["+rerank"] = rerank_shape
        for label, (shipped_column, _, _) in MODELS.items():
            value, shape, note = read_cell(specs_for(label)[dataset])
            cells[label] = value
            if value is None:
                missing.append(f"{dataset} {label}: {note}")
            elif shape:
                shapes[label] = shape
            ship, ship_shape, ship_note = read_cell(specs[shipped_column])
            refs[label] = ship
            if ship is None:
                missing.append(f"{dataset} {shipped_column} (shipped): {ship_note}")
            elif ship_shape:
                shapes[f"shipped:{shipped_column}"] = ship_shape
        # A delta is only a delta if both sides scored the same benchmark.
        distinct = set(shapes.values())
        if len(distinct) > 1:
            raise SystemExit(f"{dataset}: cells disagree on the benchmark size {shapes}")
        table[dataset] = (next(iter(distinct), None), cells)
        reference[dataset] = (refs, rerank)

    print("\nreal-event-selected checkpoints (best_real), native descriptor, 322x322, no PCA")
    for dataset in ROWS:
        shape, cells = table[dataset]
        refs, rerank = reference[dataset]
        size = f"{shape[0]:,} x {shape[1]:,}" if shape else "--"
        print(f"\n{dataset}   {size}   (+rerank baseline "
              f"{'--' if rerank is None else f'{rerank[0]:.3f} / {rerank[1]:.3f}'})")
        print(f"  {'model':<17s}{'step':>6s}  {'R@1':>7s} {'R@10':>7s}   "
              f"{'d R@1':>7s} {'d R@10':>7s}  vs shipped     "
              f"{'d R@1':>7s} {'d R@10':>7s}  vs +rerank")
        for label, (shipped_column, arch, _) in MODELS.items():
            value, ship = cells[label], refs[label]
            step = label.rsplit("_s", 1)[-1]
            if value is None:
                print(f"  {label:<17s}{step:>6s}       --      --")
                continue
            def delta(other):
                if other is None:
                    return f"{'--':>7s} {'--':>7s}  "
                return f"{value[0] - other[0]:>+7.3f} {value[1] - other[1]:>+7.3f}  "
            ship_label = f"{shipped_column}"
            print(f"  {label:<17s}{step:>6s}  {value[0]:>7.3f} {value[1]:>7.3f}   "
                  f"{delta(ship)}{ship_label:<14s} {delta(rerank)}")

    if missing:
        print("\nmissing:")
        for item in missing:
            print(f"  {item}")
        return

    if cli.latex:
        print("\n% --- real-event-selected rows, R@1 & R@10 per dataset ---")
        print(f"% columns: {' & '.join(ROWS)}")
        for label, (_, arch, run) in MODELS.items():
            body = " & ".join(
                f"{table[d][1][label][0]:.3f} & {table[d][1][label][1]:.3f}" for d in ROWS)
            print(f"            {run} & {body} \\\\")


if __name__ == "__main__":
    main()
