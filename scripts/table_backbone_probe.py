"""tab:backbone — whose patch features read an event frame, with every head removed.

    pixi run python3 scripts/table_backbone_probe.py
    pixi run python3 scripts/table_backbone_probe.py --latex --stride 1

Reads `backbone_probe/results_stride<N>.json` (written by `scripts/backbone_probe.py`) and
emits the paper table. The argument it exists to support: an RGB retrieval model's advantage
over a generic self-supervised backbone does not survive the move to event frames, while
training the same architecture on event frames does transfer — so what the event regime buys
is in the features, not in the aggregator.

Every row is the SAME ViT-B/14 trunk (768-d, 529 tokens at 322) with its aggregation head
removed and replaced by a parameter-free L2-normalised mean over patch tokens. Width is held
fixed deliberately: the ViT-S rows in the underlying JSON are not comparable to these and are
excluded unless --all is given. Each RGB backbone is run under both ImageNet and
accumulate-matched normalisation and keeps its better arm, so no row is handicapped by an
input-statistics mismatch it never chose (src/methods.py::ACCUMULATE_MEAN).

Protocol: single reference traverse (sunset1 gallery), one query traverse per column, 25 m,
native cosine, no whitening — `pairwise_sunset_ref.py`'s protocol, not the pooled one.
"""

import argparse
import json
import os

ROOT = "/media/adam/vprdatasets/megaevent/backbone_probe"

# key -> (pretty name, pretraining, VPR training). Order is the argument the table makes.
ROWS = [
    ("dinov2-b", "DINOv2 ViT-B", "self-sup.\\ RGB", "---"),
    ("salad",    "DINOv2-SALAD", "self-sup.\\ RGB", "RGB (GSV-Cities)"),
    ("megaloc",  "MegaLoc",      "self-sup.\\ RGB", "RGB (large-scale)"),
    ("gept-b",   "GEPT ViT-B",   "event",           "---"),
    ("v8-b",     "Ours",         "event",           "event"),
]
VIT_S = {"dinov2-s", "gept-s"}


def best_arm(blob):
    """The better normalisation arm, by mean R@1 — never a handicapped row."""
    best = None
    for stats, cell in blob["arms"].items():
        queries = [q for q in cell["meanpatch"]]
        mean = sum(cell["meanpatch"][q]["1"] for q in queries) / len(queries)
        if best is None or mean > best[1]:
            best = (stats, mean, cell)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="include the ViT-S rows, which are NOT width-matched to the rest")
    cli = ap.parse_args()

    path = os.path.join(ROOT, f"results_stride{cli.stride}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no probe results at {path} — run scripts/backbone_probe.py "
                         f"--stride {cli.stride}")
    with open(path) as handle:
        blob = json.load(handle)
    proto, banks = blob["protocol"], blob["backbones"]
    queries = proto["queries"]

    rows = [r for r in ROWS if r[0] in banks]
    if cli.all:
        rows += [(k, k, "---", "---") for k in banks if k in VIT_S]
    missing = [r[0] for r in ROWS if r[0] not in banks]

    if proto["stride"] != 1:
        print(f"% WARNING stride {proto['stride']}: the gallery is subsampled, so these are "
              f"NOT protocol numbers — arms are comparable to each other only.")
    print(f"% single reference {proto['reference']} "
          f"({proto['n_database']} db) -> " + ", ".join(
              f"{q} ({proto['n_queries'][q]} q)" for q in queries)
          + f"; 25 m, native, {proto['representation']}, {proto['pooling']}")
    if missing:
        print(f"% missing from this run: {', '.join(missing)}")

    table = []
    for key, pretty, pre, vpr in rows:
        stats, mean, cell = best_arm(banks[key])
        per = [cell["meanpatch"][q]["1"] for q in queries]
        table.append((pretty, pre, vpr, stats, per, mean))

    if not cli.latex:
        print(f"\n{'backbone':14s}{'pretrain':16s}{'VPR train':20s}{'norm':11s}"
              + "".join(f"{q:>10s}" for q in queries) + f"{'mean':>9s}")
        for pretty, pre, vpr, stats, per, mean in table:
            plain = pre.replace("\\ ", " ")
            print(f"{pretty:14s}{plain:16s}{vpr:20s}{stats:11s}"
                  + "".join(f"{v:>10.3f}" for v in per) + f"{mean:>9.3f}")
        base = next((m for p, _, v, _, _, m in table if v == "---" and "DINOv2 ViT" in p), None)
        if base:
            print()
            for pretty, _, _, _, _, mean in table:
                print(f"  {pretty:14s} x{mean / base:.2f} vs the generic backbone")
        return

    print("\\begin{tabular}{lllrrrr}")
    print("\\toprule")
    print("Backbone & Pre-training & VPR training & "
          + " & ".join(q.capitalize() for q in queries) + " & Mean \\\\")
    print("\\midrule")
    for i, (pretty, pre, vpr, _stats, per, mean) in enumerate(table):
        if i == len(table) - 1:
            print("\\midrule")
        cells = " & ".join(f"{v:.3f}" for v in per)
        name = f"\\textbf{{{pretty}}}" if pretty == "Ours" else pretty
        end = f"\\textbf{{{mean:.3f}}}" if pretty == "Ours" else f"{mean:.3f}"
        print(f"{name} & {pre} & {vpr} & {cells} & {end} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
