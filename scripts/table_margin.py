"""The margin to explain: v8 against every control, single reference traverse.

    pixi run python3 scripts/table_margin.py
    pixi run python3 scripts/table_margin.py --dataset nsavp

Written for the mechanism investigation, whose first question is not "how does the edge
work" but "how big is the edge". It reads the reporting protocol — **one reference traverse
per cell** (`scripts/pairwise_sunset_ref.py`, `pairwise_sunset_ref_*.json`), never the pooled
protocol. A pooled gallery holds four or five traverses of the same route, so a query needs
only ONE of them to be an easy appearance match; the cell then measures whether an easy
partner existed rather than whether the model handled the condition change.

Two things this fixes about how the margin has been read:

* **Roster.** The v8 verdict compared against MegaLoc. MegaLoc is not the strongest control
  on event frames — on these cells mixvpr, qaa and boq all beat it somewhere. "Best control"
  here means best over the whole roster, which is the number a reviewer computes.
* **Scale.** R@1 is bounded, so an absolute margin inflates on hard cells. The error ratio
  (`err = 1 - R@1`, ours over theirs) is reported beside it: 0.63 means we remove a third of
  the remaining error, and it is comparable across cells of different difficulty.

Native cosine, 25 m, no whitening — the reporting convention these cells were produced under.
"""

import argparse
import json
import os

ROOT = "/media/adam/vprdatasets/megaevent"
LEDGER = os.path.join(ROOT, "v8_bench", "pairwise_sunset_ref_{}.json")
OURS_PREFIX = "megaevent"


def cells(dataset):
    with open(LEDGER.format(dataset)) as handle:
        blob = json.load(handle)
    out = []
    for name, cell in blob["results"].items():
        rows = {}
        for method, entry in cell["methods"].items():
            r1 = entry.get("recall", {}).get("native", {}).get("1")
            if r1 is not None:
                rows[method] = {"r1": r1, "n_queries": entry.get("n_queries"),
                                "n_database": entry.get("n_database")}
        out.append((name, cell.get("database"), cell.get("query"), rows))
    return blob, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", "-d", default=None,
                    choices=["brisbane_event", "nsavp"], help="default: both")
    ap.add_argument("--ours", default="megaevent v8-B",
                    help="which of our arms is the headline row")
    ap.add_argument("--full", action="store_true", help="print every method, not the top 5")
    cli = ap.parse_args()

    for dataset in ([cli.dataset] if cli.dataset else ["brisbane_event", "nsavp"]):
        blob, table = cells(dataset)
        print(f"\n=== {dataset} — {blob.get('protocol')} ===")
        summary = []
        for name, db, query, rows in table:
            if cli.ours not in rows:
                print(f"\n  {name}: {cli.ours} absent — skipped")
                continue
            ours = rows[cli.ours]["r1"]
            ranked = sorted(((v["r1"], m) for m, v in rows.items()
                             if not m.startswith(OURS_PREFIX)), reverse=True)
            best_r1, best_name = ranked[0]
            ratio = (1 - ours) / (1 - best_r1) if best_r1 < 1 else float("nan")
            summary.append((name, ours, best_r1, best_name, ratio))
            n = rows[cli.ours]
            print(f"\n  {name}   db={db} query={query}   "
                  f"{n['n_database']} x {n['n_queries']}")
            shown = ranked if cli.full else ranked[:5]
            print(f"    {'*' + cli.ours:26s} {ours:.4f}")
            for r1, m in shown:
                print(f"     {m:26s} {r1:.4f}")

        if summary:
            print(f"\n  {'cell':22s}{'ours':>8s}{'best ctrl':>10s} {'which':12s}"
                  f"{'d R@1':>8s}{'err ratio':>11s}")
            for name, ours, best, which, ratio in summary:
                print(f"  {name:22s}{ours:>8.3f}{best:>10.3f} {which:12s}"
                      f"{ours - best:>+8.3f}{ratio:>11.2f}")
            wins = sum(1 for _, o, b, _, _ in summary if o > b)
            mean_ratio = sum(s[4] for s in summary) / len(summary)
            print(f"\n  {cli.ours} leads {wins}/{len(summary)} cells; "
                  f"mean error ratio {mean_ratio:.2f} "
                  f"({(1 - mean_ratio) * 100:.0f}% of the remaining error removed)")


if __name__ == "__main__":
    main()
