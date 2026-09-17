"""The Springfield baseline table: every method, one protocol, R@1 and R@10 at 25 m.

    pixi run python3 scripts/springfield_table.py

Reads whatever ``results_*.json`` the suite has produced and prints one table. Rows whose
protocol does not match the reference exactly (same gallery size, same query count) are
printed as VOID rather than compared — a number measured on a different gallery is not a
number about the same benchmark.

The reversal split is recomputed here from the diagnostic ``orient`` bundle and each
method's own top-k dump, so the with- and without-reversal cells come from one scoring pass
rather than two, and no method can silently be scored on a different query set from another.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

SWEEPS = ("day", "dawn", "night")
KS = (1, 5, 10, 20)


def short_label(r):
    """A name that fits a column: the method's own, or the checkpoint stem for megaevent.

    A method's native event representation stays implicit; anything else (an RGB model
    on reconstructed video, a re-normalised control) is a different measurement of the
    same model and must not dedupe against it, so the representation joins the label.
    """
    m = r.get("method", {}).get("model")
    if m:
        rep = r.get("representation") or ""
        if rep and rep not in ("countmask", "mcts", "count_triplet", "onoff",
                               "polarity_counts"):
            return f"{m} {rep}"
        return m
    tag = r.get("tag", "")
    step = r.get("step")
    stem = tag.split("_s")[0] if "_s" in tag else tag
    return f"megaevent {stem}" + (f" s{step}" if step else "")


def load_rows(out_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "results_*.json"))):
        try:
            with open(path) as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        if "recall_micro" not in r or "protocol" not in r:
            continue
        rows.append((os.path.basename(path), r))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="output/springfield_full")
    ap.add_argument("--all", action="store_true",
                    help="also print rows measured on a different protocol, marked VOID")
    ap.add_argument("--reference", default=None,
                    help="results json whose protocol every other row must match; "
                         "defaults to the largest gallery found")
    cli = ap.parse_args()

    rows = load_rows(cli.out_dir)
    if not rows:
        raise SystemExit(f"no results_*.json with a protocol block under {cli.out_dir}")

    ref = max(rows, key=lambda kv: kv[1]["n_database"])[1]
    if cli.reference:
        with open(cli.reference) as f:
            ref = json.load(f)
    n_db_ref = ref["n_database"]
    n_q_ref = sum(ref["scorable"].values())
    print(f"protocol: {n_db_ref} gallery rows, {n_q_ref} query slices, "
          f"{ref['threshold_m']:g} m, chance R@1 {ref['protocol']['chance_r1']:.4f}, "
          f"{ref['protocol']['positives_per_query_mean']:.0f} positives/query")
    print()
    hdr = f"{'method':22s} {'gallery':>8s} {'queries':>8s}"
    for s in ("overall",) + SWEEPS:
        hdr += f" {s + ' R@1':>12s} {s + ' R@10':>13s}"
    print(hdr)
    print("-" * len(hdr))

    seen = {}
    for name, r in sorted(rows, key=lambda kv: -kv[1]["n_database"]):
        label = short_label(r)
        n_db = r["n_database"]
        n_q = sum(r["scorable"].values())
        key = (label, n_db, n_q)
        if key in seen:
            continue
        seen[key] = True
        void = (n_db != n_db_ref or n_q != n_q_ref)
        if void and not cli.all:
            continue
        line = f"{label:22s} {n_db:8d} {n_q:8d}"
        for s in ("overall",) + SWEEPS:
            m = r["recall_micro"].get(s)
            if not m:
                line += f" {'-':>12s} {'-':>13s}"
                continue
            line += f" {m['1']:12.4f} {m['10']:13.4f}"
        print(line + ("   VOID: protocol differs" if void else ""))
    print()
    print("VOID rows were measured on a different gallery or query set and are not "
          "comparable; rerun them under the reference protocol before quoting.")


if __name__ == "__main__":
    main()
