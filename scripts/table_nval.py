"""Assemble `tab:nval` — the effect of the number of training places N on ViT-B recall.

    pixi run python3 scripts/table_nval.py                 # markdown + LaTeX
    pixi run python3 scripts/table_nval.py --brisbane pooled

N is the trainer's ``--P``: places per sub-batch, per stream. The multi-similarity loss is
computed *per stream*, so an anchor only ever sees ``P-1`` other places per call regardless of
global batch size — which is why this lever is about the loss, not the batch.

The four runs differ in exactly one token. All are ViT-B + SALAD, ``--blocks 12 --steps 10000
--lr 5e-5 --encoder-lr-mult 0.1``, seed 0, annealed to the end of their own cosine, so nothing
here was selected on any metric.

**N=16 does not exist.** The only P=16 ViT-B ever trained was a pre-countmask v1 run whose
checkpoint is gone; it survives only as a number in `docs/artifact_v3_packet.html`. Training it
would be a fresh ~9 h run. The table reports N in {32, 48, 64}.

Two Brisbane protocols, and they are not interchangeable
--------------------------------------------------------
``--brisbane pairwise`` (default) is `scripts/brisbane_resolution.py`'s: `sunset2` as the
reference, one query condition at a time, Event-LAB's +-70 m along-track band, mean over
`{sunset1, morning, daytime, sunrise}`. **Every P value has this**, which is the only reason it
is the default.

``--brisbane pooled`` is `scripts/brisbane_pooled.py`'s and is what the paper's main Brisbane
column reports: `sunset1` as the query, `{daytime, morning, night, sunrise}` pooled into one
database, 25 m Euclidean radius, `sunset2` excluded because it is the same route under
sunset1's own illumination. The two protocols differ by ~14 points at the same checkpoint
(P64: .698 pairwise vs .823 pooled), so a `tab:nval` built on the pairwise numbers cannot be
read against the main table.

Only P64 has been scored pooled. `--brisbane pooled` prints what exists and names the missing
banks rather than silently falling back.

Both protocols are read at the shipping descriptor space, PCA (4096, 0.5), at 322^2.
"""

import argparse
import json
import os
import statistics

ROOT = "/media/adam/vprdatasets/megaevent"
TOKYO = f"{ROOT}/tokyo_v4_stage/A_out/results_r322.json"
BRISBANE_PAIRWISE = f"{ROOT}/brisbane_v4/results.json"
# brisbane_pooled.py writes one file per database composition; the paper uses the sunset2-free
# one. A multi-checkpoint run keys each result `<tag>_<label>`; the single-checkpoint shipping
# run wrote the bare tag, which is why final_no_sunset2.json is not the file read here.
BRISBANE_POOLED = f"{ROOT}/brisbane_pooled/psweep_no_sunset2.json"

SPACE = "pca4096p0.5"
# N -> (tokyo/brisbane-pairwise result key, pooled result key)
ROWS = [
    (32, "P32_s10000", "r322ba50_P32_s10000"),
    (48, "P48_s10000", "r322ba50_P48_s10000"),
    (64, "P64_s10000", "r322ba50_P64_s10000"),
]
# The same config as N=48 with --seed 1. Not a row: it is the yardstick the rows are measured
# against, since the whole N=32->64 Brisbane spread is under 2 points.
SEED_REPEAT = (48, "P48s1_s10000")


def tokyo(results, key):
    r = results[key][SPACE]
    return r["1"], r["10"]


def brisbane_pairwise(doc, key):
    """Mean R@1/R@10 over the four query conditions, stride 1 — the sweep's own protocol."""
    per = doc["results"][key]["r322"]
    conditions = doc["queries"]
    return (statistics.mean(per[q][SPACE]["1"] for q in conditions),
            statistics.mean(per[q][SPACE]["10"] for q in conditions))


def brisbane_pooled(doc, key):
    r = doc["results"][key]["recall"][SPACE]
    return r["1"], r["10"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brisbane", choices=["pairwise", "pooled"], default="pairwise")
    cli = ap.parse_args()

    tok = json.load(open(TOKYO))
    print(f"Tokyo 24/7   {TOKYO}")
    print(f"             {tok['resolution']}^2, {tok['n_database']:,} database x "
          f"{tok['n_queries']} queries, {tok['threshold_m']:.0f} m, space {SPACE}")

    if cli.brisbane == "pairwise":
        bris = json.load(open(BRISBANE_PAIRWISE))
        print(f"Bris-Event   {BRISBANE_PAIRWISE}")
        print(f"             pairwise +-70 m band, ref {bris['ref']}, mean over "
              f"{', '.join(bris['queries'])}, 322^2 stride 1")
    elif not os.path.exists(BRISBANE_POOLED):
        raise SystemExit(
            f"no pooled sweep at {BRISBANE_POOLED}.\n"
            f"The shipping run only ever scored P64 pooled, and it wrote the bare tag rather "
            f"than a per-checkpoint key, so there is nothing to read for P32/P48. Build all "
            f"three in one pass (P64's banks are already cached, so only P32/P48 render):\n\n"
            f"  pixi run python -u scripts/brisbane_pooled.py \\\n"
            f"    --query sunset1 --database daytime morning night sunrise \\\n"
            f"    --resolutions 322 --arms on --out-json psweep_no_sunset2.json \\\n"
            + "".join(f"    --ckpt {k}={ROOT}/runs/b_full_P{n}_v4_vpr/step10000.pt \\\n"
                      for n, k, _ in ROWS)
            + f"\nOr use --brisbane pairwise, which every N already has.")
    else:
        bris = json.load(open(BRISBANE_POOLED))
        print(f"Bris-Event   {BRISBANE_POOLED}")
        print(f"             pooled {bris['threshold_m']:.0f} m, query {bris['query']}, "
              f"database {', '.join(bris['database'])}")

    rows = []
    for n, key, pooled_key in ROWS:
        t1, t10 = tokyo(tok["results"], key)
        if cli.brisbane == "pairwise":
            b1, b10 = brisbane_pairwise(bris, key)
        elif pooled_key in bris["results"]:
            b1, b10 = brisbane_pooled(bris, pooled_key)
        else:
            rows.append((n, t1, t10, None, None))
            continue
        rows.append((n, t1, t10, b1, b10))

    def cell(v):
        return "--" if v is None else f"{v:.3f}"

    print(f"\n{'N':>4s} {'Tokyo R@1':>10s} {'Tokyo R@10':>11s} "
          f"{'Bris R@1':>9s} {'Bris R@10':>10s}")
    for n, t1, t10, b1, b10 in rows:
        print(f"{n:>4d} {t1:>10.3f} {t10:>11.3f} {cell(b1):>9s} {cell(b10):>10s}")

    missing = [n for n, _, _, b1, _ in rows if b1 is None]
    if missing:
        print(f"\n  N={', '.join(str(m) for m in missing)} have no pooled score. Produce "
              f"them with one multi-checkpoint pass — load_all shares the frames, so the "
              f"extra checkpoints are nearly free:")
        print("    scripts/brisbane_pooled.py --query sunset1 --database daytime morning "
              "night sunrise \\\n      --resolutions 322 --arms on --out-json "
              "psweep_no_sunset2.json \\\n      --ckpt "
              + " ".join(f"{k}=runs/b_full_P{n}_v4_vpr/step10000.pt"
                         for n, k, _ in ROWS if n in missing))

    # Seed variance. Quoted because the N=32->64 spread is small enough that a reader is
    # entitled to ask whether it is one seed's luck.
    s_n, s_key = SEED_REPEAT
    st1, st10 = tokyo(tok["results"], s_key)
    base = next(r for r in rows if r[0] == s_n)
    print(f"\n  seed variance at N={s_n}: Tokyo R@1 {base[1]:.3f} (seed 0) vs {st1:.3f} "
          f"(seed 1), delta {abs(base[1] - st1):.3f}")
    if cli.brisbane == "pairwise":
        sb1, sb10 = brisbane_pairwise(bris, s_key)
        print(f"                          Bris  R@1 {base[3]:.3f} (seed 0) vs {sb1:.3f} "
              f"(seed 1), delta {abs(base[3] - sb1):.3f}")
        spread = max(r[3] for r in rows) - min(r[3] for r in rows)
        print(f"  N=32->64 Brisbane R@1 spread {spread:.3f} — compare against the seed delta "
              f"above before claiming the trend is resolved.")

    print("\n% --- tab:nval body ---")
    for n, t1, t10, b1, b10 in rows:
        print(f"            {n} & {t1:.3f} & {t10:.3f} & {cell(b1)} & {cell(b10)} \\\\")


if __name__ == "__main__":
    main()
