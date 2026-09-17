"""tab:recall_real — the real-event recall table, every method in one place.

    pixi run python3 scripts/table_recall_real.py            # LaTeX body
    pixi run python3 scripts/table_recall_real.py --plain    # readable grid

Twelve cells: Brisbane sunset1 -> {daytime, morning, sunrise}, NSAVP R0_FS0 -> R0_FA0 and
R0_RS0 -> R0_RA0, and NYC-Event's published random 10% split. The first eleven come from
``scripts/pairwise_sunset_ref.py``'s single-reference ledgers, the last from the per-method
``results_*.json`` main.py writes under ``evaluations/nycevent``.

The RGB controls (MegaLoc, SALAD, MixVPR, CricaVPR, BoQ, QAA, SuperVLAD) read the same
accumulate render the v8 models train on, at each model's own published resolution; the event
methods keep their published configuration. Every cell is 25 m Euclidean, native descriptors,
no PCA, single best match. Event-GeM is its 0.1.0 release *with* the re-rank, which is the
number its paper reports.

Brisbane/NSAVP cells are single-traverse galleries and NYC's is the published split, so a row
is comparable across methods within a column and never across columns.
"""

import argparse
import json

ROOT = "/media/adam/vprdatasets/megaevent"
PAIRWISE = ROOT + "/v8_bench/pairwise_sunset_ref_{ds}.json"
NYC = ROOT + "/evaluations/nycevent"

# column key -> (source dataset, cell name)
CELLS = [
    ("brisbane_event", "sunset1->daytime"),
    ("brisbane_event", "sunset1->morning"),
    ("brisbane_event", "sunset1->sunrise"),
    ("nsavp", "R0_FS0->R0_FA0"),
    ("nsavp", "R0_RS0->R0_RA0"),
    ("nycevent", "random10"),
]

# (LaTeX label, pairwise roster name, NYC results file)
#
# SpikeVPR is cross-dataset by construction — the NYC cell is its NSAVP-trained model, the
# same convention the Brisbane and NSAVP columns use (each column's model is trained on a
# different dataset than it is evaluated on).
ROWS = [
    ("MegaLoc~\\cite{Berton2025}",   "MegaLoc",         "results_megaloc_accumulate_r322.json"),
    ("SALAD~\\cite{Izquierdo2024}",  "SALAD",           "results_salad_accumulate_r322.json"),
    ("MixVPR~\\cite{Alibey2023}",    "MixVPR",          "results_mixvpr_accumulate_r320.json"),
    ("CricaVPR~\\cite{Lu2024}",      "CricaVPR",        "results_cricavpr_accumulate_r224.json"),
    ("BoQ~\\cite{Alibey2024}",       "BoQ",             "results_boq_accumulate_r322.json"),
    ("SuperVLAD~\\cite{Lu2024b}",    "SuperVLAD",       "results_supervlad_accumulate_r322.json"),
    ("QAA~\\cite{Lu2025}",           "QAA",             "results_qaa_accumulate_r322.json"),
    (None, None, None),  # rule
    ("EventVLAD~\\cite{Lee2021}",    "EventVLAD",
     "../../nycevent_eval_eventvlad/nycevent/results_eventvlad.json"),
    ("EventGeM~\\cite{Hines2026}",   "Event-GeM 0.1.0 +rerank", "EG010_RERANK"),
    ("SpikeVPR~\\cite{Keime2026}",   "SpikeVPR",        "results_spikevpr_r34_nsavp_0e729ac109.json"),
    (None, None, None),  # rule
    ("Ours (ViT/S)",                 "megaevent v8-S",
     "nycevent/results_s_v8_accum_s500_s500_9aaa0c325c_accumulate_r322.json"),
    ("Ours (ViT/B)",                 "megaevent v8-B",
     "nycevent/results_b_v8_accum_s750_s750_3f4eb54780_accumulate_r322.json"),
]


def load():
    """{(dataset, cell): {method: {k: recall}}} over every column."""
    out = {}
    for ds in ("brisbane_event", "nsavp"):
        doc = json.load(open(PAIRWISE.format(ds=ds)))
        for cell, body in doc["results"].items():
            out[(ds, cell)] = {name: list(m["recall"].values())[0]
                               for name, m in body["methods"].items()}
    nyc = {}
    for _, name, path in ROWS:
        if name is None:
            continue
        if path == "EG010_RERANK":
            nyc[name] = json.load(open(f"{ROOT}/v8_bench/eg010_rerank_nyc.json")
                                  )["random_split"]["reranked"]
        else:
            nyc[name] = json.load(open(f"{NYC}/{path}"))["recall"]["native"]
    out[("nycevent", "random10")] = nyc
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plain", action="store_true", help="readable grid instead of LaTeX")
    ap.add_argument("--dp", type=int, default=2, help="decimal places in the printed cells")
    cli = ap.parse_args()

    data = load()
    labels = [r[0] for r in ROWS if r[0]]
    names = [r[1] for r in ROWS if r[1]]

    # Rank on full precision, print rounded. A tie at the printed precision still gets the
    # marks its underlying values earn — the ranking is never re-derived from the string.
    marks = {}
    for cell in CELLS:
        for k in ("1", "10"):
            order = sorted(names, key=lambda n: -data[cell][n][k])
            marks[(cell, k, order[0])] = "best"
            marks[(cell, k, order[1])] = "second"

    def fmt(cell, name, k, latex):
        val = f"{data[cell][name][k]:.{cli.dp}f}"
        mark = marks.get((cell, k, name))
        if not latex or mark is None:
            return val
        return f"\\textbf{{{val}}}" if mark == "best" else f"\\underline{{{val}}}"

    if cli.plain:
        heads = ["s1:day", "s1:morn", "s1:sunr", "FS0:FA0", "RS0:RA0", "NYC 10%"]
        print(f"{'method':<22s}" + "".join(f"{h:>18s}" for h in heads))
        print(f"{'':<22s}" + "".join(f"{'R@1':>9s}{'R@10':>9s}" for _ in heads))
        for label, name, _ in ROWS:
            if label is None:
                print("  " + "-" * 128)
                continue
            row = "".join(f"{fmt(c, name, '1', False):>9s}{fmt(c, name, '10', False):>9s}"
                          for c in CELLS)
            print(f"{label.split('~')[0]:<22s}{row}")
        return

    for label, name, _ in ROWS:
        if label is None:
            print("            \\midrule\n")
            continue
        cells = " \n            ".join(
            f"& {fmt(c, name, '1', True)} & {fmt(c, name, '10', True)}" for c in CELLS)
        print(f"            {label} \n            {cells} \\\\\n")


if __name__ == "__main__":
    main()
