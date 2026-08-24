"""The corrected recorded-set ledger: native AND best-space columns, LOSO NYC.

    pixi run python3 scripts/table_bestspace.py [--latex]

`tab:native`'s "no PCA anywhere" convention is not neutral: whitening moves every RGB
control by +0.06..+0.14 R@1 (MixVPR's whitened Brisbane 0.872 beats our best *native*
0.853), so a native-only table overstates the megaevent lead. This assembler prints, for
the three recorded datasets, each method's native cell, its whitened (pca4096p0.5 — or
Event-GeM's own shipped pca128p0.5+rerank) cell, and the max of the two; NYC appears
under leave-one-session-out (the random-split protocol is near-duplicate retrieval — see
scripts/nycevent_all_sessions.py) with the void random-split number alongside for scale.

Reads only cached result JSONs; every cell can be traced to its file with --sources.
"""

import argparse
import json
import os

R = "/media/adam/vprdatasets/megaevent"


def cell(path, *keys):
    """recall R@1 out of a nested results JSON, or None."""
    try:
        with open(path) as h:
            node = json.load(h)
        for k in keys:
            node = node[k]
        return float(node)
    except (FileNotFoundError, KeyError, TypeError):
        return None


# (label, dataset -> (json, result tag, native key, whitened key))
def rows():
    out = []
    ships = [
        ("ours B-MLoc  s3500", {"brisbane": ("brisbane_v5/noise_projmegaloc_no_sunset2_pca.json", "r322ba50"),
                                "nsavp": ("nsavp_pooled/results_route0_v5_reproduce.json", "r322ba50")}),
        ("ours B-SALAD s8000", {"brisbane": ("brisbane_ship/ship_no_sunset2_pca.json", "r322ba50_b_salad_s8000"),
                                "nsavp": ("nsavp_pooled/results_route0_ship_pca.json", "r322ba50_b_salad_s8000")}),
        ("ours S-MLoc  s3500", {"brisbane": ("brisbane_ship/ship_no_sunset2_pca.json", "r322ba50_s_mloc_s3500"),
                                "nsavp": ("nsavp_pooled/results_route0_ship_pca.json", "r322ba50_s_mloc_s3500")}),
        ("ours S-SALAD s2000", {"brisbane": ("brisbane_ship/ship_no_sunset2_pca.json", "r322ba50_s_salad_s2000"),
                                "nsavp": ("nsavp_pooled/results_route0_ship_pca.json", "r322ba50_s_salad_s2000")}),
    ]
    for label, spec in ships:
        out.append((label, {ds: (f, tag, "native", "pca4096p0.5") for ds, (f, tag) in spec.items()}))
    controls = [("MegaLoc", "megaloc", "r322ba50"), ("SALAD", "salad", "r322ba50"),
                ("MixVPR", "mixvpr", "r320ba50"), ("CricaVPR", "cricavpr", "r224ba50")]
    for label, m, tag in controls:
        out.append((label, {"brisbane": (f"{m}_pooled/brisbane_event.json", tag, "native", "pca4096p0.5"),
                            "nsavp": (f"{m}_pooled/nsavp.json", tag, "native", "pca4096p0.5")}))
    out.append(("Event-GeM +rerank",
                {"brisbane": ("eventgem_pooled/brisbane_event.json", "r240ba50",
                              "native+rerank", "pca128p0.5+rerank"),
                 "nsavp": ("eventgem_pooled/nsavp.json", "r240ba50",
                           "native+rerank", "pca128p0.5+rerank")}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--sources", action="store_true", help="print each cell's source file")
    cli = ap.parse_args()

    table = []
    for label, spec in rows():
        row = {"label": label}
        for ds, (f, tag, nkey, wkey) in spec.items():
            n = cell(os.path.join(R, f), "results", tag, "recall", nkey, "1")
            w = cell(os.path.join(R, f), "results", tag, "recall", wkey, "1")
            row[ds] = (n, w)
            if cli.sources:
                print(f"  [{label} / {ds}] {f} :: {tag}")
        table.append(row)

    loso = {}
    try:
        with open(f"{R}/evaluations/nycevent/loso.json") as h:
            lo = json.load(h)
        loso = {name: (m["loso_mean_r1"], m["reported_random_split_r1"])
                for name, m in lo["models"].items()}
    except FileNotFoundError:
        pass

    def fmt(v):
        return f"{v:.3f}" if v is not None else "  —  "

    sep = " & " if cli.latex else "  "
    end = r" \\" if cli.latex else ""
    print(f"\n{'method':22s}{sep}{'Bris nat':>8s}{sep}{'Bris wht':>8s}{sep}{'best':>6s}"
          f"{sep}{'NSAVP nat':>9s}{sep}{'NSAVP wht':>9s}{sep}{'best':>6s}{end}")
    for row in table:
        cells = []
        for ds in ("brisbane", "nsavp"):
            n, w = row[ds]
            best = max(v for v in (n, w) if v is not None) if (n or w) else None
            cells += [fmt(n), fmt(w), fmt(best)]
        print(f"{row['label']:22s}{sep}{cells[0]:>8s}{sep}{cells[1]:>8s}{sep}{cells[2]:>6s}"
              f"{sep}{cells[3]:>9s}{sep}{cells[4]:>9s}{sep}{cells[5]:>6s}{end}")

    if loso:
        print(f"\nNYC-Event, leave-one-session-out (native; the random-split protocol is "
              f"near-duplicate retrieval and shown only for scale):")
        print(f"{'model':31s}{'LOSO mean':>10s}{'random-split':>14s}")
        for name, (m, rep) in loso.items():
            print(f"{name:31s}{m:>10.3f}{rep:>14.3f}")


if __name__ == "__main__":
    main()
