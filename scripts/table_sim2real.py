"""Assemble `tab:sim2real` — real event streams vs I2E simulation of the co-recorded frames.

    pixi run python3 scripts/table_sim2real.py
    pixi run python3 scripts/table_sim2real.py --masked      # the vignette-controlled arms

Brisbane-Event-VPR is the only dataset here that can answer the question, because the DAVIS346
recorded intensity frames and events through one sensor on one clock. Running I2E over the APS
frames therefore yields synthetic events of *exactly* the scenes the real stream saw, index
aligned slice for slice.

Protocol is the paper's pooled one — query `sunset1`, 25 m Euclidean radius, 322^2 — so the
`real` column is the main Brisbane table. Two deliberate departures, both applied to **both**
arms so the comparison is unaffected:

* **`night` is excluded from the database.** Its APS ran at 1.73 Hz against the 20 Hz slice
  grid: 1,191 real frames for 13,773 slices, 91% with no frame within half a slice. Its
  synthetic arm would be micro-saccades over stale duplicates. Every other traverse is
  0.2-0.4% out of tolerance. Costs the real arm 0.0017 R@1.
* **APS alignment.** The handful of slices with no frame within half a slice are dropped from
  the real arm too. Masking only the synthetic arm would score a real frame against a
  duplicated one and report the difference as a domain gap.

**What this table is for, and what would falsify the claim.** The objection it answers is that
MegaEvent was fine-tuned on I2E-generated events, so benchmarking it on I2E-generated Tokyo
24/7 gives it a domain advantage the baselines do not have. MegaEvent's own real->i2e drop is
therefore *not* the interesting number — it is expected to fare relatively better on the
synthetic arm. What matters is the **margin between MegaEvent and each baseline, across arms**.
A margin that grows on the synthetic arm is the objection confirmed, not refuted, however each
method's absolute number moves. The script prints those margins because the per-method deltas
alone invite exactly the wrong reading.

`--masked` reads the vignette-controlled arms. I2E thresholds ``Δlog(luma + 1e-3)``, whose
sensitivity goes as ``1/(luma + 1e-3)``, so 57-69% of the synthetic stream lands in the DAVIS's
near-black bonnet region against 8% of the real stream. A keypoint-and-homography method has
more to lose from that than a global descriptor does, so a margin shift on the unmasked arms
cannot be attributed to domain familiarity until the masked pair says the same thing.
"""

import argparse
import json
import os

OUT = "/media/adam/vprdatasets/megaevent/sim2real"

# method -> (row label, json stem per arm, result-key suffix per arm, descriptor space)
# The stems differ because each pooled script names its own output; the key suffixes differ
# because a synthetic arm carries the source in its tag instead of the filter window.
#
# Each method is reported in the space it actually ships in, not in one space imposed on all
# of them. MegaEvent whitens (4096, 0.5) -- that is its trained configuration. Event-GeM
# whitens and then re-ranks; the re-rank is its place match, not an add-on. EventVLAD and
# SpikeVPR use their descriptors directly, so whitening them would be a tuning step upstream
# never applies, and for SpikeVPR it is not a neutral one: on the synthetic arm its bank
# collapses (random pairs cosine 0.544, above the 0.510 between a slice and its own twin) and
# whitening masks that by removing the common-mode component.
METHODS = [
    ("MegaEvent ViT-B", "megaevent_{arm}", "r322{tag}_P64_s10000", "pca4096p0.5"),
    ("MegaEvent ViT-S", "megaevent_{arm}", "r322{tag}_S_step10000", "pca4096p0.5"),
    ("Event-GeM",       "eventgem_{arm}",  "r240{tag}",             "pca128p0.5+rerank"),
    ("EventVLAD",       "{vlad}",          "eventvlad{tag2}",       "native"),
    ("SpikeVPR",        "{spike}",         "spikevpr_nsavp{tag2}",  "native"),
]
BASELINES = ("Event-GeM", "EventVLAD", "SpikeVPR")
REFERENCE = "MegaEvent ViT-B"


def arm_names(arm, masked):
    """Filenames and result-key fragments for one arm. `real` keeps its historical tags."""
    src = f"{arm}_masked" if masked else arm
    if src == "real":
        return {"arm": "real", "tag": "ba50", "tag2": "_ba50",
                "vlad": "brisbane_event", "spike": "brisbane_event_nsavp"}
    return {"arm": src, "tag": f"_{src}", "tag2": f"_{src}",
            "vlad": f"brisbane_event_{src}", "spike": f"brisbane_event_nsavp_{src}"}


def read(stem, key, space):
    path = os.path.join(OUT, f"{stem}.json")
    if not os.path.exists(path):
        return None
    doc = json.load(open(path))
    results = doc["results"]
    if key not in results:
        # A single-result file names its key from the run tag; fall back when only one exists.
        if len(results) != 1:
            return None
        key = next(iter(results))
    entry = results[key]
    if space not in entry["recall"]:
        return None
    rec = entry["recall"][space]
    return {"r1": rec["1"], "r10": rec["10"],
            "n_db": entry["n_database"], "n_q": entry["n_queries"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masked", action="store_true",
                    help="read the vignette-masked arms instead of the raw ones")
    cli = ap.parse_args()

    real_n, i2e_n = arm_names("real", cli.masked), arm_names("i2e", cli.masked)
    label = "vignette-masked" if cli.masked else "raw"
    print(f"tab:sim2real ({label} arms) — pooled 25 m, query sunset1, "
          f"database daytime+morning+sunrise, 322^2\n")

    rows, shapes = [], set()
    for name, stem, key, space in METHODS:
        got = {}
        for arm, names in (("real", real_n), ("i2e", i2e_n)):
            got[arm] = read(stem.format(**names), key.format(**names), space)
            if got[arm]:
                shapes.add((got[arm]["n_db"], got[arm]["n_q"]))
        rows.append((name, space, got["real"], got["i2e"]))

    def cell(v, field):
        return "  --  " if v is None else f"{v[field]:.4f}"

    print(f"{'method':18s} {'space':20s} {'real R@1':>9s} {'i2e R@1':>8s} {'dR@1':>8s}"
          f" {'real R@10':>10s} {'i2e R@10':>9s} {'dR@10':>8s}")
    for name, space, r, i in rows:
        d1 = f"{i['r1'] - r['r1']:+.4f}" if r and i else "   --   "
        d10 = f"{i['r10'] - r['r10']:+.4f}" if r and i else "   --   "
        print(f"{name:18s} {space:20s} {cell(r,'r1'):>9s} {cell(i,'r1'):>8s} {d1:>8s}"
              f" {cell(r,'r10'):>10s} {cell(i,'r10'):>9s} {d10:>8s}")

    if len(shapes) > 1:
        print(f"\n  !! arms were scored on different galleries: {sorted(shapes)}. "
              f"The deltas are not comparable — check --database and --aps-aligned.")
    elif shapes:
        n_db, n_q = shapes.pop()
        print(f"\n  all arms: {n_db:,} database x {n_q:,} queries")

    # The margins. This is the actual test; per-method deltas invite the wrong reading.
    ref = next((r for r in rows if r[0] == REFERENCE), None)
    if ref and ref[2] and ref[3]:
        print(f"\n  margin ({REFERENCE} - baseline), and how it shifts real -> i2e:")
        print(f"  {'baseline':18s} {'R@1 real':>9s} {'R@1 i2e':>8s} {'shift':>8s}"
              f" {'R@10 real':>10s} {'R@10 i2e':>9s} {'shift':>8s}")
        for name, _, r, i in rows:
            if name not in BASELINES or not r or not i:
                continue
            m1r, m1i = ref[2]["r1"] - r["r1"], ref[3]["r1"] - i["r1"]
            m10r, m10i = ref[2]["r10"] - r["r10"], ref[3]["r10"] - i["r10"]
            print(f"  {name:18s} {m1r:>+9.4f} {m1i:>+8.4f} {m1i - m1r:>+8.4f}"
                  f" {m10r:>+10.4f} {m10i:>+9.4f} {m10i - m10r:>+8.4f}")
        print("\n  A positive shift means MegaEvent's advantage GREW on the synthetic arm,")
        print("  which is the reviewer objection confirmed rather than answered.")

    print("\n% --- tab:sim2real body ---")
    for name, _, r, i in rows:
        if not r or not i:
            print(f"            % {name}: missing an arm")
            continue
        print(f"            {name} & {r['r1']:.3f} & {r['r10']:.3f} & "
              f"{i['r1']:.3f} & {i['r10']:.3f} \\\\")


if __name__ == "__main__":
    main()
