"""Assemble `tab:native` — every method on every dataset, in its own descriptor space.

    pixi run python3 scripts/table_native.py
    pixi run python3 scripts/table_native.py --latex

**No PCA whitening anywhere.** A whitening basis is fit on one model's own database bank, so
reporting it is a per-model tuning step rather than a measurement, and the methods do not
gain from it equally (megaevent +1.6 to +11.7 R@1, Event-GeM +5 to +11 after re-ranking).
The native column is what each method actually produces.

Held fixed across every cell: the event stream, the countmask/MCTS render from it, a 25 m
Euclidean ground-truth radius via ``src.imagevpr.build_gt``, and single-best-match R@K over
scorable queries (``src.inference.recall_at_k``, or the streamed ``recall_from_ranked`` that
is certified equal to it). The script refuses to print a row whose cells disagree on the
database or scorable-query count, which is what makes the columns comparable.

Not fixed, by design — each method keeps its own published configuration:

* **megaevent** the four checkpoints in ``ckpts/``, countmask at 322x322, cosine. Two
  backbones (DINOv2 ViT-B/14 and ViT-S/14) crossed with two heads: SALAD as published
  (64x128 + 256 = 8448-d direct, 88.0M / 22.9M params) and MegaLoc's widened head
  (``--salad-cluster-dim 256`` -> 64x256 + 256 = 16640, then ``Linear(16640->8448)``,
  228.6M / 163.5M). All four are trained with the v5/v6 noise augmentation. Note the ViT-S
  ``MLoc`` column is MegaLoc's *geometry* randomly initialised, not its weights:
  ``salad.py:load_megaloc_aggregator`` asserts ``num_channels == 768``, so only ViT-B can
  inherit those 140.6M parameters.
* **ViT-B v4** (``b_full_P64_v4``) is a reference column, not a shipped model. It is
  ``ViT-B SALAD`` without the noise augmentation and nothing else, so the pair is the noise
  ablation measured on these benchmarks rather than on the trainer's in-loop metric.
* **Event-GeM** MCTS at 240x320 (SuperEvent's own geometry), GeM(p=5), cosine. Reported at
  both stages: the 128-D global descriptor alone, and the top-50 homography re-rank that is
  its actual place match. Quoting only the global stage understates it by up to 30 R@1.
* **The four RGB controls** — MixVPR, CricaVPR, SALAD, MegaLoc — countmask, cosine, never
  retrained on events, each at its own published input size. They control for whether
  training on events was worth doing at all, given that a countmask frame is still an image
  and an RGB retrieval model may simply read it. Read them as a ladder:

  - **MixVPR** 10.9M, 4096-d, 320. A ResNet50 truncated after ``layer3`` into four
    feature-mixer MLPs — the only control with no DINOv2 and no transformer in it, so it
    separates "a good RGB retrieval model can read countmask" from "a DINOv2 patch backbone
    can read countmask".
  - **CricaVPR** 106.8M, 10752-d, 224. Same ViT-B backbone, 14 multi-scale GeM region tokens
    through a cross-image encoder. Its 224 is structural, not a choice, and its descriptors
    depend on the batch — see ``src/methods.py::CricaVPRMethod``, which records the pinned
    batch of 16 in every result's ``meta``.
  - **SALAD** 88.0M, 8448-d, 322. megaevent's architecture *exactly* — DINOv2 ViT-B/14,
    SALAD 64x128+256, against ViT-B's 88.0M — so the gap to it is what the event fine-tuning
    bought with everything else held fixed.
  - **MegaLoc** 228.6M, 8448-d, 322. The same ViT-B backbone under a 140.6M-parameter
    learned compression head, trained far beyond SALAD's GSV-Cities. A gap that opens
    against MegaLoc but not against SALAD is a statement about head capacity and training
    scale, not about events.

  SALAD and MegaLoc share a normalisation *and* a resize, so the difference between those
  two columns is the network alone. Across all four the resize differs, so that comparison
  is between published configurations — the same footing on which Event-GeM's 240x320 sits.

The provenance column is the thing to read the table by. Tokyo 24/7, MSLS and Pitts250k are
I2E simulations of photographs; NYC-Event, Brisbane-Event and NSAVP came off a real DAVIS or
Prophesee sensor. I2E is also megaevent's fine-tuning domain, so the simulated group is the
one its training distribution matches.
"""

import argparse
import json
import os

ROOT = "/media/adam/vprdatasets/megaevent"
EVAL = f"{ROOT}/evaluations"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The four shipped checkpoints, by the bank tag src/methods.py builds from
# basename + step + sha256[:10]. Each is its run's ``best.pt`` (--eval-select combined),
# except ViT-B v4, which predates that convention and is the 10k endpoint.
B_MLOC = "megaevent_vitb_mloc_s3500_b1c23c1bc1_countmask_r322"    # b_noise_projmegaloc_P64_v5
B_SAL = "megaevent_vitb_salad_s8000_dde11113fc_countmask_r322"    # b_noise_P64_v5
S_MLOC = "megaevent_vits_mloc_s3500_1ed09057dd_countmask_r322"    # s_noise_projnaive_ft4_P64_v6
S_SAL = "megaevent_vits_salad_s2000_9dea56d731_countmask_r322"    # s_noise_ft4_P64_v6
V4 = "results_step10000_s9999_403872f88a_countmask_r322.json"     # b_full_P64_v4/step10000
GEM = "results_eventgem_se_240x320.json"
LOC = "results_megaloc_countmask_r322.json"
SAL = "results_salad_countmask_r322.json"
MIX = "results_mixvpr_countmask_r320.json"
CRI = "results_cricavpr_countmask_r224.json"
# Every pooled bank is the same 50 ms slices under the same BA filter; the number is each
# model's own published input size, which is why the RGB controls do not share one tag.
TAG = "r322ba50"                # pooled megaevent / SALAD / MegaLoc: 322
MIX_TAG = "r320ba50"            # pooled MixVPR: 320, the size its aggregator is built for
CRI_TAG = "r224ba50"            # pooled CricaVPR: 224, forced by its 16x16 patch slicing
GEM_TAG = "r240ba50"            # pooled Event-GeM: its own 240x320 grid, same slices

# Eleven columns will not fit on one line, so they print as blocks over the same rows. The
# size guard still runs across every column, which is what makes the blocks comparable.
SHIPPED = ["ViT-B MLoc", "ViT-B SALAD", "ViT-S MLoc", "ViT-S SALAD"]
# The RGB controls, weakest first — they read as a capacity ladder: MixVPR 10.9M (a CNN, no
# DINOv2 anywhere), CricaVPR 106.8M, SALAD 88.0M (megaevent's architecture exactly),
# MegaLoc 228.6M. None has ever seen an event.
RGB = ["MixVPR", "CricaVPR", "SALAD", "MegaLoc"]
BLOCKS = [("megaevent (ckpts/)", SHIPPED + ["ViT-B v4"]),
          ("event baselines", ["Event-GeM", "+rerank"]),
          ("RGB controls", RGB)]
COLUMNS = [column for _, block in BLOCKS for column in block]
# dataset -> (provenance, {column: (kind, path, key, space)})
#   image  doc["recall"][space]                     (src/imagevpr.py, score_cached_banks.py)
#   pooled doc["results"][key]["recall"][space]     (brisbane_pooled.py, nsavp_pooled.py, ...)
#   traj   doc["results"][key][space]               (tokyo_trajectory.py)
DATASETS = {
    "Tokyo 24/7": ("I2E sim", {
        "ViT-B MLoc": ("traj", f"{ROOT}/tokyo_v5/results_r322.json", "step3500", "native"),
        "ViT-B SALAD": ("traj", f"{ROOT}/tokyo_ship/out/results_r322.json",
                        "b_salad_s8000", "native"),
        "ViT-S MLoc": ("traj", f"{ROOT}/tokyo_ship/out/results_r322.json",
                       "s_mloc_s3500", "native"),
        "ViT-S SALAD": ("traj", f"{ROOT}/tokyo_ship/out/results_r322.json",
                        "s_salad_s2000", "native"),
        "ViT-B v4": ("traj", f"{ROOT}/tokyo_v4_stage/A_out/results_r322.json",
                     "P64_s10000", "native"),
        "Event-GeM": ("image", f"{REPO}/features/tokyo247/results_eventgem.json", None, "native"),
        "+rerank": ("image", f"{REPO}/features/tokyo247/results_eventgem.json", None, "native+rerank"),
        "MixVPR": ("image", f"{EVAL}/tokyo247/{MIX}", None, "native"),
        "CricaVPR": ("image", f"{EVAL}/tokyo247/{CRI}", None, "native"),
        "SALAD": ("image", f"{EVAL}/tokyo247/{SAL}", None, "native"),
        "MegaLoc": ("image", f"{EVAL}/tokyo247/{LOC}", None, "native"),
    }),
    "MSLS": ("I2E sim", {
        "ViT-B MLoc": ("image", f"{EVAL}/msls/results_{B_MLOC}.json", None, "native"),
        "ViT-B SALAD": ("image", f"{EVAL}/msls/results_{B_SAL}.json", None, "native"),
        "ViT-S MLoc": ("image", f"{EVAL}/msls/results_{S_MLOC}.json", None, "native"),
        "ViT-S SALAD": ("image", f"{EVAL}/msls/results_{S_SAL}.json", None, "native"),
        "ViT-B v4": ("image", f"{EVAL}/msls/{V4}", None, "native"),
        "Event-GeM": ("image", f"{EVAL}/msls/{GEM}", None, "native"),
        "+rerank": ("image", f"{EVAL}/msls/{GEM}", None, "native+rerank"),
        "MixVPR": ("image", f"{EVAL}/msls/{MIX}", None, "native"),
        "CricaVPR": ("image", f"{EVAL}/msls/{CRI}", None, "native"),
        "SALAD": ("image", f"{EVAL}/msls/{SAL}", None, "native"),
        "MegaLoc": ("image", f"{EVAL}/msls/{LOC}", None, "native"),
    }),
    "Pitts250k": ("I2E sim", {
        "ViT-B MLoc": ("image", f"{EVAL}/pitts250k/results_{B_MLOC}.json", None, "native"),
        "ViT-B SALAD": ("image", f"{EVAL}/pitts250k/results_{B_SAL}.json", None, "native"),
        "ViT-S MLoc": ("image", f"{EVAL}/pitts250k/results_{S_MLOC}.json", None, "native"),
        "ViT-S SALAD": ("image", f"{EVAL}/pitts250k/results_{S_SAL}.json", None, "native"),
        "ViT-B v4": ("image", f"{EVAL}/pitts250k/{V4}", None, "native"),
        # The 2026-08-07 Event-GeM run died in the pca1 re-rank before writing its JSON. The
        # global stage was re-scored from its cached banks (score_cached_banks.py) and agrees
        # with the log to 4 dp; the re-ranked cell is the log's own 3 dp and is marked with *.
        "Event-GeM": ("image", f"{EVAL}/pitts250k/results_eventgem_se_240x320_native.json",
                      None, "native"),
        "+rerank": ("logged", 0.196, 0.299, "2026-08-07_12-42-19.log"),
        "MixVPR": ("image", f"{EVAL}/pitts250k/{MIX}", None, "native"),
        "CricaVPR": ("image", f"{EVAL}/pitts250k/{CRI}", None, "native"),
        "SALAD": ("image", f"{EVAL}/pitts250k/{SAL}", None, "native"),
        "MegaLoc": ("image", f"{EVAL}/pitts250k/{LOC}", None, "native"),
    }),
    "NYC-Event": ("recorded", {
        "ViT-B MLoc": ("image", f"{EVAL}/nycevent/results_{B_MLOC}.json", None, "native"),
        "ViT-B SALAD": ("image", f"{EVAL}/nycevent/results_{B_SAL}.json", None, "native"),
        "ViT-S MLoc": ("image", f"{EVAL}/nycevent/results_{S_MLOC}.json", None, "native"),
        "ViT-S SALAD": ("image", f"{EVAL}/nycevent/results_{S_SAL}.json", None, "native"),
        "ViT-B v4": ("image", f"{EVAL}/nycevent/{V4}", None, "native"),
        "Event-GeM": ("image", f"{ROOT}/nycevent_eval_eventgem/nycevent/results_eventgem.json",
                      None, "native"),
        "+rerank": ("image", f"{ROOT}/nycevent_eval_eventgem/nycevent/results_eventgem.json",
                    None, "native+rerank"),
        "MixVPR": ("image", f"{EVAL}/nycevent/{MIX}", None, "native"),
        "CricaVPR": ("image", f"{EVAL}/nycevent/{CRI}", None, "native"),
        "SALAD": ("image", f"{EVAL}/nycevent/{SAL}", None, "native"),
        "MegaLoc": ("image", f"{EVAL}/nycevent/{LOC}", None, "native"),
    }),
    "Brisbane-Event": ("recorded", {
        "ViT-B MLoc": ("pooled", f"{ROOT}/brisbane_v5/noise_projmegaloc_no_sunset2.json",
                       f"{TAG}_pm_s3500", "native"),
        "ViT-B SALAD": ("pooled", f"{ROOT}/brisbane_ship/ship_no_sunset2.json",
                        f"{TAG}_b_salad_s8000", "native"),
        "ViT-S MLoc": ("pooled", f"{ROOT}/brisbane_ship/ship_no_sunset2.json",
                       f"{TAG}_s_mloc_s3500", "native"),
        "ViT-S SALAD": ("pooled", f"{ROOT}/brisbane_ship/ship_no_sunset2.json",
                        f"{TAG}_s_salad_s2000", "native"),
        "ViT-B v4": ("pooled", f"{ROOT}/brisbane_pooled/psweep_no_sunset2.json",
                     f"{TAG}_P64_s10000", "native"),
        "Event-GeM": ("pooled", f"{ROOT}/eventgem_pooled/brisbane_event.json", GEM_TAG, "native"),
        "+rerank": ("pooled", f"{ROOT}/eventgem_pooled/brisbane_event.json", GEM_TAG,
                    "native+rerank"),
        "MixVPR": ("pooled", f"{ROOT}/mixvpr_pooled/brisbane_event.json", MIX_TAG, "native"),
        "CricaVPR": ("pooled", f"{ROOT}/cricavpr_pooled/brisbane_event.json", CRI_TAG,
                     "native"),
        "SALAD": ("pooled", f"{ROOT}/salad_pooled/brisbane_event.json", TAG, "native"),
        "MegaLoc": ("pooled", f"{ROOT}/megaloc_pooled/brisbane_event.json", TAG, "native"),
    }),
    "NSAVP R0": ("recorded", {
        "ViT-B MLoc": ("pooled", f"{ROOT}/nsavp_pooled/results_route0_v5.json", TAG, "native"),
        "ViT-B SALAD": ("pooled", f"{ROOT}/nsavp_pooled/results_route0_ship.json",
                        f"{TAG}_b_salad_s8000", "native"),
        "ViT-S MLoc": ("pooled", f"{ROOT}/nsavp_pooled/results_route0_ship.json",
                       f"{TAG}_s_mloc_s3500", "native"),
        "ViT-S SALAD": ("pooled", f"{ROOT}/nsavp_pooled/results_route0_ship.json",
                        f"{TAG}_s_salad_s2000", "native"),
        "ViT-B v4": ("pooled", f"{ROOT}/nsavp_pooled/results_route0.json",
                     f"{TAG}_b_ship", "native"),
        "Event-GeM": ("pooled", f"{ROOT}/eventgem_pooled/nsavp.json", GEM_TAG, "native"),
        "+rerank": ("pooled", f"{ROOT}/eventgem_pooled/nsavp.json", GEM_TAG, "native+rerank"),
        "MixVPR": ("pooled", f"{ROOT}/mixvpr_pooled/nsavp.json", MIX_TAG, "native"),
        "CricaVPR": ("pooled", f"{ROOT}/cricavpr_pooled/nsavp.json", CRI_TAG, "native"),
        "SALAD": ("pooled", f"{ROOT}/salad_pooled/nsavp.json", TAG, "native"),
        "MegaLoc": ("pooled", f"{ROOT}/megaloc_pooled/nsavp.json", TAG, "native"),
    }),
}


def read_cell(spec):
    """-> ((R@1, R@10), (n_database, scorable), note) or (None, None, why-it-is-missing)."""
    kind, path, key, space = spec
    if kind == "logged":
        return (path, key), None, f"3 dp, from {space}"
    try:
        with open(path) as handle:
            doc = json.load(handle)
    except FileNotFoundError:
        return None, None, f"missing {path}"
    try:
        if kind == "image":
            recall = doc["recall"][space]
            shape = (doc["n_database"], doc["scorable_queries"])
        elif kind == "pooled":
            result = doc["results"][key]
            recall = result["recall"][space]
            shape = (result["n_database"], result["scorable"])
        else:
            recall = doc["results"][key][space]
            shape = (doc["n_database"], doc["n_queries"])
    except KeyError as exc:
        return None, None, f"no {exc} in {os.path.basename(path)}"
    return (recall["1"], recall["10"]), shape, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--latex", action="store_true", help="also print the tabular body")
    ap.add_argument("--delta", action="store_true",
                    help="also print each baseline's R@1 as ours - baseline, against both "
                         "shipped MegaLoc-head checkpoints. Positive = megaevent ahead. "
                         "With --latex, emits the baseline rows ready to paste.")
    cli = ap.parse_args()

    rows, missing = [], []
    for dataset, (provenance, specs) in DATASETS.items():
        cells, shapes = {}, {}
        for column in COLUMNS:
            value, shape, note = read_cell(specs[column])
            cells[column] = value
            if value is None:
                missing.append(f"{dataset} {column}: {note}")
            if shape:
                shapes[column] = shape
        # Every cell in a row must be the same benchmark. A method that silently scored a
        # different gallery would otherwise sit in the table looking comparable.
        distinct = set(shapes.values())
        if len(distinct) > 1:
            raise SystemExit(f"{dataset}: cells disagree on the benchmark size {shapes}")
        rows.append((dataset, provenance, next(iter(distinct), None), cells))

    for title, block in BLOCKS:
        head = "".join(f"{c:>17s}" for c in block)
        print(f"\n{title}")
        print(f"{'dataset':<15s} {'events':<9s} {'db x queries':>18s}  {head}")
        for dataset, provenance, shape, cells in rows:
            size = f"{shape[0]:,} x {shape[1]:,}" if shape else "--"
            line = ""
            for column in block:
                value = cells[column]
                line += "               --" if value is None else f"{value[0]:>8.3f} {value[1]:>8.3f}"
            print(f"{dataset:<15s} {provenance:<9s} {size:>18s}  {line}")
        print(f"{'':<15s} {'':<9s} {'':>18s}  " + "".join(f"{'R@1':>8s} {'R@10':>8s}"
                                                          for _ in block))

    if missing:
        print("\nmissing:")
        for item in missing:
            print(f"  {item}")
        return

    # Where megaevent's best stands, split by provenance: the simulated group is the one
    # megaevent's own I2E fine-tuning domain matches. Four references, because they answer
    # different questions — SALAD alone is the architecture-matched one (see the docstring).
    print()
    references = {
        "best baseline": lambda c: max([c["+rerank"][0]] + [c[m][0] for m in RGB]),
        "best RGB control": lambda c: max(c[m][0] for m in RGB),
        "SALAD (same 88.0M)": lambda c: c["SALAD"][0],
        "MegaLoc (228.6M)": lambda c: c["MegaLoc"][0],
    }
    for name, pick in references.items():
        for group in ("I2E sim", "recorded"):
            subset = [r for r in rows if r[1] == group]
            deltas = [max(c[m][0] for m in SHIPPED) - pick(c) for _, _, _, c in subset]
            wins = sum(1 for d in deltas if d > 0)
            print(f"  best megaevent - {name:<19s} {group:<8s} R@1 "
                  f"{sum(deltas) / len(deltas):+.3f}   (ahead on {wins}/{len(subset)})")

    if cli.delta:
        print_deltas(rows, cli.latex)

    if cli.latex:
        print("\n% --- tab:native body ---")
        for dataset, _, _, cells in rows:
            body = " & ".join(f"{cells[c][0]:.3f} & {cells[c][1]:.3f}" for c in COLUMNS)
            print(f"            {dataset} & {body} \\\\")


# The two shipped rows the paper's baseline table is read against, by their column names here.
SHIPPED_ROWS = [("ViT/S MegaLoc", "ViT-S MLoc"), ("ViT/B MegaLoc", "ViT-B MLoc")]


def print_deltas(rows, latex=False):
    """R@1 as ``ours - baseline`` for each baseline row: positive means megaevent is ahead.

    One pair per cell, against each of the two shipped MegaLoc-head checkpoints, because
    which one a baseline beats is the whole point on the recorded datasets — ViT-S wins
    Brisbane and loses NSAVP, ViT-B the other way round, so a single "ours" would hide the
    thing worth reading.
    """
    order = [dataset for dataset, _, _, _ in rows]
    print("\nR@1 delta, ours - baseline (positive = megaevent ahead)")
    for label, column in SHIPPED_ROWS:
        print(f"\n  vs {label}")
        print(f"    {'baseline':<12s}" + "".join(f"{d:>16s}" for d in order))
        for baseline in ["Event-GeM", "+rerank"] + RGB:
            line = ""
            for _, _, _, cells in rows:
                ours, theirs = cells[column], cells[baseline]
                line += "              --" if (ours is None or theirs is None) \
                    else f"{ours[0] - theirs[0]:>+16.3f}"
            print(f"    {baseline:<12s}{line}")

    if latex:
        # One row per baseline, one cell per dataset, both deltas in the cell. Column order
        # follows DATASETS, which is the order tab:native prints.
        print("\n% --- baseline rows as R@1 deltas (ours - baseline), ViT/S / ViT/B ---")
        print(f"% columns: {' & '.join(order)}")
        for baseline in ["Event-GeM", "+rerank"] + RGB:
            body = []
            for _, _, _, cells in rows:
                theirs = cells[baseline]
                body.append(" / ".join(
                    "--" if (cells[col] is None or theirs is None)
                    else f"{cells[col][0] - theirs[0]:+.2f}" for _, col in SHIPPED_ROWS))
            print(f"            {baseline} & " + " & ".join(body) + " \\\\")


if __name__ == "__main__":
    main()
