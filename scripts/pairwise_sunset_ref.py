"""Single-reference Brisbane-Event and NSAVP results — sunset as the gallery, not the query.

    pixi run python3 scripts/pairwise_sunset_ref.py --dataset brisbane_event
    pixi run python3 scripts/pairwise_sunset_ref.py --dataset nsavp
    pixi run python3 scripts/pairwise_sunset_ref.py --dataset brisbane_event --verify

Every reported Brisbane/NSAVP number uses the **pooled** protocol: sunset1 (resp. R0_FA0)
queries, and the four/five remaining traverses concatenate into one gallery so the nearest
descriptor anywhere in it wins. That lets a query match whichever condition happens to be
easiest — a query's pooled score is its best single partner +0.03 — so it conflates "can this
model match across illumination" with "was an easy partner available".

This is the un-pooled view: **one reference traverse, one query traverse per cell**, sunset as
the reference throughout.

    brisbane_event   ref sunset1  ->  daytime, morning, sunrise
    nsavp            ref R0_FS0   ->  R0_FA0

Excluded, and why. ``sunset2`` is the second sunset run and has no accumulate-render control
banks. ``night`` is dropped on both — ``run_v8_sim2real.sh`` already sets that precedent on
Brisbane, and night's R@1 floors for every model, so it moves with nothing. NSAVP keeps route 0
forward only: R1 is a different route with zero overlap, and the reverse traverses are a
heading test rather than an illumination one. That leaves R0_FA0 as the only surviving NSAVP
query, so NSAVP is a single cell.

Everything else is held at the reported settings: 50 ms slices, the BA filter at dt=50000 us,
a 25 m Euclidean radius via ``src.imagevpr.build_gt``, native descriptors, no PCA, and
single-best-match R@K over scorable queries. Only the gallery membership changes, which is
what ``--verify`` proves: it re-scores the *pooled* membership through this same code path and
checks it reproduces the cached ledger JSONs.

Two things worth knowing about the numbers:

* A single-traverse gallery is ~4x smaller than the pooled one (14.3k vs 54.6k rows on
  Brisbane), so R@1 here is **not** comparable to any published cell. Read across methods
  within this table, never against the pooled table.
* The Brisbane database is *fixed* at sunset1 across all three cells — only the query set
  changes. That is cleaner than the pooled leave-one-out sweep, where the gallery moved too.

Event-GeM keeps its own MCTS 240x320 render while every other row is the accumulate render;
that mismatch is inherited from the published ledger, where each method keeps its published
configuration. Its released banks can also sit a tail frame or two off eventlab's frames-50
grid, so its arm gets its own truncated geometry and its own GT — hence the per-method
(database x scorable) columns instead of the single size guard ``table_native.py`` uses.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from brisbane_resolution import _Args  # noqa: E402
import brisbane_pooled as bp  # noqa: E402
import nsavp_pooled as npl  # noqa: E402
from src.imagevpr import build_gt  # noqa: E402

ROOT = "/media/adam/vprdatasets/megaevent"
V8 = f"{ROOT}/v8_bench"
# The 0.1.0 worktree's own feature tree — never the data root's features/ cache, which
# predates the release (see scripts/eg010_pooled.py).
EG_FEATURES = f"{ROOT}/eventgem_010/eventgem/features"
KS = (1, 5, 10, 20)
THRESHOLD_M = 25.0

CONFIGS = {
    "brisbane_event": {
        "eventlab_dir": "/media/adam/vprdatasets/eventgem",
        # The full set the geometry is derived over. src.traversegps.track_origin picks the
        # local-ENU origin from the sequence list, so passing the same list every prior run
        # used keeps the origin, the clock calibration and the route-length check identical.
        # A common translation would cancel out of a 25 m radius anyway; this also keeps the
        # --verify comparison exact.
        "geom_traverses": ["daytime", "morning", "night", "sunrise", "sunset1"],
        # Explicit (reference, query) pairs rather than one reference plus a query list,
        # because NSAVP needs two *different* references — forward and reverse are separate
        # cells, not one gallery with two queries.
        "pairs": [("sunset1", "daytime"), ("sunset1", "morning"), ("sunset1", "sunrise")],
        "pooled_query": "sunset1",
        "pooled_database": ["daytime", "morning", "night", "sunrise"],
        "db_chunk": None,
        "chunk": 256,
    },
    "nsavp": {
        "eventlab_dir": "/media/adam/vprdatasets/eventlab",
        "geom_traverses": ["R0_FA0", "R0_FN0", "R0_FS0", "R0_RA0", "R0_RN0", "R0_RS0"],
        # Forward and reverse, each sunset-referenced against its own direction. Night
        # (R0_FN0/R0_RN0) excluded. These are two independent benchmarks with different
        # gallery and query sizes — never averaged into a single NSAVP number.
        "pairs": [("R0_FS0", "R0_FA0"), ("R0_RS0", "R0_RA0")],
        "pooled_query": "R0_FA0",
        "pooled_database": ["R0_FN0", "R0_FS0", "R0_RA0", "R0_RN0", "R0_RS0"],
        "db_chunk": npl.DB_CHUNK,
        "chunk": 256,
    },
}

# (name, kind, {dataset: (bank_dir, filename_template)}, {dataset: (verify_json, key_path)})
#
# The accumulate controls put `accum` *between* the tag and the traverse
# (r322ba50_accum_daytime_megaloc.npy) while the v8 banks do not (r322ba50_daytime_v8.npy),
# so each row carries its own template rather than sharing one format string.
MODELS = [
    ("megaevent v8-B", "bank", {
        "brisbane_event": (f"{V8}/brisbane", "r322ba50_{t}_v8.npy"),
        "nsavp": (f"{V8}/nsavp", "r322ba50_{t}_v8.npy"),
    }, {
        "brisbane_event": (f"{V8}/brisbane/brisbane_4trav.json",
                           ["results", "r322ba50", "recall", "native"]),
        "nsavp": (f"{V8}/nsavp/results.json",
                  ["results", "r322ba50", "recall", "native"]),
    }),
    ("megaevent v8-S", "bank", {
        "brisbane_event": (f"{V8}/brisbane_s", "r322ba50_{t}_v8s.npy"),
        "nsavp": (f"{V8}/nsavp_s", "r322ba50_{t}_v8s.npy"),
    }, {
        "brisbane_event": (f"{V8}/brisbane_s/brisbane_4trav.json",
                           ["results", "r322ba50", "recall", "native"]),
        "nsavp": (f"{V8}/nsavp_s/results.json",
                  ["results", "r322ba50", "recall", "native"]),
    }),
    ("MegaLoc", "bank", {
        "brisbane_event": (f"{ROOT}/megaloc_pooled", "r322ba50_accum_{t}_megaloc.npy"),
        "nsavp": (f"{ROOT}/megaloc_pooled", "r322ba50_accum_{t}_megaloc.npy"),
    }, {
        ds: (f"{ROOT}/megaloc_pooled/{ds}_accumulate.json",
             ["results", "r322ba50_accum", "recall", "native"]) for ds in CONFIGS
    }),
    # The fairness arm: the same MegaLoc, on the same accumulate frames, normalised with the
    # statistics of that render instead of ImageNet's (src/methods.py::ACCUMULATE_MEAN —
    # accumulate is a white background, which ImageNet centres 2.0-4.5 sigma off). Every
    # other control row above is still ImageNet-normalised, so this row is the measurement
    # of what that convention costs, not a like-for-like competitor to them.
    ("MegaLoc acstats", "bank", {
        "brisbane_event": (f"{ROOT}/megaloc_pooled", "r322ba50_accum_acstats_{t}_megaloc.npy"),
        "nsavp": (f"{ROOT}/megaloc_pooled", "r322ba50_accum_acstats_{t}_megaloc.npy"),
    }, {
        ds: (f"{ROOT}/megaloc_pooled/{ds}_accumulate_acstats.json",
             ["results", "r322ba50_accum_acstats", "recall", "native"]) for ds in CONFIGS
    }),
    ("SALAD", "bank", {
        "brisbane_event": (f"{ROOT}/salad_pooled", "r322ba50_accum_{t}_salad.npy"),
        "nsavp": (f"{ROOT}/salad_pooled", "r322ba50_accum_{t}_salad.npy"),
    }, {
        ds: (f"{ROOT}/salad_pooled/{ds}_accumulate.json",
             ["results", "r322ba50_accum", "recall", "native"]) for ds in CONFIGS
    }),
    ("MixVPR", "bank", {
        "brisbane_event": (f"{ROOT}/mixvpr_pooled", "r320ba50_accum_{t}_mixvpr.npy"),
        "nsavp": (f"{ROOT}/mixvpr_pooled", "r320ba50_accum_{t}_mixvpr.npy"),
    }, {
        ds: (f"{ROOT}/mixvpr_pooled/{ds}_accumulate.json",
             ["results", "r320ba50_accum", "recall", "native"]) for ds in CONFIGS
    }),
    ("CricaVPR", "bank", {
        "brisbane_event": (f"{ROOT}/cricavpr_pooled", "r224ba50_accum_{t}_cricavpr.npy"),
        "nsavp": (f"{ROOT}/cricavpr_pooled", "r224ba50_accum_{t}_cricavpr.npy"),
    }, {
        ds: (f"{ROOT}/cricavpr_pooled/{ds}_accumulate.json",
             ["results", "r224ba50_accum", "recall", "native"]) for ds in CONFIGS
    }),
    # The three controls whose networks live in the VPR-methods-evaluation checkout
    # (src/vprbench.py, run through scripts/rgb_pooled.py). All DINOv2 at 322 like MegaLoc
    # and SALAD, so they extend the ladder without changing the resize: BoQ cross-attends 64
    # learned queries at 12288-d, QAA is the most recent method in that harness, SuperVLAD is
    # the compact 3072-d point.
    ("BoQ", "bank", {
        ds: (f"{ROOT}/boq_pooled", "r322ba50_accum_{t}_boq.npy") for ds in CONFIGS
    }, {
        ds: (f"{ROOT}/boq_pooled/{ds}_accumulate.json",
             ["results", "r322ba50_accum", "recall", "native"]) for ds in CONFIGS
    }),
    ("QAA", "bank", {
        ds: (f"{ROOT}/qaa_pooled", "r322ba50_accum_{t}_qaa.npy") for ds in CONFIGS
    }, {
        ds: (f"{ROOT}/qaa_pooled/{ds}_accumulate.json",
             ["results", "r322ba50_accum", "recall", "native"]) for ds in CONFIGS
    }),
    ("SuperVLAD", "bank", {
        ds: (f"{ROOT}/supervlad_pooled", "r322ba50_accum_{t}_supervlad.npy") for ds in CONFIGS
    }, {
        ds: (f"{ROOT}/supervlad_pooled/{ds}_accumulate.json",
             ["results", "r322ba50_accum", "recall", "native"]) for ds in CONFIGS
    }),
    # SpikeVPR is deliberately cross-dataset: the Brisbane column uses the NSAVP-trained
    # model and vice versa, so neither cell has train/test overlap. That is the protocol its
    # pooled run published, and the bank tag carries the *training* set, not the eval set.
    ("SpikeVPR", "bank", {
        "brisbane_event": (f"{ROOT}/spikevpr_pooled", "spikevpr_nsavp_ba50_{t}.npy"),
        "nsavp": (f"{ROOT}/spikevpr_pooled", "spikevpr_brisbane_ba50_{t}.npy"),
    }, {
        "brisbane_event": (f"{ROOT}/spikevpr_pooled/brisbane_event_nsavp.json",
                           ["results", "spikevpr_nsavp_ba50", "recall", "native"]),
        "nsavp": (f"{ROOT}/spikevpr_pooled/nsavp_brisbane.json",
                  ["results", "spikevpr_brisbane_ba50", "recall", "native"]),
    }),
    # Its reconstruction stack consumes a short temporal window, so every bank is exactly two
    # frames shorter than the eventlab grid — hence own_grid rather than bank.
    ("EventVLAD", "own_grid", {
        ds: (f"{ROOT}/eventvlad_pooled", "eventvlad_ba50_{t}_eventvlad.npy") for ds in CONFIGS
    }, {
        ds: (f"{ROOT}/eventvlad_pooled/{ds}.json",
             ["results", "eventvlad_ba50", "recall", "native"]) for ds in CONFIGS
    }),
    ("Event-GeM 0.1.0", "own_grid", {}, {
        ds: (f"{V8}/eg010_pooled_{ds}.json", ["recall"]) for ds in CONFIGS
    }),
]


def models_for(dataset, kind=None):
    """The roster rows that have banks for this dataset. Event-GeM resolves its own paths
    from the release tree rather than a (dir, template) pair, so it is always in."""
    return [m for m in MODELS
            if (kind is None or m[1] == kind)
            and (m[0] == "Event-GeM 0.1.0" or dataset in m[2])]
# The +rerank row is not computed here — it needs the 0.1.0 release's own dependency stack.
# eventgem_010/pooled_rerank.py writes this, and it is folded in if present.
RERANK_JSON = V8 + "/eg010_rerank_pairwise_{ds}.json"


# ---------------------------------------------------------------------------
# Geometry and banks
# ---------------------------------------------------------------------------
def geometry(dataset):
    """{traverse: (xy, covered, speed, info)} — identical to scripts/eg010_pooled.py's."""
    cfg = CONFIGS[dataset]
    if dataset == "nsavp":
        return npl.traverse_geometry(os.path.join(cfg["eventlab_dir"], dataset),
                                     cfg["geom_traverses"], 50)
    args = _Args(cfg["eventlab_dir"], dataset, 50, False, False)
    args.filter_dt_us = 50_000
    return bp.traverse_geometry(args, cfg["geom_traverses"], None)


def bank_files(dataset, traverses):
    """{model: {traverse: {label: path}}} in the shape bp.pool_database wants."""
    files = {}
    for name, _, dirs, _ in models_for(dataset, "bank"):
        bank_dir, template = dirs[dataset]
        files[name] = {t: {name: os.path.join(bank_dir, template.format(t=t))}
                       for t in traverses}
        for t in traverses:
            path = files[name][t][name]
            if not os.path.exists(path):
                raise SystemExit(f"{name}: missing bank {path}")
    return files


def eg010_load_bank(dataset, traverse):
    """The released 0.1.0 global bank. Same semantics as scripts/eg010_pooled.py:53."""
    paths = sorted(glob.glob(f"{EG_FEATURES}/{dataset}/*/{dataset}_{traverse}_features.pt"))
    if not paths:
        raise SystemExit(f"no released feature bank for {dataset}/{traverse} under "
                         f"{EG_FEATURES} — harvest it with a 0.1.0 pair run first")
    banks = [torch.load(p, map_location="cpu") for p in paths]
    for b in banks[1:]:
        # allclose, not equal: independent GPU runs drift at the 1e-6 level.
        if b.shape != banks[0].shape or not torch.allclose(b, banks[0], atol=1e-4):
            raise SystemExit(f"{dataset}/{traverse}: pair-dir copies differ ({paths})")
    return torch.nn.functional.normalize(banks[0].float(), dim=1)


# How each own-grid row's descriptor index maps onto eventlab's frame grid. See own_grid_arm.
OWN_GRID_ALIGN = {"EventVLAD": "centre"}
# Scoring chunk sizes, where a row's published run used something other than this dataset's.
# topk_ranked's db_chunk merge is exact for distinct similarities but *not* for ties, and
# EventVLAD's descriptors all carry the same norm, so ties are common enough to move R@1 by
# ~1e-3. Matching each row's own published chunking is what makes --verify reproduce exactly.
OWN_GRID_CHUNK = {"Event-GeM 0.1.0": (64, 4096)}   # scripts/eg010_pooled.py's


def npy_load_bank(path):
    return torch.from_numpy(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32))


def own_grid_arm(dataset, geom, traverses, load, label, align="tail", tol=2):
    """{traverse: (desc, xy)} for a method that does not sit on eventlab's frames-50 grid.

    Two rows need this, for different reasons, and the difference matters:

    * ``tail`` — Event-GeM. Its released banks slice from the recording start like eventcv
      does but can run a tail frame short, so descriptor *i* is still eventlab frame *i* and
      truncating to the shorter of the two is sound.
    * ``centre`` — EventVLAD. It is a triplet model: each descriptor is the *middle* slice of
      a 3-slice window, so descriptor *i* is eventlab frame *i+1* and the grid loses its first
      and last frame. This is ``scripts/eventvlad_pooled.py:147 centre_geometry``, and getting
      it wrong is a silent one-slice (~0.7 m) shift on every row rather than an error.

    Either way a gap past ``tol`` means the framing conventions really have drifted and the
    row-to-coordinate mapping is unproven, which is a hard error, not a silent misalignment.
    """
    out = {}
    for t in traverses:
        bank = load(dataset, t)
        xy, cov = geom[t][0], geom[t][1]
        if align == "centre":
            xy, cov = xy[1:-1], cov[1:-1]
        elif align != "tail":
            raise SystemExit(f"unknown alignment {align!r} for {label}")
        n = min(len(bank), len(cov))
        if abs(len(bank) - len(cov)) > tol:
            raise SystemExit(f"{t}: {label} bank {len(bank)} vs {align} geometry {len(cov)} "
                             f"frames — more than {tol} apart, alignment unproven")
        keep = cov[:n]
        out[t] = (bank[:n][torch.as_tensor(keep)], xy[:n][keep])
    return out


def own_grid_arms(dataset, geom, traverses):
    """{model: {traverse: (desc, xy)}} for every own-grid row on this dataset."""
    arms = {}
    for name, _, dirs, _ in models_for(dataset, "own_grid"):
        if name == "Event-GeM 0.1.0":
            load, align = eg010_load_bank, "tail"
        else:
            bank_dir, template = dirs[dataset]

            def load(ds, t, _d=bank_dir, _tpl=template):
                return npy_load_bank(os.path.join(_d, _tpl.format(t=t)))

            align = OWN_GRID_ALIGN[name]
        arms[name] = own_grid_arm(dataset, geom, traverses, load, name, align)
    return arms


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(db_desc, db_xy, q_desc, q_xy, device, db_chunk, chunk, threshold=THRESHOLD_M):
    """-> ({k: recall}, n_database, n_queries, scorable).

    Streamed; never materialises [n_db, n_q]. ``scorable`` is the recall denominator —
    queries with no ground-truth positive anywhere in the gallery are discarded rather than
    counted as misses, and a single-traverse gallery is where the two can diverge.
    """
    gt = build_gt(db_xy, q_xy, threshold)
    ranked = bp.topk_ranked(db_desc, q_desc, device, k=max(KS), chunk=chunk,
                            db_chunk=db_chunk)
    rec, _, scorable = bp.recall_from_ranked(ranked, gt, ks=KS)
    del gt, ranked
    return rec, int(db_desc.shape[0]), int(q_desc.shape[0]), int(scorable.sum())


def score_cell(dataset, geom, files, arms, database, query, device):
    """{model: ({k: recall}, n_database, scorable)} for one (database, query) pairing.

    Outer loop is the pairing so the eventlab-grid GT is built once and shared: the geometry
    is model-independent, and a 25 m radius query costs more than the ranking does.
    """
    cfg = CONFIGS[dataset]
    out = {}
    q_xy, q_cov, _, _ = geom[query]
    q_xy = q_xy[q_cov]
    for name, _, _, _ in models_for(dataset, "bank"):
        db_desc, db_xy, _ = bp.pool_database(files[name], geom, database, name)
        q_bank = np.load(files[name][query][name], mmap_mode="r")
        q_desc = torch.from_numpy(np.asarray(q_bank[q_cov]))
        out[name] = score(db_desc, db_xy, q_desc, q_xy, device,
                          cfg["db_chunk"], cfg["chunk"])
        del db_desc, q_desc, q_bank
    for name, arm in arms.items():
        # Its own grid, so its own GT — the row counts differ from eventlab's by a frame or two.
        db_desc = torch.cat([arm[t][0] for t in database])
        db_xy = np.concatenate([arm[t][1] for t in database])
        chunk, db_chunk = OWN_GRID_CHUNK.get(name, (cfg["chunk"], cfg["db_chunk"]))
        out[name] = score(db_desc, db_xy, arm[query][0], arm[query][1], device,
                          db_chunk, chunk)
        del db_desc
    return out


# ---------------------------------------------------------------------------
# --verify: the pooled membership through this same code path
# ---------------------------------------------------------------------------
def dig(doc, keys):
    for key in keys:
        doc = doc[key]
    return doc


def verify(dataset, geom, files, arms, device):
    """Re-score the pooled configuration and check it against the cached ledger JSONs.

    This is the gate that says only the gallery membership changed: geometry, ground truth
    and the recall definition all have to still land on the published number. The pooled rows
    have sunset1/R0_FA0 as the *query*, so they certify the machinery, not the new orientation.
    """
    cfg = CONFIGS[dataset]
    print(f"\n--verify: pooled {cfg['pooled_query']} <- {cfg['pooled_database']}")
    got = score_cell(dataset, geom, files, arms, cfg["pooled_database"],
                     cfg["pooled_query"], device)
    print(f"\n  {'method':<20s}{'R@1':>9s}{'ref':>9s}{'R@10':>9s}{'ref':>9s}"
          f"{'db':>10s}{'scorable':>10s}  status")
    failures = []
    for name, _, _, refs in models_for(dataset):
        if name not in got:
            continue
        rec, n_db, n_q, scorable = got[name]
        path, keys = refs[dataset]
        try:
            with open(path) as handle:
                want = dig(json.load(handle), keys)
        except (FileNotFoundError, KeyError) as exc:
            print(f"  {name:<20s}{rec[1]:>9.4f}{'--':>9s}{rec[10]:>9.4f}{'--':>9s}"
                  f"{n_db:>10,}{scorable:>10,}  NO REFERENCE ({exc})")
            continue
        d1, d10 = abs(rec[1] - want["1"]), abs(rec[10] - want["10"])
        ok = max(d1, d10) < 5e-4
        failures += [] if ok else [f"{name}: R@1 {rec[1]:.4f} vs {want['1']:.4f}, "
                                   f"R@10 {rec[10]:.4f} vs {want['10']:.4f}"]
        print(f"  {name:<20s}{rec[1]:>9.4f}{want['1']:>9.4f}{rec[10]:>9.4f}"
              f"{want['10']:>9.4f}{n_db:>10,}{scorable:>10,}  "
              f"{'OK' if ok else f'MISMATCH (max {max(d1, d10):.2e})'}")
    if failures:
        raise SystemExit("\npooled reproduction FAILED:\n  " + "\n  ".join(failures))
    print("\n  pooled reproduction OK — geometry, GT and scoring unchanged")


# ---------------------------------------------------------------------------
def load_rerank(dataset):
    """{(reference, query): {"1":..,"10":..}} from the 0.1.0 re-rank driver, or {}.

    Keyed by the whole pair: with two references in play a bare query name is ambiguous, and
    the reverse cell's R0_RA0 would otherwise collide with nothing but could silently pick up
    a forward-referenced number if the roster ever grew.
    """
    try:
        with open(RERANK_JSON.format(ds=dataset)) as handle:
            doc = json.load(handle)
    except FileNotFoundError:
        return {}
    return {(cell["database"][0], cell["query"]): cell["recall"]
            for cell in doc.get("results", {}).values()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="brisbane_event", choices=sorted(CONFIGS))
    ap.add_argument("--threshold-m", type=float, default=THRESHOLD_M)
    ap.add_argument("--methods", nargs="+", default=None, metavar="NAME",
                    help="restrict the roster to these rows (default: every row with banks "
                         "for this dataset). Names as printed, e.g. --methods 'megaevent v8-B' "
                         "'Event-GeM 0.1.0'")
    ap.add_argument("--verify", action="store_true",
                    help="also re-score the pooled membership and check it against the "
                         "cached ledger JSONs before reporting the pairwise cells")
    ap.add_argument("--out-json", default=None,
                    help=f"default {V8}/pairwise_sunset_ref_<dataset>.json")
    cli = ap.parse_args()

    cfg = CONFIGS[cli.dataset]
    pairs = [tuple(pair) for pair in cfg["pairs"]]
    available = [name for name, _, _, _ in models_for(cli.dataset)]
    if cli.methods:
        unknown = [m for m in cli.methods if m not in available]
        if unknown:
            raise SystemExit(f"unknown method(s) {unknown}; this dataset has {available}")
        roster = [name for name in available if name in cli.methods]
    else:
        roster = available

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{cli.dataset}: " + ", ".join(f"{r} -> {q}" for r, q in pairs)
          + f"   {cli.threshold_m:g} m Euclidean, native, no PCA")
    print(f"  roster: {', '.join(roster)}")

    geom = geometry(cli.dataset)
    # --verify needs the pooled membership too, so load every traverse the run can touch.
    needed = sorted({t for pair in pairs for t in pair}
                    | ({cfg["pooled_query"], *cfg["pooled_database"]} if cli.verify else set()))
    files = bank_files(cli.dataset, needed)
    arms = own_grid_arms(cli.dataset, geom, needed)

    if cli.verify:
        verify(cli.dataset, geom, files, arms, device)

    cells = {}
    for ref, query in pairs:
        cells[(ref, query)] = score_cell(cli.dataset, geom, files, arms, [ref], query, device)
        print(f"  scored {ref} -> {query}", flush=True)

    rerank = load_rerank(cli.dataset)
    has_rerank = any(pair in rerank for pair in pairs)
    rows = roster + (["  +rerank"] if has_rerank else [])

    # When every cell shares one reference the column is just the query and the reference goes
    # in the title; with two references (NSAVP forward/reverse) each column names its own pair.
    shared_ref = pairs[0][0] if len({r for r, _ in pairs}) == 1 else None
    labels = [q if shared_ref else f"{r}->{q}" for r, q in pairs]
    # A mean across cells only means something when they share a gallery. Forward and reverse
    # are different galleries *and* different headings, so averaging them would invent a single
    # NSAVP score that no protocol produced.
    show_mean = len(pairs) > 1 and shared_ref is not None

    title = (f"reference = {shared_ref} (single traverse)" if shared_ref
             else "single-traverse reference per cell")
    print(f"\n\n{cli.dataset} — {title}, {cli.threshold_m:g} m, native\n")
    print(f"  {'method':<20s}" + "".join(f"{lab:>19s}" for lab in labels)
          + (f"{'mean':>19s}" if show_mean else ""))
    print(f"  {'':<20s}" + "".join(f"{'R@1':>10s}{'R@10':>9s}" for _ in labels)
          + (f"{'R@1':>10s}{'R@10':>9s}" if show_mean else ""))
    for name in rows:
        line = ""
        for pair in pairs:
            if name.strip() == "+rerank":
                got = rerank.get(pair)
                line += (f"{got['1']:>10.3f}{got['10']:>9.3f}" if got
                         else f"{'--':>10s}{'--':>9s}")
            else:
                rec = cells[pair][name][0]
                line += f"{rec[1]:>10.3f}{rec[10]:>9.3f}"
        if show_mean:
            if name.strip() == "+rerank":
                vals = [rerank[pair] for pair in pairs if pair in rerank]
                line += (f"{np.mean([v['1'] for v in vals]):>10.3f}"
                         f"{np.mean([v['10'] for v in vals]):>9.3f}"
                         if len(vals) == len(pairs) else f"{'--':>10s}{'--':>9s}")
            else:
                line += (f"{np.mean([cells[p][name][0][1] for p in pairs]):>10.3f}"
                         f"{np.mean([cells[p][name][0][10] for p in pairs]):>9.3f}")
        print(f"  {name:<20s}{line}")

    # Per-method sizes rather than one guard: Event-GeM's and EventVLAD's grids sit a frame or
    # two off eventlab's, and the cells do not share a gallery when the references differ.
    print("\n  gallery x scorable queries\n")
    print(f"  {'method':<20s}" + "".join(f"{lab:>19s}" for lab in labels))
    for name in roster:
        print(f"  {name:<20s}" + "".join(
            f"{cells[p][name][1]:>10,}{cells[p][name][3]:>9,}" for p in pairs))

    out = {"dataset": cli.dataset,
           "protocol": f"single reference traverse, {cli.threshold_m:g} m Euclidean, "
                       f"native cosine, no PCA",
           "pairs": [{"reference": r, "query": q} for r, q in pairs],
           "roster": roster, "threshold_m": cli.threshold_m,
           "note": "each gallery is one traverse, several times smaller than the pooled "
                   "protocol's — not comparable to any pooled cell. Cells with different "
                   "references are independent benchmarks and are not averaged.",
           "results": {}}
    for pair in pairs:
        ref, query = pair
        cell = {name: cells[pair][name] for name in roster}
        entry = {"query": query, "database": [ref],
                 "methods": {name: {"n_database": n_db, "n_queries": n_q,
                                    "scorable": scorable,
                                    "recall": {"native": {str(k): float(rec[k]) for k in KS}}}
                             for name, (rec, n_db, n_q, scorable) in cell.items()}}
        if pair in rerank:
            entry["methods"]["Event-GeM 0.1.0 +rerank"] = {
                "recall": {"native+rerank": rerank[pair]},
                "source": RERANK_JSON.format(ds=cli.dataset)}
        out["results"][f"{ref}->{query}"] = entry

    path = cli.out_json or f"{V8}/pairwise_sunset_ref_{cli.dataset}.json"
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(out, handle, indent=1)
    os.replace(tmp, path)
    print(f"\n-> {path}")
    missing = [f"{r}->{q}" for r, q in pairs if (r, q) not in rerank]
    if missing and "Event-GeM 0.1.0" in roster:
        print(f"   (+rerank not yet run for {', '.join(missing)} — "
              f"eventgem_010/pooled_rerank.py writes {RERANK_JSON.format(ds=cli.dataset)})")


if __name__ == "__main__":
    main()
