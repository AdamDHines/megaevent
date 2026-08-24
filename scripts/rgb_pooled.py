"""The RGB controls — MegaLoc, SALAD, CricaVPR and MixVPR — on the pooled Brisbane / NSAVP 25 m.

Run from the repository root::

    pixi run python -u scripts/rgb_pooled.py --model megaloc  --dataset brisbane_event
    pixi run python -u scripts/rgb_pooled.py --model salad    --dataset nsavp
    pixi run python -u scripts/rgb_pooled.py --model mixvpr   --dataset brisbane_event
    pixi run python -u scripts/rgb_pooled.py --model cricavpr --dataset nsavp

Geometry, pooling, ranking and the 25 m ground truth are all ``scripts/brisbane_pooled.py``'s
and ``scripts/nsavp_pooled.py``'s, reached the same way ``eventgem_pooled.py``,
``eventvlad_pooled.py``, ``spikevpr_pooled.py`` and ``lens_pooled.py`` reach them. The frames
are the *same* ones megaevent sees, from the same call: ``brisbane_resolution.extract`` over
``inf.EventStreamDataset``, ``--representation`` (countmask or accumulate), hot-pixel on,
background-activity filter at 50,000 us. Only three things differ from a megaevent run — the
normalisation constants (ImageNet, not the trained ones), the resize each model publishes,
and the network.

``--representation accumulate`` is the arm that matches the v8 megaevent models, which train
on the GEPT-native white-background render rather than countmask. Its banks and JSON carry an
``_accum`` tag, so the countmask arm on disk is never touched or reused.

**What this measures.** None of these models has ever seen an event. They are the control for
the question the event baselines cannot answer: not "is megaevent the better event method", but
"was training on events worth doing at all", given that a countmask frame is still an image and
an RGB retrieval model may simply read it. They answer it at four capacities:

* ``salad`` is megaevent's architecture exactly — DINOv2 ViT-B/14, SALAD 64x128+256, 8448-d,
  322 — at 88.0M parameters against megaevent ViT-B's 88.0M. The gap to it is what the event
  fine-tuning bought, with everything else held fixed.
* ``megaloc`` is 228.6M: the same ViT-B backbone under a 140.6M-parameter learned compression
  head, trained far beyond GSV-Cities. The gap to it is what a reviewer will ask about.
* ``cricavpr`` is 106.8M and a third reading of the same ViT-B backbone: 14 multi-scale GeM
  region tokens through a cross-image transformer encoder, 10752-d at 224.
* ``mixvpr`` is 10.9M and the only one with no transformer in it at all — a ResNet50 truncated
  after ``layer3`` feeding four feature-mixer MLPs, 4096-d at 320. It is the control for
  whether a DINOv2 patch backbone is doing the reading, or whether any decent RGB retrieval
  model would.

**The four do not share a resize, and that is deliberate.** ``megaloc`` and ``salad`` are at
322, ``mixvpr`` at 320 and ``cricavpr`` at 224 — each model's own published evaluation size,
and for ``cricavpr`` a hard structural requirement (``src/methods.py::CricaVPRMethod``). So the
megaloc-vs-salad difference is still the network alone, but a comparison across all four is a
comparison of published configurations, which is what ``scripts/table_native.py`` reports and
what its docstring already says of Event-GeM's 240x320.

``cricavpr`` also carries a fixed batch size of 16, because its encoder attends over the batch
axis and its descriptors therefore depend on which frames shared a batch. Slice order fixes
that, so a rerun reproduces a run; ``--batch-size`` overriding it changes the numbers.

Note that these two datasets are *real* DAVIS / Prophesee recordings, where Tokyo 24/7,
Pitts250k and MSLS are I2E simulations of photographs and therefore much closer to what these
models were trained on.

Whitening is off by default (``--pca`` to add it back). It is not part of what is being
compared here, and a space fit on one model's bank and not another's is a tuning step rather
than a measurement.
"""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src import inference as inf  # noqa: E402
from src import vprbench  # noqa: E402
from src.methods import (  # noqa: E402
    CRICAVPR_BATCH, CRICAVPR_DESC_DIM, CRICAVPR_HUB, CRICAVPR_RESOLUTION,
    MEGALOC_DESC_DIM, MEGALOC_HUB, MEGALOC_RESOLUTION,
    MIXVPR_CKPT, MIXVPR_DESC_DIM, MIXVPR_RESOLUTION,
    SALAD_DESC_DIM, SALAD_HUB,
    build_cricavpr, build_dino_salad, build_mixvpr, megaloc_transform,
)
from brisbane_resolution import _Args, extract  # noqa: E402
import brisbane_pooled as bp  # noqa: E402
import eventgem_pooled as egp  # noqa: E402
import nsavp_pooled as npl  # noqa: E402
from tokyo_trajectory import parse_pca  # noqa: E402

ROOT = "/media/adam/vprdatasets/megaevent"
DT_MS = 50
FILTER_DT_US = 50000            # microseconds — eventcv's raw timestamp unit, not ms


def _load_megaloc(device):
    # trust_repo: the checkout is already in the hub cache, and a prompt would hang a
    # backgrounded run. source="github" keeps it resolving the same way upstream does.
    return torch.hub.load(MEGALOC_HUB, "get_trained_model", source="github",
                          trust_repo=True).eval().to(device)


def _load_salad(device):
    return build_dino_salad().eval().to(device)


def _load_mixvpr(device):
    return build_mixvpr().eval().to(device)


def _load_cricavpr(device):
    return build_cricavpr().eval().to(device)


def _load_boq(device):
    return vprbench.build_boq().eval().to(device)


def _load_qaa(device):
    return vprbench.build_qaa().eval().to(device)


def _load_supervlad(device):
    return vprbench.build_supervlad("SuperVLAD").eval().to(device)


# name -> (loader, descriptor width, provenance, trained-on, native resolution, batch size)
# The last two are each model's *own* published evaluation configuration, not this script's:
# see the src.methods class docstrings for why 224 is structural for cricavpr, why 320 is
# structural for mixvpr, and why cricavpr's batch size is part of the method rather than a
# memory knob. None means "take the CLI default".
MODELS = {
    "megaloc": (_load_megaloc, MEGALOC_DESC_DIM, MEGALOC_HUB, "RGB images, no event data",
                MEGALOC_RESOLUTION, None),
    "salad": (_load_salad, SALAD_DESC_DIM, SALAD_HUB, "GSV-Cities RGB, no event data",
              MEGALOC_RESOLUTION, None),
    "mixvpr": (_load_mixvpr, MIXVPR_DESC_DIM, MIXVPR_CKPT, "GSV-Cities RGB, no event data",
               MIXVPR_RESOLUTION, None),
    "cricavpr": (_load_cricavpr, CRICAVPR_DESC_DIM, CRICAVPR_HUB,
                 "GSV-Cities RGB, no event data", CRICAVPR_RESOLUTION, CRICAVPR_BATCH),
    # Borrowed from the VPR-methods-evaluation checkout (src/vprbench.py). All three are
    # DINOv2 at 322 like salad and megaloc, so they extend the ladder without changing the
    # resize: boq 12288-d cross-attends 64 learned queries, qaa is the most recent method in
    # that harness, and supervlad is the compact 3072-d point.
    "boq": (_load_boq, vprbench.BOQ_DESC_DIM, vprbench.BOQ_SRC,
            "GSV-Cities RGB, no event data", vprbench.BOQ_RESOLUTION, None),
    "qaa": (_load_qaa, vprbench.QAA_DESC_DIM, vprbench.QAA_SRC,
            "GSV-Cities RGB, no event data", vprbench.QAA_RESOLUTION, None),
    "supervlad": (_load_supervlad, vprbench.SUPERVLAD_DESC_DIM, vprbench.SUPERVLAD_SRC,
                  "GSV-Cities RGB, no event data", vprbench.SUPERVLAD_RESOLUTION, None),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="megaloc", choices=sorted(MODELS),
                        help="which RGB control to run; see the module docstring for what "
                             "each one is a control for")
    parser.add_argument("--dataset", default="brisbane_event", choices=sorted(egp.DATASETS))
    parser.add_argument("--resolution", type=int, default=None,
                        help="square input size; defaults to the chosen model's own "
                             "evaluation size (322 for megaloc and salad, which is also "
                             "the size every megaevent traverse result here uses; 320 for "
                             "mixvpr; 224 for cricavpr, where it is structural rather than "
                             "a choice — see src/methods.py::CricaVPRMethod)")
    parser.add_argument("--out-dir", default=None,
                        help=f"default {ROOT}/<model>_pooled")
    parser.add_argument("--query", default=None,
                        help="override the dataset's default query traverse")
    parser.add_argument("--database", nargs="+", default=None,
                        help="override the pooled database. Brisbane's default excludes "
                             "sunset2: it shares lighting with the sunset1 query and would "
                             "carry the gallery.")
    parser.add_argument("--pca", nargs="+", metavar="DIM,POWER", default=None,
                        help="whitening settings to report beside the native descriptor. "
                             "Off by default — see the module docstring.")
    parser.add_argument("--threshold-m", type=float, default=25.0)
    parser.add_argument("--thresholds-m", type=float, nargs="+",
                        default=[25.0, 50.0, 75.0, 100.0])
    parser.add_argument("--score-chunk", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"default {inf.BATCH_SIZE}, except cricavpr, whose "
                             f"cross-image encoder attends over the batch — there it is "
                             f"part of the method and defaults to upstream's "
                             f"{CRICAVPR_BATCH}")
    parser.add_argument("--representation", default="countmask",
                        choices=["countmask", "accumulate"],
                        help="which frame the control reads. 'countmask' is the black-bg "
                             "eventcv render every published RGB-control number uses. "
                             "'accumulate' is the GEPT-native white-bg render the v8 "
                             "megaevent models train on (src/npzdata.py::accumulate_numpy, "
                             "rendered locally — eventcv has no accumulate). Banks and the "
                             "output JSON carry an _accum tag so the arms never collide.")
    parser.add_argument("--norm-stats", default="imagenet",
                        choices=["imagenet", "countmask"],
                        help="normalisation constants for the control. 'imagenet' is each "
                             "model's own shipped preprocessing (the default and the "
                             "published protocol). 'countmask' swaps in megaevent's "
                             "event-training statistics — the fairness arm answering "
                             "whether ImageNet stats handicap the controls on frames "
                             "whose true channel means are ~0.10/0.28/0.10. Banks and the "
                             "output JSON carry a _cmstats tag so the arms never collide.")
    parser.add_argument("--workers", type=int, default=inf.NUM_WORKERS)
    cli = parser.parse_args()

    label = cli.model
    loader, desc_dim, hub, trained_on, resolution, batch_size = MODELS[cli.model]
    if cli.resolution is None:
        cli.resolution = resolution
    if cli.batch_size is None:
        cli.batch_size = batch_size or inf.BATCH_SIZE
    eventlab_dir, query, database, npz_root = egp.DATASETS[cli.dataset]
    cli.eventlab_dir = eventlab_dir
    cli.query = cli.query or query
    cli.database = list(cli.database or database)
    cli.npz_root = npz_root
    cli.pca = parse_pca(cli.pca, ())
    cli.db_chunk = npl.DB_CHUNK if cli.dataset == "nsavp" else None
    cli.source = "real"
    cli.aps_aligned = False
    cli.label = label
    cli.out_dir = cli.out_dir or os.path.join(ROOT, f"{cli.model}_pooled")
    if cli.threshold_m not in cli.thresholds_m:
        cli.thresholds_m = sorted([cli.threshold_m, *cli.thresholds_m])
    os.makedirs(cli.out_dir, exist_ok=True)

    tag = f"r{cli.resolution}ba{DT_MS}"
    # extract() skips a traverse whose bank file already exists, so a representation that
    # did not change the tag would silently score against the countmask banks on disk.
    if cli.representation != "countmask":
        tag += "_accum"
    if cli.norm_stats != "imagenet":
        tag += "_cmstats"
    cli.fig_tag = f"{cli.dataset}_{tag}_{label}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences = [cli.query, *cli.database]

    # no_event_filter=False, so _Args derives filter_dt_us = dt_ms * 1000 = 50,000 us: the
    # `ba50` arm every pooled result in this repo is reported under.
    args = _Args(cli.eventlab_dir, cli.dataset, DT_MS, False, False, npz_root=npz_root)
    args.representation = cli.representation

    print(f"{label}/{cli.dataset}: query {cli.query} -> database {'+'.join(cli.database)}")
    print(f"  {cli.threshold_m:g} m radius, dt {DT_MS} ms, {cli.representation} at "
          f"{cli.resolution}x{cli.resolution}, hot-pixel on, background activity "
          f"dt={FILTER_DT_US} us, device {device}")

    model = loader(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  {label} ({hub}, {params / 1e6:.1f}M params, {desc_dim}-d), "
          f"ImageNet normalisation")

    bp.assert_filter_active(args, cli.query, FILTER_DT_US)

    # NSAVP carries its own pose files; Brisbane's geometry comes from the NMEA track. Same
    # build_gt downstream either way, which is what keeps the two protocols comparable.
    if cli.dataset == "nsavp":
        root = os.path.join(cli.eventlab_dir, cli.dataset)
        geom = npl.traverse_geometry(root, sequences, DT_MS)
    else:
        geom = bp.traverse_geometry(args, sequences, npz_root, aps_align=False)

    loaded = [(label, 0, model)]
    transform = megaloc_transform(cli.resolution, stats=cli.norm_stats)
    bank_files, bank_frames = {}, {}
    # cricavpr's descriptors depend on which frames shared a batch, so halving the batch to
    # survive a busy card would change the measurement rather than its footprint. Everywhere
    # else the batch is a pure memory knob and the retry is free.
    oom_backoff = cli.model != "cricavpr"
    for sequence in sequences:
        files, n = extract(loaded, transform, sequence, args, cli.out_dir, tag, device,
                           cli.batch_size, cli.workers, oom_backoff=oom_backoff)
        bank_files[sequence], bank_frames[sequence] = files, n
    del model, loaded
    torch.cuda.empty_cache()

    # The bank row count is only knowable once eventcv has framed the recording, so the grid
    # is re-derived against it rather than trusted — a mismatch would attach every coordinate
    # after the first to the wrong frame.
    if cli.dataset == "nsavp":
        geom = npl.traverse_geometry(root, sequences, DT_MS, bank_frames)
    else:
        for sequence in sequences:
            if bank_frames[sequence] != len(geom[sequence][0]):
                raise SystemExit(
                    f"{sequence}: eventcv yields {bank_frames[sequence]} slices but the pose "
                    f"grid has {len(geom[sequence][0])}")

    result = bp.score_configuration(bank_files, geom, args, cli, label, device, tag)
    result.update({
        "method": cli.model, "source": "real", "filter_arm": "on",
        "event_filter_dt_us": FILTER_DT_US, "representation": cli.representation,
        "resolution": cli.resolution, "hub": hub,
        "descriptor_dim": desc_dim, "parameters": int(params),
        "trained_on": trained_on, "label": label, "norm_stats": cli.norm_stats,
        # Recorded for every model, but it is only *load-bearing* for cricavpr, whose
        # cross-image encoder attends over the batch: there a different batch size is a
        # different measurement, not a different memory footprint.
        "batch_size": int(cli.batch_size), "cross_image_batch": cli.model == "cricavpr",
    })

    output = {"dataset": cli.dataset, "method": cli.model, "hub": hub,
              "query": cli.query, "database": cli.database,
              "threshold_m": cli.threshold_m, "dt_ms": DT_MS,
              "pca": [list(setting) for setting in cli.pca], "hot_pixel": True,
              "results": {tag: result}}
    suffix = "" if cli.representation == "countmask" else "_accumulate"
    suffix += "" if cli.norm_stats == "imagenet" else "_cmstats"
    out_json = os.path.join(cli.out_dir, f"{cli.dataset}{suffix}.json")
    tmp = out_json + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(output, handle, indent=2)
    os.replace(tmp, out_json)                   # a reader never sees a half-written file

    print(f"\nR@1 / R@5 / R@10 / R@20 @ {cli.threshold_m:g} m")
    for name, recall in result["recall"].items():
        print(f"  {name:16s} " + "  ".join(f"{recall[str(k)]:.4f}" for k in bp.KS))
    print(f"\n-> {out_json}")


if __name__ == "__main__":
    main()
