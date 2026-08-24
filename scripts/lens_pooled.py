"""LENS v2 on the pooled 25 m Brisbane protocol.

Run from the repository root::

    pixi run python -u scripts/lens_pooled.py --lens-quantise chip
    pixi run python -u scripts/lens_pooled.py --lens-quantise fp32
    pixi run python -u scripts/lens_pooled.py --lens-model /path/to/step10000.pt

Geometry, pooling, whitening, ranking and the 25 m ground truth are all
``scripts/brisbane_pooled.py``'s, reached the same way ``eventgem_pooled.py``,
``eventvlad_pooled.py`` and ``spikevpr_pooled.py`` reach them — so LENS's gallery, GPS
coverage mask, positives and background-activity filtering are identical to theirs, and a
difference in a number is a difference in the method.

What is specific to LENS is only the frame and the network: each 50 ms slice is rebinned in
the event domain to a ``(2, 128, 128)`` ON/OFF **count** frame and pushed through a
111,694-parameter spiking conv net, in LENS's own pixi environment (see
:mod:`src.lens_bridge`). One descriptor lands on one slice and the pose grid is used
unshifted, so no ``centre_geometry`` analogue is needed.

**This is LENS's out-of-distribution end on density, and it still does well — which is
worth understanding before quoting either number.** LENS trains on I2E frames at ~48
events/px on its own 128x128 grid; a Brisbane 50 ms slice renders 3.3 there, 15x sparser.
Tokyo 24/7 renders 37-45 and is in distribution. That is the mirror image of SpikeVPR's
exposure, so the two SNNs here are each strong where the other is starved. The pooled
protocol is also **cross-condition** by construction (sunset1 against
daytime/morning/night/sunrise), the axis LENS is documented weak on: 69.4 same-condition
against 19.2 daytime and 15.2 morning.

That reasoning predicted a low number, and the measurement refuted it: `lens_v2_best` int8
scores **native R@1 .6188 / R@10 .8354**, whitened **.7380 / .8987**. The prediction was
wrong because it compared across protocols. This one hands each query a mean of **427**
positives out of 54,620 (0.78% density) where the stride-10 pairwise suite gives ~22, so
it is a far more forgiving question and *every* method scores higher on it — megaevent's
ViT-B gets .7463 here against .475 on daytime pairwise. A pooled number and a pairwise
number are not comparable, in either direction.

The internal signature is consistent with the known weakness even so: `night` supplies only
**3.9%** of top-1 matches while holding 24.9% of the gallery, with morning (46.9%) and
sunrise (40.5%) carrying it.

``--lens-quantise`` picks the network. ``chip`` is the int8 ``DynapcnnNetwork`` that
deploys, and on Brisbane it *beats* fp32 (sunset1 R@1 60.8 -> 67.1); they are different
models, so each writes its own bank, its own tag and its own JSON.
"""

import argparse
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
import eventgem_pooled as egp  # noqa: E402
import nsavp_pooled as npl  # noqa: E402
from src import inference as inf  # noqa: E402
from src import lens_bridge as bridge  # noqa: E402
from src import traversenpz as tnpz  # noqa: E402

DEFAULT_OUT = "/media/adam/vprdatasets/megaevent/lens_pooled"
# LENS descriptors are 1024-D, so megaevent's 4096/2048 defaults do not exist here. 1024 is
# a full rotation plus whitening; 512 is the halved basis every method is also reported at.
# 0.5 is the power every other method in this benchmark is reported under.
PCA_SETTINGS = ((1024, 0.5), (512, 0.5))
DT_MS = 50
FILTER_DT_US = 50000            # microseconds — eventcv's raw timestamp unit, not ms
DESC_DIM = bridge.DESC_DIM

DATASETS = {
    "brisbane_event": {
        "eventlab_dir": egp.DATASETS["brisbane_event"][0],
        "query": egp.DATASETS["brisbane_event"][1],
        "database": egp.DATASETS["brisbane_event"][2],
        "npz_root": egp.DATASETS["brisbane_event"][3],
        "db_chunk": None,
    },
    "nsavp": {
        "eventlab_dir": egp.DATASETS["nsavp"][0],
        "query": egp.DATASETS["nsavp"][1],
        "database": egp.DATASETS["nsavp"][2],
        "npz_root": None,
        # NSAVP's pooled gallery is ~101k descriptors; the database streams past the
        # ranker in slices so an 8 GB card keeps room for the similarity block.
        "db_chunk": npl.DB_CHUNK,
    },
}


def n_slices(dataset, traverse, eventlab_dir):
    """How many ``dt_ms`` slices the recording yields, from the parent environment.

    Read here rather than trusted from the child so the descriptor cache can be validated
    and the pose grid cross-checked *before* an extraction that costs minutes.
    """
    import eventcv as ecv

    from types import SimpleNamespace

    args = SimpleNamespace(eventlab_dir=eventlab_dir, dataset=dataset)
    reader = ecv.open(inf.sequence_path(args, traverse), dt_ms=DT_MS,
                      sensor_size=inf.sensor_size(dataset), hot_pixel_filter=True,
                      offset=egp.traverse_offset_ms(dataset, traverse))
    return int(reader.background_activity_filter(FILTER_DT_US).n_slices)


def extract(cli, dataset, traverse, checkpoint, out_dir, tag):
    """One traverse's descriptor bank, reused if it is already on disk. -> ``({label: path}, n)``"""
    path = os.path.join(out_dir, f"{tag}_{traverse}.npy")
    n = n_slices(dataset, traverse, cli.eventlab_dir)
    if os.path.exists(path):
        bank = np.load(path, mmap_mode="r")
        if bank.shape == (n, DESC_DIM):
            print(f"    {tag} {traverse}: {n} slices — bank cached, skipping")
            return {cli.label: path}, n
        raise ValueError(f"{path}: cached shape {bank.shape}, expected {(n, DESC_DIM)}")

    job = bridge.traverse_job(
        inf.sequence_path(_Args(cli.eventlab_dir, dataset, DT_MS, False, False), traverse),
        sensor=inf.sensor_size(dataset), dt_ms=DT_MS,
        offset_ms=egp.traverse_offset_ms(dataset, traverse), hot_pixel=True,
        filter_dt_us=FILTER_DT_US, checkpoint=checkpoint, out=path,
        quantise=cli.lens_quantise, batch_size=cli.lens_batch_size,
        workers=cli.workers, lens_repo=cli.lens_repo, label=f"{dataset}/{traverse}")
    bank = bridge.run(job, cli.lens_repo, keep_bank=True)
    if bank.shape != (n, DESC_DIM):
        raise SystemExit(f"{traverse}: bank is {bank.shape}, expected {(n, DESC_DIM)}")
    return {cli.label: path}, n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="brisbane_event", choices=sorted(DATASETS))
    parser.add_argument("--lens-model", default="v2_best",
                        help="a named LENS checkpoint ('v2_best') or a path to one")
    parser.add_argument("--lens-repo", default=bridge.DEFAULT_LENS_REPO)
    parser.add_argument("--lens-quantise", default="chip", choices=["fp32", "chip"],
                        help="int8 chip_sim (what deploys, and the better model on "
                             "Brisbane) or the fp32 weights. Different models — each "
                             "writes its own bank and its own JSON.")
    parser.add_argument("--lens-batch-size", type=int, default=bridge.BATCH_SIZE,
                        help="pinned and part of the tag; sinabs makes the batch "
                             "dimension visible to the network")
    parser.add_argument("--workers", type=int, default=bridge.WORKERS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--query", default=None,
                        help="override the dataset's default query traverse")
    parser.add_argument("--database", nargs="+", default=None,
                        help="override the pooled database. The headline run excludes "
                             "sunset2: it shares lighting with the sunset1 query and "
                             "would carry the gallery.")
    parser.add_argument("--threshold-m", type=float, default=25.0)
    parser.add_argument("--thresholds-m", type=float, nargs="+",
                        default=[25.0, 50.0, 75.0, 100.0])
    parser.add_argument("--score-chunk", type=int, default=256)
    cli = parser.parse_args()

    spec = DATASETS[cli.dataset]
    cli.eventlab_dir = spec["eventlab_dir"]
    cli.query = cli.query or spec["query"]
    cli.database = list(cli.database or spec["database"])
    cli.pca = list(PCA_SETTINGS)
    cli.db_chunk = spec["db_chunk"]
    cli.npz_root = spec["npz_root"]
    cli.source = "real"
    cli.aps_aligned = False
    cli.max_events = None
    if cli.threshold_m not in cli.thresholds_m:
        cli.thresholds_m = sorted([cli.threshold_m, *cli.thresholds_m])
    os.makedirs(cli.out_dir, exist_ok=True)

    checkpoint = bridge.resolve_checkpoint(cli.lens_model, cli.lens_repo)
    digest = bridge.checkpoint_sha256(checkpoint)
    provenance = bridge.describe(checkpoint)
    stem = os.path.splitext(os.path.basename(checkpoint))[0]
    tag = f"lens_{stem}_{digest[:10]}_{cli.lens_quantise}_ba{DT_MS}"
    cli.fig_tag = f"{cli.dataset}_{tag}"
    cli.label = f"lens_{stem}_{cli.lens_quantise}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _Args(cli.eventlab_dir, cli.dataset, DT_MS, False, False)
    sequences = [cli.query, *cli.database]

    print(f"lens/{cli.dataset}: query {cli.query} -> database {'+'.join(cli.database)}")
    print(f"  checkpoint {os.path.basename(checkpoint)} (sha256 {digest[:10]}, "
          f"step {provenance['step']}, ann={provenance['ann']}, "
          f"spike_threshold={provenance['spike_threshold']}, {cli.lens_quantise})")
    print(f"  {cli.threshold_m:g} m radius, {DT_MS} ms ON/OFF count frames at "
          f"{bridge.INPUT_SHAPE[1]}x{bridge.INPUT_SHAPE[2]}, hot-pixel on, background "
          f"activity dt={FILTER_DT_US} us, device {device}")

    # NSAVP carries its own pose files and has no NMEA/APS clock story, so its geometry
    # comes from `nsavp_pooled`; Brisbane's comes from the NMEA track. Same `build_gt`
    # downstream either way, which is what keeps the two protocols comparable.
    if cli.dataset == "nsavp":
        geom = npl.traverse_geometry(os.path.join(cli.eventlab_dir, cli.dataset),
                                     sequences, DT_MS)
    else:
        geom = bp.traverse_geometry(args, sequences, spec["npz_root"], aps_align=False)

    bank_files = {}
    for sequence in sequences:
        files, n = extract(cli, cli.dataset, sequence, checkpoint, cli.out_dir, tag)
        if n != len(geom[sequence][0]):
            raise SystemExit(
                f"{sequence}: eventcv yields {n} slices but the pose grid has "
                f"{len(geom[sequence][0])} — every coordinate after the first would be "
                f"attached to the wrong frame")
        bank_files[sequence] = files

    result = bp.score_configuration(bank_files, geom, args, cli, cli.label, device, tag)
    result.update({
        "method": "lens", "source": "real", "filter_arm": "on",
        "event_filter_dt_us": FILTER_DT_US,
        "representation": (f"ON/OFF event counts at {bridge.INPUT_SHAPE[1]}x"
                           f"{bridge.INPUT_SHAPE[2]}"),
        "event_window": "full slice",
        "checkpoint": checkpoint, "checkpoint_sha256": digest,
        "quantise": cli.lens_quantise, "batch_size": cli.lens_batch_size,
        "descriptor_dim": DESC_DIM, "parameters": 111694, **provenance,
    })

    output = {"dataset": cli.dataset, "method": "lens", "checkpoint": checkpoint,
              "quantise": cli.lens_quantise, "query": cli.query,
              "database": cli.database, "threshold_m": cli.threshold_m, "dt_ms": DT_MS,
              "pca": [list(setting) for setting in PCA_SETTINGS], "hot_pixel": True,
              "results": {tag: result}}
    out_json = os.path.join(cli.out_dir, f"{cli.dataset}_{stem}_{cli.lens_quantise}.json")
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
