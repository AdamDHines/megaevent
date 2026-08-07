"""SpikeVPR on the pooled 25 m Brisbane and NSAVP protocols.

Run from the repository root::

    pixi run python -u scripts/spikevpr_pooled.py --dataset brisbane_event --model nsavp
    pixi run python -u scripts/spikevpr_pooled.py --dataset nsavp --model brisbane

The cross-dataset pairing above is the point: each traverse dataset is scored with the
checkpoint trained on the *other* one, so neither number can be explained by the model
having seen that route.

Geometry, pooling, whitening, ranking and the 25 m ground truth are all
``scripts/brisbane_pooled.py``'s, reached the same way ``eventgem_pooled.py`` and
``eventvlad_pooled.py`` reach them — so SpikeVPR's gallery, GPS coverage mask, positives and
background-activity filtering are identical to theirs, and a difference in a number is a
difference in the method.

What is specific to SpikeVPR is only the frame: each 50 ms slice is rendered to a
``[2, 260, 346]`` ON/OFF count frame by :func:`src.npzdata.onoff_from_stream`, and the
forward pass runs in ``envs/spikevpr`` (see :mod:`src.spikevpr_bridge`). Unlike EventVLAD
there is no temporal triplet, so one descriptor lands on one slice and the pose grid is used
unshifted — no ``centre_geometry`` analogue is needed.

Measured against SpikeVPR's own training density, both datasets are in distribution here: a
Brisbane 50 ms slice renders a median 0.278 events/px against the 0.167 its Brisbane
checkpoint was trained on, and an NSAVP slice 1.878 against the ~1.24 of its 33 ms eval
window — both inside their own traverse's p10-p90 spread. The image sets are the ones that
are not (Tokyo 24/7 is ~43x), which is why ``--max-events`` exists but is not used here.
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
from src import spikevpr_bridge as bridge  # noqa: E402
from src import traversenpz as tnpz  # noqa: E402

DEFAULT_OUT = "/media/adam/vprdatasets/megaevent/spikevpr_pooled"
# 4096-D descriptors, so both bases are real — unlike eventgem's 128-D case, where the two
# settings differ only in the exponent. 0.5 is the setting every other method is reported
# under here; 2048 is the dimension src.inference.pca_fit defaults to.
PCA_SETTINGS = ((4096, 0.5), (2048, 0.5))
DT_MS = 50
FILTER_DT_US = 50000            # microseconds — eventcv's raw timestamp unit, not ms
DESC_DIM = bridge.OUT_CHANNELS * bridge.OUT_ROWS

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
        # NSAVP's pooled gallery is ~101k x 4096; the database streams past the ranker in
        # slices so an 8 GB card keeps room for the similarity block.
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


def extract(cli, dataset, traverse, checkpoint, neuron, out_dir, tag):
    """One traverse's descriptor bank, reused if it is already on disk. -> ``({label: path}, n)``"""
    path = os.path.join(out_dir, f"{tag}_{traverse}.npy")
    # A Sim2Real arm has no recording to slice: its events are I2E micro-saccades over the
    # co-recorded APS frames, one .npz per slice. The bridge already carries an npz mode for
    # image sets, and one row per .npz in sorted order is exactly the traverse contract.
    synthetic = getattr(cli, "source", "real") != "real"
    npz_paths = (tnpz.frame_paths(cli.npz_root, dataset, cli.source, traverse)
                 if synthetic else None)
    n = len(npz_paths) if synthetic else n_slices(dataset, traverse, cli.eventlab_dir)
    if os.path.exists(path):
        bank = np.load(path, mmap_mode="r")
        if bank.shape == (n, DESC_DIM):
            print(f"    {tag} {traverse}: {n} slices — bank cached, skipping")
            return {cli.label: path}, n
        raise ValueError(f"{path}: cached shape {bank.shape}, expected {(n, DESC_DIM)}")

    common = dict(checkpoint=checkpoint, neuron=neuron, out=path,
                  max_events=cli.max_events, spikevpr_repo=cli.spikevpr_repo,
                  label=f"{dataset}/{traverse}")
    if synthetic:
        job = bridge.npz_job(npz_paths, **common)
    else:
        job = bridge.traverse_job(
            inf.sequence_path(_Args(cli.eventlab_dir, dataset, DT_MS, False, False), traverse),
            sensor=inf.sensor_size(dataset), dt_ms=DT_MS,
            offset_ms=egp.traverse_offset_ms(dataset, traverse), hot_pixel=True,
            filter_dt_us=FILTER_DT_US, **common)
    bank = bridge.run(job, cli.spikevpr_env, keep_bank=True)
    if bank.shape != (n, DESC_DIM):
        raise SystemExit(f"{traverse}: bank is {bank.shape}, expected {(n, DESC_DIM)}")
    return {cli.label: path}, n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--model", required=True, choices=sorted(bridge.SPIKEVPR_CHECKPOINTS),
                        help="which released SpikeVPR checkpoint to score with. The "
                             "cross-dataset pairing is brisbane_event<-nsavp and "
                             "nsavp<-brisbane.")
    parser.add_argument("--spikevpr-repo", default=bridge.DEFAULT_REPO)
    parser.add_argument("--spikevpr-env", default=bridge.DEFAULT_ENV)
    parser.add_argument("--max-events", type=int, default=None,
                        help="render each slice from only its first N events. Off by "
                             "default and not needed here — see the module docstring.")
    parser.add_argument("--aps-aligned", action="store_true",
                      help="drop frames with no APS frame within half a slice. "
                           "Implied by a non-real --source; pass it on `real` so "
                           "both arms of a Sim2Real pair score the same rows.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--query", default=None,
                        help="override the dataset's default query traverse")
    parser.add_argument("--database", nargs="+", default=None,
                        help="override the pooled database. The Sim2Real arms drop "
                             "`night`: its APS ran at 1.7 Hz against the 20 Hz slice "
                             "grid, so 91%% of its slices have no frame within half a "
                             "slice and its synthetic events would be saccades over "
                             "stale duplicates.")
    parser.add_argument("--threshold-m", type=float, default=25.0)
    parser.add_argument("--thresholds-m", type=float, nargs="+",
                        default=[25.0, 50.0, 75.0, 100.0])
    parser.add_argument("--score-chunk", type=int, default=256)
    parser.add_argument("--source", default="real", choices=list(tnpz.SOURCES),
                        help="which arm the events come from; see "
                             "scripts/brisbane_pooled.py")
    cli = parser.parse_args()

    spec = DATASETS[cli.dataset]
    cli.eventlab_dir = spec["eventlab_dir"]
    cli.query = cli.query or spec["query"]
    cli.database = list(cli.database or spec["database"])
    cli.pca = list(PCA_SETTINGS)
    cli.db_chunk = spec["db_chunk"]
    cli.npz_root = spec["npz_root"]
    synthetic = cli.source != "real"
    cli.aps_aligned = cli.aps_aligned or synthetic
    if synthetic and cli.dataset != "brisbane_event":
        raise SystemExit(f"--source {cli.source} exists only for brisbane_event")
    if cli.threshold_m not in cli.thresholds_m:
        cli.thresholds_m = sorted([cli.threshold_m, *cli.thresholds_m])
    os.makedirs(cli.out_dir, exist_ok=True)

    checkpoint, neuron = bridge.resolve_checkpoint(cli.model, cli.spikevpr_repo)
    digest = bridge.checkpoint_sha256(checkpoint)
    cap = f"_e{cli.max_events}" if cli.max_events else ""
    tag = (f"spikevpr_{cli.model}_ba{DT_MS}{cap}" if not synthetic
           else f"spikevpr_{cli.model}_{cli.source}{cap}")
    # The result key and the figure filename are the same thing for a one-dataset run, but
    # the two datasets write into one directory, so the figures need the dataset too.
    cli.fig_tag = f"{cli.dataset}_{tag}"
    cli.label = f"spikevpr_{cli.model}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _Args(cli.eventlab_dir, cli.dataset, DT_MS, False, False)
    sequences = [cli.query, *cli.database]

    print(f"spikevpr/{cli.dataset}: query {cli.query} -> database {'+'.join(cli.database)}")
    print(f"  checkpoint {os.path.basename(checkpoint)} (trained on {cli.model}, "
          f"MixVPR {neuron}, sha256 {digest[:10]})")
    print(f"  {cli.threshold_m:g} m radius, {DT_MS} ms ON/OFF count frames at "
          f"{bridge.GRID[0]}x{bridge.GRID[1]}, hot-pixel on, background activity "
          f"dt={FILTER_DT_US} us, device {device}")

    if cli.dataset == "nsavp":
        geom = npl.traverse_geometry(os.path.join(cli.eventlab_dir, cli.dataset),
                                     sequences, DT_MS)
    else:
        geom = bp.traverse_geometry(args, sequences, spec["npz_root"],
                                    aps_align=(synthetic or cli.aps_aligned))

    bank_files = {}
    for sequence in sequences:
        files, n = extract(cli, cli.dataset, sequence, checkpoint, neuron, cli.out_dir, tag)
        if n != len(geom[sequence][0]):
            raise SystemExit(
                f"{sequence}: eventcv yields {n} slices but the pose grid has "
                f"{len(geom[sequence][0])} — every coordinate after the first would be "
                f"attached to the wrong frame")
        bank_files[sequence] = files

    result = bp.score_configuration(bank_files, geom, args, cli, cli.label, device, tag)
    result.update({
        "method": "spikevpr", "source": cli.source,
        "filter_arm": "on" if not synthetic else "off",
        "event_filter_dt_us": FILTER_DT_US if not synthetic else None,
        "representation": f"ON/OFF event counts at {bridge.GRID[0]}x{bridge.GRID[1]}",
        "event_window": ("full slice" if not cli.max_events
                         else f"first {cli.max_events} events"),
        "checkpoint": checkpoint, "checkpoint_sha256": digest,
        "trained_on": cli.model, "neuron": neuron, "encoder": bridge.ENCODER,
        "descriptor_dim": DESC_DIM,
    })

    output = {"dataset": cli.dataset, "method": "spikevpr", "trained_on": cli.model,
              "query": cli.query, "database": cli.database,
              "threshold_m": cli.threshold_m, "dt_ms": DT_MS,
              "pca": [list(setting) for setting in PCA_SETTINGS], "hot_pixel": True,
              "results": {tag: result}}
    # The arm is part of the filename: a Sim2Real pair writes twice into one
    # directory, and without this the second run would clobber the first.
    stem = (f"{cli.dataset}_{cli.model}" if not synthetic
            else f"{cli.dataset}_{cli.model}_{cli.source}")
    out_json = os.path.join(cli.out_dir, f"{stem}.json")
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
