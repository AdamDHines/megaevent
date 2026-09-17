"""EventVLAD on the pooled 25 m Brisbane and NSAVP protocols.

Run from the repository root::

    pixi run python -u scripts/eventvlad_pooled.py --dataset brisbane_event
    pixi run python -u scripts/eventvlad_pooled.py --dataset nsavp

Raw traverses follow EventVLAD's author implementation: each descriptor is built from three
consecutive 50 ms count frames and is attached to the middle frame.  Geometry and scoring are
the same helpers used by ``eventgem_pooled.py`` so the methods see identical galleries, GPS
coverage masks, 25 m positives, and background-activity filtering.
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import cv2
import eventcv as ecv
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
from src import traversenpz as tnpz  # noqa: E402
from src.methods import (  # noqa: E402
    EVENTVLAD_BATCH, EVENTVLAD_DENOISE_SIZE, EventVLADMethod,
)


DEFAULT_OUT = "/media/adam/vprdatasets/megaevent/eventvlad_pooled"
PCA_SETTINGS = ((1000, 0.5),)
DT_MS = 50
FILTER_DT_US = 50000
RENDER_BLOCK = 512


def _normalise_count(frame, percentile=99.0):
    """Event-LAB ``eventvlad_denoiser.normalize_frame`` for one count slice."""
    frame = np.asarray(frame)
    if frame.ndim == 3:
        if frame.shape[0] == 1:
            frame = frame[0]
        elif frame.shape[0] == 2:
            frame = frame.sum(axis=0)
        else:
            frame = frame.mean(axis=0)
    frame = frame.astype(np.float32, copy=False)
    vmax = float(np.percentile(np.abs(frame), percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(np.abs(frame)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return np.clip(frame / vmax, 0.0, 1.0)


def _reader(dataset, traverse, path):
    reader = ecv.open(
        path, dt_ms=DT_MS, sensor_size=inf.sensor_size(dataset),
        hot_pixel_filter=True, offset=egp.traverse_offset_ms(dataset, traverse),
    )
    reader = reader.background_activity_filter(FILTER_DT_US)
    return reader.with_repr("count")


def _triplets(frames):
    """Consecutive count frames -> author-style [N-2,3,256,256] denoiser input."""
    planes = []
    for frame in frames:
        plane = _normalise_count(frame)
        height, width = plane.shape
        plane = plane[:height - height % 32, :width - width % 32]
        planes.append(cv2.resize(
            plane, (EVENTVLAD_DENOISE_SIZE, EVENTVLAD_DENOISE_SIZE),
            interpolation=cv2.INTER_AREA))
    planes = torch.from_numpy(np.ascontiguousarray(np.stack(planes)))
    return torch.stack((planes[:-2], planes[1:-1], planes[2:]), dim=1)


def _npz_planes(paths, indices):
    """Raw count frames for a block of slice indices, from a dumped Sim2Real arm.

    Stands in for ``reader.batch(...)``. The triplet construction above is unchanged: three
    *consecutive slices*, exactly as the real arm builds them. ``load_count_triplet`` — which
    splits one I2E saccade into sub-windows — is deliberately not used, because a recorded
    50 ms slice has no equivalent split and the two arms would then differ in two things.
    """
    load = tnpz.count_plane_loader()
    return np.stack([load(paths[i]) for i in indices])


@torch.no_grad()
def extract(method, dataset, traverse, out_dir, tag, source="real", npz_root=None):
    path = os.path.join(out_dir, f"{tag}_{traverse}_eventvlad.npy")
    source_args = SimpleNamespace(
        eventlab_dir=DATASETS[dataset]["eventlab_dir"], dataset=dataset)
    npz_paths = None
    if source == "real":
        reader = _reader(dataset, traverse, inf.sequence_path(source_args, traverse))
        n_slices = int(reader.n_slices)
    else:
        reader = None
        npz_paths = tnpz.frame_paths(npz_root, dataset, source, traverse)
        n_slices = len(npz_paths)
    n_desc = n_slices - 2
    if os.path.exists(path):
        bank = np.load(path, mmap_mode="r")
        if bank.shape == (n_desc, 1000):
            print(f"    {tag} {traverse}: {n_desc} triplets — bank cached, skipping")
            return {"eventvlad": path}, n_slices
        raise ValueError(f"{path}: cached shape {bank.shape}, expected {(n_desc, 1000)}")

    pending = path + ".tmp"
    bank = np.lib.format.open_memmap(
        pending, mode="w+", dtype=np.float32, shape=(n_desc, 1000))
    written, start = 0, time.time()
    for first in range(0, n_desc, RENDER_BLOCK):
        stop = min(first + RENDER_BLOCK, n_desc)
        block = list(range(first, stop + 2))
        raw = (np.asarray(reader.batch(block)) if reader is not None
               else _npz_planes(npz_paths, block))
        batch = _triplets(raw)
        for offset in range(0, batch.size(0), EVENTVLAD_BATCH):
            desc = method.encode(batch[offset:offset + EVENTVLAD_BATCH]).cpu().numpy()
            at = first + offset
            bank[at:at + len(desc)] = desc
        written = stop
        rate = written / (time.time() - start)
        print(f"    {tag} {traverse}: {written}/{n_desc}  {rate:.1f} img/s  "
              f"eta {(n_desc - written) / rate / 60:.1f} min", flush=True)
    bank.flush()
    del bank
    os.replace(pending, path)
    return {"eventvlad": path}, n_slices


def centre_geometry(geom):
    """Align the frame grid with each triplet's middle slice."""
    out = {}
    for traverse, (xy, covered, speed, info) in geom.items():
        info = dict(info)
        info["n_frames"] = len(xy) - 2
        info["uncovered"] = int((~covered[1:-1]).sum())
        out[traverse] = (xy[1:-1], covered[1:-1], speed[1:-1], info)
    return out


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
        "db_chunk": npl.DB_CHUNK,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--eventlab-repo", default="/home/adam/repo/Event-LAB")
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
    parser.add_argument("--source", default="real", choices=list(tnpz.SOURCES),
                        help="which arm the events come from; see "
                             "scripts/brisbane_pooled.py")
    parser.add_argument("--align-sources", nargs="+", default=None,
                        help="frame trees whose select.json rows are OR-dropped by the "
                             "alignment mask; default aps, plus gopro for i2e_gopro")
    cli = parser.parse_args()

    spec = DATASETS[cli.dataset]
    cli.eventlab_dir = spec["eventlab_dir"]
    cli.query = cli.query or spec["query"]
    cli.database = list(cli.database or spec["database"])
    cli.threshold_m = 25.0
    cli.thresholds_m = [25.0, 50.0, 75.0, 100.0]
    cli.pca = list(PCA_SETTINGS)
    cli.score_chunk = 256
    cli.db_chunk = spec["db_chunk"]
    synthetic = cli.source != "real"
    cli.aps_aligned = cli.aps_aligned or synthetic
    cli.align_sources = (tuple(cli.align_sources) if cli.align_sources
                         else (("aps", "gopro") if cli.source == "i2e_gopro"
                               else ("aps",)))
    if synthetic and cli.dataset != "brisbane_event":
        raise SystemExit(f"--source {cli.source} exists only for brisbane_event")
    cli.fig_tag = (f"{cli.dataset}_eventvlad_ba50" if not synthetic
                   else f"{cli.dataset}_eventvlad_{cli.source}")
    os.makedirs(cli.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _Args(cli.eventlab_dir, cli.dataset, DT_MS, False, False)
    args.eventlab_repo = cli.eventlab_repo
    method = EventVLADMethod(args, device)
    sequences = [cli.query, *cli.database]
    # The filter arm is part of the real tag; a synthetic arm has no filter, so its
    # tag names the source instead and cannot collide with the real bank.
    tag = "eventvlad_ba50" if not synthetic else f"eventvlad_{cli.source}"

    print(f"eventvlad/{cli.dataset}: query {cli.query} -> database {'+'.join(cli.database)}")
    print(f"  25 m radius, {DT_MS} ms count-frame triplets, hot-pixel on, "
          f"background activity dt={FILTER_DT_US} us, device {device}")

    if cli.dataset == "nsavp":
        root = os.path.join(cli.eventlab_dir, cli.dataset)
        geom = npl.traverse_geometry(root, sequences, DT_MS)
    else:
        geom = bp.traverse_geometry(args, sequences, spec["npz_root"],
                                    aps_align=(synthetic or cli.aps_aligned),
                                    align_sources=cli.align_sources)

    bank_files = {}
    for sequence in sequences:
        files, n_slices = extract(method, cli.dataset, sequence, cli.out_dir, tag,
                                  source=cli.source, npz_root=spec["npz_root"])
        if n_slices != len(geom[sequence][0]):
            raise SystemExit(
                f"{sequence}: EventCV yields {n_slices} slices but geometry has "
                f"{len(geom[sequence][0])}")
        bank_files[sequence] = files

    geom = centre_geometry(geom)
    result = bp.score_configuration(
        bank_files, geom, args, cli, "eventvlad", device, tag)
    result.update({
        "method": "eventvlad", "filter_arm": "on" if not synthetic else "off",
        "source": cli.source,
        "event_filter_dt_us": FILTER_DT_US,
        "representation": "three consecutive count frames",
        "descriptor_alignment": "middle slice",
        "meta": method.meta,
    })

    output = {
        "dataset": cli.dataset, "method": "eventvlad", "query": cli.query,
        "database": cli.database, "threshold_m": cli.threshold_m, "dt_ms": DT_MS,
        "pca": [list(setting) for setting in PCA_SETTINGS], "hot_pixel": True,
        "results": {tag: result},
    }
    # The arm is part of the filename: a Sim2Real pair writes twice into one
    # directory, and without this the second run would clobber the first.
    stem = cli.dataset if not synthetic else f"{cli.dataset}_{cli.source}"
    out_json = os.path.join(cli.out_dir, f"{stem}.json")
    tmp = out_json + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(output, handle, indent=2)
    os.replace(tmp, out_json)

    print("\nR@1 / R@5 / R@10 / R@20 @ 25 m")
    for name, recall in result["recall"].items():
        print(f"  {name:14s} " + "  ".join(f"{recall[str(k)]:.4f}" for k in bp.KS))
    print(f"\n-> {out_json}")


if __name__ == "__main__":
    main()
