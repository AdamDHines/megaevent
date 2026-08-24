"""Where a figure's frames come from, for both kinds of benchmark this repo scores.

``scripts/topk_figures.py`` writes retrieval panels: a query image, its top-k, the ground
truth beside them. Everything it does after "here is database row *i*" is identical whether
that row is a file on disk or a 50 ms slice of a recording — but *getting the image* is not,
and that is the whole reason ``brisbane_event`` and ``nsavp`` were out of scope when that
script was first written.

So the difference is isolated here, in one object:

* :class:`NpzFrames` — an independent-image dataset. Row *i* is a ``.npz``; rendering it is
  :func:`src.scoring.display_frame`.
* :class:`PooledFrames` — a pooled traverse. Row *i* is ``(traverse, frame)``, and rendering
  it means re-slicing the recording through eventcv under **the same representation,
  hot-pixel setting and background-activity window the descriptor bank was extracted with**.
  Anything else would draw a picture of a frame the model never saw.

:func:`load_benchmark` then assembles either kind into the same :class:`Benchmark`, so the
writer downstream has one code path. Nothing about the numbers is re-implemented: the pooled
side calls ``brisbane_pooled.traverse_geometry`` / ``nsavp_pooled.traverse_geometry``,
``brisbane_pooled.pool_database`` and ``src.imagevpr.build_gt``, which is what makes a figure
drawn here provably the same benchmark the tables report.
"""

from dataclasses import dataclass
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from src.imagevpr import build_gt  # noqa: E402
from src.scoring import display_frame  # noqa: E402

DATA_ROOT = "/media/adam/vprdatasets/megaevent"


# ---------------------------------------------------------------------------
# 1. Frame sources
# ---------------------------------------------------------------------------
class NpzFrames:
    """One rendered ``.npz`` per row — the independent-image datasets.

    ``representation`` must match the descriptor bank's — the raw-event ``.npz``
    files themselves are representation-free, so nothing else pins the two together.
    """

    def __init__(self, paths, keys, representation="countmask"):
        self.paths, self.keys = list(paths), list(keys)
        self.representation = representation

    def __len__(self):
        return len(self.paths)

    def render(self, i):
        return display_frame(self.paths[i], self.representation)

    def key(self, i):
        return self.keys[i]

    def provenance(self, i):
        return {"source": os.path.abspath(self.paths[i])}


class PooledFrames:
    """One 50 ms slice of one traverse per row — the pooled traverse datasets.

    ``pool_database`` lays the traverses out back to back, each contributing only its
    GPS-covered frames in order, so a pooled row maps back to ``(traverse, local frame)`` by
    walking ``np.flatnonzero(covered)`` per traverse. That map is built once, here, rather
    than re-derived at each call site — getting it wrong would draw a real frame from the
    wrong second of the recording and nothing downstream could tell.

    Readers are opened lazily and kept, because a montage touches a handful of frames from
    two or three traverses and opening five HDF5 recordings up front to render six images
    would dominate the run.
    """

    def __init__(self, args, sequences, covered_masks):
        self.args = args
        self.sequences = list(sequences)
        seq_of_row, frame_of_row = [], []
        for s, seq in enumerate(self.sequences):
            local = np.flatnonzero(covered_masks[seq])
            seq_of_row.append(np.full(local.size, s, dtype=np.int32))
            frame_of_row.append(local.astype(np.int64))
        self.seq_of_row = np.concatenate(seq_of_row) if seq_of_row else np.zeros(0, np.int32)
        self.frame_of_row = np.concatenate(frame_of_row) if frame_of_row else np.zeros(0, np.int64)
        self._readers = {}

    def __len__(self):
        return int(self.seq_of_row.size)

    def _dataset(self, seq):
        if seq not in self._readers:
            # Imported here rather than at module scope: an image-dataset figure run should
            # not have to import eventcv or open a recording at all.
            from torchvision import transforms

            from brisbane_resolution import _slice_source

            self._readers[seq] = _slice_source(seq, transforms.Compose([]), self.args)
        return self._readers[seq]

    def locate(self, i):
        """``(traverse, frame index within that traverse)`` for pooled row ``i``."""
        return self.sequences[int(self.seq_of_row[i])], int(self.frame_of_row[i])

    def render(self, i):
        seq, frame = self.locate(i)
        # [3, H, W] float in [0, 1] at the sensor's own resolution — the render, not the
        # 322x322 resize the model was fed. imsave takes float RGB as-is.
        return np.transpose(self._dataset(seq)[frame].numpy(), (1, 2, 0))

    def key(self, i):
        seq, frame = self.locate(i)
        return f"{seq}:{frame:06d}"

    def provenance(self, i):
        seq, frame = self.locate(i)
        return {"traverse": seq, "frame_index": frame}


# ---------------------------------------------------------------------------
# 2. Benchmarks
# ---------------------------------------------------------------------------
@dataclass
class Benchmark:
    """One scored benchmark, with everything a figure needs and nothing it does not."""

    dataset: str
    db: torch.Tensor                    # [n_db, D], L2-normalised
    queries: torch.Tensor               # [n_q, D]
    gt: np.ndarray                      # bool [n_db, n_q]
    db_frames: object
    q_frames: object
    db_xy: np.ndarray | None            # metres, for the distance behind each verdict
    q_xy: np.ndarray | None
    threshold_m: float
    meta: dict

    def distance(self, db_i, q):
        """Metres between a database row and a query, or ``None`` where there is no metric.

        MSLS spans six cities in six UTM zones, so a cross-city pair has no distance at all —
        ``None`` rather than a number that would silently compare eastings from different
        zones. See :func:`msls_coords`.
        """
        if self.db_xy is None or self.q_xy is None:
            return None
        a, b = self.db_xy[db_i], self.q_xy[q]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return None
        d = float(np.linalg.norm(a - b))
        groups = self.meta.get("groups")
        if groups is not None and groups[0][db_i] != groups[1][q]:
            return None
        return d


def msls_coords(cli, dataset):
    """MSLS per-image UTM eastings/northings and city labels, for figure distances only.

    :class:`src.imagesets.ImageSet` leaves ``db_utm``/``q_utm`` unset for MSLS, and rightly:
    the six cities project into six different UTM zones, so a single coordinate array is not a
    map and ``src/imagevpr.py`` would draw a meaningless error scatter from it. But a *pairwise*
    distance within a city is perfectly well defined, and that is what a retrieval panel wants
    — "the top-1 is 31 m away" against "the top-1 is in another country". So the same metadata
    :func:`src.imagesets._read_msls_split` reads is re-read here, kept beside a city label, and
    consumed only by :meth:`Benchmark.distance`, which refuses to subtract across cities.
    """
    from src.imagesets import MSLS_CITIES, _read_msls_split

    meta_root = os.path.join(cli.data_dir, "msls", "test")
    event_root = os.path.join(cli.data_dir, "msls", "test_countmask", "numpy")
    by_key = {}
    for city in MSLS_CITIES:
        for split in ("database", "query"):
            keys, _, coords, _ = _read_msls_split(meta_root, event_root, city, split)
            for key, xy in zip(keys, coords):
                by_key[key] = (city, xy)

    def gather(keys):
        cities = np.array([by_key[k][0] if k in by_key else "" for k in keys])
        xy = np.array([by_key[k][1] if k in by_key else (np.nan, np.nan) for k in keys],
                      dtype=np.float64)
        return xy, cities

    db_xy, db_city = gather(dataset.db_keys)
    q_xy, q_city = gather(dataset.q_keys)
    return db_xy, q_xy, (db_city, q_city)


def load_image_benchmark(cli, dataset, banks):
    """An independent-image benchmark. ``banks`` is ``(db, queries)``, already aligned."""
    db_xy, q_xy, groups = dataset.db_utm, dataset.q_utm, None
    if cli.dataset == "msls":
        db_xy, q_xy, groups = msls_coords(cli, dataset)
    db, queries = banks
    meta = {"kind": "image", "database": cli.ref, "queries": cli.query,
            "representation": cli.representation,
            "excluded_database": dataset.excluded_db, "excluded_queries": dataset.excluded_q}
    if groups is not None:
        meta["groups"] = groups
        meta["cities"] = sorted(set(groups[0]) - {""})
    return Benchmark(cli.dataset, db, queries, dataset.gt,
                     NpzFrames(dataset.db_paths, dataset.db_keys, cli.representation),
                     NpzFrames(dataset.q_paths, dataset.q_keys, cli.representation),
                     db_xy, q_xy, cli.positive_dist_threshold, meta)


def pooled_args(cli, spec):
    """The namespace ``src.inference``'s traverse helpers read, matching the bank's arm.

    ``r322ba50`` means the background-activity filter was on at a 50 ms window during
    extraction, so it has to be on here too — a frame rendered unfiltered is a different
    picture of the same instant, and the figure would be showing something the descriptor
    was not computed from. The arm is read off the bank tag rather than passed separately,
    so the two cannot disagree.
    """
    from brisbane_resolution import _Args

    filtered = "ba" in spec["bank_tag"]
    args = _Args(spec["eventlab_dir"], cli.dataset, cli.dt_ms, False, not filtered,
                 source="real", npz_root=spec.get("npz_root"))
    args.representation = cli.representation
    return args


def load_pooled_benchmark(cli, spec):
    """A pooled traverse benchmark, assembled by the same code that scores it."""
    from brisbane_pooled import pool_database
    from brisbane_pooled import traverse_geometry as brisbane_geometry
    from nsavp_pooled import traverse_geometry as nsavp_geometry

    args = pooled_args(cli, spec)
    query, database = cli.query, list(cli.database)
    sequences = [query, *database]
    if cli.dataset == "nsavp":
        geom = nsavp_geometry(os.path.join(spec["eventlab_dir"], cli.dataset), sequences,
                              cli.dt_ms)
    else:
        geom = brisbane_geometry(args, sequences, spec.get("npz_root"))

    label = cli.bank_label
    bank_files = {seq: {label: os.path.join(cli.bank_dir,
                                            f"{spec['bank_tag']}_{seq}_{label}.npy")}
                  for seq in sequences}
    missing = [f[label] for f in bank_files.values() if not os.path.exists(f[label])]
    if missing:
        raise SystemExit(f"no descriptor bank at {missing[0]} — extract it first "
                         f"(scripts/{'nsavp' if cli.dataset == 'nsavp' else 'brisbane'}"
                         f"_pooled.py)")

    # pool_database raises if a bank's row count disagrees with the GPS grid, which is the
    # one failure that would attach every coordinate after the first to the wrong frame.
    db, db_xy, _ = pool_database(bank_files, geom, database, label)
    q_bank = np.load(bank_files[query][label], mmap_mode="r")
    xy, covered, speed, _ = geom[query]
    if q_bank.shape[0] != len(xy):
        raise SystemExit(f"{query}: bank {q_bank.shape[0]} rows vs GPS grid {len(xy)}")
    queries = torch.from_numpy(np.asarray(q_bank[covered]))
    q_xy = xy[covered]
    del q_bank

    gt = build_gt(db_xy, q_xy, cli.positive_dist_threshold)
    meta = {"kind": "pooled", "query_traverse": query, "database_traverses": database,
            "bank_tag": spec["bank_tag"], "bank_label": label, "dt_ms": cli.dt_ms,
            "filter_arm": "off" if args.filter_dt_us is None else "on",
            "event_filter_dt_us": args.filter_dt_us,
            "representation": cli.representation,
            "database_rows_per_traverse": {s: int(geom[s][1].sum()) for s in database},
            "stationary_query_fraction": float((speed[covered] < 0.5).mean())}
    return Benchmark(cli.dataset, db, queries, gt,
                     PooledFrames(args, database, {s: geom[s][1] for s in database}),
                     PooledFrames(args, [query], {query: covered}),
                     db_xy, q_xy, cli.positive_dist_threshold, meta)
