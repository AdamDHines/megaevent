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
* :class:`SpringfieldFrames` — the full Springfield capture. Row *i* is ``(session,
  partition, slice)``, rendered through ``springfield_eval.make_dataset`` under the settings
  its bank manifest records. Its ranking is not recomputed either: :func:`load_springfield_benchmark`
  reads the top-20 per query from the diag dump, the ranking behind the published number.

The loaders then assemble every kind into the same :class:`Benchmark`, so the writer
downstream has one code path. Nothing about the numbers is re-implemented: the pooled
side calls ``brisbane_pooled.traverse_geometry`` / ``nsavp_pooled.traverse_geometry``,
``brisbane_pooled.pool_database`` and ``src.imagevpr.build_gt``, which is what makes a figure
drawn here provably the same benchmark the tables report.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import json
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


class SpringfieldFrames:
    """One 50 ms slice of one capture partition per row — the dump-driven Springfield kind.

    A row is ``(session, partition file, slice within that partition)``. Gallery rows come
    straight from the diag dump's ``db_sid / db_part / db_slice``; query rows arrive as
    ``(session, session-level slice)`` and are placed in a partition by
    :func:`place_session_slices`, through the cumulative ``n_slices`` of the bank manifests in
    ``list_partitions`` order — the order ``springfield_diag.load_cached_session`` laid each
    session out in when it wrote the dump.

    Rendering is ``springfield_eval.make_dataset`` under the settings recorded in the bank
    manifest (representation, dt, hot-pixel filter, background-activity window), so the frame
    drawn is the frame the descriptor was computed from. Readers are capped at four, as in
    ``springfield_full._LRUFrames`` (a lazy reader per 7.5 GB partition), and rendered frames
    are memoised: the montage re-renders exactly the frames the per-query PNGs just drew, and
    opening a partition costs seconds on the data disk.
    """

    MAX_READERS = 4
    MEMO = 96

    def __init__(self, sid, sdir_of_sid, part, local, render, label_name, labels):
        self.sid = np.asarray(sid).astype(str)
        self.sdir_of_sid = dict(sdir_of_sid)
        self.part = np.asarray(part).astype(str)
        self.local = np.asarray(local, dtype=np.int64)
        self.render_args = dict(render)
        self.label_name, self.labels = label_name, np.asarray(labels).astype(str)
        if not (len(self.sid) == len(self.part) == len(self.local) == len(self.labels)):
            raise ValueError("SpringfieldFrames: row arrays disagree in length")
        self._readers = OrderedDict()
        self._memo = OrderedDict()

    def __len__(self):
        return int(self.sid.size)

    def locate(self, i):
        """``(session, partition basename, slice within that partition)`` for row ``i``."""
        return str(self.sid[i]), str(self.part[i]), int(self.local[i])

    def h5path(self, i):
        sid, part, _ = self.locate(i)
        return os.path.join(self.sdir_of_sid[sid], part)

    def _reader(self, path):
        if path not in self._readers:
            # Imported here rather than at module scope: an image-dataset figure run should
            # not have to import eventcv or open a recording at all.
            from torchvision import transforms

            from springfield_eval import make_dataset

            if len(self._readers) >= self.MAX_READERS:
                self._readers.popitem(last=False)
            r = self.render_args
            self._readers[path] = make_dataset(path, transforms.Compose([]),
                                               r["representation"], r["dt_ms"],
                                               r["hot_pixel"], r["filter_dt_us"])
        return self._readers[path]

    def render(self, i):
        i = int(i)
        if i in self._memo:
            self._memo.move_to_end(i)
            return self._memo[i]
        # [3, H, W] float in [0, 1] at the sensor's 1280x720 — springfield_eval.FrameCache's
        # conversion, so a frame here is byte-identical to the diag strips' render of it.
        frame = self._reader(self.h5path(i))[self.locate(i)[2]].numpy()
        image = (np.transpose(frame, (1, 2, 0)) * 255).astype(np.uint8)
        self._memo[i] = image
        if len(self._memo) > self.MEMO:
            self._memo.popitem(last=False)
        return image

    def key(self, i):
        from springfield_eval import part_label

        sid, part, local = self.locate(i)
        return f"{sid}:{part_label(part)}#{local:06d}"

    def provenance(self, i):
        sid, part, local = self.locate(i)
        return {"session": sid, "partition": part, "slice": local,
                self.label_name: str(self.labels[i])}

    def caption(self, i):
        """The montage's row label: the condition beside the query index."""
        return f"{self.labels[i]} q{int(i)}"


# ---------------------------------------------------------------------------
# 2. Benchmarks
# ---------------------------------------------------------------------------
@dataclass
class Benchmark:
    """One scored benchmark, with everything a figure needs and nothing it does not.

    Two ways to say what is correct. The image and pooled kinds carry a dense ``gt``; the
    dump-driven Springfield kind carries none — ``[132,569 x 15,886]`` bool is 2.1 GB for what
    is a 25 m radius test — and answers from ``db_xy``/``q_xy`` instead. Every question the
    writer asks (is this row a positive, how many positives has this query, what does this
    pair score) goes through the methods below, so the two kinds cannot drift.
    """

    dataset: str
    db: torch.Tensor | None             # [n_db, D], L2-normalised; None when dump-driven
    queries: torch.Tensor | None        # [n_q, D]
    gt: np.ndarray | None               # bool [n_db, n_q]; None means a radius test on xy
    db_frames: object
    q_frames: object
    db_xy: np.ndarray | None            # metres, for the distance behind each verdict
    q_xy: np.ndarray | None
    threshold_m: float
    meta: dict
    ranking: tuple | None = None        # (ranked [depth, n_q], scores) precomputed, best first
    q_labels: np.ndarray | None = None  # per-query condition label (Springfield: the sweep)
    extra: dict = field(default_factory=dict)   # per-query arrays worth writing to the manifest
    sim_fn: object = None               # (db_i, q) -> cosine, when there is no bank tensor

    @property
    def shape(self):
        """``(n_db, n_q)``."""
        if self.gt is not None:
            return self.gt.shape
        return len(self.db_xy), len(self.q_xy)

    def _within(self, db_rows, q):
        d = np.linalg.norm(self.db_xy[db_rows].astype(np.float64)
                           - self.q_xy[q].astype(np.float64), axis=-1)
        return d <= self.threshold_m

    def hit(self, ranked):
        """``[depth, n_q]`` bool — is the row ranked at each depth a positive of its query.

        The dense form is exactly ``brisbane_pooled.recall_from_ranked``'s indexing; the radius
        form is ``src.imagesets.build_radius_gt``'s test (``<= threshold``) on the same
        coordinates, applied to the ranked rows only.
        """
        n_q = ranked.shape[1]
        if self.gt is not None:
            return self.gt[ranked, np.arange(n_q)[None, :]]
        return self._within(ranked, np.arange(n_q)[None, :])

    def positives(self, q):
        """Database rows within the radius of query ``q``."""
        if self.gt is not None:
            return np.flatnonzero(self.gt[:, q])
        return np.flatnonzero(self._within(slice(None), q))

    def correct(self, db_i, q):
        if self.gt is not None:
            return bool(self.gt[db_i, q])
        return bool(self._within(db_i, q))

    def n_positives_all(self):
        """``[n_q]`` positive counts — the scorability mask's source."""
        if self.gt is not None:
            return self.gt.sum(0)
        from sklearn.neighbors import NearestNeighbors

        # The call build_radius_gt makes, minus the dense matrix it fills from it.
        neighbours = NearestNeighbors(n_jobs=-1).fit(self.db_xy)
        counts = np.zeros(len(self.q_xy), dtype=np.int64)
        for s in range(0, len(self.q_xy), 2048):
            found = neighbours.radius_neighbors(self.q_xy[s:s + 2048], radius=self.threshold_m,
                                                return_distance=False)
            counts[s:s + 2048] = [len(rows) for rows in found]
        return counts

    def similarity(self, db_i, q):
        """Cosine between one database row and one query — both banks are L2-normalised."""
        if self.db is not None:
            return float(self.db[db_i].to(torch.float32) @ self.queries[q].to(torch.float32))
        return float(self.sim_fn(db_i, q))

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


def place_session_slices(bank_dir, tag, sdir, sid, session_slices):
    """``(partition basename[], local slice[])`` for session-level slice indices.

    ``springfield_diag`` numbers a session's slices across its partitions in
    ``list_partitions`` order, using each partition's ``n_slices`` from the bank manifest
    written at extraction; this inverts that numbering with the same manifests
    (``springfield_diag._slice_to_part``, vectorised over a session).
    """
    from springfield_eval import list_partitions

    parts = list_partitions(sdir)
    counts = []
    for path in parts:
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(bank_dir, sid, f"{tag}_{stem}.json")) as handle:
            counts.append(int(json.load(handle)["n_slices"]))
    edges = np.concatenate([[0], np.cumsum(counts)])
    rows = np.asarray(session_slices, dtype=np.int64)
    if rows.size and rows.max() >= edges[-1]:
        raise SystemExit(f"{sid}: slice {int(rows.max())} beyond the {int(edges[-1])} slices "
                         f"its bank manifests describe — dump and banks disagree")
    which = np.searchsorted(edges, rows, side="right") - 1
    return (np.array([os.path.basename(parts[i]) for i in which]),
            rows - edges[which])


def load_springfield_benchmark(cli, spec):
    """The full Springfield capture, ranked by the diag dump rather than re-ranked here.

    ``scripts/springfield_diag.py --stage dump`` streams every query slice against the pooled
    gallery once and keeps the top-20 rows and cosines, the best-correct row / cosine / exact
    rank, both coordinate sets and the row -> (session, partition, slice) map. That is the
    ranking behind the published number — the caller checks the results JSON's R@1 against
    it — and it is everything a panel needs. Re-ranking would mean the 132,569 x 8448 pooled
    bank (4.5 GB) plus a dense 25 m ground truth (2.1 GB), to draw fifteen queries.

    The frames are rendered under the bank manifest's own settings, cross-checked against the
    results JSON, so what is drawn is what the descriptor saw. Similarities not in the dump
    (the nearest-positive column) are dot products of rows read from the per-partition banks,
    which are unit-norm — checked, since a dot product of anything else is not the cosine the
    dump ranked with.
    """
    dump_path = cli.dump or spec["dump"]
    results_path = cli.results or spec["results"]
    bank_dir = cli.bank_dir_override or spec["bank_dir"]
    root = spec["root"]
    with open(results_path) as handle:
        results = json.load(handle)
    tag = results["tag"]
    if os.path.basename(dump_path) != f"dump_{tag}.npz":
        raise SystemExit(f"{dump_path} was not written for the run in {results_path} "
                         f"(tag {tag}) — pass a matching --dump/--results pair")
    d = np.load(dump_path, allow_pickle=False)
    sids = np.array([str(s) for s in d["sids"]])
    sweeps = np.array([str(s) for s in d["sweeps"]])
    db_sid, db_part = d["db_sid"].astype(str), d["db_part"].astype(str)

    # Render settings: the bank manifest of the first gallery row, held to the results JSON.
    manifest_path = os.path.join(bank_dir, db_sid[0],
                                 f"{tag}_{os.path.splitext(db_part[0])[0]}.json")
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    render = {}
    for key in ("representation", "dt_ms", "hot_pixel", "filter_dt_us"):
        if manifest[key] != results[key]:
            raise SystemExit(f"{manifest_path}: {key}={manifest[key]!r} but the results JSON "
                             f"says {results[key]!r}")
        render[key] = manifest[key]
    if cli.representation != render["representation"]:
        print(f"  WARNING: rendering {cli.representation} frames for a "
              f"{render['representation']} bank (--representation) — the figure will not "
              f"show what the model saw")
        render["representation"] = cli.representation

    db_sdirs = {sid: os.path.join(root, "database", sid) for sid in np.unique(db_sid)}
    db_frames = SpringfieldFrames(db_sid, db_sdirs, db_part, d["db_slice"], render,
                                  "arm", d["db_arm"].astype(str))

    q_sid_index, q_slice = d["q_sid"], d["q_slice"]
    q_sid, q_sweep = sids[q_sid_index], sweeps[q_sid_index]
    q_sdirs = {sid: os.path.join(root, f"query_{sweep}", sid) for sid, sweep in zip(sids, sweeps)}
    q_part = np.empty(len(q_sid), dtype=object)
    q_local = np.zeros(len(q_sid), dtype=np.int64)
    for i, sid in enumerate(sids):
        rows = np.flatnonzero(q_sid_index == i)
        q_part[rows], q_local[rows] = place_session_slices(bank_dir, tag, q_sdirs[sid], sid,
                                                           q_slice[rows])
    q_frames = SpringfieldFrames(q_sid, q_sdirs, q_part.astype(str), q_local, render,
                                 "sweep", q_sweep)

    banks = {}

    def bank_row(frames, i):
        sid, part, local = frames.locate(i)
        path = os.path.join(bank_dir, sid, f"{tag}_{os.path.splitext(part)[0]}.npy")
        if path not in banks:
            banks[path] = np.load(path, mmap_mode="r")
        row = np.asarray(banks[path][local], dtype=np.float64)
        if abs(np.linalg.norm(row) - 1.0) > 1e-3:
            raise SystemExit(f"{path}[{local}] is not unit-norm — a dot product would not be "
                             f"the cosine the dump ranked with")
        return row

    def similarity(db_i, q):
        return float(bank_row(db_frames, db_i) @ bank_row(q_frames, q))

    arms, arm_counts = np.unique(d["db_arm"].astype(str), return_counts=True)
    sweep_names, sweep_counts = np.unique(q_sweep, return_counts=True)
    meta = {"kind": "springfield", "tag": tag,
            "dump": os.path.abspath(dump_path), "results": os.path.abspath(results_path),
            "bank_dir": os.path.abspath(bank_dir), "root": root,
            "checkpoint": results.get("checkpoint"), "step": results.get("step"),
            **render, "filter_arm": "off" if render["filter_dt_us"] is None else "on",
            "ranking_depth": int(d["top_i"].shape[0]),
            "database_rows_per_arm": {str(a): int(c) for a, c in zip(arms, arm_counts)},
            "query_sessions": int(len(sids)),
            "query_slices_per_sweep": {str(s): int(c) for s, c in zip(sweep_names, sweep_counts)},
            "published_r1": float(results["recall_micro"]["overall"]["1"])}
    print(f"  ranking from {dump_path}\n  {len(sids)} query sessions "
          + "  ".join(f"{s}={c}" for s, c in zip(sweep_names, sweep_counts))
          + f"; gallery " + "  ".join(f"{a}={c}" for a, c in zip(arms, arm_counts)))
    return Benchmark(cli.dataset, None, None, None, db_frames, q_frames,
                     d["db_xy"].astype(np.float64), d["q_xy"].astype(np.float64),
                     float(results["threshold_m"]), meta,
                     ranking=(d["top_i"], d["top_c"]), q_labels=q_sweep,
                     extra={"best_correct_rank": d["bc_rank"],
                            "best_correct_similarity": d["bc_c"],
                            "best_correct_row": d["bc_i"]},
                     sim_fn=similarity)
