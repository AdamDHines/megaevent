"""Event-GeM's global stage under the pooled traverse protocol, on Brisbane and NSAVP.

Run from the repo root::

    pixi run python3 scripts/eventgem_pooled.py --dataset brisbane_event
    pixi run python3 scripts/eventgem_pooled.py --dataset nsavp

Scores the Event-GeM baseline against exactly the protocol ``scripts/brisbane_pooled.py`` and
``scripts/nsavp_pooled.py`` put the shipping models through — same traverses, same 25 m Euclidean
radius on the same coordinates, same database-only PCA whitening, same streamed top-20 — so the
numbers land in one table without a protocol footnote. Geometry, pooling and scoring are imported
from those two scripts rather than reimplemented; only the descriptor differs.

**The frames are streamed from the HDF5, not materialised.** ``src/methods.py``'s
:class:`EventGeMMethod` reads pre-rendered ``.npz`` frames, which is why ``main.py`` makes
``--method eventgem`` on a traverse require ``--source``. Only ``sunset1``/``sunset2`` were ever
dumped, and dumping the other nine traverses here would be ~275k files for a single pass. eventcv
renders SuperEvent's MCTS representation directly (``with_repr("mcts")`` -> ``[10, H, W]`` float32
in [0, 1]), so the same frames come off the stream at the same 50 ms framing the megaevent banks
used — which is also what makes the two methods' descriptor rows line up with one shared
coordinate grid.

**Both stages.** Every descriptor space is scored twice — the GeM global ranking, and that
ranking re-ordered by Event-GeM's homography verification, reported as its ``+rerank`` twin.
``LocalReranker.rerank_shortlists`` does the second pass over the top-``--eventgem-top-k``
shortlist rather than a full ``[n_db, n_q]`` matrix, which a pooled traverse cannot afford to
build (NSAVP's is 100,958 x 20,954 = 8.5 GB). That is exact, not an approximation: re-ranking
only ever *subtracts* ``inlier_weight`` per inlier from a shortlisted candidate, so nothing
outside the shortlist can move and the re-ordered top-20 is the same one the full column would
have given.

Whitening is reported at the shared power 0.5 and at Event-GeM's own 1.0 (``extra_pca_powers`` in
``src/methods.py``, where upstream measures full whitening as worth ~10 points of R@50). The
descriptor is 128-d, so ``pca_fit`` clamps both to a full-rank 128 basis and only the power
differs.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

import eventcv as ecv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src import inference as inf  # noqa: E402
from src import nsavpgps as ng  # noqa: E402
from brisbane_resolution import _Args  # noqa: E402
import brisbane_pooled as bp  # noqa: E402
import nsavp_pooled as npl  # noqa: E402
from tokyo_trajectory import parse_pca  # noqa: E402

DEFAULT_OUT = "/media/adam/vprdatasets/megaevent/eventgem_pooled"
# Per dataset: (eventlab dir, query, database, npz root for the slice-time check)
DATASETS = {
    "brisbane_event": ("/media/adam/vprdatasets/eventgem", "sunset1",
                       ("daytime", "morning", "night", "sunrise"),
                       "/media/adam/vprdatasets/megaevent/brisbane_npz"),
    "nsavp": ("/media/adam/vprdatasets/eventlab", "R0_FA0",
              ("R0_FN0", "R0_FS0", "R0_RA0", "R0_RN0", "R0_RS0"), None),
}
# 128-d descriptors: `pca_fit` clamps the dim to the descriptor width, so both are the full-rank
# basis and the only difference is the power.
PCA_SETTINGS = ((128, 0.5), (128, 1.0))
MCTS_MAX_WINDOW_MS = 30.0       # eventcv's default, and what Event-GeM gets by never passing it
EVENTGEM_P = 5.0                # --gem-p, upstream's exponent
BATCH = 32


def traverse_offset_ms(dataset, traverse):
    """Brisbane's per-traverse framing offset, in ms — the one ``EventStreamDataset`` applies.

    Read from the same ``brisbane_event.yaml`` key rather than re-derived, because the frame
    grid these descriptors sit on has to be the grid ``src/traversegps.py`` puts coordinates on.
    Any other dataset frames from the first event, at offset 0.
    """
    if dataset != "brisbane_event":
        return 0
    with open(os.path.join(inf.EVENTLAB_DATASETS, "brisbane_event.yaml")) as handle:
        spec = yaml.safe_load(handle)
    return spec["other"]["offset"][traverse] * 1000


class MctsStreamDataset(Dataset):
    """Fixed ``dt_ms`` slices of one recording as SuperEvent MCTS frames, ``[10, H, W]`` float32.

    The reader is opened lazily and never pickled — eventcv's is not fork-safe, so each
    DataLoader worker builds its own. Same contract as
    :class:`src.inference.EventStreamDataset`; what differs is that MCTS is already in [0, 1]
    (so there is no divide-by-255) and that the resize happens in the **event domain**, on the
    reader, before rendering. ``src/npzdata.py::load_mcts`` documents why: rebinning the events
    keeps each pixel's decay value exact, where resizing a rendered frame would average pixels
    that fired at different times.
    """

    def __init__(self, dataset, traverse, path, size, dt_ms, sensor,
                 hot_pixel=True, filter_dt_us=None):
        self.dataset, self.traverse, self.path = dataset, traverse, path
        self.size, self.dt_ms, self.sensor = tuple(size), dt_ms, tuple(sensor)
        self.hot_pixel, self.filter_dt_us = bool(hot_pixel), filter_dt_us
        self.offset = traverse_offset_ms(dataset, traverse)
        self._reader = None
        reader = self._open()
        self._n = int(reader.n_slices)
        frame = np.asarray(reader[0])
        if frame.shape != (10, self.size[0], self.size[1]):
            raise ValueError(f"{traverse}: mcts renders {frame.shape}, expected "
                             f"(10, {self.size[0]}, {self.size[1]})")
        self._reader = None

    def _open(self):
        reader = ecv.open(self.path, dt_ms=self.dt_ms, sensor_size=self.sensor,
                          hot_pixel_filter=self.hot_pixel, offset=self.offset)
        if self.filter_dt_us is not None:
            reader = reader.background_activity_filter(int(self.filter_dt_us))
        reader = reader.resize(width=self.size[1], height=self.size[0])
        return reader.with_repr("mcts", max_window_ms=MCTS_MAX_WINDOW_MS)

    @property
    def reader(self):
        if self._reader is None:
            self._reader = self._open()
        return self._reader

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return torch.from_numpy(np.ascontiguousarray(np.asarray(self.reader[i])))

    def __getstate__(self):
        return {**self.__dict__, "_reader": None}


class PooledFrames(Dataset):
    """The pooled gallery (or the query set) as one indexable MCTS frame source.

    Re-ranking needs the *frames* behind the descriptor rows, and a pooled row index means
    nothing to any single traverse. ``rows`` is the same ``(traverse, local frame)`` mapping
    :func:`brisbane_pooled.pool_database` builds its descriptor array from — rebuilt here from
    the same ``covered`` masks, in the same traverse order, so row *i* of the bank and row *i*
    of this dataset are the same 50 ms slice.
    """

    def __init__(self, streams, rows):
        self.streams = streams                  # {traverse: MctsStreamDataset}
        self.rows = np.asarray(rows, dtype=object) if not isinstance(rows, np.ndarray) else rows

    @classmethod
    def build(cls, streams, sequences, covered_of):
        rows = [(seq, i) for seq in sequences
                for i in np.nonzero(np.asarray(covered_of[seq]))[0]]
        return cls(streams, np.array(rows, dtype=object))

    def select(self, indices):
        return PooledFrames(self.streams, self.rows[np.asarray(indices)])

    @property
    def keys(self):
        """Manifest identifiers. Unique per basename, which is what the guard compares."""
        return [f"{seq}_{int(i):06d}.mcts" for seq, i in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        seq, local = self.rows[i]
        return self.streams[seq][int(local)]


@torch.no_grad()
def extract(model, cfg, crop, traverse, args, cli, out_dir, tag, device):
    """One pass over a traverse -> ``({label: path}, n_frames)``, matching ``extract``'s contract.

    Only SuperEvent's trunk and FPN run. ``SuperEvent.forward`` would also evaluate the
    detector and descriptor heads, and the descriptor head alone materialises a
    ``[B, 256, H, W]`` map that costs more than everything else put together — see
    ``src/methods.py::EventGeMMethod.descriptors``, which this mirrors exactly.
    """
    import time

    files = {"eventgem": os.path.join(out_dir, f"{tag}_{traverse}_eventgem.npy")}
    ds = MctsStreamDataset(args.dataset, traverse, inf.sequence_path(args, traverse),
                           cli.eventgem_size, args.dt_ms, inf.sensor_size(args.dataset),
                           hot_pixel=not args.no_hot_pixel, filter_dt_us=args.filter_dt_us)
    n = len(ds)
    if os.path.exists(files["eventgem"]):
        print(f"    {tag} {traverse}: {n} slices — bank cached, skipping")
        return files, n

    off_top, off_left, hc, wc = crop
    loader = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, num_workers=cli.workers,
                        pin_memory=True, drop_last=False)
    bank, written, start = None, 0, time.time()
    for batch in loader:
        x = batch.to(device, non_blocking=True)
        if (hc, wc) != tuple(cli.eventgem_size):
            x = x[:, :, off_top:off_top + hc, off_left:off_left + wc]
        f = model.fpn(model.backbone(x)).float()
        # GeM, feature_extraction.py:68. p=5 is a near-max pool, so the clamp is what keeps
        # the backward-compatible zero features from collapsing the root.
        pooled = torch.nn.functional.avg_pool2d(
            f.clamp(min=1e-6).pow(EVENTGEM_P), (f.shape[-2], f.shape[-1])).pow(1.0 / EVENTGEM_P)
        desc = torch.nn.functional.normalize(
            pooled.squeeze(-1).squeeze(-1), p=2, dim=1).cpu().numpy()
        if bank is None:
            bank = np.lib.format.open_memmap(files["eventgem"], mode="w+", dtype=np.float32,
                                             shape=(n, desc.shape[1]))
        bank[written:written + desc.shape[0]] = desc
        written += desc.shape[0]
        if written % (cli.batch_size * 40) == 0 or written == n:
            rate = written / (time.time() - start)
            print(f"    {tag} {traverse}: {written}/{n}  {rate:.1f} img/s  "
                  f"eta {(n - written) / rate / 60:.1f} min", flush=True)
    bank.flush()
    return files, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="brisbane_event", choices=sorted(DATASETS))
    ap.add_argument("--query", default=None)
    ap.add_argument("--database", nargs="+", default=None)
    ap.add_argument("--pca", nargs="+", metavar="DIM,POWER", default=None)
    ap.add_argument("--arms", nargs="+", default=["on"], choices=["on", "off"],
                    help="background-activity filter arms. 'on' matches the protocol the "
                         "shipping models were scored under; upstream Event-GeM applies no "
                         "such filter, so 'off' is the like-for-like-with-upstream arm.")
    ap.add_argument("--threshold-m", type=float, default=25.0)
    ap.add_argument("--thresholds-m", type=float, nargs="+", default=[25.0, 50.0, 75.0, 100.0])
    ap.add_argument("--event-filter-dt-us", type=int, default=50000)
    ap.add_argument("--eventgem-repo", default="./external/eventgem")
    ap.add_argument("--eventgem-size", type=int, nargs=2, default=[240, 320],
                    help="(H, W) grid. SuperEvent's own geometry, a cropped DAVIS346.")
    # Event-GeM's local stage. Defaults are upstream's own, matching main.py.
    ap.add_argument("--no-rerank", dest="rerank", action="store_false",
                    help="global stage only; skip the homography verification pass")
    ap.add_argument("--eventgem-top-k", type=int, default=50,
                    help="candidates per query that Event-GeM re-ranks by homography")
    ap.add_argument("--ransac-thresh", type=float, default=5.0)
    ap.add_argument("--inlier-weight", type=float, default=0.05)
    ap.add_argument("--match-filter", default="mutual", choices=["mutual", "ratio"])
    ap.add_argument("--match-ratio", type=float, default=0.8)
    ap.add_argument("--eventlab-dir", default=None)
    ap.add_argument("--npz-root", default=None)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--no-hot-pixel", action="store_true")
    ap.add_argument("--score-chunk", type=int, default=256)
    ap.add_argument("--db-chunk", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=8)
    cli = ap.parse_args()

    eventlab_dir, query, database, npz_root = DATASETS[cli.dataset]
    cli.query = cli.query or query
    cli.database = cli.database or list(database)
    cli.eventlab_dir = cli.eventlab_dir or eventlab_dir
    cli.npz_root = cli.npz_root or npz_root
    cli.out_json = cli.out_json or f"{cli.dataset}.json"
    if cli.threshold_m not in cli.thresholds_m:
        cli.thresholds_m = sorted([cli.threshold_m, *cli.thresholds_m])
    cli.pca = parse_pca(cli.pca, PCA_SETTINGS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cli.out_dir, exist_ok=True)
    sequences = [cli.query, *cli.database]

    args = _Args(cli.eventlab_dir, cli.dataset, cli.dt_ms, cli.no_hot_pixel, True)
    args.eventgem_repo = cli.eventgem_repo

    from src import eventgemlocal as egl
    egl.add_to_path(args)
    model, cfg, _ = egl.build_superevent(egl.superevent_root(args), device)
    crop = egl.crop_offsets(*cli.eventgem_size, egl.input_multiple(cfg))

    print(f"eventgem/{cli.dataset}: query {cli.query} -> database {'+'.join(cli.database)}")
    print(f"  {cli.threshold_m:g} m radius, dt {cli.dt_ms} ms, mcts "
          f"{cli.eventgem_size[0]}x{cli.eventgem_size[1]}, gem_p {EVENTGEM_P}, arms {cli.arms}")

    if cli.dataset == "nsavp":
        root = os.path.join(cli.eventlab_dir, cli.dataset)
        geom = npl.traverse_geometry(root, sequences, cli.dt_ms)
    else:
        geom = bp.traverse_geometry(args, sequences, cli.npz_root)

    all_results = {}
    for arm in cli.arms:
        args.no_event_filter = (arm == "off")
        args.filter_dt_us = None if arm == "off" else cli.event_filter_dt_us
        tag = f"r{cli.eventgem_size[0]}" + ("" if arm == "off" else f"ba{cli.dt_ms}")
        print(f"\n{'=' * 78}\n{tag}  (filter {arm})\n{'=' * 78}")

        bank_files, bank_frames = {}, {}
        for seq in sequences:
            files, n = extract(model, cfg, crop, seq, args, cli, cli.out_dir, tag, device)
            bank_files[seq], bank_frames[seq] = files, n
        torch.cuda.empty_cache()

        for seq in sequences:
            if bank_frames[seq] != len(geom[seq][0]):
                raise SystemExit(
                    f"{seq}: eventcv yields {bank_frames[seq]} slices but the coordinate grid "
                    f"has {len(geom[seq][0])}. The descriptors and the ground truth are on "
                    f"different frame grids.")

        cli.rank_fn = None
        if cli.rerank:
            streams = {seq: MctsStreamDataset(
                args.dataset, seq, inf.sequence_path(args, seq), cli.eventgem_size,
                args.dt_ms, inf.sensor_size(args.dataset), hot_pixel=not args.no_hot_pixel,
                filter_dt_us=args.filter_dt_us) for seq in sequences}
            covered = {seq: geom[seq][1] for seq in sequences}
            db_frames = PooledFrames.build(streams, list(cli.database), covered)
            q_frames = PooledFrames.build(streams, [cli.query], covered)
            reranker = egl.LocalReranker(
                model, cfg, fast_nms, device, cli.eventgem_size,
                cache_dir=os.path.join(cli.out_dir, "keypoints"),
                tag=f"{cli.dataset}_{tag}", top_k=cli.eventgem_top_k,
                ransac_thresh=cli.ransac_thresh, inlier_weight=cli.inlier_weight,
                match_filter=cli.match_filter, match_ratio=cli.match_ratio)
            cli.rank_fn = make_rank_fn(reranker, db_frames, q_frames, cli)

        res = bp.score_configuration(bank_files, geom, args, cli, "eventgem", device, tag)
        res["filter_arm"] = arm
        res["event_filter_dt_us"] = None if arm == "off" else cli.event_filter_dt_us
        res["method"] = "eventgem"
        res["stage"] = "global"
        res["mcts_size"] = list(cli.eventgem_size)
        all_results[tag] = res

        out_json = os.path.join(cli.out_dir, cli.out_json)
        tmp = out_json + ".tmp"
        with open(tmp, "w") as handle:
            json.dump({"dataset": cli.dataset, "method": "eventgem", "stage": "global",
                       "query": cli.query, "database": list(cli.database),
                       "threshold_m": cli.threshold_m, "dt_ms": cli.dt_ms,
                       "pca": [list(s) for s in cli.pca],
                       "hot_pixel": not cli.no_hot_pixel, "results": all_results},
                      handle, indent=2)
        os.replace(tmp, out_json)

    print(f"\n{'=' * 78}\nR@1 / R@10 by tolerance (best descriptor space)")
    for tag, res in all_results.items():
        best = max(res["recall"], key=lambda s: res["recall"][s]["1"])
        cells = "  ".join(
            f"{t:g}m {res['recall_by_threshold'][f'{t:g}'][best]['1']:.3f}/"
            f"{res['recall_by_threshold'][f'{t:g}'][best]['10']:.3f}"
            for t in cli.thresholds_m)
        print(f"  {tag:14s} [{best}]  {cells}")
    print(f"\n-> {os.path.join(cli.out_dir, cli.out_json)}")


if __name__ == "__main__":
    main()
