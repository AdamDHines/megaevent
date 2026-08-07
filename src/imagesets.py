"""Dataset layouts and ground truth for independent-image event VPR benchmarks."""

from dataclasses import dataclass, replace
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.npzdata import list_npz, split_dir, utm_from_paths


MSLS_CITIES = ("miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala")
MSLS_UTM_ZONES = {
    "miami": 17,
    "athens": 34,
    "buenosaires": 21,
    "stockholm": 33,
    "bengaluru": 43,
    "kampala": 36,
}


@dataclass
class ImageSet:
    db_paths: list
    q_paths: list
    gt: np.ndarray
    db_keys: list
    q_keys: list
    db_utm: np.ndarray | None = None
    q_utm: np.ndarray | None = None
    excluded_db: int = 0
    excluded_q: int = 0

    def limited(self, limit):
        """Deterministic smoke-test subset with at least one positive per query."""
        if not limit:
            return self
        q_keep = np.flatnonzero(self.gt.any(axis=0))[:limit]
        if not len(q_keep):
            raise RuntimeError("cannot limit an image set with no scorable queries")

        required = []
        for q in q_keep:
            db = int(np.flatnonzero(self.gt[:, q])[0])
            if db not in required:
                required.append(db)
        fill = (i for i in range(len(self.db_paths)) if i not in required)
        db_keep = required[:limit]
        db_keep.extend(next(fill) for _ in range(min(limit - len(db_keep),
                                                     len(self.db_paths) - len(db_keep))))
        db_keep = np.asarray(sorted(db_keep), dtype=np.int64)
        q_keep = np.asarray(q_keep, dtype=np.int64)
        return replace(
            self,
            db_paths=[self.db_paths[i] for i in db_keep],
            q_paths=[self.q_paths[i] for i in q_keep],
            gt=self.gt[np.ix_(db_keep, q_keep)],
            db_keys=[self.db_keys[i] for i in db_keep],
            q_keys=[self.q_keys[i] for i in q_keep],
            db_utm=self.db_utm[db_keep] if self.db_utm is not None else None,
            q_utm=self.q_utm[q_keep] if self.q_utm is not None else None,
        )


def build_radius_gt(db_utm, q_utm, threshold_m):
    """Boolean ``[database, query]`` positives within a metric radius."""
    neighbours = NearestNeighbors(n_jobs=-1).fit(db_utm)
    positives = neighbours.radius_neighbors(q_utm, radius=threshold_m,
                                             return_distance=False)
    gt = np.zeros((len(db_utm), len(q_utm)), dtype=bool)
    for q, indexes in enumerate(positives):
        gt[indexes, q] = True
    return gt


def project_utm(lon_lat, zone, northern=True):
    """WGS84 longitude/latitude degrees to one UTM zone, as ``[easting, northing]``."""
    lon_lat = np.asarray(lon_lat, dtype=np.float64)
    lon, lat = np.radians(lon_lat[:, 0]), np.radians(lon_lat[:, 1])
    a, f, k0 = 6_378_137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    tan_lat = np.tan(lat)
    n = a / np.sqrt(1 - e2 * sin_lat ** 2)
    t = tan_lat ** 2
    c = ep2 * cos_lat ** 2
    aa = cos_lat * (lon - lon0)
    m = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024)
             * np.sin(2 * lat)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * lat)
             - (35 * e2 ** 3 / 3072) * np.sin(6 * lat))
    east = (k0 * n * (aa + (1 - t + c) * aa ** 3 / 6
                          + (5 - 18 * t + t ** 2 + 72 * c - 58 * ep2)
                          * aa ** 5 / 120) + 500_000)
    north = k0 * (m + n * tan_lat * (aa ** 2 / 2
                                      + (5 - t + 9 * c + 4 * c ** 2) * aa ** 4 / 24
                                      + (61 - 58 * t + t ** 2 + 600 * c - 330 * ep2)
                                      * aa ** 6 / 720))
    if not northern:
        north += 10_000_000
    return np.column_stack((east, north))


def _numeric_npz(directory):
    paths = list_npz(directory)
    try:
        ids = [int(os.path.splitext(os.path.basename(path))[0]) for path in paths]
    except ValueError as err:
        raise ValueError(f"non-numeric .npz name in {directory}") from err
    order = np.argsort(ids)
    return [paths[i] for i in order], [ids[i] for i in order]


def load_pitts(root):
    db_paths, db_ids = _numeric_npz(os.path.join(root, "ref_countmask", "numpy"))
    q_paths, q_ids = _numeric_npz(os.path.join(root, "query_countmask", "numpy"))
    db_row = {key: row for row, key in enumerate(db_ids)}
    q_col = {key: col for col, key in enumerate(q_ids)}
    if len(db_row) != len(db_ids) or len(q_col) != len(q_ids):
        raise ValueError("Pitts contains duplicate numeric image ids")

    gt_path = os.path.join(root, "ground_truth_new.npy")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"no Pitts ground truth at {gt_path}")
    rows = np.load(gt_path, allow_pickle=True)
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError(f"Pitts ground truth must be [N,2], got {rows.shape}")

    gt = np.zeros((len(db_paths), len(q_paths)), dtype=bool)
    seen = set()
    for query_id, positive_ids in rows:
        query_id = int(query_id)
        if query_id in seen:
            raise ValueError(f"duplicate Pitts ground-truth row for query {query_id}")
        seen.add(query_id)
        if query_id not in q_col:
            raise ValueError(f"Pitts ground truth names missing query {query_id}")
        try:
            indexes = [db_row[int(db_id)] for db_id in positive_ids]
        except KeyError as err:
            raise ValueError(f"Pitts ground truth names missing reference {err.args[0]}") from err
        gt[indexes, q_col[query_id]] = True
    missing = sorted(set(q_ids) - seen)
    if missing:
        raise ValueError(f"Pitts ground truth omits {len(missing)} queries, first {missing[0]}")
    return ImageSet(db_paths, q_paths, gt,
                    [str(key) for key in db_ids], [str(key) for key in q_ids])


def _read_msls_split(meta_root, event_root, city, split):
    directory = os.path.join(meta_root, city, split)
    required = [os.path.join(directory, name) for name in
                ("subtask_index.csv", "raw.csv")]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "MSLS official test ground truth is incomplete; missing "
            + ", ".join(missing)
            + ". Run scripts/recover_msls_test_metadata.py to recover the public "
              "Mapillary capture metadata.")

    subtask = pd.read_csv(required[0], index_col=0)
    raw = pd.read_csv(required[1], index_col=0).set_index("key", drop=False)
    post_path = os.path.join(directory, "postprocessed.csv")
    post = (pd.read_csv(post_path, index_col=0).set_index("key", drop=False)
            if os.path.exists(post_path) else None)
    selected = subtask.loc[subtask["all"].astype(bool), "key"].tolist()
    missing_keys = [key for key in selected
                    if key not in raw.index or (post is not None and key not in post.index)]
    if missing_keys:
        raise ValueError(f"{city}/{split}: metadata omits key {missing_keys[0]}")

    if "available" in raw:
        available = raw["available"].fillna(False).astype(bool)
        selected = [key for key in selected if available.at[key]]
    excluded = int(subtask["all"].astype(bool).sum()) - len(selected)
    keys = [key for key in selected if not bool(raw.at[key, "pano"])]
    paths = [os.path.join(event_root, city, split, "images", f"{key}.npz") for key in keys]
    absent = [path for path in paths if not os.path.exists(path)]
    if absent:
        raise FileNotFoundError(f"{city}/{split}: no converted event file for {absent[0]}")
    if post is not None:
        coords = post.loc[keys, ["easting", "northing"]].to_numpy(dtype=np.float64)
    else:
        lon_lat = raw.loc[keys, ["lon", "lat"]].to_numpy(dtype=np.float64)
        zone = MSLS_UTM_ZONES.get(city, int((np.median(lon_lat[:, 0]) + 180) // 6) + 1)
        coords = project_utm(lon_lat, zone, northern=np.median(lon_lat[:, 1]) >= 0)
    return keys, paths, coords, excluded


def load_msls(root, threshold_m):
    meta_root = os.path.join(root, "test")
    event_root = os.path.join(root, "test_countmask", "numpy")
    db_paths, q_paths, db_keys, q_keys = [], [], [], []
    positive_rows = []
    excluded_db = excluded_q = 0

    for city in MSLS_CITIES:
        city_db_keys, city_db_paths, db_coords, city_excluded_db = _read_msls_split(
            meta_root, event_root, city, "database")
        city_q_keys, city_q_paths, q_coords, city_excluded_q = _read_msls_split(
            meta_root, event_root, city, "query")
        excluded_db += city_excluded_db
        excluded_q += city_excluded_q
        offset = len(db_paths)
        neighbours = NearestNeighbors(algorithm="brute").fit(db_coords)
        positives = neighbours.radius_neighbors(q_coords, radius=threshold_m,
                                                 return_distance=False)

        db_paths.extend(city_db_paths)
        db_keys.extend(city_db_keys)
        for key, path, indexes in zip(city_q_keys, city_q_paths, positives):
            if not len(indexes):
                continue
            q_keys.append(key)
            q_paths.append(path)
            positive_rows.append(np.asarray(indexes, dtype=np.int64) + offset)

    gt = np.zeros((len(db_paths), len(q_paths)), dtype=bool)
    for q, indexes in enumerate(positive_rows):
        gt[indexes, q] = True
    return ImageSet(db_paths, q_paths, gt, db_keys, q_keys,
                    excluded_db=excluded_db, excluded_q=excluded_q)


def load_geographic(args):
    db_paths = list_npz(split_dir(args, args.ref))
    q_paths = list_npz(split_dir(args, args.query))
    db_utm, q_utm = utm_from_paths(db_paths), utm_from_paths(q_paths)
    gt = build_radius_gt(db_utm, q_utm, args.positive_dist_threshold)
    return ImageSet(db_paths, q_paths, gt,
                    [os.path.splitext(os.path.basename(path))[0] for path in db_paths],
                    [os.path.splitext(os.path.basename(path))[0] for path in q_paths],
                    db_utm, q_utm)


def load_image_set(args):
    root = os.path.join(args.data_dir, args.dataset)
    if args.dataset == "pitts":
        dataset = load_pitts(root)
    elif args.dataset == "msls":
        dataset = load_msls(root, args.positive_dist_threshold)
    else:
        dataset = load_geographic(args)
    return dataset.limited(getattr(args, "limit", None))
