import csv
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.imagesets import ImageSet, load_msls, load_pitts, project_utm
from src.scoring import map_at_k


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb"):
        pass


def _write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ImageSetTests(unittest.TestCase):
    def test_pitts_numeric_order_and_ground_truth_orientation(self):
        with tempfile.TemporaryDirectory() as root:
            for key in (2, 0, 1):
                _touch(os.path.join(root, "ref_countmask", "numpy", f"{key:07d}.npz"))
            for key in (1, 0):
                _touch(os.path.join(root, "query_countmask", "numpy", f"{key:07d}.npz"))
            rows = np.empty((2, 2), dtype=object)
            rows[0] = [1, [2]]
            rows[1] = [0, [0, 1]]
            np.save(os.path.join(root, "ground_truth_new.npy"), rows)

            dataset = load_pitts(root)
            self.assertEqual(dataset.db_keys, ["0", "1", "2"])
            self.assertEqual(dataset.q_keys, ["0", "1"])
            np.testing.assert_array_equal(dataset.gt,
                                          [[True, False], [True, False], [False, True]])

    def test_ground_truth_aware_limit_keeps_queries_scorable(self):
        gt = np.eye(4, dtype=bool)
        dataset = ImageSet(list("abcd"), list("wxyz"), gt,
                           list("abcd"), list("wxyz"))
        limited = dataset.limited(2)
        self.assertEqual(limited.gt.shape, (2, 2))
        self.assertTrue(limited.gt.any(axis=0).all())

    def _write_msls_city(self, root, city, db, query):
        for split, records in (("database", db), ("query", query)):
            base = os.path.join(root, "test", city, split)
            subtask_rows, raw_rows, post_rows = [], [], []
            for index, (key, east, north, pano) in enumerate(records):
                subtask_rows.append({"": index, "key": key, "all": True})
                raw_rows.append({"": index, "key": key, "lon": east, "lat": north,
                                 "pano": pano})
                post_rows.append({"": index, "key": key,
                                  "easting": east, "northing": north})
                _touch(os.path.join(root, "test_countmask", "numpy", city, split,
                                    "images", f"{key}.npz"))
            _write_csv(os.path.join(base, "subtask_index.csv"), ["", "key", "all"],
                       subtask_rows)
            _write_csv(os.path.join(base, "raw.csv"),
                       ["", "key", "lon", "lat", "pano"], raw_rows)
            _write_csv(os.path.join(base, "postprocessed.csv"),
                       ["", "key", "easting", "northing"], post_rows)

    def test_msls_filters_panoramas_and_keeps_cities_isolated(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_msls_city(root, "one",
                                  [("db1", 0, 0, False), ("dbp", 5, 0, True)],
                                  [("q1", 0, 0, False), ("qp", 0, 0, True)])
            self._write_msls_city(root, "two",
                                  [("db2", 0, 0, False)],
                                  [("q2", 0, 0, False)])
            with mock.patch("src.imagesets.MSLS_CITIES", ("one", "two")):
                dataset = load_msls(root, 25.0)

            self.assertEqual(dataset.db_keys, ["db1", "db2"])
            self.assertEqual(dataset.q_keys, ["q1", "q2"])
            np.testing.assert_array_equal(dataset.gt, np.eye(2, dtype=bool))

    def test_msls_uses_raw_gps_when_postprocessed_metadata_is_absent(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_msls_city(root, "one",
                                  [("db1", 153.0000, -27.0000, False)],
                                  [("q1", 153.0001, -27.0000, False),
                                   ("q2", 153.0010, -27.0000, False)])
            for split in ("database", "query"):
                base = os.path.join(root, "test", "one", split)
                os.unlink(os.path.join(base, "postprocessed.csv"))
                if split == "query":
                    raw_path = os.path.join(base, "raw.csv")
                    with open(raw_path, newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    for row in rows:
                        row["available"] = row["key"] != "q2"
                    _write_csv(raw_path,
                               ["", "key", "lon", "lat", "pano", "available"], rows)
            with mock.patch("src.imagesets.MSLS_CITIES", ("one",)):
                dataset = load_msls(root, 25.0)

            self.assertEqual(dataset.q_keys, ["q1"])
            self.assertEqual(dataset.excluded_q, 1)
            np.testing.assert_array_equal(dataset.gt, [[True]])

    def test_utm_projection_matches_official_msls_metadata(self):
        # First Copenhagen database row in the official metadata archive (zone 32N).
        actual = project_utm([[12.562913, 55.691903]], zone=32)
        expected = [[723923.0044317835, 6177544.100753511]]
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=0)

    def test_msls_map_matches_official_definition(self):
        gt = np.asarray([[True, False], [True, True], [False, False]])
        ranked = np.asarray([[0, 0], [2, 1], [1, 2]])
        metrics = map_at_k(ranked, gt, np.asarray([True, True]), ks=(1, 3))
        self.assertAlmostEqual(metrics[1], 0.5)
        # q0: (1 + 2/3) / 2; q1: 1/2. Their mean is 2/3.
        self.assertAlmostEqual(metrics[3], 2 / 3)


if __name__ == "__main__":
    unittest.main()
