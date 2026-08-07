"""The pitts250k layout and the geographic ground truth it feeds.

``pitts250k`` is the official NetVLAD test split: database and query tiles come from
different Street View captures and positives are a 25 m GPS band. That makes it the
opposite of the ``pitts`` set already in the repo, whose supplied ground truth pairs each
query with the other 23 tiles of its *own* panorama. These tests pin the distinction down
where it is easiest to break silently -- the UTM that ends up in the file name, and the
tile index that keeps 24 tiles of one panorama from colliding onto one output name.
"""
import os
import sys
import unittest

import numpy as np

from src.imagesets import build_radius_gt
from src.npzdata import utm_from_paths

I2E_ROOT = "/home/adam/repo/I2E"
if I2E_ROOT not in sys.path:
    sys.path.insert(0, I2E_ROOT)


def _layout():
    try:
        import pitts250k
    except ImportError as err:                  # pragma: no cover - I2E not checked out
        raise unittest.SkipTest(f"no I2E checkout at {I2E_ROOT}: {err}")
    return pitts250k


class Pitts250kLayoutTests(unittest.TestCase):
    def test_tile_index_is_unique_over_the_24_tiles(self):
        layout = _layout()
        seen = {layout.tile_number(pitch, yaw)
                for pitch in (1, 2) for yaw in range(1, 13)}
        self.assertEqual(seen, set(range(24)))

    def test_tile_number_rejects_out_of_range(self):
        layout = _layout()
        with self.assertRaises(layout.Pitts250kError):
            layout.tile_number(1, 13)
        with self.assertRaises(layout.Pitts250kError):
            layout.tile_number(0, 1)

    def test_perspective_name_parsing(self):
        layout = _layout()
        self.assertEqual(layout.parse_perspective_name("000062_pitch1_yaw11"),
                         ("000062", 1, 11))
        with self.assertRaises(layout.Pitts250kError):
            layout.parse_perspective_name("000062_yaw11")

    def test_utm_survives_the_name_round_trip(self):
        """The evaluator reads coordinates back out of the file name, so that is the
        only place the UTM has to be exact -- to the 1 cm the field encodes."""
        layout = _layout()
        for east, north in ((585042.68, 4476938.85), (583289.0, 4475171.5),
                            (586437.25, 4477189.0)):
            lat, lon = layout.to_latlon(east, north, layout.DEFAULT_ZONE,
                                        layout.DEFAULT_BAND)
            name = layout.get_dst_image_name(lat, lon, pano_id="000062", tile_num=0,
                                             extension=".npz")
            back = utm_from_paths([name])[0]
            self.assertAlmostEqual(back[0], east, places=1)
            self.assertAlmostEqual(back[1], north, places=1)
            self.assertEqual(name.split("@")[3:5], ["17", "T"])

    def test_one_panorama_yields_24_distinct_names_at_one_coordinate(self):
        """Every tile of a panorama shares its UTM, so only the tile field separates
        them. A collision here would silently drop 23 of every 24 images."""
        layout = _layout()
        lat, lon = layout.to_latlon(585042.68, 4476938.85, layout.DEFAULT_ZONE,
                                    layout.DEFAULT_BAND)
        names = {layout.get_dst_image_name(lat, lon, pano_id="000062",
                                           tile_num=layout.tile_number(pitch, yaw),
                                           pitch=pitch, extension=".npz")
                 for pitch in (1, 2) for yaw in range(1, 13)}
        self.assertEqual(len(names), 24)
        coords = utm_from_paths(sorted(names))
        self.assertEqual(len(np.unique(coords, axis=0)), 1)


class Pitts250kGroundTruthTests(unittest.TestCase):
    def test_positives_are_a_symmetric_25m_band(self):
        db = np.array([[0.0, 0.0], [10.0, 0.0], [30.0, 0.0], [100.0, 0.0]])
        q = np.array([[0.0, 0.0], [100.0, 0.0]])
        gt = build_radius_gt(db, q, 25.0)
        self.assertEqual(gt.shape, (4, 2))
        np.testing.assert_array_equal(gt[:, 0], [True, True, False, False])
        np.testing.assert_array_equal(gt[:, 1], [False, False, False, True])

    def test_orientation_is_database_by_query(self):
        """``[db, q]`` is the orientation every scorer in the repo assumes; a transpose
        here would score the benchmark against itself without failing loudly."""
        db = np.zeros((7, 2))
        q = np.zeros((3, 2))
        self.assertEqual(build_radius_gt(db, q, 25.0).shape, (7, 3))


if __name__ == "__main__":
    unittest.main()
