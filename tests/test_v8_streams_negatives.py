"""Place-id layout across streams: measured, not assumed.

The ``place_offset`` blocks never actually made ids globally unique (see the
``streams.py`` module docstring): builders mint ids far outside their nominal 100M
blocks, and sf_xl_frontal/lateral collide *by construction*. That is harmless under the
per-stream loss and poisonous for anything that merges streams — the shared-queue XBM
already did, so its v2 "measured negative" verdict is confounded. These tests pin the
collision mechanics, the ``check_id_ranges`` guard that now reports them, and the
GSV-Cities minting that overruns its block from city 10 on.
"""

import contextlib
import io
import os
import tempfile
import unittest

import megaevent.training.dataset as ds  # noqa: E402
from megaevent.training import streams


def _sf_xl_id(offset, easting, northing, M=20):
    """The exact minting formula of ``build_sf_xl_index`` (dataset.py)."""
    return offset + (easting // M) * 1_000_000 + (northing // M)


class SfXlCollisionTests(unittest.TestCase):
    def test_frontal_and_lateral_collide_two_km_apart(self):
        # Offsets differ by 1e8 = 100 easting cells x 1e6, so a frontal cell 100 cells
        # (2 km at M=20) east of a lateral cell shares its id exactly.
        frontal, lateral, M = 100_000_000, 200_000_000, 20
        ce, cn = 551_000, 4_180_000  # San Francisco-scale UTM
        self.assertEqual(_sf_xl_id(frontal, ce + 100 * M, cn), _sf_xl_id(lateral, ce, cn))

    def test_ids_dwarf_the_nominal_block(self):
        # ~2.7e10 for San Francisco: the "100M block" story was never true for SF-XL.
        self.assertGreater(_sf_xl_id(100_000_000, 551_000, 4_180_000), 10_000_000_000)


class CheckIdRangesTests(unittest.TestCase):
    def _run(self, indices):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ranges = streams.check_id_ranges(indices)
        return ranges, buf.getvalue()

    def test_disjoint_streams_pass_quietly(self):
        ranges, out = self._run(
            {"a": [(0, ["x"]), (99, ["x"])], "b": [(1000, ["x"]), (1999, ["x"])]}
        )
        self.assertEqual(ranges, {"a": (0, 99), "b": (1000, 1999)})
        self.assertNotIn("WARNING", out)

    def test_overlap_is_reported_for_every_colliding_pair(self):
        _, out = self._run(
            {
                "a": [(0, ["x"]), (150, ["x"])],
                "b": [(100, ["x"]), (200, ["x"])],
                "c": [(10_000, ["x"])],
            }
        )
        self.assertIn("WARNING", out)
        self.assertIn("a", out)
        self.assertIn("b", out)
        self.assertNotIn("c [", out.split("WARNING")[1].split("\n")[0])

    def test_touching_ranges_count_as_overlap(self):
        # A shared boundary id IS a collision — one id, two places.
        _, out = self._run({"a": [(0, ["x"]), (100, ["x"])], "b": [(100, ["x"]), (200, ["x"])]})
        self.assertIn("WARNING", out)


class GsvCityBlockTests(unittest.TestCase):
    def test_city_ten_overruns_the_100m_block(self):
        # 11 cities x city_offset 10M puts city 10's ids at >= 1e8 — outside gsv_cities'
        # nominal block. Built from a real (synthetic) tree so the whole filename-parsing
        # path is exercised, not just the arithmetic.
        with tempfile.TemporaryDirectory() as root:
            for i in range(11):
                city = f"City{i:02d}"
                os.makedirs(os.path.join(root, city))
                for img in range(4):
                    open(
                        os.path.join(root, city, f"{city}_0000042_2020_{img:02d}.jpg"), "w"
                    ).close()
            index = ds.build_gsv_cities_from_tree(root, min_img_per_place=4)
        ids = sorted(pid for pid, _ in index)
        self.assertEqual(len(ids), 11)
        self.assertEqual(ids[0], 42)  # city 0: bare pid
        self.assertEqual(ids[10], 10 * 10_000_000 + 42)
        self.assertGreaterEqual(ids[10], 100_000_000)  # overruns the nominal block


if __name__ == "__main__":
    unittest.main()
