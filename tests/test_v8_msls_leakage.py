"""MSLS is both a training stream and an evaluation set — the exclusions must be guarded.

Two guards: a typo'd ``exclude_cities`` entry is an error (existing behaviour, pinned),
and the six official MSLS *test* cities can never enter a training index no matter what
flags are passed (new, ``dataset.MSLS_TEST_CITIES``). The second exists because the
megaevent benchmark scores exactly those cities; a metadata tree that mixes the test
split into train_val would otherwise leak silently.
"""

import os
import tempfile
import unittest

import megaevent.training.dataset as ds  # noqa: E402


def _meta_root(tmp, cities):
    root = os.path.join(tmp, "train_val")
    for city in cities:
        os.makedirs(os.path.join(root, city, "database"))
    return root


class MslsLeakageTests(unittest.TestCase):
    def _build(self, tmp, tree_cities, **kwargs):
        return ds.build_msls_index(
            images_dir=os.path.join(tmp, "images"),
            meta_root=_meta_root(tmp, tree_cities),
            refresh=True,
            cache_dir=tmp,
            **kwargs,
        )

    def test_test_city_in_metadata_tree_is_an_error(self):
        for leaked in ("miami", "stockholm"):
            with (
                tempfile.TemporaryDirectory() as tmp,
                self.assertRaises(ValueError, msg=leaked) as ctx,
            ):
                self._build(tmp, ["zurich", leaked])
            self.assertIn(leaked, str(ctx.exception))

    def test_no_flag_can_admit_a_test_city(self):
        # Even naming it in `cities` explicitly must fail — the guard is unconditional.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._build(tmp, ["athens"], cities=["athens"])

    def test_unknown_exclude_city_is_an_error_not_a_warning(self):
        # A typo here would silently leave a validation city training (dataset.py's own
        # rationale); pinned so it stays an error.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                self._build(tmp, ["zurich"], exclude_cities=["cph"])
            self.assertIn("cph", str(ctx.exception))

    def test_constant_matches_the_official_test_split(self):
        self.assertEqual(
            sorted(ds.MSLS_TEST_CITIES),
            ["athens", "bengaluru", "buenosaires", "kampala", "miami", "stockholm"],
        )


if __name__ == "__main__":
    unittest.main()
