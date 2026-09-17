"""v8 machinery: the accumulate-space noise augmentation and the pooled-eval GT.

Two invariants that the representation switch (memory: gept-diet-is-accumulate-not-
countmask) hangs on:

1. ``AccumulateDomainRandomise`` must respect accumulate's WHITE background — "no events"
   is 1.0, so gain must leave background untouched, dropout must write 1.0 (not 0.0 —
   a black blob means "maximal events of both polarities" in this space), and salt must
   darken the correct channels for each polarity (pos -> G,B; neg -> R,G).
2. ``pooled_gt`` must be a [nq, nr] Euclidean radius check in metres — the benchmark's
   ``build_gt`` semantics — because the trainer now SELECTS on it.
"""

import random
import unittest

import numpy as np
import torch

import megaevent.training.dataset as ds  # noqa: E402
from megaevent.training.evalsuite import pooled_gt  # noqa: E402


def _frame(base=1.0):
    x = torch.full((3, 8, 10), float(base))
    return x


class TestAccumulateDomainRandomise(unittest.TestCase):
    def test_gain_preserves_white_background(self):
        aug = ds.AccumulateDomainRandomise(gain=(0.5, 2.0), dropout_max=0.0, salt_prob=0.0)
        random.seed(0)
        torch.manual_seed(0)
        out = aug(_frame(1.0))
        self.assertTrue(torch.all(out == 1.0), "gain moved the empty background")

    def test_gain_scales_intensity_not_channel(self):
        # one pixel holding a positive event: B and G carry intensity 0.4 (= 1 - 0.6)
        x = _frame(1.0)
        x[1, 3, 4] = 0.6
        x[2, 3, 4] = 0.6
        aug = ds.AccumulateDomainRandomise(gain=(0.5, 0.5 + 1e-12), dropout_max=0.0, salt_prob=0.0)
        random.seed(0)
        out = aug(x)
        self.assertAlmostEqual(float(out[2, 3, 4]), 1.0 - 0.5 * 0.4, places=5)
        self.assertAlmostEqual(float(out[0, 3, 4]), 1.0, places=6)

    def test_dropout_writes_background_not_black(self):
        x = _frame(1.0)
        x[:, 2, 2] = 0.3  # an active pixel
        aug = ds.AccumulateDomainRandomise(gain=(1.0, 1.0), dropout_max=1.0, salt_prob=0.0)
        random.seed(3)  # any draw: pixels are original or exactly 1.0
        torch.manual_seed(3)
        out = aug(x)
        changed = (out != x).any(dim=0)
        self.assertTrue(
            torch.all(out[:, changed] == 1.0), "a dropped pixel must become white background"
        )
        self.assertFalse(torch.any(out == 0.0), "0.0 would mean maximal events, not none")

    def test_salt_darkens_polarity_consistent_channels(self):
        aug = ds.AccumulateDomainRandomise(gain=(1.0, 1.0), dropout_max=0.0, salt_prob=1.0)
        random.seed(1)
        torch.manual_seed(1)
        out = aug(_frame(1.0))
        salted = (out != 1.0).any(dim=0)
        self.assertTrue(bool(salted.any()), "salt_prob=1 produced no events")
        unit = 1.0 / 255.0  # all-white frame -> fallback unit
        for r, c in torch.nonzero(salted):
            self.assertAlmostEqual(
                float(out[1, r, c]),
                1.0 - unit,
                places=6,
                msg="any event must darken the activity channel G",
            )
            red, blue = float(out[0, r, c]), float(out[2, r, c])
            self.assertTrue(
                (red == 1.0) != (blue == 1.0), "exactly one polarity channel must darken"
            )

    def test_build_transforms_selects_background_aware_class(self):
        cfg = type(
            "C",
            (),
            dict(
                tencode_mean=[0.9, 0.8, 0.9],
                tencode_std=[0.2, 0.3, 0.2],
                H=224,
                W=224,
                representation="accumulate",
                aug_domain_rand=True,
                aug_hflip=False,
                aug_gain_jitter=(0.7, 1.4),
                aug_dropout_max=0.8,
                aug_salt_prob=0.02,
            ),
        )()
        tf = ds.build_transforms(cfg, train=True)
        kinds = [type(t).__name__ for t in tf.transforms]
        self.assertIn("AccumulateDomainRandomise", kinds)
        self.assertNotIn("CountmaskDomainRandomise", kinds)
        cfg.representation = "tencode"
        with self.assertRaises(ValueError):
            ds.build_transforms(cfg, train=True)


class TestPooledGT(unittest.TestCase):
    def test_radius_and_orientation(self):
        db = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 24.0]])
        q = np.array([[0.0, 0.0], [100.0, 100.0]])
        gt = pooled_gt(db, q, threshold_m=25.0)
        self.assertEqual(gt.shape, (2, 3))  # [nq, nr]
        self.assertTrue(gt[0, 0])  # 0 m
        self.assertFalse(gt[0, 1])  # 30 m > 25
        self.assertTrue(gt[0, 2])  # 24 m <= 25
        self.assertFalse(gt[1].any())  # unanswerable query


if __name__ == "__main__":
    unittest.main()
