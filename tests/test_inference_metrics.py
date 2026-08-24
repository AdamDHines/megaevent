"""The scoring path's three unguarded junctures: ``load_gt``, ``recall_at_k``, ``sim_matrix``.

Until 2026-08-19 none of these had a test: a cached bank was dot-producted blind (EventVLAD's
deliberately-unnormalised descriptors flow through the same ``sim_matrix`` as everyone's
cosines), the Event-LAB band was bilinearly resampled onto the descriptor grid with no check
that the band survived, and a ground truth with zero scorable queries produced NaN rather than
an error. These tests pin the guards added for each, plus the metric arithmetic itself on
matrices small enough to score by hand.
"""
import os
import tempfile
import unittest

import numpy as np
import torch

from src.inference import load_gt, recall_at_k, sim_matrix


class LoadGtTests(unittest.TestCase):
    def _write(self, arr):
        path = os.path.join(self._dir.name, "gt.npy")
        np.save(path, arr)
        return path

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)

    def test_exact_shape_passes_through_untouched(self):
        gt = np.zeros((40, 30), dtype=np.float32)
        gt[np.arange(30), np.arange(30)] = 1.0
        band, shape = load_gt(self._write(gt), 40, 30)
        self.assertEqual(shape, (40, 30))
        np.testing.assert_array_equal(band, gt.astype(bool))

    def test_resample_preserves_a_tolerance_band(self):
        # A ±5-row band around the diagonal of a 200x100 grid, resampled to 100x50. A band
        # that wide survives a 2x grid change; the guard must not fire on the normal case.
        gt = np.zeros((200, 100), dtype=np.float32)
        rows = np.arange(200)[:, None]
        cols = np.arange(100)[None, :]
        gt[np.abs(rows - 2 * cols) <= 5] = 1.0
        band, shape = load_gt(self._write(gt), 100, 50)
        self.assertEqual(shape, (200, 100))
        self.assertEqual(band.shape, (100, 50))
        ratio = band.mean() / gt.mean()
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 2.0)
        # Every query column that had a positive keeps one: the band's reason to exist.
        self.assertTrue(band.any(axis=0).all())

    def test_destroyed_band_raises_instead_of_scoring(self):
        # A one-frame-wide band resampled 100x thinner dilutes below the 0.5 threshold and
        # vanishes. That must stop the run: scoring against it reports NaN as a result.
        gt = np.zeros((1000, 1000), dtype=np.float32)
        gt[np.arange(1000), np.arange(1000)] = 1.0
        with self.assertRaises(SystemExit):
            load_gt(self._write(gt), 10, 10)


class RecallAtKTests(unittest.TestCase):
    def test_hand_computed_recalls(self):
        # 6 references x 4 queries. Positives: q0 -> r0 (its top-1), q1 -> r5 (ranked 2nd),
        # q2 -> r1 (ranked last), q3 -> nothing (unscorable, dropped from the denominator).
        sim = np.array([
            [0.9, 0.1, 0.1, 0.5],
            [0.8, 0.2, 0.0, 0.4],
            [0.1, 0.3, 0.9, 0.3],
            [0.2, 0.9, 0.8, 0.9],
            [0.3, 0.4, 0.7, 0.1],
            [0.4, 0.8, 0.6, 0.2],
        ], dtype=np.float32)
        gt = np.zeros((6, 4), dtype=bool)
        gt[0, 0] = gt[5, 1] = gt[1, 2] = True
        rec = recall_at_k(sim, gt, ks=(1, 2, 5))
        self.assertAlmostEqual(rec[1], 1 / 3)           # only q0 hits at rank 1
        self.assertAlmostEqual(rec[2], 2 / 3)           # q1's r5 arrives at rank 2
        self.assertAlmostEqual(rec[5], 2 / 3)           # q2's r1 is ranked 6th of 6
        # K beyond the database is clamped, at which point every scorable query hits.
        self.assertAlmostEqual(recall_at_k(sim, gt, ks=(20,))[20], 1.0)


class SimMatrixTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        gen = torch.Generator().manual_seed(0)
        self.ref = torch.nn.functional.normalize(torch.randn(30, 16, generator=gen), dim=1)
        self.qry = torch.nn.functional.normalize(torch.randn(10, 16, generator=gen), dim=1)

    def test_normalised_banks_produce_cosine(self):
        sim = sim_matrix(self.ref, self.qry, self.device, chunk=7)
        np.testing.assert_allclose(sim, (self.ref @ self.qry.T).numpy(), atol=1e-6)

    def test_unnormalised_bank_is_refused(self):
        with self.assertRaises(ValueError):
            sim_matrix(self.ref * 3.0, self.qry, self.device)
        with self.assertRaises(ValueError):
            sim_matrix(self.ref, self.qry * 0.5, self.device)

    def test_allow_unnormalized_is_a_plain_dot_product(self):
        # EventVLAD's path: deliberately unnormalised, and the flag says so at the call site.
        ref = self.ref * 3.0
        sim = sim_matrix(ref, self.qry, self.device, allow_unnormalized=True)
        np.testing.assert_allclose(sim, (ref @ self.qry.T).numpy(), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
