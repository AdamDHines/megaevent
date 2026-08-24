"""The two-stage MegaEvent verification pass (scripts/megaevent_rerank.py).

Mirrors tests/test_eventgem_rerank.py for our own token re-ranker: the mutual-NN +
RANSAC pass must promote a geometrically consistent candidate over higher-cosine
impostors, must be an exact no-op at inlier_weight 0 (the identity check the script also
enforces at runtime), must respect the activity mask, and the token-centre coordinate
grid must agree with the ViT's row-major token order.
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from megaevent_rerank import COORDS, GRID, rerank_query  # noqa: E402

CPU = torch.device("cpu")


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def _tokens(seed, n=GRID * GRID, d=16):
    return _unit(np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32))


class CoordinateGridTests(unittest.TestCase):
    def test_row_major_token_centres(self):
        # Token 0 is the top-left patch; token GRID is the start of the second row.
        self.assertEqual(COORDS.shape, (GRID * GRID, 2))
        np.testing.assert_array_equal(COORDS[0], [7.0, 7.0])
        np.testing.assert_array_equal(COORDS[1], [21.0, 7.0])        # x advances first
        np.testing.assert_array_equal(COORDS[GRID], [7.0, 21.0])     # then y
        self.assertEqual(float(COORDS.max()), (GRID - 1) * 14.0 + 7.0)


class RerankQueryTests(unittest.TestCase):
    def setUp(self):
        self.q = _tokens(0)
        self.act = np.ones(GRID * GRID, np.float32)
        impostors = [_tokens(s) for s in (1, 2, 3, 4)]
        self.cand = np.stack([self.q, *impostors])           # candidate 0 = same frame
        self.cand_act = np.tile(self.act, (5, 1))
        self.scores = np.array([0.5, 0.9, 0.8, 0.7, 0.6], np.float32)

    def test_geometric_consistency_beats_cosine(self):
        new = rerank_query(self.q, self.act, self.cand, self.cand_act, self.scores,
                           5.0, 0.05, 0.0, CPU)
        self.assertEqual(int(np.argmax(new)), 0)
        # Full mutual agreement on an identical frame: all tokens are inliers.
        self.assertAlmostEqual(float(new[0]), 0.5 + GRID * GRID * 0.05, places=3)

    def test_weight_zero_is_an_exact_identity(self):
        new = rerank_query(self.q, self.act, self.cand, self.cand_act, self.scores,
                           5.0, 0.0, 0.0, CPU)
        np.testing.assert_array_equal(new, self.scores)

    def test_activity_mask_can_silence_the_match(self):
        # With every query token below the activity floor there is nothing to verify:
        # scores must come back untouched, not zero-matched.
        new = rerank_query(self.q, np.zeros_like(self.act), self.cand, self.cand_act,
                           self.scores, 5.0, 0.05, 0.5, CPU)
        np.testing.assert_array_equal(new, self.scores)

    def test_translated_frame_recovers_a_homography(self):
        # Candidate = the query's token grid shifted one column: mutual matches land one
        # token over, a pure translation RANSAC fits with many inliers.
        shifted = self.q.reshape(GRID, GRID, -1)
        shifted = np.roll(shifted, 1, axis=1).reshape(GRID * GRID, -1)
        cand = np.stack([shifted, _tokens(9)])
        scores = np.array([0.4, 0.9], np.float32)
        new = rerank_query(self.q, self.act, cand, np.tile(self.act, (2, 1)), scores,
                           5.0, 0.05, 0.0, CPU)
        self.assertEqual(int(np.argmax(new)), 0)
        self.assertGreater(float(new[0]), 0.4 + 400 * 0.05)  # most tokens inline


if __name__ == "__main__":
    unittest.main()
