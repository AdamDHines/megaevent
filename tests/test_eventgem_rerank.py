"""The shortlist construction Event-GeM's re-ranker builds before it verifies anything.

``LocalReranker.rerank`` partitions the distance matrix in column blocks rather than in
one call, because a whole-matrix ``argpartition`` materialises an int64 index of the full
shape before the ``[:k]`` slice discards nearly all of it (5.6 GB on pitts250k). The
partition is along axis 0 and therefore independent per column, so the block version must
return *exactly* the same shortlists -- these tests are what says so, including at the
chunk boundary and when ``k`` reaches the whole database.
"""
import unittest

import numpy as np

from src.eventgemlocal import SHORTLIST_CHUNK


def _chunked(dist, k):
    """The shortlist loop as ``LocalReranker.rerank`` performs it."""
    n_q = dist.shape[1]
    out = np.empty((k, n_q), dtype=np.int64)
    for s in range(0, n_q, SHORTLIST_CHUNK):
        block = dist[:, s:s + SHORTLIST_CHUNK]
        part = np.argpartition(block, k - 1, axis=0)[:k]
        order = np.argsort(np.take_along_axis(block, part, axis=0), axis=0)
        out[:, s:s + SHORTLIST_CHUNK] = np.take_along_axis(part, order, axis=0)
    return out


def _whole(dist, k):
    """The single-call form, kept here as the reference the chunking must reproduce."""
    part = np.argpartition(dist, k - 1, axis=0)[:k]
    order = np.argsort(np.take_along_axis(dist, part, axis=0), axis=0)
    return np.take_along_axis(part, order, axis=0)


class ShortlistChunkingTests(unittest.TestCase):
    SHAPES = (
        (500, 1300, 50),                        # several full blocks plus a remainder
        (200, SHORTLIST_CHUNK, 50),             # exactly one block
        (200, SHORTLIST_CHUNK + 1, 50),         # one block plus a single-column tail
        (97, 33, 7),                            # smaller than a block
        (60, 10, 60),                           # k == n_db: the partition keeps everything
    )

    def test_matches_the_whole_matrix_partition(self):
        rng = np.random.default_rng(0)
        for n_db, n_q, k in self.SHAPES:
            with self.subTest(n_db=n_db, n_q=n_q, k=k):
                dist = rng.random((n_db, n_q)).astype(np.float32)
                np.testing.assert_array_equal(_chunked(dist, k), _whole(dist, k))

    def test_shortlists_are_sorted_nearest_first(self):
        rng = np.random.default_rng(1)
        dist = rng.random((300, 700)).astype(np.float32)
        short = _chunked(dist, 50)
        picked = np.take_along_axis(dist, short, axis=0)
        self.assertTrue((np.diff(picked, axis=0) >= 0).all())

    def test_shortlists_are_the_k_smallest_distances(self):
        """Re-ranking only ever subtracts from a shortlisted candidate, so anything left
        out of the shortlist can never be promoted -- the shortlist must really be the
        k nearest, not merely k sorted candidates."""
        rng = np.random.default_rng(2)
        dist = rng.random((240, 260)).astype(np.float32)
        k = 50
        short = _chunked(dist, k)
        picked = np.sort(np.take_along_axis(dist, short, axis=0), axis=0)
        expected = np.sort(dist, axis=0)[:k]
        np.testing.assert_allclose(picked, expected)


if __name__ == "__main__":
    unittest.main()
