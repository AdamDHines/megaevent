"""The streamed pooled scorer must equal the reference scorer — as a unit test, not a flag.

``scripts/brisbane_pooled.py::topk_ranked`` + ``recall_from_ranked`` is what every pooled
traverse number (Brisbane, NSAVP, the RGB controls, the all-queries sweeps) is scored with;
``src.inference.sim_matrix`` + ``recall_at_k`` is the reference implementation the image sets
use. ``scripts/score_cached_banks.py --validate`` proves them equal at runtime, but only when
someone passes the flag on that one script. This pins the equivalence permanently, on synthetic
banks, including the chunked-database merge path (``db_chunk``) that NSAVP's 3.4 GB gallery
forces on an 8 GB card.
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from brisbane_pooled import recall_from_ranked, topk_ranked  # noqa: E402
from src.inference import recall_at_k, sim_matrix  # noqa: E402

KS = (1, 5, 10, 20)


def _banks(n_db, n_q, dim, seed):
    gen = torch.Generator().manual_seed(seed)
    db = torch.nn.functional.normalize(torch.randn(n_db, dim, generator=gen), dim=1)
    q = torch.nn.functional.normalize(torch.randn(n_q, dim, generator=gen), dim=1)
    rng = np.random.default_rng(seed)
    gt = rng.random((n_db, n_q)) < 0.02                 # sparse: some queries unscorable
    return db, q, gt


class StreamedScoringParityTests(unittest.TestCase):
    device = torch.device("cpu")

    def _assert_parity(self, n_db, n_q, dim, seed, **topk_kwargs):
        db, q, gt = _banks(n_db, n_q, dim, seed)
        reference = recall_at_k(sim_matrix(db, q, self.device), gt, ks=KS)
        ranked = topk_ranked(db, q, self.device, chunk=13, **topk_kwargs)
        streamed, _, scorable = recall_from_ranked(ranked, gt, ks=KS)
        self.assertEqual(int(scorable.sum()), int((gt.sum(0) > 0).sum()))
        for k in KS:
            self.assertAlmostEqual(streamed[k], reference[k], places=9,
                                   msg=f"R@{k} disagrees at db={n_db} q={n_q} {topk_kwargs}")

    def test_single_pass_database(self):
        self._assert_parity(300, 40, 24, seed=0)

    def test_chunked_database_merge(self):
        # db_chunk far smaller than the gallery, not a divisor of it, and smaller than k:
        # the running top-k merge across chunks must still be the global top-k.
        self._assert_parity(300, 40, 24, seed=1, db_chunk=17)

    def test_db_chunk_equal_to_gallery_changes_nothing(self):
        db, q, gt = _banks(200, 30, 16, seed=2)
        whole = topk_ranked(db, q, self.device, chunk=7)
        chunked = topk_ranked(db, q, self.device, chunk=7, db_chunk=200)
        np.testing.assert_array_equal(whole, chunked)

    def test_unscorable_queries_share_a_denominator(self):
        # All-negative ground truth for half the queries: both scorers must drop exactly
        # those columns rather than counting them as misses.
        db, q, gt = _banks(150, 20, 16, seed=3)
        gt[:, 10:] = False
        reference = recall_at_k(sim_matrix(db, q, self.device), gt, ks=(1,))
        ranked = topk_ranked(db, q, self.device)
        streamed, _, scorable = recall_from_ranked(ranked, gt, ks=(1,))
        self.assertEqual(int(scorable.sum()), int((gt[:, :10].sum(0) > 0).sum()))
        self.assertAlmostEqual(streamed[1], reference[1], places=9)


if __name__ == "__main__":
    unittest.main()
