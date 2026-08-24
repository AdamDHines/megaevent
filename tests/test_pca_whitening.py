"""``pca_fit`` / ``pca_apply``: the transform every whitened cell in the paper passes through.

Whitening moves R@1 by up to +0.12 on NSAVP, so these two functions carry real weight and had
no test. Pinned here: the fit is deterministic (its randomized SVD is seeded in a forked RNG),
whitening at power 0.5 actually equalises the per-axis variance it claims to, ``pca_apply``
returns unit rows, and the subsampled fit used on large galleries agrees with the full fit.
"""
import unittest

import numpy as np
import torch

from src.inference import pca_apply, pca_fit
from src.scoring import pca_fit_subsampled


def _anisotropic(n=600, dim=32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    scales = torch.linspace(5.0, 0.1, dim)
    return torch.randn(n, dim, generator=gen) * scales + 2.0


class PcaWhiteningTests(unittest.TestCase):
    device = torch.device("cpu")

    def test_fit_is_deterministic(self):
        X = _anisotropic()
        a = pca_fit(X, self.device, dim=16, power=0.5)
        b = pca_fit(X, self.device, dim=16, power=0.5)
        torch.testing.assert_close(a["V"], b["V"])
        torch.testing.assert_close(a["scale"], b["scale"])

    def test_fit_does_not_disturb_callers_rng(self):
        torch.manual_seed(123)
        expected = torch.randn(4)
        torch.manual_seed(123)
        pca_fit(_anisotropic(), self.device, dim=8, power=0.5)
        torch.testing.assert_close(torch.randn(4), expected)

    def test_power_half_equalises_axis_variance(self):
        # Before the final L2 normalise, y = (x - mu) @ V * scale with scale =
        # (var + eps)^-0.5 must have ~unit variance on every retained axis — that is what
        # "whitening" means, and what a wrong exponent or an un-centred fit would break.
        X = _anisotropic()
        p = pca_fit(X, self.device, dim=16, power=0.5)
        Y = (X - p["mu"]) @ p["V"] * p["scale"]
        var = Y.var(dim=0)
        self.assertTrue(bool((var > 0.8).all() and (var < 1.2).all()),
                        msg=f"whitened axis variances span {var.min():.3f}..{var.max():.3f}")

    def test_apply_returns_unit_rows(self):
        X = _anisotropic()
        out = pca_apply(X, pca_fit(X, self.device, dim=16, power=0.5), self.device, chunk=100)
        self.assertEqual(out.shape, (X.shape[0], 16))
        torch.testing.assert_close(out.norm(dim=1), torch.ones(X.shape[0]),
                                   atol=1e-5, rtol=0)

    def test_subsampled_fit_matches_full_fit_when_nothing_is_dropped(self):
        # pca_fit_subsampled exists purely to bound SVD memory; at n <= the sample budget it
        # must be the identical fit, or "which code path built this bank" becomes a variable.
        X = _anisotropic(n=400)
        full = pca_fit(X, self.device, power=0.5)
        sub = pca_fit_subsampled(X, self.device, n=1000, power=0.5)
        torch.testing.assert_close(full["V"], sub["V"])
        torch.testing.assert_close(full["scale"], sub["scale"])


if __name__ == "__main__":
    unittest.main()
