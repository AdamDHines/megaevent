"""The backbone-probe adapters: four module layouts, one token interface.

``scripts/backbone_probe.py`` compares encoders by stripping the aggregation head and
mean-pooling the patch tokens. That is only a fair comparison if every adapter hands back
the *same* quantity, and the four backbones do not agree on a format: Meta's ViT (GEPT,
stock DINOv2, SALAD's inner model) returns a dict of ``[B, N, D]`` tokens, while MegaLoc's
backbone and our ``VPRModel.forward_encoder`` return ``([B, D, h, w], [B, D])``. The reshape
that reconciles them is the one thing that could silently scramble a comparison, so it is
pinned here.
"""
import sys
import unittest

import torch

sys.path.insert(0, "scripts")
from backbone_probe import _from_pair, _from_vit          # noqa: E402


class PairAdapterTests(unittest.TestCase):
    def _pair(self, feat, cls):
        return _from_pair(lambda _x: (feat, cls)).forward_features(None)

    def test_reshape_preserves_the_token_set(self):
        feat, cls = torch.randn(2, 8, 3, 5), torch.randn(2, 8)
        out = self._pair(feat, cls)
        self.assertEqual(tuple(out["x_norm_patchtokens"].shape), (2, 15, 8))
        # the pooled descriptor must not depend on the layout
        torch.testing.assert_close(out["x_norm_patchtokens"].mean(dim=1),
                                   feat.mean(dim=(2, 3)))

    def test_each_token_maps_to_its_own_cell(self):
        # a map whose values encode (channel, row, col) — any transpose error shows up
        feat = torch.arange(1 * 4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 3)
        out = self._pair(feat, torch.zeros(1, 4))
        for r in range(2):
            for c in range(3):
                torch.testing.assert_close(out["x_norm_patchtokens"][0, r * 3 + c],
                                           feat[0, :, r, c])

    def test_cls_passes_through(self):
        cls = torch.randn(3, 6)
        out = self._pair(torch.randn(3, 6, 2, 2), cls)
        torch.testing.assert_close(out["x_norm_clstoken"], cls)


class VitAdapterTests(unittest.TestCase):
    def test_vit_adapter_is_a_passthrough(self):
        want = {"x_norm_patchtokens": torch.randn(1, 529, 384),
                "x_norm_clstoken": torch.randn(1, 384)}
        got = _from_vit(type("V", (), {"forward_features": staticmethod(lambda x: want)})
                        ).forward_features(None)
        self.assertIs(got, want)


if __name__ == "__main__":
    unittest.main()
