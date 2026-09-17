"""``eval_transform`` order and the ``white_frame`` inversion trap.

Two pins. First, the deliberate Normalize-then-Resize order is only sound because it
commutes with the usual Resize-then-Normalize for unit-sum interpolation kernels — that
claim was asserted in a docstring and never tested. Second, every shipped checkpoint's
config carries ``white_frame: True`` while the models were trained on black-background
frames; the pipeline is safe only because ``_repr_kwargs`` hardcodes ``white_frame=False``
and ``white_frame`` is not in ``_RESTORE``. Any future reader that trusts the saved config
inverts every frame — these tests are the tripwire.

Third pin: the RGB controls' normalisation constants must match the representation they are
handed. ``accumulate`` is a white-background render whose channel means are ~0.95/0.88/0.93,
so ImageNet stats centre it 2.0-4.5 sigma off and countmask stats (black-background
constants) are worse still. Getting this wrong silently handicaps every control in the
comparison, which is exactly what it did on the v8 accumulate banks.
"""
import types
import unittest

import torch

from src.inference import _repr_kwargs, _RESTORE, eval_transform
from src.methods import _NORM_STATS, megaloc_transform


def _cfg(**over):
    base = dict(tencode_mean=[0.0975, 0.2804, 0.0977], tencode_std=[0.1866, 0.4155, 0.1867],
                H=64, W=64)
    base.update(over)
    return types.SimpleNamespace(**base)


class EvalTransformTests(unittest.TestCase):
    def test_normalize_then_resize_commutes_with_the_usual_order(self):
        # Bicubic kernels have unit-sum weights, so interp(a*x+b) == a*interp(x)+b and the
        # two orders agree to float tolerance. If the interpolation mode or antialiasing
        # ever changes this stops being automatic — which is why it is pinned.
        from torchvision import transforms
        cfg = _cfg()
        torch.manual_seed(0)
        x = torch.rand(3, 260, 346)                     # DAVIS346-shaped countmask
        ours = eval_transform(cfg)(x)
        usual = transforms.Normalize(cfg.tencode_mean, cfg.tencode_std)(
            transforms.Resize((cfg.H, cfg.W),
                              interpolation=transforms.InterpolationMode.BICUBIC)(x))
        torch.testing.assert_close(ours, usual, atol=2e-4, rtol=0)

    def test_output_shape_is_the_configured_grid(self):
        self.assertEqual(tuple(eval_transform(_cfg())(torch.rand(3, 260, 346)).shape),
                         (3, 64, 64))


class WhiteFrameTrapTests(unittest.TestCase):
    def test_countmask_render_is_hardwired_black_background(self):
        self.assertIs(_repr_kwargs("countmask", 50)["white_frame"], False)

    def test_white_frame_is_not_restored_from_checkpoints(self):
        # Shipped configs say white_frame=True (a training-era leftover kept by
        # export_ckpt.KEEP). It must never reach the renderer through _RESTORE.
        self.assertNotIn("white_frame", _RESTORE)


class ControlNormStatsTests(unittest.TestCase):
    """megaloc_transform's stats must match the representation the control is reading."""

    # A frame that is almost all background under each render: accumulate is white, countmask
    # is black. These are the cases the constants exist to centre.
    ACCUMULATE_BG = 0.95
    COUNTMASK_BG = 0.0

    def _centred(self, value, stats):
        out = megaloc_transform(32, stats=stats)(torch.full((3, 8, 8), float(value)))
        return float(out.mean().abs())

    def test_every_representation_has_a_stats_pair(self):
        self.assertEqual(sorted(_NORM_STATS), ["accumulate", "countmask", "imagenet"])

    def test_accumulate_stats_centre_an_accumulate_frame(self):
        matched = self._centred(self.ACCUMULATE_BG, "accumulate")
        self.assertLess(matched, 0.5)
        # and both alternatives push it far off centre
        self.assertGreater(self._centred(self.ACCUMULATE_BG, "imagenet"), 2.0)
        self.assertGreater(self._centred(self.ACCUMULATE_BG, "countmask"), 2.0)

    def test_countmask_stats_are_wrong_for_accumulate(self):
        # The trap: "use the event stats" is not enough — the wrong event stats are worse
        # than ImageNet on this render.
        self.assertGreater(self._centred(self.ACCUMULATE_BG, "countmask"),
                           self._centred(self.ACCUMULATE_BG, "imagenet"))

    def test_countmask_stats_centre_a_countmask_frame(self):
        self.assertLess(self._centred(self.COUNTMASK_BG, "countmask"), 0.9)
        self.assertGreater(self._centred(self.COUNTMASK_BG, "imagenet"), 1.9)

    def test_unknown_stats_are_rejected(self):
        with self.assertRaises(ValueError):
            megaloc_transform(32, stats="tencode")


if __name__ == "__main__":
    unittest.main()
