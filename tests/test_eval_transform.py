"""``eval_transform`` order and the ``white_frame`` inversion trap.

Two pins. First, the deliberate Normalize-then-Resize order is only sound because it
commutes with the usual Resize-then-Normalize for unit-sum interpolation kernels — that
claim was asserted in a docstring and never tested. Second, every shipped checkpoint's
config carries ``white_frame: True`` while the models were trained on black-background
frames; the pipeline is safe only because ``_repr_kwargs`` hardcodes ``white_frame=False``
and ``white_frame`` is not in ``_RESTORE``. Any future reader that trusts the saved config
inverts every frame — these tests are the tripwire.
"""
import types
import unittest

import torch

from src.inference import _repr_kwargs, _RESTORE, eval_transform


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


if __name__ == "__main__":
    unittest.main()
