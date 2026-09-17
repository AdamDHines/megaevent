"""The training transform pipeline: order, polarity-swap semantics, domain randomise.

The pre-2026-08-19 pipeline ran ``Normalize`` *before* the polarity swap, which is only
equivalent to swapping raw polarities when R and B share mean/std — nearly true for
countmask (why three sweeps trained fine), false for tencode (mean R 0.298 vs B 0.229).
The fix moves ``Normalize`` last; these tests pin the new order, prove the old order was
wrong in general, and pin the domain-randomise invariants the sim-to-real story rests on.
"""

import random
import types
import unittest

import torch
from torchvision import transforms

import megaevent.training.dataset as ds  # noqa: E402


def _cfg(**over):
    base = dict(
        tencode_mean=[0.0975, 0.2804, 0.0977],
        tencode_std=[0.1866, 0.4155, 0.1867],
        H=224,
        W=224,
        aug_domain_rand=False,
        representation="countmask",
        aug_gain_jitter=(0.7, 1.4),
        aug_dropout_max=0.8,
        aug_salt_prob=0.02,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


class PipelineOrderTests(unittest.TestCase):
    def test_normalize_is_the_last_train_op(self):
        ops = ds.build_transforms(_cfg(), train=True).transforms
        self.assertIsInstance(ops[-1], transforms.Normalize)
        kinds = [type(o).__name__ for o in ops]
        # Augmentations act on raw counts, in this order, before description.
        self.assertEqual(
            kinds,
            [
                "ToTensor",
                "RandomSwapEventRedBlue",
                "RandomHorizontalFlip",
                "RandomResizedCrop",
                "Normalize",
            ],
        )

    def test_domain_randomise_precedes_everything_but_totensor(self):
        ops = ds.build_transforms(_cfg(aug_domain_rand=True), train=True).transforms
        kinds = [type(o).__name__ for o in ops]
        self.assertEqual(kinds[:2], ["ToTensor", "CountmaskDomainRandomise"])
        self.assertEqual(kinds[-1], "Normalize")

    def test_no_hflip_removes_only_the_flip(self):
        kinds = [
            type(o).__name__
            for o in ds.build_transforms(_cfg(aug_hflip=False), train=True).transforms
        ]
        self.assertEqual(
            kinds, ["ToTensor", "RandomSwapEventRedBlue", "RandomResizedCrop", "Normalize"]
        )

    def test_domain_randomise_refuses_non_countmask(self):
        with self.assertRaises(ValueError):
            ds.build_transforms(_cfg(aug_domain_rand=True, representation="tencode"), train=True)

    def test_eval_transform_unchanged(self):
        kinds = [type(o).__name__ for o in ds.build_transforms(_cfg(), train=False).transforms]
        self.assertEqual(kinds, ["ToTensor", "Normalize", "Resize"])

    def test_eval_img_size_decouples_selection_from_the_training_crop(self):
        # --eval-img-size 322: in-loop evaluation moves to the reporting resolution while
        # the training crop stays at H=224 — the fix for selection optimising a metric
        # nobody reports (224-vs-322 reorders checkpoints, measured 2026-08-04).
        cfg = _cfg(eval_img_size=322)
        self.assertEqual(ds.build_transforms(cfg, train=False).transforms[-1].size, (322, 322))
        self.assertEqual(
            ds.build_transforms(cfg, train=True).transforms[-2].size, (224, 224)
        )  # RandomResizedCrop, then Normalize


class SwapSemanticsTests(unittest.TestCase):
    def test_swap_on_raw_counts_is_polarity_inversion(self):
        x = torch.rand(3, 8, 8)
        random.seed(0)  # p=1.0 fires regardless
        y = ds.RandomSwapEventRedBlue(p=1.0)(x)
        torch.testing.assert_close(y[0], x[2])
        torch.testing.assert_close(y[2], x[0])
        torch.testing.assert_close(y[1], x[1])

    def test_why_normalize_first_was_wrong(self):
        # With unequal per-channel stats (tencode's), swap-then-normalize and
        # normalize-then-swap disagree — the historical order silently corrupted any
        # non-countmask run. This is the measurement behind the reorder.
        mean, std = [0.298, 0.101, 0.229], [0.320, 0.144, 0.255]
        norm = transforms.Normalize(mean, std)
        x = torch.rand(3, 4, 4)
        swap = lambda t: t[[2, 1, 0]]  # noqa: E731
        self.assertFalse(torch.allclose(norm(swap(x)), swap(norm(x))))
        # And with countmask's near-equal R/B stats the two orders nearly agree — why the
        # bug was benign for every shipped (countmask) run.
        cm_norm = transforms.Normalize([0.0975, 0.2804, 0.0977], [0.1866, 0.4155, 0.1867])
        self.assertLess((cm_norm(swap(x)) - swap(cm_norm(x))).abs().max().item(), 0.01)


class DomainRandomiseTests(unittest.TestCase):
    def _frame(self):
        # A countmask-like frame: sparse counts in R/B, binary activity mask in G.
        torch.manual_seed(0)
        x = torch.zeros(3, 16, 16)
        active = torch.rand(16, 16) < 0.3
        x[0][active] = torch.rand(int(active.sum())) * 0.5
        x[2][active] = torch.rand(int(active.sum())) * 0.5
        x[1][active] = 1.0
        return x

    def test_output_stays_in_unit_range_and_green_stays_binary(self):
        aug = ds.CountmaskDomainRandomise(gain=(0.7, 1.4), dropout_max=0.8, salt_prob=0.02)
        for seed in range(5):
            random.seed(seed)
            torch.manual_seed(seed)
            y = aug(self._frame())
            self.assertGreaterEqual(float(y.min()), 0.0)
            self.assertLessEqual(float(y.max()), 1.0)
            self.assertTrue(
                bool(((y[1] == 0) | (y[1] == 1)).all()), "activity mask must stay binary"
            )

    def test_gain_never_touches_the_activity_mask(self):
        x = self._frame()
        aug = ds.CountmaskDomainRandomise(gain=(1.3, 1.4), dropout_max=0.0, salt_prob=0.0)
        random.seed(1)
        torch.testing.assert_close(aug(x)[1], x[1])

    def test_dropout_zeroes_all_three_channels_together(self):
        # A dropped pixel becomes background — never an "active pixel with zero counts",
        # which real data cannot produce.
        x = self._frame()
        aug = ds.CountmaskDomainRandomise(gain=(1.0, 1.0), dropout_max=1.0, salt_prob=0.0)
        random.seed(3)
        torch.manual_seed(3)
        y = aug(x)
        dropped = (x[1] == 1.0) & (y[1] == 0.0)
        self.assertGreater(int(dropped.sum()), 0)
        self.assertTrue(bool((y[0][dropped] == 0).all() and (y[2][dropped] == 0).all()))


if __name__ == "__main__":
    unittest.main()
