"""The LR schedule and the selection/reporting resolution contract.

``get_lr`` historically returned exactly 0.0 at step 0, making the first optimizer step
a silent no-op (audit B10); the fix counts warmup from step+1. The trainer also now
warns when the in-loop selection resolution differs from the 322 reporting resolution —
selection at 224 was measured (2026-08-04) to *reorder* checkpoints, so best.pt was the
argmax of a metric nobody reports.
"""

import math
import unittest

from megaevent.training.utils import get_lr  # noqa: E402


class GetLrTests(unittest.TestCase):
    LR, WARMUP, STEPS, MIN_LR = 5e-5, 500, 10_000, 0.0

    def _lr(self, step):
        return get_lr(step, self.WARMUP, self.LR, self.STEPS, self.MIN_LR)

    def test_step_zero_takes_a_real_step(self):
        self.assertGreater(self._lr(0), 0.0)
        self.assertAlmostEqual(self._lr(0), self.LR / self.WARMUP)

    def test_warmup_is_linear_and_reaches_peak(self):
        self.assertAlmostEqual(self._lr(self.WARMUP - 1), self.LR)
        self.assertAlmostEqual(self._lr(249), self.LR * 250 / 500)
        self.assertAlmostEqual(self._lr(self.WARMUP), self.LR)  # cosine start = peak

    def test_cosine_decays_to_min_lr(self):
        mid = (self.WARMUP + self.STEPS) // 2
        expected = self.MIN_LR + 0.5 * (
            1 + math.cos(math.pi * (mid - self.WARMUP) / (self.STEPS - self.WARMUP))
        ) * (self.LR - self.MIN_LR)
        self.assertAlmostEqual(self._lr(mid), expected)
        self.assertAlmostEqual(self._lr(self.STEPS), self.MIN_LR)
        self.assertAlmostEqual(self._lr(self.STEPS + 999), self.MIN_LR)

    def test_monotone_up_then_down(self):
        ups = [self._lr(s) for s in range(0, self.WARMUP)]
        downs = [self._lr(s) for s in range(self.WARMUP, self.STEPS, 100)]
        self.assertEqual(ups, sorted(ups))
        self.assertEqual(downs, sorted(downs, reverse=True))


class ReportingResolutionTests(unittest.TestCase):
    def test_constant_is_the_pooled_protocol_resolution(self):
        from megaevent.training import train

        self.assertEqual(train.REPORTING_RESOLUTION, 322)


if __name__ == "__main__":
    unittest.main()
