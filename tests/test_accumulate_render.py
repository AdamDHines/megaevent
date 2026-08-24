"""megaevent's local ``accumulate`` renderer: parity with gept's port + the eval fallback.

eventcv cannot render ``accumulate`` (the GEPT pretraining diet the v8 wave trains on),
so :mod:`src.npzdata` vendors the renderer and :class:`src.inference.EventStreamDataset`
falls back to raw slices + that port. Three things pinned here:

1. Byte parity with gept's ``representations.accumulate_numpy`` (itself byte-verified
   against the upstream ``accumulate_to_rgb``) — on synthetic events always, and on a
   real Brisbane frame when the data volume is mounted.
2. The polarity trap: ``p > 0`` must split {-1,+1} streams correctly (upstream GEP's
   ``astype(bool)`` makes -1 truthy and would paint every event positive).
3. White background: zero events -> an all-255 frame, never black.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.npzdata import accumulate_numpy  # noqa: E402

GEPT_VPR = "/home/adam/repo/gept/src/vpr"
REAL_FRAME = ("/media/adam/vprdatasets/megaevent/brisbane_npz/brisbane_event/"
              "real/sunset1/frame_007500.npz")


class TestAccumulateRender(unittest.TestCase):
    def test_white_background_on_empty(self):
        out = accumulate_numpy(np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                               np.array([], dtype=np.int64), np.array([], dtype=np.int8),
                               8, 10)
        self.assertEqual(out.shape, (3, 8, 10))
        self.assertTrue(np.all(out == 255), "no events must render WHITE, not black")

    def test_polarity_split_on_signed_stream(self):
        # one positive event at (2,3), one negative at (5,4), p in {-1,+1}
        x = np.array([2, 5]); y = np.array([3, 4])
        t = np.array([0, 1]); p = np.array([1, -1], dtype=np.int8)
        out = accumulate_numpy(x, y, t, p, 8, 10)
        r, g, b = out[0], out[1], out[2]
        self.assertEqual(int(r[3, 2]), 255)       # positive -> red: R stays full
        self.assertLess(int(b[3, 2]), 255)        #   ... G and B darken
        self.assertLess(int(g[3, 2]), 255)
        self.assertEqual(int(b[4, 5]), 255)       # negative -> blue: B stays full
        self.assertLess(int(r[4, 5]), 255)
        self.assertLess(int(g[4, 5]), 255)

    def test_parity_with_gept_port_synthetic(self):
        if not os.path.isdir(GEPT_VPR):
            self.skipTest("gept checkout not present")
        sys.path.append(GEPT_VPR)
        from representations import accumulate_numpy as gept_acc
        rng = np.random.default_rng(0)
        n = 5000
        x = rng.integers(0, 40, n); y = rng.integers(0, 30, n)
        t = rng.integers(0, 50000, n); p = rng.choice([-1, 1], n).astype(np.int8)
        ours = accumulate_numpy(x, y, t, p, 30, 40)
        theirs = gept_acc(x, y, t, p, 30, 40)
        np.testing.assert_array_equal(ours, theirs)

    def test_parity_with_gept_port_real_frame(self):
        if not (os.path.isdir(GEPT_VPR) and os.path.exists(REAL_FRAME)):
            self.skipTest("gept checkout or data volume not present")
        sys.path.append(GEPT_VPR)
        from representations import accumulate_numpy as gept_acc
        d = np.load(REAL_FRAME)
        ours = accumulate_numpy(d["x"], d["y"], d["t"], d["p"], 260, 346)
        theirs = gept_acc(d["x"], d["y"], d["t"], d["p"], 260, 346)
        np.testing.assert_array_equal(ours, theirs)

    def test_eval_fallback_is_wired(self):
        from src import inference as inf
        self.assertIn("accumulate", inf._LOCAL_RENDERERS)
        self.assertIs(inf._LOCAL_RENDERERS["accumulate"], accumulate_numpy)

    def test_image_set_path_dispatches_by_representation(self):
        from src import methods
        from src.npzdata import load_accumulate, load_countmask
        self.assertIs(methods._NPZ_RENDERERS["countmask"], load_countmask)
        self.assertIs(methods._NPZ_RENDERERS["accumulate"], load_accumulate)


if __name__ == "__main__":
    unittest.main()
