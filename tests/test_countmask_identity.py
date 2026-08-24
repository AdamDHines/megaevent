"""The docstring claim ``npzdata.load_countmask`` ≡ the HDF5 traverse renderer — proven.

Every image-set frame is rendered by ``ecv.from_numpy(...).countmask()`` and every
traverse frame by ``ecv.open(<hdf5>).with_repr("countmask")``; the claim that the two
paths cannot diverge lived only in a docstring. Here one synthetic event stream is pushed
through both (round-tripped through eventcv's own HDF5 writer) and the frames must match
byte for byte. Also pinned: the countmask spec itself against a pure-numpy reference
(per-frame pooled 99th-percentile alpha, binary activity mask, black background), and the
µs-vs-ms filter-window trap (E8) at the representation level.
"""
import os
import tempfile
import unittest

import eventcv as ecv
import numpy as np

from src.npzdata import load_countmask

W, H = 32, 24


def _events(n=400, seed=0, t_max_us=50_000):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, t_max_us, n))
    return np.stack([rng.integers(0, W, n), rng.integers(0, H, n),
                     t, rng.integers(0, 2, n)], axis=1).astype(np.int64)


def _reference_countmask(ev):
    """The spec, straight from the paper: R=clip(pos,a)/a, B=clip(neg,a)/a, G=activity."""
    pos = np.zeros((H, W), np.int64)
    neg = np.zeros((H, W), np.int64)
    for x, y, _, p in ev:
        (pos if p == 1 else neg)[y, x] += 1
    counts = np.concatenate([pos[pos > 0], neg[neg > 0]]).astype(np.float64)
    alpha = np.percentile(counts, 99.0)
    r = np.clip(pos, 0, alpha) / alpha * 255.0
    b = np.clip(neg, 0, alpha) / alpha * 255.0
    g = ((pos + neg) > 0) * 255.0
    return np.stack([r, g, b]).astype(np.uint8)


class CountmaskIdentityTests(unittest.TestCase):
    def test_npz_path_matches_hdf5_path_byte_for_byte(self):
        ev = _events()
        with tempfile.TemporaryDirectory() as tmp:
            h5 = os.path.join(tmp, "stream.h5")
            ecv.from_numpy(ev, sensor_size=(W, H), time_unit="us", order="xytp").save(h5)
            reader = ecv.open(h5, dt_ms=50, sensor_size=(W, H), hot_pixel_filter=False) \
                        .with_repr("countmask", window_ms=50, white_frame=False)
            hdf5_frame = np.asarray(reader[0])
            npz = os.path.join(tmp, "stream.npz")
            np.savez(npz, x=ev[:, 0], y=ev[:, 1], t=ev[:, 2], p=ev[:, 3],
                     resolution=np.array([H, W]))
            npz_frame = load_countmask(npz)
        self.assertEqual(hdf5_frame.dtype, np.uint8)
        np.testing.assert_array_equal(npz_frame, hdf5_frame)

    def test_renderer_matches_the_paper_spec(self):
        ev = _events(seed=1)
        rendered = ecv.from_numpy(ev, sensor_size=(W, H), time_unit="us",
                                  order="xytp").countmask(white_frame=False).numpy()
        np.testing.assert_array_equal(rendered, _reference_countmask(ev))

    def test_filter_window_units_are_microseconds(self):
        # The E8 trap: a 50 ms window passed as `50` asks for 50 µs and guts the stream.
        # Retention under the correct 50,000 µs must far exceed retention under 50 µs.
        ev = _events(n=2000, seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            h5 = os.path.join(tmp, "stream.h5")
            ecv.from_numpy(ev, sensor_size=(W, H), time_unit="us", order="xytp").save(h5)
            active = {}
            for label, dt_us in (("none", None), ("us", 50_000), ("trap", 50)):
                r = ecv.open(h5, dt_ms=50, sensor_size=(W, H), hot_pixel_filter=False)
                if dt_us is not None:
                    r = r.background_activity_filter(dt_us)
                frame = np.asarray(r.with_repr("countmask", window_ms=50,
                                               white_frame=False)[0])
                active[label] = int((frame[1] > 0).sum())
        self.assertGreater(active["us"], 0.5 * active["none"])
        self.assertLess(active["trap"], 0.5 * active["us"])


if __name__ == "__main__":
    unittest.main()
