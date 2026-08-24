"""Brisbane geometry: clock-calibration guard, offset application, coverage mask.

Two Brisbane trees exist on this machine and only one carries ``clock_offsets.json``; the
uncalibrated one puts "the same place" a median 19-104 m from itself across traverses.
``clock_offsets`` historically returned empty dicts (offset 0) when the file was absent —
these tests pin the new hard failure, and check ``frame_coords`` end to end on a synthetic
NMEA track: interpolation onto slice centres, the clock shift actually moving coordinates,
and the ``covered`` mask refusing to extrapolate beyond the GPS span.
"""
import datetime
import json
import os
import tempfile
import unittest

import numpy as np

from src import traversegps as tg

SEQ, DS = "faketraverse", "brisbane_event"


def _nmea_time(epoch):
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime("%H%M%S.00")


def _write_traverse(root, t0, n_fixes=60, lat0=-27.5, lon0=153.0, dlat_per_s=1e-5):
    """A straight-line 1 Hz GPS track plus the frames-50 metadata, in tg's layout."""
    d = os.path.join(root, DS, SEQ)
    os.makedirs(os.path.join(d, f"{SEQ}-frames-50"))
    day = datetime.datetime.fromtimestamp(t0, datetime.timezone.utc).strftime("%d%m%y")
    lines = [f"$GPRMC,{_nmea_time(t0)},A,2730.000,S,15300.000,E,10.0,0.0,{day},,,A"]
    for i in range(n_fixes):
        # NMEA ddmm.mmmm: build from the decimal latitude directly.
        latd = abs(lat0 + i * dlat_per_s)
        lond = abs(lon0)
        nmea_lat = f"{int(latd) * 100 + latd % 1 * 60:012.7f}"
        nmea_lon = f"{int(lond) * 100 + lond % 1 * 60:013.7f}"
        lines.append(f"$GPGGA,{_nmea_time(t0 + i)},{nmea_lat},S,{nmea_lon},E,1,08,1.0,"
                     f"10.0,M,0.0,M,,")
    with open(os.path.join(d, f"{SEQ}_ground_truth.nmea"), "w") as h:
        h.write("\n".join(lines) + "\n")
    meta = {"start_tick": int((t0 + 5) * 1e6), "ticks_per_second": 1e6,
            "timewindow_ms": 50, "total_frames": 399}   # 400 slices, 20 s of video
    with open(os.path.join(d, f"{SEQ}-frames-50", "metadata.json"), "w") as h:
        json.dump(meta, h)
    return lat0, lon0


class ClockOffsetGuardTests(unittest.TestCase):
    def test_missing_file_without_requirement_returns_empty(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, DS))
            self.assertEqual(tg.clock_offsets(root, DS), ({}, {}))

    def test_missing_file_with_requirement_is_fatal(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, DS))
            with self.assertRaises(SystemExit):
                tg.clock_offsets(root, DS, required=(SEQ,))

    def test_file_missing_a_required_sequence_is_fatal(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, DS))
            with open(os.path.join(root, DS, tg.CLOCK_OFFSET_FILE), "w") as h:
                json.dump({"offsets": {"sunset1": -0.96}, "drifts": {}}, h)
            with self.assertRaises(SystemExit):
                tg.clock_offsets(root, DS, required=(SEQ,))
            offs, _ = tg.clock_offsets(root, DS, required=("sunset1",))
            self.assertAlmostEqual(offs["sunset1"], -0.96)

    def test_the_real_brisbane_tree_is_calibrated(self):
        # The tree every published number uses must keep its calibration in place.
        root = "/media/adam/vprdatasets/eventgem"
        if not os.path.isdir(os.path.join(root, DS)):
            self.skipTest("Brisbane data tree not mounted")
        offs, _ = tg.clock_offsets(root, DS, required=("sunset1", "sunset2", "daytime",
                                                       "morning", "night", "sunrise"))
        self.assertEqual(len(offs), 6)


class FrameCoordsTests(unittest.TestCase):
    T0 = 1_600_000_000.0

    def _coords(self, offsets=None):
        with tempfile.TemporaryDirectory() as root:
            lat0, lon0 = _write_traverse(root, self.T0)
            if offsets is None:
                with open(os.path.join(root, DS, tg.CLOCK_OFFSET_FILE), "w") as h:
                    json.dump({"offsets": {SEQ: 0.0}, "drifts": {}}, h)
                return tg.frame_coords(root, DS, SEQ, lat0, lon0)
            return tg.frame_coords(root, DS, SEQ, lat0, lon0, offsets=offsets)

    def test_uncalibrated_tree_cannot_produce_coordinates(self):
        with tempfile.TemporaryDirectory() as root:
            lat0, lon0 = _write_traverse(root, self.T0)
            with self.assertRaises(SystemExit):
                tg.frame_coords(root, DS, SEQ, lat0, lon0)

    def test_slice_centres_interpolate_onto_the_track(self):
        xy, covered, speed, info = self._coords()
        self.assertEqual(len(xy), 400)                  # total_frames + 1
        self.assertEqual(info["clock_offset_s"], 0.0)
        # The track moves north at 1e-5 deg/s ~ 1.113 m/s; frame 0's centre is t0+5.025 s.
        expected_north = np.deg2rad(5.025 * 1e-5) * tg.EARTH_RADIUS_M
        self.assertAlmostEqual(xy[covered][0][1], expected_north, delta=0.02)
        self.assertAlmostEqual(float(np.median(speed[covered])), 1.113, delta=0.03)

    def test_clock_offset_shifts_the_interpolation_instant(self):
        base, cov, _, _ = self._coords(offsets=({SEQ: 0.0}, {}))
        shifted, cov2, _, info = self._coords(offsets=({SEQ: 2.0}, {}))
        self.assertEqual(info["clock_offset_s"], 2.0)
        # +2 s at ~1.113 m/s northward = ~2.23 m further north at every covered frame.
        both = cov & cov2
        dn = shifted[both, 1] - base[both, 1]
        self.assertAlmostEqual(float(np.median(dn)), 2.226, delta=0.05)

    def test_frames_beyond_the_gps_span_are_uncovered_not_extrapolated(self):
        # Track covers t0..t0+59; slices span t0+5..t0+25 — all inside. Shift the clock
        # +40 s and the late slices fall off the track's end: covered must go False there.
        _, covered, _, _ = self._coords(offsets=({SEQ: 40.0}, {}))
        self.assertFalse(covered[-1])
        self.assertTrue(covered[0])


if __name__ == "__main__":
    unittest.main()
