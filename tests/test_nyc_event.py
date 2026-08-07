import csv
import importlib.util
import os
import tempfile
import unittest
import zipfile

import numpy as np

from src.npzdata import load_countmask


SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                      "scripts", "prepare_nyc_event.py")
SPEC = importlib.util.spec_from_file_location("prepare_nyc_event", SCRIPT)
nyc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nyc)


class NycEventPreparationTests(unittest.TestCase):
    @staticmethod
    def _evt3_word(kind, payload):
        return ((kind << 12) | (payload & 0x0fff)).to_bytes(2, "little")

    def test_utm18_matches_wgs84_reference(self):
        east, north = nyc.utm18(40.690221, -73.9892037)
        self.assertAlmostEqual(east, 585406.9515308123, places=3)
        self.assertAlmostEqual(north, 4504860.777801894, places=3)

    def test_nearest_gps_and_timestamp_are_stable(self):
        first = nyc.timestamp_us("2022-12-06_18-27-22_940313",
                                 "%Y-%m-%d_%H-%M-%S_%f")
        second = nyc.timestamp_us("2022-12-06_18-27-23_937633",
                                  "%Y-%m-%d_%H-%M-%S_%f")
        rows = [{"time_us": first, "name": "first"},
                {"time_us": second, "name": "second"}]
        self.assertEqual(nyc.nearest_gps(rows, [first, second], first + 100)["name"], "first")
        midpoint = (first + second) // 2
        self.assertEqual(nyc.nearest_gps(rows, [first, second], midpoint)["name"], "first")

    def test_split_is_exact_and_deterministic(self):
        rows = [{"sample": f"frame_{i:03d}.npz", "path": f"/{i}"} for i in range(101)]
        first = nyc.assign_splits(rows, seed=0)
        second = nyc.assign_splits(list(reversed(rows)), seed=0)
        self.assertEqual(first, second)
        self.assertEqual(sum(row["split"] == "queries" for row in first), 10)
        self.assertEqual(sum(row["split"] == "database" for row in first), 91)

    def test_split_materialisation_uses_hard_links(self):
        with tempfile.TemporaryDirectory() as root:
            all_dir = os.path.join(root, "all")
            os.mkdir(all_dir)
            rows = []
            for index, split in enumerate(("database", "queries")):
                name = f"frame_{index}.npz"
                path = os.path.join(all_dir, name)
                with open(path, "wb") as handle:
                    handle.write(bytes([index]))
                rows.append({"sample": name, "path": path, "split": split})
            nyc.materialise_splits(rows, root)
            for row in rows:
                linked = os.path.join(root, row["split"], row["sample"])
                self.assertEqual(os.stat(row["path"]).st_ino, os.stat(linked).st_ino)

    def test_raw_header_and_gps_csv(self):
        with tempfile.TemporaryDirectory() as root:
            raw = os.path.join(root, "sample.raw")
            with open(raw, "wb") as handle:
                handle.write(b"% date 2022-12-06 18:27:24\n% format EVT3\n")
                handle.write(b"% geometry 1280x720\n\x00\x80")
            self.assertEqual(nyc.raw_header(raw)["geometry"], "1280x720")

            gps = os.path.join(root, "gps.csv")
            with open(gps, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Longitude", "Latitude", "HeadMotion", "Timestamp"])
                writer.writerow([-73.9, 40.7, 10, "2022-12-06_18-27-22_940313"])
                writer.writerow([-73.8, 40.8, 20, "2022-12-06_18-27-23_937633"])
            rows = nyc.read_gps(gps)
            self.assertEqual(len(rows), 2)
            self.assertLess(rows[0]["time_us"], rows[1]["time_us"])

    def test_prepare_traverse_from_evt3_archive(self):
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, "sensor_data_test")
            all_dir = os.path.join(root, "all")
            records = os.path.join(root, "records")
            staging = os.path.join(root, "staging")
            for path in (folder, all_dir, records, staging):
                os.mkdir(path)

            raw = os.path.join(root, "data_test.raw")
            with open(raw, "wb") as handle:
                handle.write(b"% date 2022-12-06 20:45:57\n")
                handle.write(b"% format EVT3\n% geometry 1280x720\n")
                for second in range(10):
                    timestamp = 1_837 + second * nyc.SAMPLE_PERIOD_US
                    handle.write(self._evt3_word(0x8, (timestamp % (1 << 24)) >> 12))
                    handle.write(self._evt3_word(0x6, timestamp & 0x0fff))
                    handle.write(self._evt3_word(0x0, 10))
                    handle.write(self._evt3_word(0x2, 20 + second))
            archive = os.path.join(folder, "data_test.zip")
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.write(raw, os.path.basename(raw))

            gps = os.path.join(folder, "GPS_data_test.csv")
            with open(gps, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Longitude", "Latitude", "HeadMotion", "Timestamp"])
                start = nyc.timestamp_us("2022-12-06 20:45:57", "%Y-%m-%d %H:%M:%S")
                for second in range(10):
                    stamp = nyc.dt.datetime.fromtimestamp(
                        (start + second * nyc.SAMPLE_PERIOD_US) / 1_000_000, nyc.dt.UTC)
                    writer.writerow([-73.9, 40.7, second,
                                     stamp.strftime("%Y-%m-%d_%H-%M-%S_%f")])

            rows, report = nyc.prepare_traverse(folder, all_dir, records, staging)
            self.assertEqual(report["samples"], 10)
            self.assertEqual(len(rows), 10)
            self.assertFalse(os.listdir(staging))
            with np.load(rows[0]["path"]) as events:
                self.assertEqual(events["resolution"].tolist(), [720, 1280])
                self.assertEqual(events["t"].tolist(), [1_837])
                self.assertEqual(events["p"].tolist(), [-1])
            frame = load_countmask(rows[0]["path"])
            self.assertEqual(frame.shape, (3, 720, 1280))
            self.assertEqual(int(frame.max()), 255)


if __name__ == "__main__":
    unittest.main()
