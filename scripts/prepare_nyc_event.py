"""Prepare NYC-Event-VPR's real EVT3 recordings for MegaEvent's 1 Hz image-set path.

The dataset authors render the RAW stream at 30 fps, retain one frame per second, sample
10% of the resulting frames as queries, and use the remaining 90% as references.  This
script preserves that protocol while keeping the sensor events rather than an RGB-like
render: each retained 33,333 us window is written in I2E's ``x/y/t/p/resolution`` NPZ
schema, then :mod:`src.npzdata` renders the checkpoint's countmask at inference time.

Run from the repository root::

    pixi run python3 scripts/prepare_nyc_event.py

Output is ``<data-dir>/nycevent/numpy/{all,database,queries}``.  ``all`` owns the data;
the two benchmark splits are hard links, so the event payload is stored only once.
"""

import argparse
import bisect
import calendar
import csv
import datetime as dt
import glob
import json
import math
import os
import shutil
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eventcv as ecv  # noqa: E402


DEFAULT_RAW = "/media/adam/vprdatasets/megaevent/NYC-Event-VPR_raw_data"
DEFAULT_DATA = "/media/adam/vprdatasets/megaevent"
WINDOW_US = 33_333
SAMPLE_PERIOD_US = 1_000_000
MAX_GPS_DELTA_US = 550_000
SENSOR_WH = (1280, 720)

# WGS84 / UTM constants. NYC is wholly in zone 18N; keeping the short projection here avoids
# adding a runtime geospatial dependency just to put metric coordinates in a filename.
_K0 = 0.9996
_E = 0.00669438
_E2 = _E * _E
_E3 = _E2 * _E
_EP2 = _E / (1 - _E)
_R = 6_378_137.0
_M1 = 1 - _E / 4 - 3 * _E2 / 64 - 5 * _E3 / 256
_M2 = 3 * _E / 8 + 3 * _E2 / 32 + 45 * _E3 / 1024
_M3 = 15 * _E2 / 256 + 45 * _E3 / 1024
_M4 = 35 * _E3 / 3072


def utm18(latitude, longitude):
    """WGS84 latitude/longitude -> UTM zone 18N easting/northing in metres."""
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    central = math.radians(-75.0)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    tan_lat = sin_lat / cos_lat
    tan2, tan4 = tan_lat * tan_lat, tan_lat ** 4
    n = _R / math.sqrt(1 - _E * sin_lat ** 2)
    c = _EP2 * cos_lat ** 2
    a = cos_lat * ((lon - central + math.pi) % (2 * math.pi) - math.pi)
    a2, a3 = a * a, a ** 3
    a4, a5, a6 = a ** 4, a ** 5, a ** 6
    m = _R * (_M1 * lat - _M2 * math.sin(2 * lat)
              + _M3 * math.sin(4 * lat) - _M4 * math.sin(6 * lat))
    east = (_K0 * n * (a + a3 / 6 * (1 - tan2 + c)
                        + a5 / 120 * (5 - 18 * tan2 + tan4 + 72 * c - 58 * _EP2))
            + 500_000)
    north = _K0 * (m + n * tan_lat * (
        a2 / 2 + a4 / 24 * (5 - tan2 + 9 * c + 4 * c ** 2)
        + a6 / 720 * (61 - 58 * tan2 + tan4 + 600 * c - 330 * _EP2)))
    return east, north


def timestamp_us(value, fmt):
    """Parse the dataset's timezone-free wall clock into a stable integer time base."""
    stamp = dt.datetime.strptime(value, fmt)
    return calendar.timegm(stamp.timetuple()) * 1_000_000 + stamp.microsecond


def raw_header(path):
    """Return lower-case RAW header fields without touching the binary payload."""
    fields = {}
    with open(path, "rb") as handle:
        while handle.peek(1)[:1] == b"%":
            line = handle.readline().decode("ascii", errors="replace").strip()
            body = line.lstrip("%").strip()
            key, _, value = body.partition(" ")
            fields[key.lower()] = value.strip()
    return fields


def read_gps(path):
    """Sorted GPS rows with a parsed microsecond wall clock."""
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                lat, lon = float(row["Latitude"]), float(row["Longitude"])
                stamp = timestamp_us(row["Timestamp"], "%Y-%m-%d_%H-%M-%S_%f")
                heading = float(row["HeadMotion"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            rows.append({"time_us": stamp, "timestamp": row["Timestamp"],
                         "latitude": lat, "longitude": lon, "heading": heading})
    rows.sort(key=lambda row: row["time_us"])
    if len(rows) < 2:
        raise ValueError(f"{path}: only {len(rows)} usable GPS fixes")
    return rows


def nearest_gps(rows, times, sample_us):
    """Nearest GPS row to ``sample_us``; ties prefer the earlier fix."""
    index = bisect.bisect_left(times, sample_us)
    choices = [i for i in (index - 1, index) if 0 <= i < len(rows)]
    return min((rows[i] for i in choices), key=lambda row: (abs(row["time_us"] - sample_us),
                                                             row["time_us"]))


def is_uniform_countmask(stream):
    """The authors' Sobel>0 filter at threshold zero: reject only constant frames."""
    frame = stream.countmask(white_frame=False).numpy()
    return not (np.any(frame[:, 1:, :] != frame[:, :-1, :])
                or np.any(frame[:, :, 1:] != frame[:, :, :-1]))


def sample_name(gps, traverse, raw_member, sample_us):
    east, north = utm18(gps["latitude"], gps["longitude"])
    member = os.path.splitext(os.path.basename(raw_member))[0]
    # Fields 1 and 2 are the only contract src.npzdata.utm_from_paths consumes. The rest make
    # every row auditable back to its recording and GPS fix.
    return (f"@{east:.3f}@{north:.3f}@18@T@{gps['latitude']:.8f}@"
            f"{gps['longitude']:.8f}@{traverse}@{member}@{gps['heading']:.3f}@"
            f"{sample_us}@@@{gps['timestamp']}@.npz")


def save_window(path, stream, window_start_us):
    events = stream.numpy()
    if events.shape[0]:
        x = events[:, 0].astype(np.uint16)
        y = events[:, 1].astype(np.uint16)
        t = (events[:, 2].astype(np.int64) - window_start_us).astype(np.int32)
        p = np.where(events[:, 3] > 0, 1, -1).astype(np.int8)
    else:
        x = y = np.empty(0, dtype=np.uint16)
        t = np.empty(0, dtype=np.int32)
        p = np.empty(0, dtype=np.int8)
    temporary = path + ".part"
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, x=x, y=y, t=t, p=p,
                            resolution=np.array([SENSOR_WH[1], SENSOR_WH[0]], np.uint16))
    os.replace(temporary, path)
    return int(events.shape[0])


def extract_member(archive, member, staging_dir):
    """Extract one member atomically; callers remove it after event decoding."""
    os.makedirs(staging_dir, exist_ok=True)
    destination = os.path.join(staging_dir, os.path.basename(member))
    temporary = destination + ".part"
    with zipfile.ZipFile(archive) as zipped:
        info = zipped.getinfo(member)
        free = shutil.disk_usage(staging_dir).free
        if free < info.file_size + 1_000_000_000:
            raise OSError(f"{staging_dir}: {free / 1e9:.1f} GB free, but extracting "
                          f"{member} needs {info.file_size / 1e9:.1f} GB")
        with zipped.open(info) as source, open(temporary, "wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    os.replace(temporary, destination)
    return destination


def write_csv(path, rows, fieldnames):
    temporary = path + ".tmp"
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


FIELDS = ("sample", "path", "traverse", "raw_member", "window_start_us", "window_end_us",
          "sample_wall_time_us", "gps_timestamp", "gps_delta_us", "latitude", "longitude",
          "heading", "easting", "northing", "n_events")


def prepare_traverse(folder, all_dir, record_dir, staging_dir, window_us=WINDOW_US,
                     max_gps_delta_us=MAX_GPS_DELTA_US):
    traverse = os.path.basename(folder).removeprefix("sensor_data_")
    reports = os.path.join(record_dir, f"{traverse}.csv")
    if os.path.exists(reports):
        with open(reports, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if all(os.path.exists(row["path"]) for row in rows):
            print(f"{traverse}: {len(rows)} prepared samples (resume)")
            return rows, {"resume": True, "samples": len(rows)}

    gps_files = glob.glob(os.path.join(folder, "GPS_data_*.csv"))
    archives = glob.glob(os.path.join(folder, "data_*.zip"))
    if len(gps_files) != 1:
        raise ValueError(f"{folder}: expected one GPS CSV, found {len(gps_files)}")
    if not archives:
        print(f"{traverse}: skipped (no event archive)")
        return [], {"skipped": "no event archive"}
    if len(archives) != 1:
        raise ValueError(f"{folder}: expected one event archive, found {len(archives)}")

    gps_rows = read_gps(gps_files[0])
    gps_times = [row["time_us"] for row in gps_rows]
    rows, skipped_gap, skipped_uniform, duplicates = [], 0, 0, 0
    seen_wall_times = set()
    archive = archives[0]
    with zipfile.ZipFile(archive) as zipped:
        members = sorted(info.filename for info in zipped.infolist()
                         if not info.is_dir() and info.filename.lower().endswith(".raw"))
    if not members:
        raise ValueError(f"{archive}: contains no RAW member")

    for member in members:
        raw_path = extract_member(archive, member, staging_dir)
        try:
            header = raw_header(raw_path)
            if not header.get("format", "").lower().startswith("evt3"):
                raise ValueError(f"{member}: expected EVT3, got {header.get('format')!r}")
            if header.get("geometry") != "1280x720":
                raise ValueError(f"{member}: expected 1280x720, got {header.get('geometry')!r}")
            header_us = timestamp_us(header["date"], "%Y-%m-%d %H:%M:%S")
            reader = ecv.open(raw_path, sensor_size=SENSOR_WH)
            lo_us = math.ceil(reader.time_span_ms[0] * 1000)
            hi_us = math.floor(reader.time_span_ms[1] * 1000)
            # The author's first 1 Hz frame starts at the RAW recording epoch even when the
            # first CD event arrives a few milliseconds later (1.837 ms in one NYC file).
            first = math.floor(lo_us / SAMPLE_PERIOD_US)
            last = math.floor(hi_us / SAMPLE_PERIOD_US)
            print(f"{traverse}: {os.path.basename(member)} -> {max(0, last-first+1)} candidates")

            for second in range(first, last + 1):
                window_start = second * SAMPLE_PERIOD_US
                wall_time = header_us + window_start
                if wall_time in seen_wall_times:
                    duplicates += 1
                    continue
                gps = nearest_gps(gps_rows, gps_times, wall_time)
                delta = abs(gps["time_us"] - wall_time)
                if delta > max_gps_delta_us:
                    skipped_gap += 1
                    continue
                stream = reader.slice(t0_ms=window_start / 1000,
                                      t1_ms=(window_start + window_us) / 1000)
                if len(stream) == 0 or is_uniform_countmask(stream):
                    skipped_uniform += 1
                    continue
                name = sample_name(gps, traverse, member, wall_time)
                destination = os.path.join(all_dir, name)
                if os.path.exists(destination):
                    with np.load(destination) as existing:
                        n_events = int(existing["x"].shape[0])
                else:
                    n_events = save_window(destination, stream, window_start)
                east, north = utm18(gps["latitude"], gps["longitude"])
                rows.append({"sample": name, "path": os.path.abspath(destination),
                             "traverse": traverse, "raw_member": member,
                             "window_start_us": window_start,
                             "window_end_us": window_start + window_us,
                             "sample_wall_time_us": wall_time,
                             "gps_timestamp": gps["timestamp"], "gps_delta_us": delta,
                             "latitude": gps["latitude"], "longitude": gps["longitude"],
                             "heading": gps["heading"], "easting": east, "northing": north,
                             "n_events": n_events})
                seen_wall_times.add(wall_time)
        finally:
            os.remove(raw_path)

    rows.sort(key=lambda row: (int(row["sample_wall_time_us"]), row["sample"]))
    write_csv(reports, rows, FIELDS)
    report = {"resume": False, "raw_members": len(members), "samples": len(rows),
              "gps_gap_skips": skipped_gap, "uniform_skips": skipped_uniform,
              "duplicate_boundary_skips": duplicates}
    print(f"{traverse}: {len(rows)} samples, {skipped_gap} GPS-gap, "
          f"{skipped_uniform} empty/uniform, {duplicates} duplicate")
    return rows, report


def assign_splits(rows, seed=0, query_fraction=0.1):
    """Return rows with an exact, deterministic global 10/90 assignment."""
    ordered = sorted(rows, key=lambda row: row["sample"])
    n_query = int(round(len(ordered) * query_fraction))
    generator = np.random.default_rng(seed)
    query_indices = set(int(i) for i in generator.choice(len(ordered), n_query, replace=False))
    return [{**row, "split": "queries" if i in query_indices else "database"}
            for i, row in enumerate(ordered)]


def materialise_splits(rows, numpy_dir):
    expected = {"database": set(), "queries": set()}
    for row in rows:
        expected[row["split"]].add(row["sample"])
    for split, names in expected.items():
        directory = os.path.join(numpy_dir, split)
        os.makedirs(directory, exist_ok=True)
        actual = {name for name in os.listdir(directory) if name.endswith(".npz")}
        stale = actual - names
        if stale:
            raise RuntimeError(f"{directory}: {len(stale)} files belong to another split; "
                               "use a fresh output directory rather than mixing protocols")
        for name in sorted(names - actual):
            os.link(os.path.join(numpy_dir, "all", name), os.path.join(directory, name))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", default=DEFAULT_RAW)
    parser.add_argument("--data-dir", default=DEFAULT_DATA,
                        help="output root; writes <data-dir>/nycevent/numpy")
    parser.add_argument("--traverse", nargs="+",
                        help="only these sensor_data_* suffixes (also implies --no-finalize)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-us", type=int, default=WINDOW_US)
    parser.add_argument("--max-gps-delta-ms", type=float, default=MAX_GPS_DELTA_US / 1000)
    parser.add_argument("--no-finalize", action="store_true",
                        help="prepare canonical samples but do not create database/query links")
    args = parser.parse_args()
    if args.window_us <= 0 or args.window_us > SAMPLE_PERIOD_US:
        parser.error("--window-us must be in 1..1000000")

    dataset_dir = os.path.join(args.data_dir, "nycevent")
    numpy_dir = os.path.join(dataset_dir, "numpy")
    all_dir = os.path.join(numpy_dir, "all")
    record_dir = os.path.join(dataset_dir, "records")
    staging_dir = os.path.join(dataset_dir, ".staging")
    for directory in (all_dir, record_dir, staging_dir):
        os.makedirs(directory, exist_ok=True)

    folders = sorted(path for path in glob.glob(os.path.join(args.raw_root, "sensor_data_*"))
                     if os.path.isdir(path))
    if args.traverse:
        wanted = set(args.traverse)
        folders = [path for path in folders
                   if os.path.basename(path).removeprefix("sensor_data_") in wanted]
        missing = wanted - {os.path.basename(path).removeprefix("sensor_data_")
                            for path in folders}
        if missing:
            parser.error(f"unknown traverse(s): {sorted(missing)}")
    if not folders:
        raise FileNotFoundError(f"no sensor_data_* directories under {args.raw_root}")

    all_rows, traverse_reports = [], {}
    for folder in folders:
        rows, report = prepare_traverse(
            folder, all_dir, record_dir, staging_dir, window_us=args.window_us,
            max_gps_delta_us=round(args.max_gps_delta_ms * 1000))
        all_rows.extend(rows)
        traverse_reports[os.path.basename(folder).removeprefix("sensor_data_")] = report

    finalize = not args.no_finalize and not args.traverse
    if finalize:
        prepared_records = sorted(glob.glob(os.path.join(record_dir, "*.csv")))
        all_rows = []
        for path in prepared_records:
            with open(path, newline="") as handle:
                all_rows.extend(csv.DictReader(handle))
        split_rows = assign_splits(all_rows, args.seed)
        materialise_splits(split_rows, numpy_dir)
        write_csv(os.path.join(dataset_dir, "manifest.csv"), split_rows, FIELDS + ("split",))
        protocol = {"name": "authors_style_seed0" if args.seed == 0
                    else f"authors_style_seed{args.seed}",
                    "sample_rate_hz": 1, "source_render_rate_hz": 30,
                    "window_us": args.window_us, "query_fraction": 0.1,
                    "seed": args.seed, "positive_radius_m": 25.0,
                    "n_samples": len(split_rows),
                    "n_database": sum(row["split"] == "database" for row in split_rows),
                    "n_queries": sum(row["split"] == "queries" for row in split_rows),
                    "traverses": traverse_reports}
        with open(os.path.join(dataset_dir, "protocol.json") + ".tmp", "w") as handle:
            json.dump(protocol, handle, indent=2)
        os.replace(os.path.join(dataset_dir, "protocol.json") + ".tmp",
                   os.path.join(dataset_dir, "protocol.json"))
        print(f"final: {protocol['n_database']} database x {protocol['n_queries']} queries "
              f"(seed {args.seed}) -> {numpy_dir}")
    else:
        print(f"prepared {len(all_rows)} canonical samples; split finalization skipped")


if __name__ == "__main__":
    sys.exit(main())
