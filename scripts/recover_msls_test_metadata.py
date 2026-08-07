#!/usr/bin/env python3
"""Recover public GPS/panorama metadata for the original hidden-label MSLS test set.

The official 2020 ``metadata.zip`` contains ``seq_info.csv`` and
``subtask_index.csv`` for the six test cities, but no capture coordinates. Mapillary's
current API exposes the original capture record. ``sequence_key`` is the current API
sequence ID and ``frame_number`` is its zero-based image position, so no fuzzy image
matching is involved.

Set ``MAPILLARY_ACCESS_TOKEN`` to a Mapillary client token, then run::

    python scripts/recover_msls_test_metadata.py --msls-root /path/to/msls

The script is resumable through ``<msls-root>/.metadata_recovery`` and writes the
standard ``raw.csv`` beside each test split's existing metadata.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen


CITIES = ("miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala")
API = "https://graph.mapillary.com"
FIELDS = "id,geometry,captured_at,compass_angle,is_pano,sequence"
WORKERS = 16
LEGACY_WORKERS = 8
BATCH_SIZE = 50


def _read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def _save_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle)
    os.replace(temporary, path)


def _get_json(path, params, token):
    url = f"{API}/{path}?{urlencode(params)}"
    request = Request(url, headers={"Authorization": f"OAuth {token}"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            if attempt == 4:
                raise
            wait = int(error.headers.get("Retry-After", 2 ** attempt))
        except URLError:
            if attempt == 4:
                raise
            wait = 2 ** attempt
        time.sleep(wait)


def _fetch_sequences(sequence_ids, token, cache, cache_path):
    missing = sorted(set(sequence_ids) - set(cache))
    if not missing:
        return
    print(f"sequences: fetching {len(missing)}")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(_get_json, "image_ids", {"sequence_id": key}, token): key
            for key in missing
        }
        for done, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            payload = future.result()
            cache[key] = [row["id"] for row in payload["data"]]
            if done % 50 == 0 or done == len(missing):
                _save_json(cache_path, cache)
                print(f"sequences: {done}/{len(missing)}")


def _fetch_images(image_ids, token, cache, cache_path):
    missing = sorted(set(image_ids) - set(cache))
    batches = [missing[i:i + BATCH_SIZE] for i in range(0, len(missing), BATCH_SIZE)]
    if not batches:
        return
    print(f"images: fetching {len(missing)} in {len(batches)} batches")

    def fetch(batch):
        payload = _get_json("", {"ids": ",".join(batch), "fields": FIELDS}, token)
        absent = set(batch) - set(payload)
        if absent:
            raise RuntimeError(f"Mapillary omitted image {sorted(absent)[0]}")
        return payload

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch, batch): batch for batch in batches}
        for done, future in enumerate(as_completed(futures), 1):
            cache.update(future.result())
            if done % 20 == 0 or done == len(batches):
                _save_json(cache_path, cache)
                print(f"images: {done}/{len(batches)} batches")


def _legacy_image_id(key):
    request = Request(f"https://www.mapillary.com/app/?{urlencode({'pKey': key})}")
    for attempt in range(5):
        try:
            with urlopen(request, timeout=60) as response:
                final_url = urlparse(response.geturl())
                values = parse_qs(final_url.query).get("pKey", [])
                if len(values) == 1 and values[0].isdigit():
                    return values[0]
                if final_url.path == "/404":
                    return None
                if attempt == 4:
                    return None
                wait = 2 ** attempt
        except HTTPError as error:
            # urllib reports a legacy key that redirects back to itself as a 302 loop.
            # Mapillary has no current image mapping for it, just as for an explicit /404.
            if error.code == 302:
                return None
            if error.code != 429 and error.code < 500:
                raise
            if attempt == 4:
                raise
            wait = int(error.headers.get("Retry-After", 2 ** attempt))
        except URLError:
            if attempt == 4:
                raise
            wait = 2 ** attempt
        time.sleep(wait)


def _fetch_legacy_ids(keys, cache, cache_path):
    missing = sorted(set(keys) - set(cache))
    if not missing:
        return
    print(f"legacy images: resolving {len(missing)}")
    with ThreadPoolExecutor(max_workers=LEGACY_WORKERS) as pool:
        futures = {pool.submit(_legacy_image_id, key): key for key in missing}
        for done, future in enumerate(as_completed(futures), 1):
            cache[futures[future]] = future.result()
            if done % 10 == 0 or done == len(missing):
                _save_json(cache_path, cache)
                print(f"legacy images: {done}/{len(missing)}")
    unavailable = sum(cache[key] is None for key in set(keys))
    print(f"legacy images: {unavailable} unavailable")


def _write_raw(path, rows, key_to_id, images):
    temporary = path + ".tmp"
    fields = ["", "key", "lon", "lat", "ca", "captured_at", "pano", "available"]
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            image_id = key_to_id[row["key"]]
            if image_id is None or images[image_id] is None:
                writer.writerow({"": row[""], "key": row["key"], "available": False})
                continue
            record = images[image_id]
            lon, lat = record["geometry"]["coordinates"]
            captured = datetime.fromtimestamp(
                int(record["captured_at"]) / 1000, tz=timezone.utc).date().isoformat()
            writer.writerow({"": row[""], "key": row["key"], "lon": lon, "lat": lat,
                             "ca": record["compass_angle"], "captured_at": captured,
                             "pano": bool(record["is_pano"]), "available": True})
    os.replace(temporary, path)


def recover(root, token):
    splits = {}
    for city in CITIES:
        for split in ("database", "query"):
            directory = os.path.join(root, "test", city, split)
            sequence_path = os.path.join(directory, "seq_info.csv")
            subtask_path = os.path.join(directory, "subtask_index.csv")
            if not os.path.exists(sequence_path):
                raise FileNotFoundError(sequence_path)
            selected = {row["key"] for row in _read_rows(subtask_path)
                        if row["all"] == "True"}
            splits[(city, split)] = [row for row in _read_rows(sequence_path)
                                     if row["key"] in selected]

    cache_dir = os.path.join(root, ".metadata_recovery")
    os.makedirs(cache_dir, exist_ok=True)
    sequence_path = os.path.join(cache_dir, "sequences.json")
    legacy_path = os.path.join(cache_dir, "legacy_ids.json")
    image_path = os.path.join(cache_dir, "images.json")
    sequences = _load_json(sequence_path)
    _fetch_sequences((row["sequence_key"] for rows in splits.values() for row in rows),
                     token, sequences, sequence_path)

    fallback_keys = []
    for rows in splits.values():
        for row in rows:
            sequence = sequences[row["sequence_key"]]
            frame = int(row["frame_number"])
            if frame >= len(sequence):
                fallback_keys.append(row["key"])
    legacy_ids = _load_json(legacy_path)
    _fetch_legacy_ids(fallback_keys, legacy_ids, legacy_path)
    key_to_id = {}
    for rows in splits.values():
        for row in rows:
            sequence = sequences[row["sequence_key"]]
            frame = int(row["frame_number"])
            key_to_id[row["key"]] = (sequence[frame] if frame < len(sequence)
                                     else legacy_ids[row["key"]])
    if len(key_to_id) != sum(map(len, splits.values())):
        raise RuntimeError("duplicate image key in MSLS test metadata")

    images = {key: value for key, value in _load_json(image_path).items()
              if value is not None}
    _save_json(image_path, images)
    _fetch_images((image_id for image_id in key_to_id.values() if image_id is not None),
                  token, images, image_path)
    unavailable = [key for key, image_id in key_to_id.items()
                   if image_id is None or images[image_id] is None]
    for rows in splits.values():
        for row in rows:
            image_id = key_to_id[row["key"]]
            if image_id is None or images[image_id] is None:
                continue
            record = images[image_id]
            if row["key"] not in legacy_ids and record["sequence"] != row["sequence_key"]:
                raise RuntimeError(f"{row['key']}: recovered image belongs to wrong sequence")

    for (city, split), rows in splits.items():
        path = os.path.join(root, "test", city, split, "raw.csv")
        _write_raw(path, rows, key_to_id, images)
        print(f"wrote {path}: {len(rows)} rows")
    print(f"recovered {len(key_to_id) - len(unavailable)}/{len(key_to_id)} MSLS test "
          f"images; {len(unavailable)} unavailable records are marked and excluded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msls-root", required=True)
    parser.add_argument("--access-token-file",
                        help="File containing a Mapillary client token; avoids exposing it "
                             "in shell history. MAPILLARY_ACCESS_TOKEN is used otherwise.")
    args = parser.parse_args()
    if args.access_token_file:
        with open(args.access_token_file) as handle:
            token = handle.read().strip()
    else:
        token = os.environ.get("MAPILLARY_ACCESS_TOKEN")
    if not token:
        parser.error("set MAPILLARY_ACCESS_TOKEN or pass --access-token-file")
    recover(os.path.abspath(args.msls_root), token)


if __name__ == "__main__":
    main()
