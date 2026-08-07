# MegaEvent continuation handover

This document records the work completed in the current evaluation effort and the exact
state in which it was left. The immediate unfinished goal is to recover the publicly
available Mapillary metadata needed for the MSLS test split, audit the recovered subset,
and then run MegaEvent ViT-S, MegaEvent ViT-B, EventGeM, and EventVLAD on it.

The repository is deliberately dirty. Do **not** reset, clean, or revert files just to get
a tidy worktree. The changes and untracked files listed below are part of this work or
predate it and must be preserved.

## Executive status

- NYC-Event-VPR support is implemented using the original authors' 1 Hz subset protocol.
- NYC has been prepared from the real EVT3 recordings and fully evaluated with MegaEvent
  ViT-S, MegaEvent ViT-B, EventGeM, and EventVLAD at a 25 m tolerance.
- EventGeM and EventVLAD have also been run on the pooled 25 m Brisbane Event and NSAVP
  protocols.
- Pitts is implemented and fully evaluated with all four requested methods.
- The shared image-set evaluation path now understands geographic datasets, Pitts' explicit
  ground truth, and the official per-city MSLS layout.
- MSLS code is implemented and unit-tested, but full evaluation is blocked on finishing
  metadata recovery with a private Mapillary client token.
- The supplied MSLS `metadata.zip` is the correct official archive. Downloading it again is
  not useful and will not add the missing test coordinates.
- No valid `raw.csv` currently exists under the MSLS test directories. This is intentional:
  twelve invalid partial files were removed after an expired public token caused API failures
  to be cached as missing records.
- Current unit suite: 12 tests passing.

## Non-negotiable protocol decisions

### The `.npz` files are event streams, not pre-rendered countmask arrays

The converted image-set files use I2E's sparse event schema:

```text
x, y, t, p, resolution
```

`resolution` is `[height, width]`, timestamps are in microseconds, and polarity is signed.
The files are rendered into the model's requested representation at inference time.
For MegaEvent checkpoints used here that representation is `countmask`, produced by
`src.npzdata.load_countmask` through eventcv. The serialized `.npz` itself is **not** a
three-channel countmask image.

This was an explicit concern during the NYC work, so do not replace the event arrays with
rendered RGB/countmask arrays. Keeping the sparse events also allows EventGeM to build MCTS
and EventVLAD to split a stream into temporal count bins.

### All requested VPR scoring uses a 25 m positive radius

- NYC and Tokyo derive positives directly from UTM coordinates.
- Brisbane Event and NSAVP use their pooled traverse geometry with a 25 m headline radius.
- Pitts uses `ground_truth_new.npy`; the CLI still records 25 m for consistent reporting,
  but its supplied positive lists are authoritative.
- MSLS uses independent 25 m neighbourhoods within each city. Cross-city matches must never
  be considered positives.

Recall is calculated over queries with at least one positive. MSLS additionally reports the
official-style mAP at the same cutoffs.

### Use Pixi for CUDA

The machine has an RTX 2080 with 8 GB VRAM. The reliable invocation pattern is:

```bash
CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python -u ...
```

Do not fall back to the system Python for evaluations. `CONDA_OVERRIDE_CUDA=12` is needed for
Pixi's CUDA environment resolution on this machine.

## 1. NYC-Event-VPR implementation

### Protocol interpretation

The original authors render their raw stream at 30 fps, retain one frame per second, choose
10% of those retained frames as queries, and use the other 90% as the reference database.
The implementation follows that sampling/split protocol but preserves the real events:

- one candidate every 1,000,000 microseconds;
- a 33,333 microsecond event window for each retained 1 Hz sample, corresponding to one
  source frame at 30 fps;
- nearest GPS fix, with a maximum accepted offset of 550 ms;
- deterministic global 10/90 split using NumPy RNG seed 0;
- 25 m UTM ground-truth radius.

The raw source is:

```text
/media/adam/vprdatasets/megaevent/NYC-Event-VPR_raw_data
```

The prepared dataset is:

```text
/media/adam/vprdatasets/megaevent/nycevent
```

### Preparation script

`scripts/prepare_nyc_event.py` was added. It:

- discovers `sensor_data_*` traverses;
- reads the accompanying GPS CSV;
- streams `.raw` members out of the traverse ZIPs one at a time, avoiding extraction of the
  whole dataset;
- verifies EVT3 format and 1280x720 geometry;
- aligns the event timestamps to the RAW header's wall-clock epoch;
- handles recordings split across multiple RAW members and removes duplicate boundary
  samples;
- rejects samples with a GPS gap over the configured threshold;
- applies the authors' effective zero-threshold Sobel/uniform-frame rejection;
- converts WGS84 GPS coordinates to UTM zone 18N without adding a geospatial dependency;
- writes compressed sparse event arrays atomically;
- writes per-traverse audit CSVs and a global manifest;
- stores canonical samples once under `numpy/all` and uses hard links for `database` and
  `queries` so the 10/90 split does not duplicate the event payload;
- supports safe resumption.

Prepared protocol from
`/media/adam/vprdatasets/megaevent/nycevent/protocol.json`:

| Field | Value |
|---|---:|
| Total retained samples | 55,433 |
| Database | 49,890 |
| Queries | 5,543 |
| Scorable queries at 25 m | 5,538 |
| Sample rate | 1 Hz |
| Event window | 33,333 us |
| Query fraction | 10% |
| Split seed | 0 |

One traverse (`2022-12-09_19-40-27`) had no event archive and was skipped. Across the
multi-member recordings the preparation also recorded small numbers of GPS-gap, uniform,
and duplicate-boundary skips in `protocol.json`; those are expected and auditable.

### NYC tests

`tests/test_nyc_event.py` covers:

- UTM conversion against a WGS84 reference;
- stable timestamp and nearest-GPS matching;
- exact deterministic 10/90 assignment;
- hard-link split materialisation;
- RAW header and GPS CSV parsing;
- a synthetic EVT3 ZIP through the complete preparation path, including the saved sparse
  event schema and its countmask rendering.

### NYC full results

All values below are `R@1 / R@5 / R@10 / R@20` at 25 m over the 5,538 scorable queries.

| Method/space | Recall |
|---|---|
| MegaEvent ViT-S native | 0.6835 / 0.8270 / 0.8707 / 0.9052 |
| MegaEvent ViT-S PCA | 0.7521 / 0.8817 / 0.9131 / 0.9354 |
| MegaEvent ViT-B native | 0.7418 / 0.8761 / 0.9119 / 0.9355 |
| MegaEvent ViT-B PCA | 0.7864 / 0.9075 / 0.9345 / 0.9549 |
| EventVLAD native | 0.1103 / 0.2095 / 0.2580 / 0.3220 |
| EventVLAD PCA | 0.1575 / 0.2718 / 0.3333 / 0.4032 |
| EventGeM native | 0.4294 / 0.6177 / 0.6912 / 0.7606 |
| EventGeM native + rerank | 0.7514 / 0.8050 / 0.8164 / 0.8270 |
| EventGeM PCA | 0.4930 / 0.6858 / 0.7497 / 0.8115 |
| EventGeM PCA + rerank | 0.8003 / 0.8476 / 0.8584 / 0.8693 |
| EventGeM full-whitening PCA1 | 0.4079 / 0.5831 / 0.6490 / 0.7149 |
| EventGeM PCA1 + rerank | 0.7216 / 0.7625 / 0.7732 / 0.7844 |

Existing NYC result files:

```text
/media/adam/vprdatasets/megaevent/nycevent_eval/nycevent/results_megaevent.json
/media/adam/vprdatasets/megaevent/nycevent_eval_vitb/nycevent/results_megaevent.json
/media/adam/vprdatasets/megaevent/nycevent_eval_eventgem/nycevent/results_eventgem.json
/media/adam/vprdatasets/megaevent/nycevent_eval_eventvlad/nycevent/results_eventvlad.json
```

The NYC ViT-S run used `/home/adam/repo/megaevent/ckpts/s_salad_ft4.pt`, step 3300,
SHA256 `a804f53aebb8dcdcceff7cfb8b78d9c0595d3d86738896dcade09e1c75ebb3e0`.
The ViT-B run used the step-10000 checkpoint listed later.

## 2. Shared image-set and method support

### Unified dataset adapter

`src/imagesets.py` was added. Its `ImageSet` carries:

- database/query paths;
- stable database/query keys;
- a boolean ground-truth matrix in `[database, query]` orientation;
- optional UTM coordinates for geographic figures;
- counts of explicitly excluded database/query records.

It provides:

- generic geographic loading for NYC and Tokyo;
- Pitts numeric-ID loading and supplied ground truth;
- MSLS per-city loading and coordinate recovery fallback;
- a ground-truth-aware `--limit` path which keeps every selected smoke-test query scorable.

The old image evaluation assumed every dataset used `numpy/database` and `numpy/queries`
with coordinates embedded in filenames. `main.py` now has dataset-native defaults:

```text
msls      database / query
nycevent  database / queries
pitts     ref / query
tokyo247  database / queries
```

`src/imagevpr.py` now consumes the adapter instead of constructing geographic ground truth
itself. It records excluded rows, uses checkpoint-specific artifact tags, and only draws a
geographic error map when coordinates are part of the returned dataset.

### Explicit MegaEvent checkpoints

`src/methods.py` now accepts `--ckpt`. An explicit checkpoint is inspected normally, hashed,
and incorporated into both metadata and artifact tags. This prevents ViT-S and ViT-B banks
from overwriting each other and makes an evaluation reproducible even if a model-name default
changes.

The two main checkpoints used for Pitts and intended for MSLS are:

```text
ViT-S: /media/adam/vprdatasets/megaevent/runs/s_salad_ft4_v2_vpr/step10000.pt
SHA256 prefix: 5b576403be

ViT-B: /media/adam/vprdatasets/megaevent/runs/b_full_P64_v4_vpr/step10000.pt
SHA256 prefix: 403872f88a
```

### Representation paths for comparison methods

All methods consume the same sparse `.npz` event stream but render the representation they
were designed for:

- MegaEvent: eventcv countmask, then the checkpoint's normalization/resize/model path.
- EventVLAD on independent images: split each short event stream into three temporal count
  bins, normalize each as upstream does, denoise, then VGG16 + NetVLAD.
- EventGeM: ten-channel MCTS at 240x320, SuperEvent backbone, GeM global descriptor, and
  SuperEvent local keypoint homography reranking.
- Sparse Event, where used elsewhere: event counts and the Event-LAB-style random pixel
  readout.

`src/npzdata.py` remains the representation boundary. In particular, it supplies countmask,
count, MCTS, and three-bin EventVLAD inputs from the same event arrays.

### Scoring changes

`src/scoring.py` now supports official-style MSLS mAP at the normal cutoffs in addition to
recall. `src/inference.py` clamps requested `R@K` to the available gallery size so tiny
`--limit` smoke tests cannot over-index upstream's recall helper.

For MSLS, `src/imagevpr.py` also writes a text prediction file containing each scorable query
key followed by its ranked database keys. The JSON records which descriptor space produced
that ranking.

## 3. EventGeM and EventVLAD pooled traverse evaluations

### EventVLAD author-style pooled path

`scripts/eventvlad_pooled.py` was added for Brisbane Event and NSAVP. This path is different
from the independent-image path and intentionally follows EventVLAD's original temporal
input:

- real traverses are sliced into consecutive 50 ms count frames;
- background activity filtering uses 50,000 us;
- three consecutive frames form one denoiser input;
- the descriptor is attached to the middle frame;
- the author denoiser and VGG16/NetVLAD weights are loaded from
  `/home/adam/repo/Event-LAB`;
- both datasets use the same pooled geometry and 25 m scoring helpers as EventGeM.

`EventVLADMethod.encode` was separated from descriptor iteration so the pooled script can
feed already-assembled triplets without duplicating the model forward path.

### EventGeM pooled improvements

`scripts/eventgem_pooled.py`, `scripts/brisbane_pooled.py`, and `src/eventgemlocal.py` were
extended so pooled EventGeM reports both its global and local-reranked stages. The reranker:

- takes the top-50 global candidates;
- uses upstream-style SuperEvent keypoints and homography verification;
- can share one full database keypoint store across descriptor spaces when that is cheaper
  than several nearly complete shortlist-union stores;
- uses dataset-qualified figure names so Brisbane and NSAVP do not overwrite each other.

### Pooled results

All values are `R@1 / R@5 / R@10 / R@20` at 25 m.

| Dataset | Method/space | Recall |
|---|---|---|
| Brisbane Event | EventVLAD native | 0.2155 / 0.3503 / 0.4333 / 0.5230 |
| Brisbane Event | EventVLAD PCA | 0.2770 / 0.4182 / 0.4971 / 0.5852 |
| Brisbane Event | EventGeM native | 0.5145 / 0.6648 / 0.7214 / 0.7754 |
| Brisbane Event | EventGeM native + rerank | 0.7761 / 0.8069 / 0.8191 / 0.8290 |
| Brisbane Event | EventGeM PCA128 p=0.5 | 0.6827 / 0.8222 / 0.8606 / 0.8926 |
| Brisbane Event | EventGeM PCA128 p=0.5 + rerank | 0.8772 / 0.9001 / 0.9085 / 0.9163 |
| NSAVP | EventVLAD native | 0.1059 / 0.1843 / 0.2359 / 0.3040 |
| NSAVP | EventVLAD PCA | 0.1289 / 0.2122 / 0.2637 / 0.3253 |
| NSAVP | EventGeM native | 0.2937 / 0.3950 / 0.4481 / 0.5104 |
| NSAVP | EventGeM native + rerank | 0.4550 / 0.5034 / 0.5268 / 0.5580 |
| NSAVP | EventGeM PCA128 p=0.5 | 0.3427 / 0.4714 / 0.5327 / 0.6017 |
| NSAVP | EventGeM PCA128 p=0.5 + rerank | 0.5562 / 0.6059 / 0.6284 / 0.6533 |

The Brisbane galleries/queries were 54,620 / 14,280. EventVLAD NSAVP had 100,949 /
20,952 after three-frame alignment; EventGeM had 100,958 / 20,954.

Artifacts:

```text
/media/adam/vprdatasets/megaevent/eventgem_pooled/brisbane_event.json
/media/adam/vprdatasets/megaevent/eventgem_pooled/nsavp.json
/media/adam/vprdatasets/megaevent/eventvlad_pooled/brisbane_event.json
/media/adam/vprdatasets/megaevent/eventvlad_pooled/nsavp.json
```

The cached pooled artifacts are large: approximately 43 GB for EventGeM and 732 MB for
EventVLAD. Do not delete them casually; rerunning local keypoint extraction is expensive.

## 4. Pitts implementation and completed evaluation

### Dataset layout and loader

The Pitts root is:

```text
/media/adam/vprdatasets/megaevent/pitts
```

Expected layout:

```text
pitts/
  ref_countmask/numpy/<numeric-id>.npz
  query_countmask/numpy/<numeric-id>.npz
  ground_truth_new.npy
```

The loader sorts filenames numerically, not lexically, verifies uniqueness and complete query
coverage, maps the supplied ID lists to descriptor rows, and produces a boolean
`[database, query]` matrix. It rejects missing IDs, duplicate ground-truth rows, malformed
ground truth, and omitted queries rather than silently changing alignment.

Full Pitts dimensions are 23,000 database images by 1,000 queries. Every query is scorable,
with a mean of 23 supplied positives.

### Pitts results

All values are `R@1 / R@5 / R@10 / R@20`.

| Method/space | Recall |
|---|---|
| MegaEvent ViT-S native | 0.613 / 0.946 / 0.984 / 0.996 |
| MegaEvent ViT-S PCA | 0.699 / 0.974 / 0.994 / 0.998 |
| MegaEvent ViT-B native | 0.647 / 0.969 / 0.998 / 1.000 |
| MegaEvent ViT-B PCA | 0.719 / 0.985 / 0.999 / 0.999 |
| EventVLAD native | 0.005 / 0.013 / 0.029 / 0.048 |
| EventVLAD PCA | 0.004 / 0.021 / 0.031 / 0.054 |
| EventGeM native | 0.213 / 0.399 / 0.510 / 0.603 |
| EventGeM native + rerank | 0.685 / 0.695 / 0.696 / 0.701 |
| EventGeM PCA | 0.573 / 0.807 / 0.864 / 0.908 |
| EventGeM PCA + rerank | 0.948 / 0.951 / 0.951 / 0.953 |
| EventGeM full-whitening PCA1 | 0.490 / 0.721 / 0.789 / 0.859 |
| EventGeM PCA1 + rerank | 0.911 / 0.915 / 0.917 / 0.919 |

Artifacts are under:

```text
/media/adam/vprdatasets/megaevent/evaluations/pitts
```

Headline JSON files:

```text
results_step10000_s9999_5b576403be_countmask.json   # ViT-S
results_step10000_s9999_403872f88a_countmask.json   # ViT-B
results_eventgem_se_240x320.json
results_eventvlad.json
```

That directory is approximately 9.1 GB, mostly because EventGeM retains local-feature caches.

## 5. MSLS implementation

### Dataset layout

The root supplied by the user is:

```text
/media/adam/vprdatasets/megaevent/msls
```

The event data is already present at:

```text
test_countmask/numpy/<city>/<database|query>/images/<key>.npz
```

Again, despite the directory name, the `.npz` files are sparse event arrays generated from
the source images, not serialized countmask frames.

The six official test cities, in official order, are:

```text
miami, athens, buenosaires, stockholm, bengaluru, kampala
```

The metadata tree is:

```text
test/<city>/<database|query>/
```

### What is wrong with the official test metadata

The user's archive is correct:

```text
/media/adam/vprdatasets/megaevent/msls/metadata.zip
size:   180,754,151 bytes
SHA256: 9ede552cb04245a851bb44908719b3db56b364b42cf3eefce35846796e2842ce
entry timestamps: 2020-04-22
```

For validation/train cities the official archive includes capture-coordinate files such as
`raw.csv` and `postprocessed.csv`. For the six test cities it contains only:

```text
seq_info.csv
subtask_index.csv
```

It does not contain the longitude/latitude or UTM coordinates required to construct the
official 25 m positives. There is therefore nothing different the user needs to download
from the same metadata link.

The official Mapillary SLS repository was also checked. Its April 13, 2026 change says test
ground truth is available and changes the evaluator to load test cities through validation
mode, but the published archive still lacks the `raw.csv`/`postprocessed.csv` files that code
then expects. In other words, the current official repository and the current official
archive are internally inconsistent.

### Verified Mapillary recovery mapping

The hidden coordinates can be recovered for images that still exist in the current
Mapillary graph:

- `sequence_key` in `seq_info.csv` is the current Mapillary sequence ID;
- `frame_number` is the zero-based position in the ordered response from
  `graph.mapillary.com/image_ids?sequence_id=...`;
- that exact mapping was validated on a Miami test sequence and against a labeled Copenhagen
  control sequence;
- no visual or nearest-time fuzzy matching is involved.

Some old sequences no longer enumerate. For those, the old image key can be passed through
the current Mapillary web redirect (`/app/?pKey=<old key>`) to obtain the current numeric
image ID. This recovers most of the legacy cases.

Do **not** interpolate or fabricate coordinates for deleted records. They must be marked
unavailable and explicitly excluded.

### Recovery counts and cache state

Across all test metadata there are 65,862 image rows. The official `all` subtask selects
39,940 of them:

| Split | Selected rows |
|---|---:|
| Database | 38,770 |
| Query | 1,170 |
| Total | 39,940 |

Recovery facts:

- 1,131 sequence IDs have been enumerated and cached.
- 35 old sequences could not provide the needed positions through normal enumeration.
- 1,827 legacy image keys were checked through redirects.
- 441 records across the full 65,862-row metadata are genuinely unavailable.
- Of the official `all` subset, 364 are genuinely deleted: 358 database and 6 query rows.
- All 364 selected deleted rows are in Stockholm.
- 24,120 selected records currently have valid API metadata cached.
- 15,456 selected records still require a successful metadata API fetch.

Recovery caches live in:

```text
/media/adam/vprdatasets/megaevent/msls/.metadata_recovery/
  sequences.json   # complete, 1,131 sequence entries
  legacy_ids.json  # complete, 1,827 entries; 441 genuine null redirects
  images.json      # 65,418 keys currently, 40,000 valid and 25,418 bad nulls
```

The `images.json` nulls are **not** all deleted images. A public Mapillary client token taken
from the website bundle worked for roughly 40,000 records, then the API began returning OAuth
error code 368 (`Please log in to see this page`). The earlier recovery attempt incorrectly
treated those failed batches as absent images, leaving 25,418 null cache values.

The script has since been fixed so that:

- null values in `images.json` are discarded on load and the cleaned cache is saved;
- only records in the official `all` subset are fetched;
- an API HTTP 400 is fatal rather than classified as an unavailable image;
- only a true missing legacy redirect is written as unavailable;
- interruption is resumable through the three JSON caches.

Twelve `raw.csv` files generated by the bad attempt were explicitly removed. A search for
`test/*/*/raw.csv` currently returns nothing, so the evaluator cannot accidentally consume
partial metadata.

### Recovery script

`scripts/recover_msls_test_metadata.py` was added. On success it writes a `raw.csv` beside
each of the twelve city/split metadata pairs, with fields compatible with the loader plus an
`available` boolean. Deleted records receive a row with `available=False`; valid rows receive
longitude, latitude, compass angle, capture date, and panorama status.

The token is accepted either via `MAPILLARY_ACCESS_TOKEN` or `--access-token-file`. The file
form is preferred because it does not expose the token in shell history or logs.

### MSLS loader behavior

`src/imagesets.py` implements the following official-style pipeline per city:

1. Read `subtask_index.csv` and retain rows with `all=True`.
2. Join those keys against recovered `raw.csv`.
3. Explicitly exclude rows where `available=False` and record the excluded count.
4. Exclude panorama images using the recovered `pano` flag.
5. Use official `postprocessed.csv` UTM coordinates if present; otherwise project recovered
   WGS84 longitude/latitude into the city's expected UTM zone.
6. Build a brute-force 25 m neighbourhood inside that city only.
7. Append the city's database to the global gallery with an index offset.
8. Drop queries that have no surviving positive after unavailable/panorama filtering.

Expected UTM zones are hardcoded to avoid an accidental median/zone-boundary change:

```text
Miami 17N, Athens 34N, Buenos Aires 21S,
Stockholm 33N, Bengaluru 43N, Kampala 36N
```

The WGS84-to-UTM implementation was checked against official MSLS validation metadata:

- Copenhagen zone 32: maximum coordinate error 0.000612 m over 5,686 exact pairs; zero
  query-positive sets changed at 25 m.
- San Francisco zone 10: maximum coordinate error 0.000104 m over 2,269 exact pairs; zero
  query-positive sets changed at 25 m.

The implementation does not yet know the final post-filter database/query/scorable counts,
because those can only be audited after the remaining public metadata is recovered. Do not
invent expected final counts from the 39,940 pre-filter total.

### MSLS tests

`tests/test_image_sets.py` covers:

- Pitts numeric ordering and ground-truth orientation;
- ground-truth-aware limited subsets;
- MSLS panorama filtering and city isolation;
- raw GPS fallback when `postprocessed.csv` is absent;
- explicit unavailable-row exclusion and accounting;
- WGS84-to-UTM agreement with an official Copenhagen point;
- official-style MSLS mAP calculation.

## 6. Exact next actions for MSLS

### Step 1: obtain and store a private Mapillary client token

This is the only external input still needed. Ask the user to create a Mapillary developer
client token if one is not already available. Store it only here:

```text
/tmp/mapillary_access_token
```

Set permissions to 600. Do not commit the token, print it, include it in a command-line
argument, or write it under the repository or `/media` tree.

### Step 2: resume recovery

Run from `/home/adam/repo/megaevent`:

```bash
CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python -u \
  scripts/recover_msls_test_metadata.py \
  --msls-root /media/adam/vprdatasets/megaevent/msls \
  --access-token-file /tmp/mapillary_access_token
```

This needs network access and writes under `/media`, so an execution sandbox may require
approval. The script should clean the 25,418 false nulls, reuse the 40,000 valid cache
records, fetch the remaining selected records, and write twelve `raw.csv` files.

Expected terminal summary before panorama filtering is approximately:

```text
recovered 39576/39940 MSLS test images; 364 unavailable records are marked and excluded
```

If the unavailable count differs, stop and inspect the API response/cache state. Do not
silently accept a large increase in missing records.

### Step 3: audit the loaded protocol before any GPU run

At minimum verify:

- exactly twelve `raw.csv` files exist;
- every selected key has a raw metadata row;
- excluded counts are 358 database and 6 query before panorama filtering;
- every retained event path exists;
- every retained coordinate is finite and falls in the expected city/UTM range;
- positives are constructed within cities only;
- the final number of scorable queries and positives-per-query distribution are plausible;
- no query with an empty positive set reaches scoring.

A direct loader audit can start with:

```bash
CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python -c \
  "from src.imagesets import load_msls; d=load_msls('/media/adam/vprdatasets/megaevent/msls',25.0); print('db',len(d.db_paths),'q',len(d.q_paths),'excluded_db',d.excluded_db,'excluded_q',d.excluded_q,'positive_range',int(d.gt.sum(0).min()),int(d.gt.sum(0).max()))"
```

Also inspect per-city counts rather than relying only on the aggregate. The final report must
state that this is the **publicly recoverable MSLS test subset**, not the complete official
hidden test set, because 364 selected Mapillary records are no longer available.

### Step 4: run four CUDA smoke tests

Use `--limit 32` for each method before committing hours and disk to full extraction. A
ground-truth-aware limit is already implemented, and limit artifacts/caches receive separate
names. A clean smoke output directory under `/tmp` is reasonable.

MegaEvent ViT-S:

```bash
CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python -u main.py \
  --dataset msls --method megaevent \
  --ckpt /media/adam/vprdatasets/megaevent/runs/s_salad_ft4_v2_vpr/step10000.pt \
  --data-dir /media/adam/vprdatasets/megaevent \
  --feature-dir /tmp/megaevent_msls_smoke \
  --positive-dist-threshold 25 --limit 32
```

MegaEvent ViT-B:

```bash
CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python -u main.py \
  --dataset msls --method megaevent \
  --ckpt /media/adam/vprdatasets/megaevent/runs/b_full_P64_v4_vpr/step10000.pt \
  --data-dir /media/adam/vprdatasets/megaevent \
  --feature-dir /tmp/megaevent_msls_smoke \
  --positive-dist-threshold 25 --limit 32
```

EventGeM:

```bash
CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python -u main.py \
  --dataset msls --method eventgem \
  --data-dir /media/adam/vprdatasets/megaevent \
  --feature-dir /tmp/megaevent_msls_smoke \
  --eventgem-repo /home/adam/repo/megaevent/external/eventgem \
  --positive-dist-threshold 25 --limit 32
```

EventVLAD:

```bash
CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python -u main.py \
  --dataset msls --method eventvlad \
  --data-dir /media/adam/vprdatasets/megaevent \
  --feature-dir /tmp/megaevent_msls_smoke \
  --eventlab-repo /home/adam/repo/Event-LAB \
  --positive-dist-threshold 25 --limit 32
```

Success means all four complete descriptor extraction, native/PCA scoring, JSON writing,
prediction writing, and figure generation without path alignment, OOM, or empty-GT errors.
Smoke-test recall values are not comparable to full results.

### Step 5: run the full evaluations

Use this output root:

```text
/media/adam/vprdatasets/megaevent/evaluations
```

MegaEvent ViT-S:

```bash
CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python -u main.py \
  --dataset msls --method megaevent \
  --ckpt /media/adam/vprdatasets/megaevent/runs/s_salad_ft4_v2_vpr/step10000.pt \
  --data-dir /media/adam/vprdatasets/megaevent \
  --feature-dir /media/adam/vprdatasets/megaevent/evaluations \
  --positive-dist-threshold 25
```

MegaEvent ViT-B:

```bash
CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python -u main.py \
  --dataset msls --method megaevent \
  --ckpt /media/adam/vprdatasets/megaevent/runs/b_full_P64_v4_vpr/step10000.pt \
  --data-dir /media/adam/vprdatasets/megaevent \
  --feature-dir /media/adam/vprdatasets/megaevent/evaluations \
  --positive-dist-threshold 25
```

EventGeM:

```bash
CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python -u main.py \
  --dataset msls --method eventgem \
  --data-dir /media/adam/vprdatasets/megaevent \
  --feature-dir /media/adam/vprdatasets/megaevent/evaluations \
  --eventgem-repo /home/adam/repo/megaevent/external/eventgem \
  --positive-dist-threshold 25
```

EventVLAD:

```bash
CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python -u main.py \
  --dataset msls --method eventvlad \
  --data-dir /media/adam/vprdatasets/megaevent \
  --feature-dir /media/adam/vprdatasets/megaevent/evaluations \
  --eventlab-repo /home/adam/repo/Event-LAB \
  --positive-dist-threshold 25
```

Keep all methods in the same `evaluations/msls` artifact directory: method/checkpoint tags
separate their banks and outputs. EventGeM local caches may be very large, so check free disk
space before its full run.

For every completed method, report:

- final database/query/scorable counts;
- explicitly excluded database/query counts;
- panorama and no-positive effects if separately audited;
- native and PCA `R@1/5/10/20`;
- EventGeM reranked twins and its full-whitening `pca1` variants;
- MSLS mAP at each cutoff;
- exact checkpoint SHA for MegaEvent;
- artifact paths;
- the “publicly recoverable subset” caveat.

## 7. Verification already completed

Latest unit test command:

```bash
CONDA_OVERRIDE_CUDA=12 MPLCONFIGDIR=/tmp pixi run python -m unittest discover -s tests
```

Result: 12 tests passed.

The following also passed after the implementation changes:

```bash
python -m compileall ...
git diff --check
```

All four Pitts `--limit` CUDA smoke tests passed before the full Pitts runs. The UTM
validation results are recorded in the MSLS section above.

## 8. Current worktree and files to preserve

Current `git status --short` before adding this handover showed:

```text
 m external/eventlab
 M main.py
 M scripts/brisbane_pooled.py
 M scripts/eventgem_pooled.py
 M src/eventgemlocal.py
 M src/imagevpr.py
 M src/inference.py
 M src/methods.py
 M src/scoring.py
?? scripts/eventvlad_pooled.py
?? scripts/prepare_nyc_event.py
?? scripts/recover_msls_test_metadata.py
?? src/imagesets.py
?? tests/
```

`CLAUDE-CONTINUE.md` is now an additional untracked handover file. Do not run `git clean`,
`git reset --hard`, or restore these paths. In particular, `external/eventlab` is an existing
modified submodule/worktree state and should not be normalized as collateral cleanup.

Primary files by responsibility:

| File | Responsibility |
|---|---|
| `scripts/prepare_nyc_event.py` | Raw NYC EVT3 to the authors-style 1 Hz sparse-event subset |
| `tests/test_nyc_event.py` | NYC preparation and representation tests |
| `src/npzdata.py` | Sparse event arrays to countmask/count/MCTS/EventVLAD inputs |
| `src/imagesets.py` | Generic, Pitts, and MSLS dataset/ground-truth adapters |
| `tests/test_image_sets.py` | Pitts/MSLS/UTM/mAP tests |
| `scripts/recover_msls_test_metadata.py` | Resumable current-Mapillary metadata recovery |
| `main.py` | Dataset routing, split defaults, explicit checkpoint CLI |
| `src/imagevpr.py` | Shared independent-image evaluation and MSLS prediction artifacts |
| `src/methods.py` | MegaEvent fingerprinting and comparison method input/model paths |
| `src/scoring.py` | Shared recall/PCA/reranking scoring and MSLS mAP |
| `src/inference.py` | Small-gallery Recall@K clamp |
| `scripts/eventvlad_pooled.py` | Author-style temporal EventVLAD on Brisbane/NSAVP |
| `scripts/eventgem_pooled.py` | Pooled EventGeM global plus local reranking |
| `scripts/brisbane_pooled.py` | Shared pooled scorer and non-colliding figure tags |
| `src/eventgemlocal.py` | Reusable local-feature stores and shortlist reranking |

## 9. Important failure modes to avoid

- Do not ask the user to download `metadata.zip` again; it has already been hash-verified as
  the official archive.
- Do not treat `test_countmask` files as rendered countmask arrays. Read their sparse events
  through `src.npzdata`.
- Do not reuse the public website token that failed with OAuth code 368.
- Do not accept the current 25,418 `images.json` null values as genuine deletions. The fixed
  recovery script must remove and retry them.
- Do not write partial `raw.csv` files after an API failure. `_write_raw` should only be
  reached after every recoverable selected image has valid metadata.
- Do not interpolate deleted MSLS coordinates.
- Do not build MSLS positives across cities.
- Do not count queries with no surviving positive in recall or mAP.
- Do not compare `--limit` smoke recall to full-run recall.
- Do not run ViT-S and ViT-B under a shared generic model tag; always pass the explicit
  checkpoint so hashes separate their caches.
- Do not delete EventGeM caches unless the user explicitly accepts the recomputation cost.
- Do not clean unrelated dirty-worktree changes.

## 10. Definition of done

The requested Pitts/MSLS task is genuinely complete when:

1. a private token has finished the resumable MSLS metadata recovery;
2. the twelve recovered `raw.csv` files and all retained event paths have passed the audit;
3. the exact excluded/deleted and post-filter counts have been recorded;
4. ViT-S, ViT-B, EventGeM, and EventVLAD smoke tests have all passed on CUDA via Pixi;
5. all four full MSLS evaluations have completed and their JSON/prediction/figure artifacts
   exist under `/media/adam/vprdatasets/megaevent/evaluations/msls`;
6. results have been reported with recall, mAP, checkpoint identity, and the explicit caveat
   that this is the publicly recoverable test subset rather than the full hidden official
   test set.
