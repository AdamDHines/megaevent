# MegaEvent - Multi-viewpoint Geo-localization with Event Cameras

This repository contains the evaluation and training code for MegaEvent, our state-of-the-art visual place recognition system for event cameras.

## Install

Install [Pixi](https://pixi.sh), clone this repository, then run:

```sh
pixi install
```

Linux and Windows use CUDA 12.8; Apple Silicon uses MPS when available, otherwise CPU.
While the models are private, authenticate an account with repository access:

```sh
pixi run hf auth login
```

## Retrieve

```sh
pixi run megaevent retrieve --reference reference.h5 --query query.h5 --top-k 5 --output results
```

EventCV reads the event files. A file is sliced into 50 ms windows (`--window-ms`);
a directory supplies one sample per event file. Results include matches and scores
in `retrievals.csv`. Add `--save-previews 10` for retrieval images.

## Evaluate

```sh
pixi run megaevent retrieve --reference reference.h5 --query query.h5 --ground-truth positives.npy --output results
```

Ground truth is a boolean `[reference, query]` matrix aligned to the windows, or
a CSV with `query_id,reference_id` columns. Evaluation adds Recall@1/5/10/20.
See [input options](docs/inputs.md) for timestamps, sensor sizes and alignment.

## Models and Training

`--model megaevent_vits14` (default) or `--model megaevent_vitb14` downloads and
caches weights into `src/ckpts` from [AdamHines/megaevent](https://huggingface.co/AdamHines/megaevent).
Use `--checkpoint model.pt` for local weights, or `--offline` for cached models.

```sh
pixi run -e train megaevent train --smoke --device cpu
```

The original v8 small/base recipes are in `configs/train/`.
See [training](docs/training.md) for data and initialization paths.
