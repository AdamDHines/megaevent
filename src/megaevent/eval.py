"""Reference/query workflow shared by the CLI and Python API."""

import csv
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoints import load_model, resolve_model
from .recall import load_positives, recall
from .events import EventDataset, StreamOptions
from .retrieval import topk
from .runtime import atomic_write, device_for, write_json


@dataclass
class RetrievalResult:
    indices: np.ndarray
    scores: np.ndarray
    reference_samples: list
    query_samples: list
    metrics: dict | None


@torch.inference_mode()
def extract(
    model,
    dataset,
    destination,
    device,
    batch_size=16,
    workers=0,
):
    def write(path):
        bank = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(len(dataset), model.config.desc_dim),
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )

        position = 0

        for batch in tqdm(loader, desc="Extracting descriptors"):
            descriptors = model(batch.to(device)).float().cpu().numpy()

            bank[position : position + len(descriptors)] = descriptors
            position += len(descriptors)

        bank.flush()

    atomic_write(destination, write)

    return np.load(destination, mmap_mode="r")


def descriptor_bank(
    model,
    dataset,
    descriptor_dir,
    device,
    batch_size=16,
    workers=0,
):
    destination = Path(descriptor_dir) / "descriptors.npy"

    if destination.exists():
        return np.load(destination, mmap_mode="r")

    # determine number of workers if workers == 0
    if workers == 0:
        workers = min(32, (os.cpu_count() or 1) + 4)

    return extract(
        model,
        dataset,
        destination,
        device,
        batch_size,
        workers,
    )


def save_previews(root, reference, query, indices, scores, count):
    root.mkdir(exist_ok=True)
    width, height = 224, 190
    for i in range(min(count, len(query))):
        cells = [(query.frame(i), "query " + query.samples[i]["id"])]
        cells.extend(
            (reference.frame(int(j)), f"{rank + 1}: {score:.4f}  " + reference.samples[j]["id"])
            for rank, (j, score) in enumerate(zip(indices[i], scores[i]))
        )
        canvas = Image.new("RGB", (width * len(cells), height), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (frame, label) in enumerate(cells):
            image = Image.fromarray(frame.transpose(1, 2, 0))
            image.thumbnail((width, height - 30))
            canvas.paste(image, (column * width, 0))
            draw.text((column * width + 4, height - 25), label[:30], fill="black")
        canvas.save(root / f"{query.samples[i]['id']}.jpg")


def retrieve(args):
    """Return top-k cosine matches, optionally write results and evaluate positives.

    Event files are continuous streams; directories contain one event sample per file.
    StreamOptions controls EventCV input interpretation independently for each side.
    """

    # set up the retrieval parameters and model
    started = time.monotonic()
    target = device_for(args.device)
    path = resolve_model(args.model, args.ckpt_dir)
    network, cfg = load_model(path, target)

    # define the reference and query datasets
    options = StreamOptions(window_ms=args.window_ms, hot_pixel_filter=args.hot_pixel_filter)
    ref = EventDataset(args.reference, cfg, options)
    qry = EventDataset(args.query, cfg, options)

    # load ground truth, if provided
    positives = (
        load_positives(args.ground_truth, ref.samples, qry.samples, args.gt_layout)
        if args.ground_truth
        else None
    )

    # eyeball mode: retrieve a random handful of the EventCV query slices, not all of them
    if args.save_previews:
        chosen = np.random.default_rng().choice(
            len(qry.samples), min(args.save_previews, len(qry.samples)), replace=False
        )
        chosen = sorted(int(i) for i in chosen)
        qry.samples = [qry.samples[i] for i in chosen]
        if positives is not None:
            positives = [positives[i] for i in chosen]

    # run the descriptor generators
    bank = Path(args.descriptor_dir) / args.model / f"{args.window_ms:g}ms"
    db = descriptor_bank(network, ref, bank / ref.path.stem, target, args.batch_size, args.workers)
    if args.save_previews:
        # the sampled slices differ every run, so never cache or reuse this bank
        queries = extract(
            network,
            qry,
            bank / qry.path.stem / "sampled.npy",
            target,
            args.batch_size,
            args.workers,
        )
    else:
        queries = descriptor_bank(
            network, qry, bank / qry.path.stem, target, args.batch_size, args.workers
        )

    # score the descriptors
    ranked, scores = topk(
        db,
        queries,
        max(args.top_k, 20 if positives is not None else args.top_k),
        target,
        args.query_chunk,
        args.reference_chunk,
    )

    metrics = recall(ranked, positives) if positives is not None else None
    result = RetrievalResult(
        ranked[:, :args.top_k], scores[:, :args.top_k], ref.samples, qry.samples, metrics
    )
    if args.output:
        destination = Path(args.output)
        destination.mkdir(parents=True, exist_ok=True)

        def write_csv(temporary):
            with open(temporary, "w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["query_id", "reference_id", "rank", "score"])
                for i, row in enumerate(result.indices):
                    for rank, j in enumerate(row):
                        writer.writerow(
                            [
                                qry.samples[i]["id"],
                                ref.samples[j]["id"],
                                rank + 1,
                                float(result.scores[i, rank]),
                            ]
                        )

        atomic_write(destination / "retrievals.csv", write_csv)
        write_json(destination / "samples.json", {"reference": ref.samples, "query": qry.samples})
        write_json(
            destination / "run.json",
            {
                "checkpoint": str(path),
                "model": args.model,
                "backbone": cfg.vit,
                "device": str(target),
                "seconds": time.monotonic() - started,
                "reference_options": asdict(ref.options),
                "query_options": asdict(qry.options),
                "resolution": [cfg.eval_img_size or cfg.H, cfg.eval_img_size or cfg.W],
                "top_k": args.top_k,
                "ground_truth": str(args.ground_truth) if args.ground_truth else None,
                "gt_layout": args.gt_layout,
                "representation": cfg.representation,
            },
        )
        if metrics is not None:
            write_json(destination / "metrics.json", metrics)
        else:
            (destination / "metrics.json").unlink(missing_ok=True)
        if args.save_previews:
            save_previews(
                destination / "previews",
                ref,
                qry,
                result.indices,
                result.scores,
                args.save_previews,
            )
    return result
