import argparse
import json
import os

from loguru import logger
from src.megaevent.logs import configure
from src.megaevent.eval import retrieve
from src.megaevent.checkpoints import resolve_model


def parser():
    args = argparse.ArgumentParser(description="Args for megaevent running")

    # megaevent mode to run
    args.add_argument("--mode", "-m", choices=["eval", "train"], default="eval",
                       help="Mode to run megaevent")

    # reference/query retrieval args
    args.add_argument("--reference", required=True, type=str,
                      help="Reference event file or directory")
    args.add_argument("--query", required=True, type=str,
                       help="Query event file or directory")
    args.add_argument("--window-ms", type=float, default=50,
                      help="Window size in milliseconds for event slicing")
    args.add_argument("--hot-pixel-filter", action="store_true",
                      help="Filter hot pixels from event streams using eventcv (recommended)")
    args.add_argument("--model", type=str, choices=["megaevent_vits14", "megaevent_vitb14"], default="megaevent_vits14", 
                          help="megaevent model to use for evaluation")

    # directory args
    args.add_argument("--ckpt-dir", type=str, default="./src/ckpts",
                          help = "Directory where checkpoints are stored")
    args.add_argument("--descriptor-dir", type=str, default="./src/descriptors",
                          help = "Directory where descriptor banks are stored")

    # recall arguments
    args.add_argument("--ground-truth", type=str, default=None,
                       help="Optional .npy matrix or query_id,reference_id CSV")
    args.add_argument("--gt-layout", type=str, choices=["reference-query", "query-reference"], default="reference-query",
                       help="Ground truth layout for .npy matrix or CSV")
    args.add_argument("--top-k", type=int, default=5,
                       help="Number of top results to return")

    # output and results args
    args.add_argument("--output", type=str,help="Default: results/<reference>_<query>",
                      default=None)
    args.add_argument("--save-previews", type=int, default=0,
                      help="Randomly sample N query slices and save their top-k retrievals as "
                           "images for visual inspection; 0 runs the full query file")

    # device args
    args.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                      help="Device to run the model on (recommended: auto)")
    args.add_argument("--batch-size", type=int, default=32,
                      help="Batch size for descriptor generation")
    args.add_argument("--workers", type=int, default=0,
                      help="Number of workers for data loading, 0 for auto-detection: recommended 0")
    args.add_argument("--query-chunk", type=int, default=4096,
                      help="Chunk size for query descriptor generation")
    args.add_argument("--reference-chunk", type=int, default=4096,
                      help="Chunk size for reference descriptor generation")

    # models = commands.add_parser("models").add_subparsers(dest="action", required=True)
    # models.add_parser("list")
    # download = models.add_parser("download")
    # download.add_argument("name", choices=MODELS)
    # download.add_argument("--offline", action="store_true")
    # export = commands.add_parser("export", help="Export a trusted local training checkpoint")
    # export.add_argument("source")
    # export.add_argument("destination")
    # train = commands.add_parser("train")
    # train.add_argument("--config")
    # train.add_argument("--data")
    # train.add_argument("--output", default="runs/example")
    # train.add_argument("--encoder-checkpoint")
    # train.add_argument("--megaloc-weights")
    # train.add_argument("--msls-meta")
    # train.add_argument("--eval-root")
    # train.add_argument("--eval-coordinates")
    # train.add_argument("--place-index", help="Custom JSON list of [place_id, [image paths]]")
    # train.add_argument("--resume", help="Trusted local training checkpoint")
    # train.add_argument("--device", default="auto")
    # train.add_argument("--workers", type=int, default=0)
    # train.add_argument("--smoke", action="store_true")
    # train.add_argument("--tensorboard", action="store_true")
    # train.add_argument("--wandb", action="store_true")
    return args


def main():
    # parse args
    root = parser()
    args = root.parse_args()

    if args.mode == "eval":

        reference = os.path.splitext(os.path.basename(args.reference))[0]
        query = os.path.splitext(os.path.basename(args.query))[0]
        if args.output is None:
            args.output = os.path.join("results", f"{reference}_{query}")
        # console + logs/retrieve/<time>_<reference>_<query>.log
        logfile = configure("retrieve", f"{reference}_{query}")
        logger.info(f"Reference {args.reference}, query {args.query}, log {logfile}")

        # check existence ofthe model and download it
        if args.model == "megaevent_vitb14" or args.model == "megaevent_vits14":
            logger.info(f"Looking for {args.model} in {args.ckpt_dir}")
            args.checkpoint = resolve_model(args.model, args.ckpt_dir)
        else:
            raise ValueError(
                f"{args.model} is not a recognised model, please use either "
                "megaevent_vitb14 or megaevent_vits14"
            )

        # run the retrieval
        result = retrieve(args)

        logger.info(
            f"Matched {len(result.query_samples)} query windows against "
            f"{len(result.reference_samples)} reference windows "
        )
        logger.info(
            f"Top-1 cosine score mean {result.scores[:, 0].mean():.4f}, "
            f"min {result.scores[:, 0].min():.4f}"
        )
        if result.metrics:
            logger.info(f"Metrics {json.dumps(result.metrics)}")
            print(json.dumps(result.metrics))
        else:
            print(f"Retrievals saved to {args.output}")

    elif args.mode == "train":   
        # TODO; full implementation
        # from megaevent.training.entrypoint import run

        # configure("train", os.path.basename(args.output))
        # run(args)
        raise NotImplementedError("Training mode is not yet implemented. Please use eval mode.")
    else:
        raise ValueError(f"Unknown mode {args.mode!r}")

if __name__ == "__main__":
    main()
