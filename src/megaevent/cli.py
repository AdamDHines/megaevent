"""Small command-line surface for retrieval, training and checkpoint management."""

import argparse
import json


def parser():
    root = argparse.ArgumentParser(prog="megaevent")
    commands = root.add_subparsers(dest="command", required=True)
    retrieve = commands.add_parser("retrieve", help="Match reference and query event streams")
    for side in ("reference", "query"):
        retrieve.add_argument(f"--{side}", required=True)
        retrieve.add_argument(
            f"--{side}-sensor-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT")
        )
        retrieve.add_argument(f"--{side}-time-unit", choices=["s", "ms", "us", "ns"])
        retrieve.add_argument(f"--{side}-offset-ms", type=float)
        retrieve.add_argument(f"--{side}-topic")
        retrieve.add_argument(f"--{side}-order", choices=["txyp", "xytp"], default="txyp")
        retrieve.add_argument(f"--{side}-keys", type=json.loads, help="EventCV key mapping as JSON")
    retrieve.add_argument("--window-ms", type=float, default=50)
    retrieve.add_argument("--hot-pixel-filter", action="store_true")
    retrieve.add_argument(
        "--model", choices=["megaevent_vits14", "megaevent_vitb14"], default="megaevent_vits14"
    )
    retrieve.add_argument("--checkpoint")
    retrieve.add_argument("--offline", action="store_true")
    retrieve.add_argument("--ground-truth")
    retrieve.add_argument(
        "--gt-layout", choices=["reference-query", "query-reference"], default="reference-query"
    )
    retrieve.add_argument("--top-k", type=int, default=5)
    retrieve.add_argument("--output", default="results")
    retrieve.add_argument("--device", default="auto")
    retrieve.add_argument("--resolution", type=int)
    retrieve.add_argument("--batch-size", type=int, default=16)
    retrieve.add_argument("--workers", type=int, default=0)
    retrieve.add_argument("--cache-dir")
    retrieve.add_argument("--query-chunk", type=int, default=128)
    retrieve.add_argument("--reference-chunk", type=int, default=4096)
    retrieve.add_argument("--save-previews", type=int, default=0)
    models = commands.add_parser("models").add_subparsers(dest="action", required=True)
    models.add_parser("list")
    download = models.add_parser("download")
    download.add_argument("name")
    download.add_argument("--offline", action="store_true")
    export = commands.add_parser("export", help="Export a trusted local training checkpoint")
    export.add_argument("source")
    export.add_argument("destination")
    train = commands.add_parser("train")
    train.add_argument("--config")
    train.add_argument("--data")
    train.add_argument("--output", default="runs/example")
    train.add_argument("--encoder-checkpoint")
    train.add_argument("--megaloc-weights")
    train.add_argument("--msls-meta")
    train.add_argument("--eval-root")
    train.add_argument("--eval-coordinates")
    train.add_argument("--place-index", help="Custom JSON list of [place_id, [image paths]]")
    train.add_argument("--resume", help="Trusted local training checkpoint")
    train.add_argument("--device", default="auto")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--tensorboard", action="store_true")
    train.add_argument("--wandb", action="store_true")
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "retrieve":
            from .eval import retrieve
            from .events import StreamOptions

            options = {}
            for side in ("reference", "query"):
                options[side + "_options"] = StreamOptions(
                    window_ms=args.window_ms,
                    hot_pixel_filter=args.hot_pixel_filter,
                    **{
                        name: getattr(args, f"{side}_{name}")
                        for name in (
                            "sensor_size",
                            "time_unit",
                            "offset_ms",
                            "order",
                            "topic",
                            "keys",
                        )
                    },
                )
            result = retrieve(
                args.reference,
                args.query,
                **options,
                **{
                    key: getattr(args, key)
                    for key in (
                        "model",
                        "checkpoint",
                        "offline",
                        "ground_truth",
                        "gt_layout",
                        "top_k",
                        "output",
                        "device",
                        "resolution",
                        "batch_size",
                        "workers",
                        "cache_dir",
                        "query_chunk",
                        "reference_chunk",
                    )
                },
                save_previews_count=args.save_previews,
            )
            print(
                json.dumps(result.metrics)
                if result.metrics
                else f"Retrievals saved to {args.output}"
            )
        elif args.command == "models":
            from .checkpoints import registry, resolve_model

            if args.action == "list":
                print(json.dumps(registry(), indent=2))
            else:
                print(resolve_model(args.name, offline=args.offline))
        elif args.command == "export":
            from .checkpoints import export_checkpoint

            export_checkpoint(args.source, args.destination)
            print(args.destination)
        else:
            from .training.entrypoint import run

            run(args)
    except (ValueError, FileNotFoundError, OSError) as exc:
        root.exit(2, f"megaevent: {exc}\n")
