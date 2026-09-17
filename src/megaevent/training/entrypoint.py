"""Portable configuration and entry point for the original v8 trainer."""

import json
from pathlib import Path

import yaml

from ..config import VPRConfig
from ..runtime import device_for, write_json
from .dataset import PlacesDataset, build_transforms, make_places_loader
from .streams import build_streams, resolve_stats
from .train import VPRTrainer, _seed_everything, _smoke


def read_config(path):
    values = yaml.safe_load(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("Training configuration must be a YAML mapping")
    cfg = VPRConfig()
    unknown = set(values) - set(vars(cfg))
    if unknown:
        raise ValueError(f"Unknown training configuration fields: {sorted(unknown)}")
    cfg.apply_vit(values.get("vit", "small"))
    for key, value in values.items():
        setattr(cfg, key, value)
    cfg.finalize()
    if cfg.eval_pca or cfg.select_on != "native":
        raise ValueError("The v8 release supports native cosine evaluation without PCA")
    if cfg.H <= 0 or cfg.W <= 0 or cfg.H % cfg.P or cfg.W % cfg.P:
        raise ValueError("Training dimensions must be positive multiples of patch size")
    if cfg.steps <= 0 or cfg.P_places < 2 or cfg.K_images < 2:
        raise ValueError("Training needs positive steps and at least two places/images per batch")
    # Held-out monitor cities must never enter the training index.
    cfg.msls_exclude_cities = sorted(set(cfg.msls_exclude_cities) | set(cfg.msls_val_cities))
    return cfg


def run(args):
    if args.smoke:
        import torch

        torch.set_num_threads(min(4, torch.get_num_threads()))
        _smoke(
            n_streams=2,
            load_pretrained=False,
            H=28,
            W=28,
            backbone_img_size=28,
            n_trainable_blocks=1,
            P_places=2,
            K_images=2,
            steps=2,
            representation="accumulate",
            aggregator="salad",
            salad_clusters=2,
            salad_cluster_dim=4,
            salad_token_dim=8,
            salad_mlp_dim=16,
            salad_proj=True,
            salad_proj_dim=12,
            device=str(device_for(args.device)),
        )
        return
    if not args.config:
        raise ValueError("Training requires --config, or use --smoke")
    cfg = read_config(args.config)
    overrides = {
        "data_root": args.data,
        "ckpt_path": args.encoder_checkpoint,
        "megaloc_weights": args.megaloc_weights,
        "msls_meta_root": args.msls_meta,
        "eval_root": args.eval_root,
        "eval_pooled_coords": args.eval_coordinates,
        "resume": args.resume,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    cfg.device, cfg.n_workers = str(device_for(args.device)), args.workers
    if cfg.eval_root and not cfg.eval_pooled:
        raise ValueError("Training validation requires eval_pooled: true (the v8 protocol)")
    cfg.output_dir = str(Path(args.output).resolve())
    cfg.tensorboard, cfg.wandb = args.tensorboard, args.wandb
    if cfg.n_workers < 0:
        raise ValueError("workers must be nonnegative")
    resume_path = (
        (Path(cfg.output_dir) / "latest.pt")
        if cfg.resume == "auto"
        else (Path(cfg.resume) if cfg.resume else None)
    )
    if resume_path is not None and resume_path.is_file():
        cfg.load_pretrained = False
        cfg.salad_init = None
    if cfg.load_pretrained and not Path(cfg.ckpt_path).is_file():
        raise FileNotFoundError("Provide --encoder-checkpoint with the GEPT encoder weights")
    if cfg.salad_init == "megaloc" and not (
        cfg.megaloc_weights and Path(cfg.megaloc_weights).is_file()
    ):
        raise FileNotFoundError("v8 base initialization needs --megaloc-weights model.safetensors")
    if cfg.eval_pooled and (not cfg.eval_root or not Path(cfg.eval_pooled_coords).is_file()):
        raise FileNotFoundError(
            "v8 selection needs --eval-root and --eval-coordinates (pooled geometry)"
        )
    if "msls" in cfg.streams and not cfg.msls_meta_root and not args.place_index:
        raise ValueError("The MSLS stream requires --msls-meta")
    _seed_everything(cfg.seed)
    if args.place_index:
        source = Path(args.place_index).resolve()
        entries = json.loads(source.read_text())
        index = [
            (int(pid), [str((source.parent / p).resolve()) for p in paths])
            for pid, paths in entries
        ]
        if len({pid for pid, _ in index}) != len(index) or any(
            len(ps) < cfg.K_images for _, ps in index
        ):
            raise ValueError("Place IDs must be unique and every place needs K_images samples")
        if any(not Path(p).is_file() for _, ps in index for p in ps):
            raise FileNotFoundError("A place-index image is missing")
        if cfg.compute_stats:
            resolve_stats(cfg, {"custom": index})
        dataset = PlacesDataset(index, cfg.K_images, build_transforms(cfg, train=True))
        streams = {"custom": make_places_loader(dataset, cfg.P_places, cfg.n_workers)}
        cfg.streams = ["custom"]
    else:
        streams = build_streams(cfg)
    if not streams:
        raise ValueError("No training streams available")
    if any(len(loader) == 0 for loader in streams.values()):
        raise ValueError("A training stream has fewer than P_places places per batch")
    write_json(Path(cfg.output_dir) / "config.json", vars(cfg))
    trainer = VPRTrainer(cfg, streams)
    try:
        trainer.start()
    finally:
        if trainer.writer:
            trainer.writer.close()
