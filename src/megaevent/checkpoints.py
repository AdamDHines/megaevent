"""Checkpoint compatibility, verified Hub downloads and portable exports."""

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from .config import VPRConfig
from .model import VPRModel
from .runtime import atomic_write

MODEL_FIELDS = (
    "vit",
    "n_embed",
    "n_head",
    "num_register_tokens",
    "n_layer",
    "P",
    "backbone_img_size",
    "H",
    "W",
    "tencode_mean",
    "tencode_std",
    "aggregator",
    "desc_dim",
    "gem_p_init",
    "gem_eps",
    "representation",
    "salad_clusters",
    "salad_cluster_dim",
    "salad_token_dim",
    "salad_mlp_dim",
    "salad_dropout",
    "salad_proj",
    "salad_proj_dim",
    "salad_out_dim",
    "eval_img_size",
)


def registry():
    return json.loads(files(__package__).joinpath("models.json").read_text())


def resolve_model(name="megaevent_vits14", dir="./src/ckpts"):
    """Return a local checkpoint path, downloading a registry model into src/ckpts if needed."""

    dir = Path(dir)
    entry = registry().get(name)
    if entry is None:
        raise ValueError(f"Unknown model {name!r}; run 'megaevent models list'")
    if not all(entry.get(k) for k in ("repo_id", "filename", "revision")):
        raise ValueError(f"{name} has not been published yet. Use --checkpoint PATH.")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    if not (dir / entry["filename"]).is_file():
        logger.info(f"Downloading {name} from {entry['repo_id']} into {dir}")
    try:
        path = Path(
            hf_hub_download(
                repo_id=entry["repo_id"],
                filename=entry["filename"],
                revision=entry["revision"],
                local_dir=str(dir),
            )
        )
    except LocalEntryNotFoundError as exc:
        raise ValueError(
            f"{name} is not cached in {dir}. Download online first or use --checkpoint."
        ) from exc
    except HfHubHTTPError as exc:
        raise ValueError(
            f"Cannot download {name} from {entry['repo_id']}: {exc}. "
            "Check your connection or use --checkpoint PATH."
        ) from exc
    logger.info(f"Checkpoint {path}")
    return path


def config_from_checkpoint(checkpoint):
    saved = checkpoint.get("config")
    if not isinstance(saved, dict) or not all(
        k in saved for k in ("vit", "aggregator", "tencode_mean", "tencode_std")
    ):
        raise ValueError("Checkpoint needs architecture and normalization config; export it first")
    cfg = VPRConfig().apply_vit(saved["vit"])
    for key in MODEL_FIELDS:
        if key in saved:
            setattr(cfg, key, saved[key])
    cfg.salad_out_dim = cfg.salad_clusters * cfg.salad_cluster_dim + cfg.salad_token_dim
    cfg.load_pretrained = False
    cfg.salad_init = None
    cfg.transfer = "linear"
    cfg.grad_checkpoint = False
    if cfg.H <= 0 or cfg.W <= 0 or cfg.H % cfg.P or cfg.W % cfg.P:
        raise ValueError("Checkpoint resolution must be a positive multiple of patch size")
    if len(cfg.tencode_mean) != 3 or len(cfg.tencode_std) != 3 or min(cfg.tencode_std) <= 0:
        raise ValueError("Invalid checkpoint normalization")
    return cfg


def load_model(path, device="cpu"):
    allow = [np._core.multiarray._reconstruct, np.ndarray, np.dtype]
    allow += [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")]
    with torch.serialization.safe_globals(allow):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    cfg = config_from_checkpoint(checkpoint)

    model = VPRModel(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, cfg


def export_checkpoint(source, destination):
    """Export a trusted local trainer checkpoint and verify before publication."""
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("Export destination must differ from the training checkpoint")
    checkpoint = read_checkpoint(source, trusted=True)
    cfg = config_from_checkpoint(checkpoint)
    model = VPRModel(cfg).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    payload = {
        "format_version": 1,
        "model": checkpoint["model"],
        "config": {k: getattr(cfg, k) for k in MODEL_FIELDS},
        "meta": {
            "step": int(checkpoint.get("step", checkpoint.get("meta", {}).get("step", -1))),
        },
    }

    def write(temporary):
        torch.save(payload, temporary)
        restored, _ = load_model(temporary, resolution=cfg.H)
        with torch.inference_mode(), torch.random.fork_rng():
            torch.manual_seed(0)
            sample = torch.randn(1, 3, cfg.H, cfg.W)
            torch.testing.assert_close(model(sample), restored(sample), rtol=0, atol=0)

    atomic_write(destination, write)
