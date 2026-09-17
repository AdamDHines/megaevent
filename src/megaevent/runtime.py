"""Portable device selection and atomic output helpers."""

import json
import os
import tempfile
from pathlib import Path

import torch


def device_for(name="auto"):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    device = torch.device(name)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {name}")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError(f"CUDA device unavailable: {name}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this machine")
    return device


def atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    def write(temporary):
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")

    atomic_write(path, write)
