"""EventCV sources and checkpoint-specific preprocessing."""

from dataclasses import asdict, dataclass
from pathlib import Path

import eventcv as ecv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .representations import accumulate_numpy


@dataclass
class StreamOptions:
    window_ms: float = 50.0
    sensor_size: tuple[int, int] | None = None
    time_unit: str | None = None
    offset_ms: float | None = None
    order: str = "txyp"
    topic: str | None = None
    keys: dict | None = None
    hot_pixel_filter: bool = False

    def kwargs(self):
        if not np.isfinite(self.window_ms) or self.window_ms <= 0:
            raise ValueError("window_ms must be positive and finite")
        if self.sensor_size and (len(self.sensor_size) != 2 or min(self.sensor_size) <= 0):
            raise ValueError("sensor_size must be positive (width, height)")
        options = asdict(self)
        options["dt_ms"] = options.pop("window_ms")
        return {k: v for k, v in options.items() if v is not None}


def eval_transform(cfg):
    size = int(getattr(cfg, "eval_img_size", None) or cfg.H)
    shape = (size, size) if getattr(cfg, "eval_img_size", None) else (cfg.H, cfg.W)
    return transforms.Compose(
        [
            transforms.Normalize(cfg.tencode_mean, cfg.tencode_std),
            transforms.Resize(shape, interpolation=transforms.InterpolationMode.BICUBIC),
        ]
    )


def render_stream(stream, representation, window_ms=50):
    if representation == "accumulate":
        # GEPT's winner-take-all white-background transform is not EventCV countmask.
        events = ecv.numpy(stream)
        width, height = stream.sensor_size
        return accumulate_numpy(
            events[:, 0], events[:, 1], events[:, 2], events[:, 3], height, width
        )
    kwargs = {"window_ms": window_ms, "white_frame": False}
    if representation not in {"countmask", "tencode"}:
        raise ValueError(f"Unsupported checkpoint representation: {representation}")
    return np.asarray(getattr(stream, representation)(**kwargs).numpy())


class EventDataset(Dataset):
    def __init__(self, source, cfg, options=None):
        self.path = Path(source).expanduser().resolve()
        self.options = options or StreamOptions()
        self.options.kwargs()
        self.representation = cfg.representation
        self.transform = eval_transform(cfg)
        self._reader = None
        self.folder = self.path.is_dir()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.paths = (
            sorted(p for p in self.path.iterdir() if p.is_file() and not p.name.startswith("."))
            if self.folder
            else [self.path]
        )
        if not self.paths:
            raise ValueError(f"No event files in {self.path}")
        self.samples = []
        if self.folder:
            for path in self.paths:
                reader = ecv.open(str(path), **self.options.kwargs())
                lo, hi = reader.time_span_ms
                self.samples.append(
                    dict(id=path.name, path=str(path), slice=None, start_ms=lo, end_ms=hi)
                )
        else:
            reader = self.reader
            lo, hi = reader.time_span_ms
            origin = max(lo, self.options.offset_ms if self.options.offset_ms is not None else lo)
            for i in range(reader.n_slices):
                self.samples.append(
                    dict(
                        id=f"{i:08d}",
                        path=str(self.path),
                        slice=i,
                        start_ms=origin + i * self.options.window_ms,
                        end_ms=min(hi, origin + (i + 1) * self.options.window_ms),
                    )
                )
        if not self.samples:
            raise ValueError(f"No event windows in {self.path}; check offsets and time units")

    @property
    def reader(self):
        if self._reader is None:
            self._reader = ecv.open(str(self.path), **self.options.kwargs())
        return self._reader

    def frame(self, index):
        if self.folder:
            kwargs = self.options.kwargs()
            kwargs.pop("dt_ms")
            hot = kwargs.pop("hot_pixel_filter")
            stream = ecv.load(str(self.paths[index]), **kwargs)
            if hot:
                stream = stream.hot_pixel_filter()
        else:
            stream = self.reader.slice(index)
        return render_stream(stream, self.representation, self.options.window_ms)

    def __getitem__(self, index):
        frame = self.frame(index)
        tensor = torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255)
        return self.transform(tensor)

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        return {**self.__dict__, "_reader": None}
