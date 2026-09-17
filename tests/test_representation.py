import hashlib

import numpy as np
import torch
from PIL import Image

from megaevent.config import VPRConfig
from megaevent.events import eval_transform
from megaevent.representations import accumulate_numpy
from megaevent.training.dataset import build_transforms


def test_accumulate_matches_submitted_renderer():
    rng = np.random.default_rng(0)
    count = 5000
    x, y = rng.integers(0, 40, count), rng.integers(0, 30, count)
    t = rng.integers(0, 50000, count)
    p = rng.choice([-1, 1], count).astype(np.int8)
    image = accumulate_numpy(x, y, t, p, 30, 40)
    # Captured from megaevent 592ebaf before removing the research implementation.
    assert hashlib.sha256(image.tobytes()).hexdigest() == (
        "f7c8e6558486634e34d470d405ee427b80b4976202cf6122ad8f63147f24f80d"
    )
    empty = np.array([], dtype=np.int64)
    assert (accumulate_numpy(empty, empty, empty, empty, 8, 10) == 255).all()


def test_train_eval_and_inference_transform_agree():
    cfg = VPRConfig()
    cfg.eval_img_size = 28
    pixels = np.random.default_rng(4).integers(0, 256, (20, 32, 3), dtype=np.uint8)
    training_eval = build_transforms(cfg, train=False)(Image.fromarray(pixels))
    inference = eval_transform(cfg)(torch.from_numpy(pixels.transpose(2, 0, 1)).float() / 255)
    torch.testing.assert_close(training_eval, inference, rtol=0, atol=0)
