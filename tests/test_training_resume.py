import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from megaevent.config import VPRConfig
from megaevent.training.train import VPRTrainer


def test_optimizer_rng_selection_and_export_state(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    cfg = VPRConfig()
    cfg.load_pretrained = False
    cfg.H = cfg.W = cfg.backbone_img_size = 28
    cfg.desc_dim = 8
    cfg.n_trainable_blocks = 1
    cfg.device = "cpu"
    cfg.output_dir = str(tmp_path)
    cfg.warmup_steps = 2
    cfg.steps = 4
    cfg.P_places = cfg.K_images = 2
    cfg.resume = None
    images = torch.randn(4, 3, 28, 28)
    labels = torch.tensor([0, 0, 1, 1])
    streams = {"test": DataLoader(TensorDataset(images, labels), batch_size=4)}
    trainer = VPRTrainer(cfg, streams)
    before = trainer.model.proj.weight.detach().clone()
    frozen = trainer.model.encoder.blocks[0].attn.qkv.weight.detach().clone()
    loss, _, _, _ = trainer.train_step(0)
    assert np.isfinite(loss)
    assert not torch.equal(before, trainer.model.proj.weight)
    torch.testing.assert_close(
        frozen, trainer.model.encoder.blocks[0].attn.qkv.weight, rtol=0, atol=0
    )
    trainer.best["main"] = {"r1": 0.75, "step": 0}
    trainer.save(0)
    expected_rng = (torch.rand(2), np.random.rand(2), random.random())
    restored = VPRTrainer(cfg, streams)
    assert restored.load_checkpoint(tmp_path / "latest.pt") == 0
    torch.testing.assert_close(torch.rand(2), expected_rng[0], rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.rand(2), expected_rng[1])
    assert random.random() == expected_rng[2]
    assert restored.best == trainer.best
    for a, b in zip(trainer.optimizer.state.values(), restored.optimizer.state.values()):
        torch.testing.assert_close(a["exp_avg"], b["exp_avg"], rtol=0, atol=0)
    trainer.model.eval()
    restored.model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(trainer.model(images), restored.model(images), rtol=0, atol=0)
    from megaevent.training import evalsuite

    monkeypatch.setattr(evalsuite, "run_pooled", lambda *args, **kwargs: {"R@1": 0.9})
    trainer.evaluate_suite(1)
    selected = torch.load(tmp_path / "best.pt", weights_only=False)
    assert selected["step"] == 0
    assert selected["best"]["main"]["step"] == 1
    assert selected["best"]["real"]["r1"] == 0.9


def test_training_cli_custom_data_and_resume(tmp_path):
    import yaml
    from PIL import Image

    from megaevent.cli import main

    torch.set_num_threads(2)
    index = []
    for place in range(3):
        images = []
        for sample in range(2):
            path = tmp_path / f"{place}_{sample}.png"
            Image.new("RGB", (28, 28), (40 + place * 60, 90, 200 - place * 30)).save(path)
            images.append(path.name)
        index.append([place, images])
    (tmp_path / "places.json").write_text(json.dumps(index))
    cfg = {
        "load_pretrained": False,
        "H": 28,
        "W": 28,
        "backbone_img_size": 28,
        "desc_dim": 8,
        "n_trainable_blocks": 1,
        "steps": 2,
        "stop_after": 1,
        "P_places": 2,
        "K_images": 2,
        "warmup_steps": 2,
        "save_every": 1,
        "representation": "accumulate",
        "log_every": 1,
    }
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    args = [
        "train",
        "--config",
        str(config_path),
        "--place-index",
        str(tmp_path / "places.json"),
        "--output",
        str(tmp_path / "run"),
        "--device",
        "cpu",
    ]
    main(args)
    first = torch.load(tmp_path / "run/latest.pt", weights_only=False)
    assert first["step"] == 0
    cfg["stop_after"] = 0
    config_path.write_text(yaml.safe_dump(cfg))
    main(args)
    last = torch.load(tmp_path / "run/latest.pt", weights_only=False)
    assert last["step"] == 1
    assert not torch.equal(first["model"]["proj.weight"], last["model"]["proj.weight"])
    assert (tmp_path / "run/metrics.jsonl").is_file()
