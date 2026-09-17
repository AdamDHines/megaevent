import csv
import json

import eventcv as ecv
import h5py
import numpy as np
import pytest
import torch

from megaevent import checkpoints
from megaevent.eval import retrieve
from megaevent.config import VPRConfig
from megaevent.recall import load_positives, recall
from megaevent.events import EventDataset, StreamOptions
from megaevent.model import VPRModel
from megaevent.retrieval import topk
from megaevent.training.entrypoint import read_config


@pytest.fixture
def event_files(tmp_path):
    events = dict(
        x=np.array([1, 2, 3, 1], dtype=np.uint16),
        y=np.array([1, 2, 3, 2], dtype=np.uint16),
        t=np.array([0, 1000, 11000, 31000], dtype=np.int64),
        p=np.array([1, 0, 1, 0], dtype=np.uint8),
    )
    np.savez(tmp_path / "events.npz", **events, resolution=[8, 8])
    with h5py.File(tmp_path / "events.h5", "w") as file:
        for key, value in events.items():
            file.create_dataset(key, data=value)
    return tmp_path / "events.npz", tmp_path / "events.h5"


def config():
    cfg = VPRConfig()
    cfg.load_pretrained = False
    cfg.representation = "accumulate"
    cfg.H = cfg.W = cfg.backbone_img_size = 28
    cfg.desc_dim = 16
    cfg.eval_img_size = None
    return cfg


def test_eventcv_inputs_and_empty_window(event_files):
    options = StreamOptions(window_ms=10, sensor_size=(8, 8), time_unit="us")
    npz, h5 = (EventDataset(path, config(), options) for path in event_files)
    assert len(npz) == len(h5) == 4
    for i in range(4):
        np.testing.assert_array_equal(npz.frame(i), h5.frame(i))
    assert (npz.frame(2) == 255).all()
    assert npz.samples[3]["start_ms"] == 30
    assert npz[0].shape == (3, 28, 28)
    reader = ecv.open(str(event_files[0]), dt_ms=10, sensor_size=(8, 8), time_unit="us")
    assert reader.n_slices == len(npz)


def test_folder_and_offset(event_files):
    options = StreamOptions(window_ms=10, sensor_size=(8, 8), time_unit="us")
    folder = EventDataset(event_files[0].parent, config(), options)
    assert [s["id"] for s in folder.samples] == ["events.h5", "events.npz"]
    np.testing.assert_array_equal(folder.frame(0), folder.frame(1))
    offset = EventDataset(
        event_files[0],
        config(),
        StreamOptions(window_ms=10, sensor_size=(8, 8), time_unit="us", offset_ms=10),
    )
    assert len(offset) == 3
    assert offset.samples[0]["start_ms"] == 10


def test_topk_chunking_ties_and_small_gallery():
    rng = np.random.default_rng(12)
    ref = rng.normal(size=(27, 10)).astype("float32")
    qry = rng.normal(size=(7, 10)).astype("float32")
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    qry /= np.linalg.norm(qry, axis=1, keepdims=True)
    ref[18] = ref[1]
    ranked, scores = topk(ref, qry, 30, query_chunk=3, reference_chunk=4)
    expected = np.argsort(-(qry @ ref.T), axis=1, kind="stable")
    np.testing.assert_array_equal(ranked, expected)
    np.testing.assert_allclose(scores, np.take_along_axis(qry @ ref.T, expected, axis=1), atol=1e-6)
    tied, _ = topk(np.ones((15, 2), "float32"), np.ones((2, 2), "float32"), 5, reference_chunk=2)
    np.testing.assert_array_equal(tied, np.tile(np.arange(5), (2, 1)))


def test_ground_truth_orientation_and_unscorable(tmp_path):
    refs = [{"id": "a"}, {"id": "b"}]
    queries = [{"id": "q"}, {"id": "none"}, {"id": "r"}]
    matrix = np.array([[1, 0, 0], [0, 0, 1]])
    np.save(tmp_path / "gt.npy", matrix)
    positives = load_positives(tmp_path / "gt.npy", refs, queries)
    assert positives == [{0}, set(), {1}]
    metrics = recall(np.array([[0, 1], [0, 1], [0, 1]]), positives)
    assert metrics["R@1"] == 0.5 and metrics["R@5"] == 1
    assert metrics["scorable_queries"] == 2
    with pytest.raises(ValueError, match="shape"):
        load_positives(tmp_path / "gt.npy", refs, queries, "query-reference")
    with (tmp_path / "gt.csv").open("w") as file:
        file.write("query_id,reference_id\nq,a\nr,b\n")
    assert load_positives(tmp_path / "gt.csv", refs, queries) == positives
    assert recall(np.zeros((3, 2), int), [set(), set(), set()])["R@1"] is None


def test_download_uses_auth_cache_revision_and_local_override(tmp_path, monkeypatch):
    import huggingface_hub

    path = tmp_path / "model.pt"
    path.write_bytes(b"checkpoint")
    entry = dict(
        repo_id="AdamDHines/megaevent",
        filename="megaevent_vits14.pt",
        revision="revision123",
    )
    monkeypatch.setattr(checkpoints, "registry", lambda: {"small": entry})
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert checkpoints.resolve_model("small", offline=True) == path
    assert calls == [
        dict(
            repo_id=entry["repo_id"],
            filename=entry["filename"],
            revision="revision123",
            local_dir=str(checkpoints.CHECKPOINT_DIR),
            local_files_only=True,
        )
    ]
    assert checkpoints.resolve_model("unknown", checkpoint=path) == path
    assert len(calls) == 1


def test_release_workflow_cache_and_metrics(event_files, tmp_path, monkeypatch):
    import megaevent.eval as eval

    cfg = config()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = cfg
            cfg.desc_dim = 3

        def forward(self, x):
            return torch.nn.functional.normalize(x.mean((2, 3)), dim=1)

    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"local")
    monkeypatch.setattr(eval, "load_model", lambda *a: (Model(), cfg))
    options = StreamOptions(window_ms=10, sensor_size=(8, 8), time_unit="us")
    np.save(tmp_path / "gt.npy", np.eye(4, dtype=bool))
    kwargs = dict(
        checkpoint=weights,
        reference_options=options,
        query_options=options,
        device="cpu",
        cache_dir=tmp_path / "cache",
        output=tmp_path / "out",
        ground_truth=tmp_path / "gt.npy",
        top_k=1,
        save_previews_count=1,
    )
    result = retrieve(*event_files, **kwargs)
    assert result.indices.shape == (4, 1)
    assert result.metrics["R@5"] == 1
    assert (tmp_path / "out/previews/00000000.jpg").exists()
    monkeypatch.setattr(eval, "extract", lambda *a: pytest.fail("Cache should be reused"))
    cached = retrieve(*event_files, **kwargs)
    np.testing.assert_array_equal(result.indices, cached.indices)
    assert json.loads((tmp_path / "out/metrics.json").read_text())["queries"] == 4
    with (tmp_path / "out/retrievals.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 4


def test_checkpoint_roundtrip_and_projection(tmp_path):
    torch.set_num_threads(2)
    cfg = config()
    cfg.aggregator = "salad"
    cfg.salad_clusters, cfg.salad_cluster_dim, cfg.salad_token_dim = 2, 4, 8
    cfg.salad_mlp_dim, cfg.salad_proj_dim, cfg.salad_proj = 16, 12, True
    cfg.finalize()
    model = VPRModel(cfg).eval()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "config": vars(cfg),
            "model": model.state_dict(),
            "step": 1,
            "rng": {"numpy": np.random.get_state()},
        },
        path,
    )
    checkpoints.load_model(path)
    checkpoints.export_checkpoint(path, tmp_path / "export.pt")
    loaded, restored = checkpoints.load_model(tmp_path / "export.pt")
    assert restored.salad_proj and restored.desc_dim == 12
    sample = torch.randn(1, 3, 28, 28)
    with torch.inference_mode():
        torch.testing.assert_close(model(sample), loaded(sample), rtol=0, atol=0)


def test_v8_recipes():
    for name, blocks in [("vits", 4), ("vitb", 12)]:
        cfg = read_config(f"configs/train/{name}_salad.yaml")
        assert cfg.representation == "accumulate" and cfg.salad_proj
        assert cfg.salad_out_dim == 16640 and cfg.desc_dim == 8448
        assert cfg.n_trainable_blocks == blocks and cfg.steps == 2000
        assert cfg.msls_exclude_cities == ["cph", "sf"]
        assert not cfg.aug_hflip and not cfg.eval_pca


def test_offline_cache_miss_is_actionable(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def missing(**kwargs):
        raise LocalEntryNotFoundError("missing")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    with pytest.raises(ValueError, match="not cached"):
        checkpoints.resolve_model(offline=True)


def test_cache_invalidates_when_input_or_options_change(event_files, tmp_path, monkeypatch):
    import megaevent.eval as eval

    cfg = config()
    dataset = EventDataset(event_files[0], cfg, StreamOptions(sensor_size=(8, 8), time_unit="us"))
    network = type("M", (), {"config": cfg})()
    calls = []

    def extract(model, dataset, path, *args):
        calls.append(path)
        np.save(path, np.zeros((len(dataset), cfg.desc_dim), dtype=np.float32))
        return np.load(path, mmap_mode="r")

    monkeypatch.setattr(eval, "extract", extract)
    arguments = (network, dataset, tmp_path / "banks", "checkpoint", torch.device("cpu"), 1, 0)
    (tmp_path / "banks").mkdir()
    eval.descriptor_bank(*arguments)
    eval.descriptor_bank(*arguments)
    assert len(calls) == 1
    dataset.options.hot_pixel_filter = True
    eval.descriptor_bank(*arguments)
    assert len(calls) == 2
    with event_files[0].open("ab") as stream:
        stream.write(b"changed")
    eval.descriptor_bank(*arguments)
    assert len(calls) == 3
