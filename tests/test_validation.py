import numpy as np
import pandas as pd
import torch

from megaevent.recall import recall_at_k_cross
from megaevent.training.i2eval import build_msls_val


def test_msls_validation_keeps_cities_disjoint(tmp_path):
    images, metadata = tmp_path / "images", tmp_path / "metadata"
    for city in ("a", "b"):
        for split in ("database", "query"):
            directory = images / city / split / "images"
            directory.mkdir(parents=True)
            for key in ("one", "two"):
                (directory / f"{key}.png").touch()
            directory = metadata / city / split
            directory.mkdir(parents=True)
            pd.DataFrame({"key": ["one", "two"], "easting": [0, 50], "northing": [0, 0]}).to_csv(
                directory / "postprocessed.csv", index=False
            )
    query, ref, gt, spans = build_msls_val(str(images), str(metadata), ["a", "b"], ext=".png")
    assert len(query) == len(ref) == 4
    np.testing.assert_array_equal(gt, np.eye(4, dtype=bool))
    assert spans == [("a", 0, 2), ("b", 2, 4)]


def test_training_validation_reuses_inference_ranking():
    ref = torch.eye(3)
    query = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    gt = np.array([[0, 1, 0], [0, 0, 0]], dtype=bool)
    metrics = recall_at_k_cross(query, ref, gt)
    assert metrics == {1: 1.0, 5: 1.0, 10: 1.0, 20: 1.0}
