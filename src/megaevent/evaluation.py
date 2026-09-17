"""Explicit ground-truth alignment and recall metrics."""

import csv
from pathlib import Path

import numpy as np


def load_positives(path, reference_samples, query_samples, layout="reference-query"):
    nr, nq = len(reference_samples), len(query_samples)
    if layout not in {"reference-query", "query-reference"}:
        raise ValueError("Unknown ground-truth layout")
    if Path(path).suffix.lower() == ".npy":
        gt = np.load(path, allow_pickle=False, mmap_mode="r")
        expected = (nr, nq) if layout == "reference-query" else (nq, nr)
        if gt.shape != expected:
            raise ValueError(
                f"Ground truth shape {gt.shape}, expected {expected}; align windows first"
            )
        positives = []
        for i in range(nq):
            row = gt[:, i] if layout == "reference-query" else gt[i]
            if not ((row == 0) | (row == 1)).all():
                raise ValueError("Ground truth must contain only boolean/0/1 values")
            positives.append(set(np.flatnonzero(row)))
        return positives
    references = {sample["id"]: i for i, sample in enumerate(reference_samples)}
    queries = {sample["id"]: i for i, sample in enumerate(query_samples)}
    positives = [set() for _ in queries]
    with open(path, newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        if not {"query_id", "reference_id"}.issubset(rows.fieldnames or []):
            raise ValueError("Ground-truth CSV needs query_id,reference_id columns")
        for row in rows:
            try:
                positives[queries[row["query_id"]]].add(references[row["reference_id"]])
            except KeyError as exc:
                raise ValueError(f"Unknown sample ID in ground truth: {row}") from exc
    return positives


def recall(ranked, positives, ks=(1, 5, 10, 20)):
    valid = [i for i, positive in enumerate(positives) if positive]
    metrics = {
        f"R@{k}": (
            sum(bool(set(ranked[i, :k]) & positives[i]) for i in valid) / len(valid)
            if valid
            else None
        )
        for k in ks
    }
    return {**metrics, "queries": len(positives), "scorable_queries": len(valid)}


def recall_at_k_cross(query_desc, ref_desc, gt, ks=(1, 5, 10, 20), device=None, chunk=128):
    """Training validation using the same bounded-memory ranker as inference."""
    from .retrieval import topk

    query = query_desc.detach().cpu().numpy()
    reference = ref_desc.detach().cpu().numpy()
    ground_truth = np.asarray(gt, dtype=bool)
    if ground_truth.shape != (len(query), len(reference)):
        raise ValueError("Validation ground-truth dimensions do not match the descriptor banks")
    valid = ground_truth.any(axis=1)
    ranked, _ = topk(reference, query, max(ks), device or "cpu", query_chunk=chunk)
    hits = np.take_along_axis(ground_truth, ranked, axis=1)
    return {
        k: float((hits[:, :k].any(axis=1) & valid).sum()) / max(1, int(valid.sum())) for k in ks
    }
