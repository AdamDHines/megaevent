"""Bounded-memory exact cosine retrieval, with deterministic tie ordering."""

import numpy as np
import torch


@torch.inference_mode()
def topk(reference, query, k=5, device="cpu", query_chunk=128, reference_chunk=4096):
    if k < 1 or query_chunk < 1 or reference_chunk < 1:
        raise ValueError("k and chunk sizes must be positive")
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("Descriptor banks must be matrices with the same descriptor dimension")
    if not len(reference) or not len(query):
        raise ValueError("Descriptor banks must not be empty")
    k = min(k, len(reference))
    indices = np.empty((len(query), k), dtype=np.int64)
    scores = np.empty((len(query), k), dtype=np.float32)
    for start in range(0, len(query), query_chunk):
        q = torch.tensor(np.asarray(query[start : start + query_chunk]), device=device)
        best_scores = torch.empty((len(q), 0), device=device)
        best_ids = torch.empty((len(q), 0), dtype=torch.long, device=device)
        for offset in range(0, len(reference), reference_chunk):
            r = torch.tensor(
                np.asarray(reference[offset : offset + reference_chunk]), device=device
            )
            sim = q @ r.T
            ids = torch.arange(offset, offset + len(r), device=device).expand(len(q), -1)
            candidates = torch.cat((best_scores, sim), dim=1)
            candidate_ids = torch.cat((best_ids, ids), dim=1)
            # Sort ID first, then stable score, so ties agree across chunk boundaries.
            order = candidate_ids.argsort(dim=1, stable=True)
            candidates = candidates.gather(1, order)
            candidate_ids = candidate_ids.gather(1, order)
            order = candidates.argsort(dim=1, descending=True, stable=True)[:, :k]
            best_scores = candidates.gather(1, order)
            best_ids = candidate_ids.gather(1, order)
        indices[start : start + len(q)] = best_ids.cpu().numpy()
        scores[start : start + len(q)] = best_scores.cpu().numpy()
    return indices, scores
