"""SALAD aggregator — optimal-transport pooling of local features into a global descriptor.

Ported verbatim from MegaLoc (``../megaloc/megaloc_model.py``, arXiv 2502.17237, MIT
license), which in turn adapts the Sinkhorn solver from OpenGlue (MIT). Kept in its own
module so ``model.py`` stays free of the OT machinery and importable in a minimal env.

Output dim = ``num_clusters * cluster_dim + token_dim`` (default 64*128+256 = 8448),
already L2-normalised. This is MegaLoc's own aggregator and the capacity upgrade over
GeM (whose 2048-d output is a linear map of a 384-d vector -> effective rank <=384).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling for the differentiable optimal-transport problem.

    Args:
        log_a: source weights. log_b: target weights. M: metric cost matrix.
        num_iters: number of iterations. reg: regularization value.
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm


class FeatureAggregator(nn.Module):
    """Optimal transport-based aggregation of local features into a global descriptor.

    Args:
        num_channels: number of input feature channels (backbone embed dim).
        num_clusters: number of cluster centers.
        cluster_dim: dimensionality of cluster descriptors.
        token_dim: dimensionality of the global scene token.
        mlp_dim: hidden dimension for the MLPs.
        dropout: dropout probability (0 to disable).
    """

    def __init__(
        self,
        num_channels=1536,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        mlp_dim=512,
        dropout=0.3,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.mlp_dim = mlp_dim

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, self.token_dim)
        )
        # MLP for local features
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, self.mlp_dim, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(self.mlp_dim, self.cluster_dim, 1),
        )
        # MLP for score matrix
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, self.mlp_dim, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(self.mlp_dim, self.num_clusters, 1),
        )
        # Dustbin parameter
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    @property
    def out_dim(self):
        return self.num_clusters * self.cluster_dim + self.token_dim

    def forward(self, x):
        """(features [B,C,H,W], token [B,C]) -> global descriptor [B, clusters*cluster_dim + token_dim]."""
        x, t = x

        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        p = p[:, :-1, :]

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        f = torch.cat(
            [
                F.normalize(t, p=2, dim=-1),
                F.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )

        return F.normalize(f, p=2, dim=-1)


def load_megaloc_aggregator(aggregator, weights_path):
    """Warm-start a :class:`FeatureAggregator` from MegaLoc's pretrained SALAD weights.

    Our ``FeatureAggregator`` is a verbatim port of MegaLoc's, so the sub-module names
    (``token_features``/``cluster_features``/``score``/``dust_bin``) match 1:1. In MegaLoc's
    checkpoint the head sits under ``aggregator.agg.*`` (``aggregator.linear.*`` is a separate
    projection we do not use). We select that sub-state-dict, strip the prefix, and load strict.

    Only shape-compatible with the ViT-B backbone (``num_channels == 768``); MegaLoc's head is
    clusters 64 / cluster_dim 256 / token_dim 256 / mlp 512 (16640-d), forced by
    ``VPRConfig.finalize`` when ``salad_init == 'megaloc'``. A strict load raises on any mismatch.
    """
    if weights_path is None:
        raise ValueError(
            "salad_init='megaloc' but no weights path — set GEPT_MEGALOC_WEIGHTS (or pass "
            "--megaloc-weights) to MegaLoc's model.safetensors (HF gberton/MegaLoc). Compute "
            "nodes are offline, so pre-download it on a login node.")
    if aggregator.num_channels != 768:
        raise ValueError(
            f"MegaLoc SALAD warm-start needs a 768-d backbone (ViT-B); this aggregator has "
            f"num_channels={aggregator.num_channels}. Use --vit base for --salad-init megaloc.")
    from safetensors.torch import load_file
    full = load_file(weights_path)
    prefix = "aggregator.agg."
    sub = {k[len(prefix):]: v for k, v in full.items() if k.startswith(prefix)}
    if not sub:
        raise KeyError(
            f"no keys under {prefix!r} in {weights_path} — not a MegaLoc checkpoint? "
            f"(found e.g. {list(full)[:3]})")
    aggregator.load_state_dict(sub, strict=True)   # raises on any mismatch
    print(f"[salad] warm-started SALAD head from MegaLoc {weights_path} "
          f"({len(sub)} tensors under {prefix})")
