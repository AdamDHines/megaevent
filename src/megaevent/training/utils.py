"""Small, dependency-free helpers for the VPR package.

``get_param_groups`` and ``get_lr`` are ported verbatim (behaviour-preserving)
from ``src/utils.py`` so the ``vpr`` package does not have to import that module,
which pulls in numba / h5py / hdf5plugin at module load. Keep these in sync if the
originals change.
"""

import math


def get_param_groups(model, wd, encoder_lr_mult=1.0, transformer_lr_mult=1.0):
    """Split params into (encoder, transformer, head-with-wd, head-no-wd) groups.

    - anything whose parameter name contains ``"encoder"``   -> encoder group  (wd=0, lr_mult=encoder_lr_mult)
    - anything whose parameter name contains ``"transformer"`` -> transformer group (wd=0, lr_mult=transformer_lr_mult)
    - remaining 2D+ weights  -> head group with weight decay
    - remaining 1D params / biases -> head group without weight decay
    Frozen params (``requires_grad=False``) are skipped, so freezing blocks before
    building the optimizer is sufficient to exclude them.

    Ported from ``src/utils.py:get_param_groups``.
    """
    en, tr = [], []
    reg_hd, noreg_hd = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder" in name:
            en.append(p)
        elif "transformer" in name:
            tr.append(p)
        else:
            if p.ndim < 2 or name.endswith(".bias"):
                noreg_hd.append(p)
            else:
                reg_hd.append(p)

    return (
        {"params": en, "weight_decay": 0.0, "lr_mult": encoder_lr_mult},
        {"params": tr, "weight_decay": 0.0, "lr_mult": transformer_lr_mult},
        {"params": reg_hd, "weight_decay": wd, "lr_mult": 1.0},
        {"params": noreg_hd, "weight_decay": 0.0, "lr_mult": 1.0},
    )


def get_param_groups_llrd(model, wd, num_layers, layer_decay, encoder_prefix="encoder"):
    """Layerwise-LR-decay param groups (BEiT / DINOv2-finetune style).

    Each parameter gets a depth-scaled ``lr_mult = layer_decay ** (top - layer_id)`` so
    early (general) features move slower than late (task-specific) ones. Depth map::

        layer 0             patch_embed + cls/pos/mask/register tokens   (lowest LR)
        layer 1..num_layers transformer blocks 0..num_layers-1
        layer num_layers+1  final norm + aggregator + projection (head, lr_mult = 1.0)

    Encoder groups carry no weight decay; only head 2D weights get ``wd`` (biases / 1D
    don't). Frozen params are skipped, so this composes with the finetune-freeze — e.g.
    for last-k finetune the frozen early blocks simply drop out and the surviving blocks
    keep their true-depth multipliers. Groups sort deepest->shallowest for readable logs.

    Uses the same ``lr_mult`` group convention as ``get_param_groups`` (train loop does
    ``group_lr = base_lr * lr_mult``), matching the repo idiom in ``src/gra.py``.
    """
    top = num_layers + 1

    def layer_id(name):
        if name.startswith(f"{encoder_prefix}.blocks."):
            return int(name.split(f"{encoder_prefix}.blocks.")[1].split(".")[0]) + 1
        if name.startswith(f"{encoder_prefix}.patch_embed") or name in (
            f"{encoder_prefix}.cls_token",
            f"{encoder_prefix}.pos_embed",
            f"{encoder_prefix}.mask_token",
            f"{encoder_prefix}.register_tokens",
        ):
            return 0
        return top  # encoder.norm + non-encoder head (aggregator / proj)

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(name)
        lr_mult = layer_decay ** (top - lid)
        is_encoder = name.startswith(encoder_prefix)
        g_wd = 0.0 if is_encoder else (wd if (p.ndim >= 2 and not name.endswith(".bias")) else 0.0)
        key = (lid, g_wd)
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": g_wd, "lr_mult": lr_mult}
        groups[key]["params"].append(p)
    return [groups[k] for k in sorted(groups, key=lambda k: (-k[0], k[1]))]


def get_lr(step, warmup_steps, lr, lr_decay_steps, min_lr):
    """Linear warmup then cosine decay to ``min_lr``. Ported from ``src/utils.py:get_lr``.

    Warmup counts from ``step + 1`` so step 0 takes a (small) real step; the historical
    form returned exactly 0.0 there, making the first optimizer step a silent no-op
    (audit B10).
    """
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step > lr_decay_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (lr_decay_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)
