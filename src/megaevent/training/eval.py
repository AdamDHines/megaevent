"""Descriptor extraction for converted validation images."""

import torch

from ..recall import recall_at_k_cross as recall_at_k_cross
from .dataset import FlatImageDataset, build_transforms, make_flat_loader


@torch.no_grad()
def extract_descriptors(model, place_index, cfg, device, batch_size=64, num_workers=4, amp=True):
    """Run ``model`` over every image in ``place_index`` -> (desc[N,D], labels[N]).

    Descriptors come back on CPU in float32 (they are L2-normalised by the model).
    """
    model.eval()
    tf = build_transforms(cfg, train=False)
    loader = make_flat_loader(FlatImageDataset(place_index, tf), batch_size, num_workers)

    use_amp = amp and str(device).startswith("cuda")
    autocast = torch.amp.autocast(device_type="cuda", enabled=use_amp)

    descs, labels = [], []
    for imgs, y in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast:
            d = model(imgs)
        descs.append(d.float().cpu())
        labels.append(y)
    return torch.cat(descs), torch.cat(labels)


# ---------------------------------------------------------------------------
# Cross-set retrieval (query traverse vs reference traverse) — real event VPR
# ---------------------------------------------------------------------------
def extract_paths(model, paths_list, cfg, device, batch_size=64, num_workers=4, amp=True):
    """Descriptors for an *ordered* list of image paths (order preserved). [N, D] on CPU."""
    fake_index = [(i, [p]) for i, p in enumerate(paths_list)]  # reuse the flat extractor
    desc, _ = extract_descriptors(model, fake_index, cfg, device, batch_size, num_workers, amp)
    return desc
