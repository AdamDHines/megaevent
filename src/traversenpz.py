"""Read a traverse's slices from dumped ``.npz`` instead of from the HDF5 recording.

Every pooled script (``scripts/brisbane_pooled.py`` and its three siblings) builds its
descriptor banks from an eventcv reader over ``<seq>.hdf5``. That is the right source for the
``real`` arm and the only one for it. The Sim2Real arms have no HDF5: their events were
simulated by I2E from the DAVIS APS frames and exist only as ``frame_%06d.npz``.

This module is the swap. It supplies the frames; the rendering, the model and the scoring all
stay where they are, so the two arms differ in exactly one thing — where the events came from.

Layout, shared with :mod:`src.traversevpr`::

    <npz-root>/<dataset>/<source>/<seq>/frame_%06d.npz

``source`` is one of ``real``, ``i2e``, ``real_masked``, ``i2e_masked``. Note that a ``real``
tree here is *not* interchangeable with the HDF5 path: only ``sunset1`` and ``sunset2`` were
ever materialised to npz, and the pooled protocol needs all six traverses. Ask for
``source="real"`` and you get the HDF5 reader; the npz ``real`` tree is for the pairwise
harness and for byte-equality audits.

The renderers live in :mod:`src.npzdata` and already match each method one-to-one —
``load_countmask`` for MegaEvent, ``load_mcts`` for Event-GeM, ``load_count_triplet`` for
EventVLAD, ``load_onoff`` for SpikeVPR — because :mod:`src.traversevpr` already scores all
five methods off this same tree, just under the pairwise protocol.
"""

import glob
import json
import os

import numpy as np
from torch.utils.data import Dataset

SOURCES = ("real", "i2e", "real_masked", "i2e_masked")
# The arms whose events are simulated rather than recorded. Everything that is a property of
# the sensor rather than of the scene -- hot-pixel removal, background-activity denoising --
# is meaningless on these, because I2E has no read noise to remove.
SYNTHETIC = ("i2e", "i2e_masked")


def source_dir(npz_root, dataset, source, seq):
    path = os.path.join(npz_root, dataset, source, seq)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"no {source} arm for {seq} at {path}. Build it: scripts/extract_aps.py "
            f"--seq {seq}, then I2E over <npz-root>/{dataset}/aps/{seq}, then "
            f"scripts/mask_vignette.py for the masked arms.")
    return path


def frame_paths(npz_root, dataset, source, seq):
    """Every ``frame_*.npz`` for one traverse, sorted — and the sort *is* the slice order.

    Six-digit zero padding makes the lexicographic sort numeric, so row *i* of a descriptor
    bank is slice *i*. That is the whole alignment contract: the pooled geometry attaches
    coordinates by row index, and ``pool_database`` asserts the row count against it.
    """
    paths = sorted(glob.glob(os.path.join(source_dir(npz_root, dataset, source, seq),
                                          "frame_*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"no frame_*.npz in {source_dir(npz_root, dataset, source, seq)}")
    return paths


def aps_validity_mask(npz_root, dataset, seq, n, strict=True):
    """``[n]`` bool — False where this slice has no APS frame within half a slice.

    Read from the ``select.json`` that ``scripts/extract_aps.py`` writes. It describes the
    *alignment*, not the event source, so it must be applied to whichever arm is scored:
    dropping these frames from the synthetic arm alone would compare a real frame against a
    duplicated one and report the difference as a domain gap.

    Both traverses that have been measured have only a handful, all at the ends — sunset1's
    event stream starts 1.24 s before its first APS frame, sunset2's last slice runs 68 ms
    past its last one.
    """
    path = os.path.join(npz_root, dataset, "aps", seq, "select.json")
    keep = np.ones(n, dtype=bool)
    if not os.path.exists(path):
        if strict:
            raise FileNotFoundError(
                f"{path}: no APS selection report for {seq}. Without it the arms cannot be "
                f"masked in step, so they are not comparable. Pass strict=False only for a "
                f"real-arm run that is deliberately reproducing the unmasked numbers.")
        return keep
    with open(path) as handle:
        report = json.load(handle)
    if int(report.get("n_slices", n)) != n:
        raise ValueError(
            f"{path} was written for {report.get('n_slices')} slices but {seq} has {n} "
            f"frames — the arms would be scored on different grids")
    bad = np.asarray(report.get("out_of_tolerance", []), dtype=int)
    keep[bad[bad < n]] = False
    return keep


class NpzTraverseDataset(Dataset):
    """One traverse's dumped slices, rendered by ``loader``.

    Mirrors :class:`src.inference.EventStreamDataset`'s contract — ``__len__`` and an
    ``__getitem__`` returning one model-ready tensor — so a DataLoader cannot tell them
    apart. Unlike that class there is no reader handle, so nothing here is fork-hostile and
    ``__getstate__`` needs no special case.

    ``loader`` takes a path and returns the finished tensor. Keeping rendering in the caller
    is deliberate: each method renders differently (countmask, MCTS, count triplet, on/off),
    and those renderers already exist in :mod:`src.npzdata`.
    """

    def __init__(self, paths, loader):
        self.paths = list(paths)
        self.loader = loader
        if not self.paths:
            raise ValueError("NpzTraverseDataset got no paths")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.loader(self.paths[i])


def mcts_loader(size):
    """Event-GeM's renderer: ``.npz`` -> ``[10, H, W]`` float32 MCTS, already in [0, 1].

    ``src.npzdata.load_mcts`` rebins in the *event* domain exactly as
    ``MctsStreamDataset`` resizes its reader before rendering, so the two arms are built the
    same way — averaging a rendered decay map instead would mix pixels that fired at
    different times.
    """
    import torch

    from src.npzdata import load_mcts

    def load(path):
        return torch.from_numpy(np.ascontiguousarray(load_mcts(path, size=tuple(size))))

    return load


def count_plane_loader():
    """EventVLAD's per-slice source: ``.npz`` -> one raw count frame.

    Deliberately *not* ``load_count_triplet``. That builds a three-plane input by splitting a
    single I2E saccade into sub-windows, which has no counterpart in a recorded 50 ms slice.
    The pooled EventVLAD arm stacks **three consecutive slices** instead, so the synthetic arm
    must do the same: same construction, same ``n-2`` rows, only the events differ.
    """
    from src.npzdata import load_count

    def load(path):
        return np.asarray(load_count(path))

    return load


def countmask_loader(transform):
    """MegaEvent's renderer: ``.npz`` -> ``[3,H,W]`` uint8 -> float [0,1] -> ``transform``.

    The same three steps :class:`src.inference.EventStreamDataset` applies to an eventcv
    frame, and ``src.npzdata.load_countmask`` is documented byte-identical to eventcv's own
    countmask — so a bank built here and one built from the HDF5 differ only by their events.
    """
    import torch

    from src.npzdata import load_countmask

    def load(path):
        frame = np.ascontiguousarray(load_countmask(path))
        return transform(torch.from_numpy(frame).float().div_(255.0))

    return load
