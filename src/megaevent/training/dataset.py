"""Place-index builders, event-image augmentation and P-by-K sampling."""

import glob
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms


# ---------------------------------------------------------------------------
# Index caching
# ---------------------------------------------------------------------------
def atomic_write_json(path, obj, **dump_kw):
    """Write ``obj`` to ``path`` atomically, safely against *concurrent writers*.

    ``os.replace`` makes the swap atomic for readers, but only if every writer owns its own
    temp file. A fixed ``path + '.tmp'`` is shared by every process writing the same cache,
    and that is the normal case here: a sweep submitted with ``--chain-width N`` starts N
    jobs which all build the same stream indices and the same stats blob at once. They then
    interleave their writes into one temp file and both rename it, so the "atomic" result is
    a truncated JSON. ``mkstemp`` gives each writer a private name in the *same* directory,
    which is what keeps the rename on one filesystem and therefore atomic.
    """
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, **dump_kw)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def cached_index(name, params, build_fn, cache_dir=None, refresh=False):
    """Memoise a place index to JSON, keyed by ``name`` + a hash of ``params``.

    Building an index means walking the image tree. GSV-Cities is 525k files; SF-XL
    is millions. On a shared HPC filesystem that is minutes of A100 time burned at
    every job start (and a sweep launches many jobs at once), so the walk is done
    once and reloaded thereafter. ``params`` must contain every value that changes
    the result — it is hashed into the filename, so a changed parameter yields a
    different cache entry rather than a stale hit.

    ``cache_dir`` defaults to the dataset root in ``params["root"]``; if that is not
    writable (read-only mounts are common on HPC) the cache is silently skipped and
    the index is built in-process.
    """
    key = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]
    root = params.get("root", "")
    cdir = cache_dir or (root if os.path.isdir(root) else None)
    path = os.path.join(cdir, f".vpr_index_{name}_{key}.json") if cdir else None

    if path and not refresh and os.path.exists(path):
        try:
            with open(path) as f:
                blob = json.load(f)
            index = [(int(pid), ps) for pid, ps in blob["index"]]
            gone, probed = _sample_missing(index)
            if gone:
                print(
                    f"[{name}] cached index is stale: {gone}/{probed} sampled files "
                    f"no longer exist (tree re-rendered or moved?) — rebuilding"
                )
            else:
                age_h = (time.time() - os.path.getmtime(path)) / 3600.0
                print(
                    f"[{name}] loaded cached index: {len(index)} places, "
                    f"{sum(len(p) for _, p in index)} images "
                    f"({os.path.basename(path)}, built {age_h:.1f}h ago)"
                )
                return index
        except (OSError, ValueError, KeyError) as err:
            print(f"[{name}] cache unreadable ({err}) — rebuilding")

    t0 = time.time()
    index = build_fn()
    print(f"[{name}] built index in {time.time() - t0:.1f}s")

    if path:
        try:
            # Private temp + rename: safe both for readers mid-write and for the several
            # sweep jobs that build this same index concurrently (see atomic_write_json).
            atomic_write_json(path, {"params": params, "index": index})
            print(f"[{name}] cached index -> {path}")
        except OSError as err:
            print(f"[{name}] could not write cache ({err}) — continuing without")
    return index


# How many paths to stat() when deciding whether a cached index still describes the tree.
_CACHE_PROBE = 32


def _sample_missing(index, n=_CACHE_PROBE, seed=0):
    """``(missing, probed)`` over up to ``n`` deterministically-sampled paths in ``index``.

    The cache is keyed on build *parameters*, not on tree content, so a tree that has been
    re-rendered in place or moved would otherwise be trained on through a stale index as if
    nothing had changed. Stat'ing a few dozen paths costs milliseconds against the minutes a
    full re-walk costs on SF-XL, and it catches the whole-tree changes that actually happen.

    Samples place-then-path rather than flattening the index: SF-XL holds ~10^6 paths and
    materialising them on every cache load would cost more than the check saves.

    What it does **not** catch: a re-render still in progress that happens to have left the
    sampled files intact. A *finished* re-render to a new extension invalidates the cache by
    changing the ``ext`` in ``params``, and ``PlacesDataset`` substitutes files that are
    present but unreadable — between them that hole is covered.
    """
    if not index:
        return 0, 0
    rng = random.Random(seed)
    probe = set()
    for _ in range(4 * n):  # a few draws may collide; stop as soon as we have n
        if len(probe) >= n:
            break
        _, ps = index[rng.randrange(len(index))]
        if ps:
            probe.add(ps[rng.randrange(len(ps))])
    return sum(not os.path.exists(p) for p in probe), len(probe)


_WALK_MEMO = {}


def _walk_images(root, ext):
    """Every ``*ext`` file under ``root``, recursively (sorted, absolute).

    Memoised per process: SF-XL's frontal and lateral streams walk the *same* tree,
    and on a multi-million-file dataset that walk is the expensive part (~5 min on a
    cold cache), so the second stream reuses the first one's listing.
    """
    if (root, ext) in _WALK_MEMO:
        return _WALK_MEMO[(root, ext)]
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in filenames:
            if fn.endswith(ext):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    _WALK_MEMO[(root, ext)] = out
    return out


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
class RandomSwapEventRedBlue:
    """Swap the R and B channels of a tencode tensor (polarity inversion).

    tencode = [R=positive polarity, G=time, B=negative polarity]; swapping R<->B
    inverts event polarity, a label-preserving event augmentation. Reimplemented
    here (no numba dep) from ``src/dataset.py:RandomSwapEventRedBlue``.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):  # x: [3, H, W]
        if random.random() < self.p:
            x = x[[2, 1, 0], :, :]
        return x


class CountmaskDomainRandomise:
    """Sim-to-real event-noise randomisation, on a ``[3,H,W]`` countmask tensor in [0,1].

    I2E events are noiseless and dense and are generated with a *single global* contrast
    threshold; a real DAVIS346 has per-pixel threshold variance, hot pixels, a refractory
    period and background noise events. countmask is
    ``[R = M_pos/alpha, G = binary activity mask, B = M_neg/alpha]`` on a black background,
    so each missing non-ideality has a direct expression in these channels:

    * **gain** — one global ``C`` in simulation, unknown per-pixel threshold and unknown
      normalising ``alpha`` in reality: scale R/B by a per-image factor. The activity mask
      is untouched, because this is the same set of events counted differently.
    * **dropout** — refractory period and bandwidth limits lose real events. Zero *all
      three* channels at a random pixel subset, so a dropped pixel becomes background
      rather than an active pixel holding a zero count (which never occurs in real data).
    * **salt** — background noise events and hot pixels: give a few pixels one event of
      random polarity and mark them active.

    Applied **before** ``Normalize``, since all three are statements about event counts.
    One event is worth ``1/alpha``, which varies per frame, so the frame's own smallest
    non-zero count is used as the unit — self-calibrating and cheap on 3x260x346.
    """

    def __init__(self, gain=(0.7, 1.4), dropout_max=0.15, salt_prob=0.005):
        self.gain = (float(gain[0]), float(gain[1]))
        self.dropout_max = float(dropout_max)
        self.salt_prob = float(salt_prob)

    def __call__(self, x):  # x: [3,H,W] in [0,1]
        x = x.clone()
        hw = x.shape[-2:]
        lo, hi = self.gain
        if hi > lo:
            g = lo + (hi - lo) * random.random()
            x[0] *= g
            x[2] *= g
        if self.dropout_max > 0:
            rate = random.random() * self.dropout_max
            x *= torch.rand(hw) >= rate  # broadcasts over channels
        if self.salt_prob > 0:
            # One event is worth 1/alpha, and alpha varies per frame, so calibrate off the
            # frame's own smallest non-zero *count* — the polarity channels only. The green
            # channel is a binary mask whose non-zero value is 1.0 and would swamp the min on
            # any frame whose counts are all small.
            pol = x[::2]
            nz = pol[pol > 0]
            unit = float(nz.min()) if nz.numel() else 1.0 / 255.0
            salt = torch.rand(hw) < random.random() * self.salt_prob
            if bool(salt.any()):
                pos = salt & (torch.rand(hw) < 0.5)
                x[0][pos] = unit
                x[2][salt & ~pos] = unit
                x[1][salt] = 1.0
        return x.clamp_(0.0, 1.0)  # gain > 1 saturates, as the percentile norm would

    def __repr__(self):
        return (
            f"{type(self).__name__}(gain={self.gain}, dropout_max={self.dropout_max}, "
            f"salt_prob={self.salt_prob})"
        )


class AccumulateDomainRandomise:
    """``CountmaskDomainRandomise``'s three non-idealities, in accumulate's inverted space.

    accumulate = ``[R = 1-neg, G = 1-pos-neg, B = 1-pos]`` on a **white** background, so
    "no events" is 1.0 and count intensity lives as ``1 - channel``. Same knobs, same
    physics, mapped through the inversion:

    * **gain** — scale count intensities: ``x -> 1 - g*(1-x)`` on all three channels (G is
      count-graded here, not binary, so it carries the same rescaled counts — "the same
      events counted differently" stays true).
    * **dropout** — a dropped pixel becomes *background*, i.e. 1.0 on all channels.
    * **salt** — one event of random polarity: a positive event darkens G and B by one
      count unit, a negative one darkens R and G (winner-take-all painting at an empty
      pixel reduces to exactly this).

    One event is worth ``1/thr`` per polarity channel; calibrated off the frame's own
    smallest non-zero intensity in the polarity-carrying channels, as the countmask
    version does. Clamped to [0,1]: gain > 1 saturates at 0, as the percentile norm would.
    """

    def __init__(self, gain=(0.7, 1.4), dropout_max=0.15, salt_prob=0.005):
        self.gain = (float(gain[0]), float(gain[1]))
        self.dropout_max = float(dropout_max)
        self.salt_prob = float(salt_prob)

    def __call__(self, x):  # x: [3,H,W] in [0,1], background = 1
        x = x.clone()
        hw = x.shape[-2:]
        lo, hi = self.gain
        if hi > lo:
            g = lo + (hi - lo) * random.random()
            x = 1.0 - g * (1.0 - x)
        if self.dropout_max > 0:
            rate = random.random() * self.dropout_max
            drop = torch.rand(hw) < rate
            x[:, drop] = 1.0
        if self.salt_prob > 0:
            inten = 1.0 - x[::2]  # [neg, pos] intensities (R, B)
            nz = inten[inten > 0]
            unit = float(nz.min()) if nz.numel() else 1.0 / 255.0
            salt = torch.rand(hw) < random.random() * self.salt_prob
            if bool(salt.any()):
                pos = salt & (torch.rand(hw) < 0.5)
                neg = salt & ~pos
                x[1][salt] -= unit
                x[2][pos] -= unit
                x[0][neg] -= unit
        return x.clamp_(0.0, 1.0)

    def __repr__(self):
        return (
            f"{type(self).__name__}(gain={self.gain}, dropout_max={self.dropout_max}, "
            f"salt_prob={self.salt_prob})"
        )


def build_transforms(cfg, train=True):
    """tencode PIL -> normalised tensor. Follows PLAN 1a (no color jitter)."""
    norm = transforms.Normalize(cfg.tencode_mean, cfg.tencode_std)
    if train:
        ops = [transforms.ToTensor()]
        if getattr(cfg, "aug_domain_rand", False):
            rep = getattr(cfg, "representation", "tencode")
            # Each op set is written against one background/channel convention. tencode's
            # green is event *time*, so neither variant is valid there and it stays an error.
            noise_cls = {
                "countmask": CountmaskDomainRandomise,
                "polmask": CountmaskDomainRandomise,
                "accumulate": AccumulateDomainRandomise,
            }.get(rep)
            if noise_cls is None:
                raise ValueError(
                    f"--aug-domain-rand has no background-aware implementation for "
                    f"representation {rep!r} (tencode's green channel is event time)"
                )
            ops.append(
                noise_cls(
                    gain=cfg.aug_gain_jitter,
                    dropout_max=cfg.aug_dropout_max,
                    salt_prob=cfg.aug_salt_prob,
                )
            )
        # Normalize comes LAST. Flip and crop commute with a per-channel affine exactly
        # (unit-sum interpolation kernels), but the polarity swap does not: swapping
        # *normalised* R/B only equals swapping raw polarities when the two channels share
        # mean/std. countmask's nearly do (which is why the pre-2026-08-19 order — norm
        # first — was benign there), tencode's differ by 0.07 in mean, so the old order
        # silently corrupted any non-countmask run. Augment counts, then describe them.
        ops.append(RandomSwapEventRedBlue(p=0.5))
        if getattr(cfg, "aug_hflip", True):
            # Mirrors scene chirality — a view that never occurs at query time, and one
            # GSV-Cities/MegaLoc deliberately do not train with. Kept default-on for
            # reproducibility of the v4-v6 recipes; v7 arms turn it off (--no-hflip).
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops += [
            transforms.RandomResizedCrop(
                (cfg.H, cfg.W),
                scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            norm,
        ]
        return transforms.Compose(ops)
    from ..events import eval_transform

    return transforms.Compose(
        [
            transforms.ToTensor(),
            *eval_transform(cfg).transforms,
        ]
    )


# ---------------------------------------------------------------------------
# Dataset + P x K sampler
# ---------------------------------------------------------------------------
class PlacesDataset(Dataset):
    """Yields one *place* per index: K images (stacked) + K identical labels.

    ``place_index`` : ``list[(place_id:int, [path, ...])]`` with ``len(paths) >= K``.
    With a ``DataLoader(batch_size=P, collate_fn=places_collate)`` this produces a
    P x K sub-batch, i.e. the MegaLoc regime (P places, K images each).
    """

    def __init__(self, place_index, img_per_place, transform, loader=None):
        assert img_per_place >= 2, "need >= 2 images per place for positives"
        self.place_index = [(pid, ps) for pid, ps in place_index if len(ps) >= img_per_place]
        if len(self.place_index) < len(place_index):
            dropped = len(place_index) - len(self.place_index)
            print(f"[PlacesDataset] dropped {dropped} places with < {img_per_place} images")
        self.K = img_per_place
        self.transform = transform
        self.loader = loader or self._default_loader

    # A tree being re-rendered, or copied, under a running job hands workers files that are
    # truncated or mid-write. A bare Image.open then raises, which kills the worker, which
    # kills the DataLoader, which kills a 48-hour job at whatever step it had reached.
    # Substituting another image from the *same place* keeps the label and the P x K shape
    # intact, so the sub-batch is still well-formed and training continues.
    _MAX_LOAD_RETRY = 4

    @staticmethod
    def _default_loader(path):
        with Image.open(path) as im:
            return im.convert("RGB")

    def _load_or_substitute(self, path, pool):
        for attempt in range(self._MAX_LOAD_RETRY):
            try:
                return self.loader(path)
            except (OSError, ValueError, SyntaxError) as err:
                if attempt == 0:
                    print(
                        f"[PlacesDataset] unreadable image {path} "
                        f"({type(err).__name__}: {err}) — substituting from the same place"
                    )
                path = random.choice(pool)
        raise RuntimeError(
            f"[PlacesDataset] {self._MAX_LOAD_RETRY} consecutive unreadable images around "
            f"{path}. This is a broken or half-written tree, not a stray bad file — fix the "
            f"data rather than raising _MAX_LOAD_RETRY."
        )

    def __len__(self):
        return len(self.place_index)

    def __getitem__(self, idx):
        place_id, all_paths = self.place_index[idx]
        chosen = random.sample(all_paths, self.K)  # K distinct images from this place
        imgs = torch.stack(
            [self.transform(self._load_or_substitute(p, all_paths)) for p in chosen], dim=0
        )  # [K,3,H,W]
        labels = torch.full((self.K,), place_id, dtype=torch.long)
        return imgs, labels


def places_collate(batch):
    """[(imgs[K,3,H,W], labels[K]), ...] (P of them) -> ([P*K,3,H,W], [P*K])."""
    imgs = torch.cat([b[0] for b in batch], dim=0)
    labels = torch.cat([b[1] for b in batch], dim=0)
    return imgs, labels


def make_places_loader(dataset, P, num_workers=8, shuffle=True, drop_last=True):
    """DataLoader over places: each batch = P places = P*K images."""
    return DataLoader(
        dataset,
        batch_size=P,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=places_collate,
    )


# ---------------------------------------------------------------------------
# SF-XL EigenPlaces group cycling
# ---------------------------------------------------------------------------
def sf_xl_group_of(place_id, place_offset, N):
    """Recover the EigenPlaces group (0..N*N-1) an SF-XL place belongs to.

    ``build_sf_xl_index`` mints ``place_id = place_offset + (ce//M)*1_000_000 + (cn//M)``
    and the group is ``(ce % (M*N) // M)*N + (cn % (M*N) // M)``. Since ``ce,cn`` are
    multiples of ``M`` this reduces to ``((ce//M) % N)*N + ((cn//M) % N)`` — a pure
    function of the id, so no side-map has to be carried through the cache.
    """
    local = int(place_id) - int(place_offset)
    cx, cy = local // 1_000_000, local % 1_000_000
    return (cx % N) * N + (cy % N)


class GroupCyclingBatchSampler(Sampler):
    """Yield P-place batches from ONE EigenPlaces group per epoch, cycling groups.

    Each epoch (= one ``iter()``, which is exactly when ``VPRTrainer._next`` re-creates a
    stream's iterator on ``StopIteration``) draws only the current group's places, so a
    sub-batch never mixes groups — classes within a group are M*N=100 m apart (no false
    positives) and cross-group classes never co-occur (no false negatives). Over the run
    all groups are covered, fixing SF-XL under-population without the false-negative risk
    of pooling every group into one index. Groups too small to fill one batch are dropped
    so every epoch yields >=1 batch (guards ``_next``'s single re-iter).
    """

    def __init__(self, groups_per_idx, P, drop_last=True, shuffle=True, seed=0):
        self.P = int(P)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.group_to_idx = defaultdict(list)
        for i, g in enumerate(groups_per_idx):
            self.group_to_idx[g].append(i)
        # only groups with at least one full P-place batch can be an epoch
        self.groups = sorted(g for g, idx in self.group_to_idx.items() if len(idx) >= self.P)
        if not self.groups:
            raise ValueError(
                f"no SF-XL group has >= P={self.P} places "
                f"(largest={max((len(v) for v in self.group_to_idx.values()), default=0)})"
            )
        self.epoch = 0
        self._rng = random.Random(seed)

    def _n_batches(self, g):
        n = len(self.group_to_idx[g])
        return n // self.P if self.drop_last else -(-n // self.P)

    def __iter__(self):
        g = self.groups[self.epoch % len(self.groups)]
        self.epoch += 1
        idxs = list(self.group_to_idx[g])
        if self.shuffle:
            self._rng.shuffle(idxs)
        end = (len(idxs) // self.P) * self.P if self.drop_last else len(idxs)
        for s in range(0, end, self.P):
            yield idxs[s : s + self.P]

    def __len__(self):
        # batches in the group that the *next* __iter__ will serve (varies per epoch)
        return self._n_batches(self.groups[self.epoch % len(self.groups)])

    def group_batch_counts(self):
        """{group: batches_per_epoch} for logging the per-group balance."""
        return {g: self._n_batches(g) for g in self.groups}


def make_group_cycling_loader(dataset, groups_per_idx, P, num_workers=8, seed=0):
    """DataLoader whose epochs cycle EigenPlaces groups (see GroupCyclingBatchSampler)."""
    sampler = GroupCyclingBatchSampler(groups_per_idx, P, drop_last=True, seed=seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=places_collate,
    )
    return loader


# ---------------------------------------------------------------------------
# Evaluation: flat (non-P x K) loading + subset selection
# ---------------------------------------------------------------------------
class FlatImageDataset(Dataset):
    """One (image, place_id) per index, in a fixed order — for descriptor extraction.

    Flattens a ``place_index`` (``list[(place_id, [path, ...])]``) into individual
    images so retrieval eval can build a gallery and run kNN. Uses the *validation*
    transform (no augmentation) — pass ``build_transforms(cfg, train=False)``.
    """

    def __init__(self, place_index, transform, loader=None):
        self.samples = [(p, int(pid)) for pid, paths in place_index for p in paths]
        self.transform = transform
        self.loader = loader or PlacesDataset._default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid = self.samples[idx]
        return self.transform(self.loader(path)), pid


def make_flat_loader(dataset, batch_size=64, num_workers=4):
    """In-order DataLoader for eval (no shuffle, no drop_last)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def subset_index(place_index, n_places, seed=0):
    """Deterministically sample ``n_places`` places (for overfit / quick eval)."""
    if n_places is None or n_places >= len(place_index):
        return list(place_index)
    return random.Random(seed).sample(list(place_index), n_places)


# ---------------------------------------------------------------------------
# Builders (one per MegaLoc dataset stream)
# ---------------------------------------------------------------------------
def _gsv_img_name(row, ext):
    """Reconstruct the amaralibey/gsv-cities filename from a CSV row.

    Convention (verified against the dataset):
      <city>_<place_id:07d>_<year:04d>_<month:02d>_<northdeg:03d>_<lat>_<lon>_<panoid><ext>
    lat/lon/panoid are used *verbatim* from the CSV (read as strings) so float
    reformatting can't corrupt the match.
    """
    return (
        f"{row['city_id']}_"
        f"{int(row['place_id']):07d}_"
        f"{int(row['year']):04d}_"
        f"{int(row['month']):02d}_"
        f"{int(row['northdeg']):03d}_"
        f"{row['lat']}_{row['lon']}_{row['panoid']}{ext}"
    )


def build_gsv_cities_index(
    dataframes_dir,
    images_dir,
    ext=".png",
    cities=None,
    min_img_per_place=4,
    require_exists=True,
    city_offset=100000,
):
    """Build a place index for GSV-Cities.

    dataframes_dir : dir of ``<City>.csv`` (RGB-side metadata / class organization).
    images_dir     : dir under which ``<City>/<filename><ext>`` live (tencode:
                     the mirrored ``Images`` tree; source RGB: ``.../Images``).
    ext            : ``.png`` for tencode output, ``.jpg`` for source RGB.
    cities         : list of city stems, or None = every CSV present.
    city_offset    : place_ids are made globally unique via ``place_id + i*offset``.
    require_exists : keep only images that exist on disk (needed while conversion
                     is partial); set False to skip the stat() for speed.

    Returns ``list[(global_place_id:int, [abs_path, ...])]``.
    """
    if cities is None:
        cities = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(dataframes_dir, "*.csv"))
        )

    index = []
    for i, city in enumerate(cities):
        csv_path = os.path.join(dataframes_dir, f"{city}.csv")
        # read every column as str; casting to int happens in _gsv_img_name
        df = pd.read_csv(csv_path, dtype=str)
        base = i * city_offset
        for local_pid, rows in df.groupby("place_id"):
            paths_ = []
            for _, row in rows.iterrows():
                p = os.path.join(images_dir, city, _gsv_img_name(row, ext))
                if (not require_exists) or os.path.exists(p):
                    paths_.append(p)
            if len(paths_) >= min_img_per_place:
                index.append((base + int(local_pid), paths_))
    print(
        f"[gsv-cities] {len(cities)} cities -> {len(index)} places "
        f"(>= {min_img_per_place} imgs) from {images_dir}"
    )
    return index


def build_gsv_cities_from_tree(
    images_dir,
    ext=".jpg",
    cities=None,
    min_img_per_place=4,
    city_offset=10_000_000,
):
    """Build a place index by scanning an I2E-style ``<City>/<file>`` image tree.

    Use when the ``Dataframes/*.csv`` metadata isn't present (e.g. the I2E tencode
    output ``<root>/jpg/<City>/<name><ext>``): the GSV-Cities filename already
    encodes the class, ``<City>_<place_id:07d>_<year>_<month>_...`` — so place_id is
    field 1 of the underscore-split name. Place ids are offset per city (``i *
    city_offset``, offset > any 7-digit id) for global uniqueness.

    Returns ``list[(global_place_id:int, [abs_path, ...])]``.
    """
    if cities is None:
        cities = sorted(d.name for d in os.scandir(images_dir) if d.is_dir())

    index = []
    for i, city in enumerate(cities):
        cdir = os.path.join(images_dir, city)
        groups = {}
        for f in os.scandir(cdir):
            if not f.name.endswith(ext):
                continue
            parts = f.name.split("_")
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            groups.setdefault(pid, []).append(f.path)
        base = i * city_offset
        for pid, paths_ in groups.items():
            if len(paths_) >= min_img_per_place:
                index.append((base + pid, sorted(paths_)))
    n_imgs = sum(len(p) for _, p in index)
    print(
        f"[gsv-cities/tree] {len(cities)} cities -> {len(index)} places "
        f"({n_imgs} imgs, >= {min_img_per_place}/place) from {images_dir}"
    )
    return index


def compute_channel_stats(
    place_index, hw=(224, 224), max_images=None, batch_size=64, num_workers=4, shuffle=True, seed=0
):
    """Per-channel (mean, std) over the images in ``place_index`` (ToTensor+Resize only).

    Needed because tencode stats differ from GEPT's alignment-frame prior, and a
    black- vs white-background conversion shifts the mean hugely (PLAN 1a). Feed the
    result into ``cfg.tencode_mean/std`` before training. Returns (mean:list, std:list).

    ``shuffle`` is not cosmetic. The pass ``break``s once ``max_images`` pixels have been
    accumulated, and callers hand it the per-stream subsets *concatenated in stream order*
    (``streams.resolve_stats``), so without shuffling the whole budget is spent inside the
    first stream or two and the last ones contribute **zero pixels** to constants that are
    then baked into every checkpoint. Seeded, so the stats stay reproducible.
    """
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(hw, interpolation=transforms.InterpolationMode.BICUBIC),
        ]
    )
    flat = FlatImageDataset(place_index, tf)
    if shuffle:
        random.Random(seed).shuffle(flat.samples)
    loader = make_flat_loader(flat, batch_size, num_workers)
    n = 0
    s = torch.zeros(3, dtype=torch.float64)
    ss = torch.zeros(3, dtype=torch.float64)
    for imgs, _ in loader:
        imgs = imgs.double()
        s += imgs.sum(dim=(0, 2, 3))
        ss += (imgs**2).sum(dim=(0, 2, 3))
        n += imgs.shape[0] * imgs.shape[2] * imgs.shape[3]
        if max_images and n >= max_images * hw[0] * hw[1]:
            break
    mean = s / n
    std = (ss / n - mean**2).clamp(min=1e-8).sqrt()
    return mean.tolist(), std.tolist()


# ---------------------------------------------------------------------------
# SF-XL / EigenPlaces class generation
# ---------------------------------------------------------------------------
# SF-XL filenames carry every field we need, so no side metadata is required:
#   @utm_east@utm_north@zone@S@lat@lon@panoid@@heading@@@@date@@.jpg
#    [1]      [2]        [3]  [4] [5] [6] [7]     [9]      [13]
# (verified against /media/adam/vprdatasets/data/sf-xl/train/*).
SFXL_UTM_EAST, SFXL_UTM_NORTH, SFXL_PANOID, SFXL_HEADING = 1, 2, 7, 9


def _sfxl_fields(path):
    """(utm_east, utm_north, heading_deg, panoid) from an SF-XL filename, or None."""
    parts = os.path.basename(path).split("@")
    if len(parts) <= SFXL_HEADING:
        return None
    try:
        return (
            float(parts[SFXL_UTM_EAST]),
            float(parts[SFXL_UTM_NORTH]),
            float(parts[SFXL_HEADING]),
            parts[SFXL_PANOID],
        )
    except ValueError:
        return None


def _eigen_focal_point(coords, focal_dist, angle_deg):
    """EigenPlaces' focal point for one class — ported from ``gmberton/EigenPlaces``.

    ``datasets/eigenplaces_dataset.py::get_focal_point``: SVD the mean-centred UTM
    coords, take ``eigenvectors[1]``, rotate it by ``angle_deg``, and step
    ``focal_dist`` metres from the centre of mass. ``angle=0`` is the frontal stream
    and ``angle=90`` the lateral one — the two SF-XL sub-batches MegaLoc uses.

    Note ``eigenvectors[1]`` indexes a **row** of the left-singular-vector matrix, not
    the second singular vector. That is what the reference implementation does and
    what produced the published EigenPlaces/MegaLoc results, so it is reproduced
    verbatim rather than "corrected".
    """
    mu = coords.mean(0)
    u, _, _ = np.linalg.svd((coords - mu).T, full_matrices=False)
    direction = u[1]
    theta = math.radians(angle_deg)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return mu + (rot @ direction) * focal_dist


def _bearing_deg(from_e, from_n, to_e, to_n):
    """Compass bearing (degrees clockwise from north) from one UTM point to another."""
    return math.degrees(math.atan2(to_e - from_e, to_n - from_n)) % 360.0


def _angular_diff(a, b):
    """Smallest absolute difference between two bearings, in degrees (0..180)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def build_sf_xl_index(
    images_dir,
    mode="frontal",
    ext=".jpg",
    M=20,
    N=5,
    focal_dist=10,
    groups=(0,),
    min_images_per_class=10,
    min_img_per_place=4,
    max_angle_diff=45.0,
    place_offset=0,
    cache_dir=None,
    refresh=False,
):
    """EigenPlaces class generation over an SF-XL tencode tree -> place index.

    Class = a ``M``-metre UTM cell; ``group`` = the ``(east, north) mod M*N`` residue,
    so classes inside one group are ``M*N`` metres apart and cannot visually overlap
    (EigenPlaces' ``get__class_id__group_id``). Only ``groups`` are kept — default
    ``(0,)``, i.e. one non-overlapping group, because MegaLoc's multi-similarity loss
    treats every other class in the sub-batch as a negative, and classes 20 m apart
    would be false negatives.

    **Adaptation to pre-cut views.** EigenPlaces crops an arbitrary heading out of a
    3328-px panorama (``get_crop``). Our SF-XL is the *pre-cut* release — 512x512
    perspective views at 6 fixed headings (0/60/.../300), 6 per panorama (verified on
    disk) — and an arbitrary heading cannot be synthesised from those. So the focal
    point is used to **select** rather than to crop: per panorama, keep the pre-cut
    view whose heading is closest to the bearing towards the class focal point, within
    ``max_angle_diff``. ``mode='frontal'`` (angle 0) and ``mode='lateral'`` (angle 90)
    therefore pick complementary views of each class, which is exactly the property
    MegaLoc's two SF-XL sub-batches rely on.

    Caveat: the absolute convention of the heading field is not pinned down — the
    reference code applies ``(180 - yaw) % 360`` when it treats the field as a *panorama*
    yaw. A constant offset or mirror would relabel which physical direction each stream
    looks at, but preserves the properties training depends on (classes are places, the
    two streams are ~90 degrees apart, views are consistent within a class). Use
    ``python src/vpr/dataset.py --sfxl-diagnose <dir>`` to inspect the split.

    Returns ``list[(global_place_id:int, [abs_path, ...])]``.
    """
    if mode not in ("frontal", "lateral"):
        raise ValueError(f"mode must be 'frontal' or 'lateral', got {mode!r}")
    angle = 0 if mode == "frontal" else 90
    groups = tuple(groups)

    params = {
        "root": images_dir,
        "mode": mode,
        "ext": ext,
        "M": M,
        "N": N,
        "focal_dist": focal_dist,
        "groups": groups,
        "min_images_per_class": min_images_per_class,
        "min_img_per_place": min_img_per_place,
        "max_angle_diff": max_angle_diff,
        "place_offset": place_offset,
    }

    def _build():
        paths_ = _walk_images(images_dir, ext)
        if not paths_:
            raise FileNotFoundError(f"no {ext} images under {images_dir}")

        # 1. bucket images into M-metre cells, keeping only the wanted group(s)
        per_class = defaultdict(list)
        skipped = 0
        for p in paths_:
            f = _sfxl_fields(p)
            if f is None:
                skipped += 1
                continue
            east, north, heading, panoid = f
            ce, cn = int(east // M * M), int(north // M * M)
            # EigenPlaces' group id, flattened to 0..N*N-1
            group = (ce % (M * N) // M) * N + (cn % (M * N) // M)
            if group not in groups:
                continue
            per_class[(ce, cn)].append((p, east, north, heading, panoid))
        if skipped:
            print(f"[sf_xl/{mode}] skipped {skipped} files with unparseable names")

        # 2. per class: focal point, then keep the best-aimed view of each panorama
        index, dropped_small, dropped_aim = [], 0, 0
        for (ce, cn), items in sorted(per_class.items()):
            if len(items) < min_images_per_class:
                dropped_small += 1
                continue
            coords = np.array([[e, n] for _, e, n, _, _ in items], dtype=np.float64)
            focal = _eigen_focal_point(coords, focal_dist, angle)

            best_per_pano = {}
            for p, east, north, heading, panoid in items:
                target = _bearing_deg(east, north, focal[0], focal[1])
                diff = _angular_diff(heading, target)
                if diff > max_angle_diff:
                    continue
                # one view per panorama: the pre-cut heading aimed most at the focal point
                if panoid not in best_per_pano or diff < best_per_pano[panoid][0]:
                    best_per_pano[panoid] = (diff, p)
            chosen = sorted(p for _, p in best_per_pano.values())
            if len(chosen) < min_img_per_place:
                dropped_aim += 1
                continue
            # cell coords -> a stable, collision-free integer id
            index.append((place_offset + (ce // M) * 1_000_000 + (cn // M), chosen))

        n_imgs = sum(len(p) for _, p in index)
        print(
            f"[sf_xl/{mode}] groups={groups} M={M} N={N} focal={focal_dist}m -> "
            f"{len(index)} places / {n_imgs} images "
            f"(dropped {dropped_small} sparse cells, {dropped_aim} poorly-aimed)"
        )
        return index

    return cached_index(f"sf_xl_{mode}", params, _build, cache_dir, refresh)


def diagnose_sf_xl(
    images_dir,
    ext=".jpg",
    M=20,
    N=5,
    focal_dist=10,
    groups=(0,),
    min_images_per_class=10,
    n_classes=8,
    seed=0,
):
    """Print the frontal/lateral split for a few SF-XL classes, to sanity-check it.

    What matters for MegaLoc is not the absolute naming but three properties, all of
    which this makes visible: classes are places, the two streams look ~90 degrees
    apart, and a class's views are consistent within a stream.
    """
    frontal = {
        pid: ps
        for pid, ps in build_sf_xl_index(
            images_dir, "frontal", ext, M, N, focal_dist, groups, min_images_per_class
        )
    }
    lateral = {
        pid: ps
        for pid, ps in build_sf_xl_index(
            images_dir, "lateral", ext, M, N, focal_dist, groups, min_images_per_class
        )
    }

    shared = sorted(set(frontal) & set(lateral))
    print(
        f"\n[sf_xl/diagnose] {len(frontal)} frontal places, {len(lateral)} lateral, "
        f"{len(shared)} in both"
    )
    if not shared:
        print("  !! no shared classes — the two streams disagree on which cells survive")
        return

    seps, f_spreads, l_spreads = [], [], []
    for pid in shared:
        fh = [_sfxl_fields(p)[2] for p in frontal[pid]]
        lh = [_sfxl_fields(p)[2] for p in lateral[pid]]
        fm, lm = _circular_mean(fh), _circular_mean(lh)
        seps.append(_angular_diff(fm, lm))
        f_spreads.append(_circular_spread(fh, fm))
        l_spreads.append(_circular_spread(lh, lm))

    for pid in random.Random(seed).sample(shared, min(n_classes, len(shared))):
        fh = sorted({_sfxl_fields(p)[2] for p in frontal[pid]})
        lh = sorted({_sfxl_fields(p)[2] for p in lateral[pid]})
        print(f"  class {pid}: frontal n={len(frontal[pid])} headings={fh}")
        print(f"  {' ' * len(str(pid))}        lateral n={len(lateral[pid])} headings={lh}")

    print(
        f"\n  frontal-vs-lateral separation: mean {np.mean(seps):.1f}deg "
        f"median {np.median(seps):.1f}deg   (want ~90)"
    )
    print(
        f"  within-class heading spread:   frontal {np.mean(f_spreads):.1f}deg, "
        f"lateral {np.mean(l_spreads):.1f}deg   (want small)"
    )


def _circular_mean(degs):
    r = np.deg2rad(np.asarray(degs, dtype=float))
    return math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean())) % 360.0


def _circular_spread(degs, mean_deg):
    return float(np.mean([_angular_diff(d, mean_deg) for d in degs])) if len(degs) else 0.0


# ---------------------------------------------------------------------------
# Remaining MegaLoc streams
# ---------------------------------------------------------------------------
# These three need the on-disk layout pinned down first. i2e_infer.py mirrors only
# *images*, so each dataset's class organization — which lives in non-image files —
# is only usable if it was copied to the HPC alongside the tencode tree:
#
#   MSLS       city CSVs (postprocessed.csv / seq_info.csv: UTM + sequence ids)
#   MegaScenes COLMAP reconstructions (sparse/) for shared-3D-point co-visibility
#   ScanNet    per-frame poses (+ intrinsics) for pose/depth co-visibility
#
# Run `python src/vpr/survey_data.py --root <DATA_ROOT>` to see what is actually there.
_NEEDS_SURVEY = (
    "\n\nRun `python src/vpr/survey_data.py --root <DATA_ROOT>` on the HPC and share the "
    "output so this builder can be written against the real layout (PLAN 1a)."
)


def build_dir_class_index(
    images_dir,
    name="dir_classes",
    ext=".png",
    min_img_per_place=4,
    place_offset=0,
    max_img_per_place=None,
    cache_dir=None,
    refresh=False,
):
    """Place index where **the directory containing the images is the class**.

    Serves the two MegaLoc streams whose organization is already expressed by the
    directory tree (verified against the converted trees on HPC):

    * **MegaScenes** ``jpg/<AAA>/<BBB>/<Scene>/<recon>/*.png`` -> class = one
      *reconstruction*, which is exactly MegaLoc's class definition for this dataset.
    * **ScanNet(++)** ``jpg/<scan_id>/dslr/*.png`` -> class = one *scan/scene*.

    This is the base organization only. MegaLoc additionally mines quadruplets with
    mutual visual overlap (>=1% shared 3D points for MegaScenes; <10 m and <30 deg for
    ScanNet), which needs the SfM/pose data — not present in the converted trees, and a
    separate refinement to measure on its own.

    ``max_img_per_place`` caps huge classes so one scene cannot dominate sampling
    (``PlacesDataset`` draws K at random from whatever it is given).
    """
    params = {
        "root": images_dir,
        "ext": ext,
        "min": min_img_per_place,
        "max": max_img_per_place,
        "offset": place_offset,
    }

    def _build():
        groups = defaultdict(list)
        for p in _walk_images(images_dir, ext):
            groups[os.path.dirname(p)].append(p)
        index, dropped = [], 0
        for i, (d, ps) in enumerate(sorted(groups.items())):
            if len(ps) < min_img_per_place:
                dropped += 1
                continue
            ps = sorted(ps)
            if max_img_per_place:
                ps = ps[:max_img_per_place]
            index.append((place_offset + i, ps))
        n_imgs = sum(len(p) for _, p in index)
        print(
            f"[{name}] {len(groups)} image dirs -> {len(index)} places / {n_imgs} images "
            f"(dropped {dropped} with < {min_img_per_place} images)"
        )
        return index

    return cached_index(name, params, _build, cache_dir, refresh)


def _read_msls_csv(path, city, split, attempts=3, backoff=2.0):
    """``pd.read_csv`` that names the city it failed on, retrying transient FS errors.

    A parallel filesystem can fail the *read* of a file that ``stat()``s perfectly well:
    Lustre/GPFS return ENODATA (errno 61) for an object that is HSM-released or has a
    damaged stripe, and it can be transient while a restore completes. The bare pandas
    traceback names neither the city nor the path, and this loop covers ~22 of them, so a
    18-hour job dies with no way to tell which file to look at.
    """
    for attempt in range(1, attempts + 1):
        try:
            return pd.read_csv(path)
        except OSError as err:
            if attempt == attempts:
                raise RuntimeError(
                    f"[msls] cannot read {city}/{split}: {path}\n"
                    f"  {type(err).__name__}: {err}\n"
                    f"The file exists but the filesystem would not return its contents, so "
                    f"this is a storage problem, not a malformed CSV — check it with "
                    f"`wc -l` on a login node and restore it from the source distribution "
                    f"if it is bad. To train without this city meanwhile, add it to "
                    f"--msls-exclude-cities; the guard accepts a superset of "
                    f"--msls-val-cities, so the drop is recorded in the run's flags rather "
                    f"than happening silently."
                ) from err
            print(
                f"[msls] {city}/{split}: read failed ({err}) — "
                f"retry {attempt}/{attempts - 1} in {backoff:g}s"
            )
            time.sleep(backoff)
        except Exception as err:  # malformed/empty CSV: no retry, but name it
            raise RuntimeError(
                f"[msls] cannot parse {city}/{split}: {path}\n  {type(err).__name__}: {err}"
            ) from err


# The six official MSLS *test* cities. The megaevent benchmark scores on them (via the
# recovered-coordinate subset), so their presence in a training index is benchmark leakage
# no flag should be able to cause. The official train_val distribution does not contain
# them; finding one under meta_root means a mixed or misconfigured metadata tree.
MSLS_TEST_CITIES = ("athens", "bengaluru", "buenosaires", "kampala", "miami", "stockholm")


def build_msls_index(
    images_dir,
    meta_root,
    ext=".jpg",
    min_img_per_place=4,
    place_offset=0,
    splits=("database", "query"),
    cities=None,
    exclude_cities=None,
    view_bins=4,
    merge_splits="auto",
    exclude_night=False,
    max_img_per_place=None,
    require_exists=True,
    cache_dir=None,
    refresh=False,
):
    """MSLS place index, joining the converted tencode tree to the source metadata.

    The converted tree (``mapillary/jpg/<city>/<split>/images/<key>.jpg``) carries only
    opaque Mapillary keys — ``i2e_infer.py`` mirrors images, not metadata — so the class
    organization comes from the **source** distribution, joined on the key:

        <meta_root>/<city>/<split>/postprocessed.csv
            ,key,easting,northing,unique_cluster,control_panel,night,view_direction

    ``easting``/``northing`` are already UTM, and ``unique_cluster`` is MSLS's own spatial
    clustering — so a place is ``(unique_cluster, view_direction bin)``. Binning the view
    direction matters: two images at one spot facing opposite ways along a street are not
    the same place visually, and the multi-similarity loss would otherwise be handed a
    false positive. This mirrors GSV-Cities ("same orientation") and the SF-XL builder
    (cell + heading), so the street-level streams stay consistent.

    ``merge_splits='auto'``: database and query are different traversals of the same city,
    and merging them is what makes MSLS valuable (cross-season/-time positives for one
    place) — **but only if the two splits share one ``unique_cluster`` id space**. That is
    checked rather than assumed: shared ids whose centroids agree to within
    ``_MSLS_MERGE_TOL_M`` are merged, otherwise the splits stay separate and say so.
    Force with ``True``/``False``.

    CliqueMining hard negatives are a further refinement needing an offline similarity
    graph from a trained model — deliberately not folded in here.
    """
    params = {
        "root": images_dir,
        "meta": meta_root,
        "ext": ext,
        "min": min_img_per_place,
        "splits": list(splits),
        "cities": cities,
        "exclude": sorted(exclude_cities) if exclude_cities else None,
        "view_bins": view_bins,
        "merge": merge_splits,
        "night": exclude_night,
        "max": max_img_per_place,
        "offset": place_offset,
    }

    def _build():
        if not os.path.isdir(meta_root):
            raise FileNotFoundError(
                f"MSLS metadata root not found: {meta_root}\nExpected the source "
                f"distribution's train_val/ (holding <city>/<split>/postprocessed.csv). "
                f"Set it with --msls-meta / cfg.msls_meta_root."
            )
        city_list = cities or sorted(
            d.name
            for d in os.scandir(meta_root)
            if d.is_dir() and os.path.isdir(os.path.join(d.path, splits[0]))
        )
        if exclude_cities:
            present = {c.lower(): c for c in city_list}
            drop = {c.strip().lower() for c in exclude_cities if c.strip()}
            # A typo here would silently leave the validation cities in the training
            # mixture, which invalidates every number the monitor produces — so it is an
            # error, not a warning.
            unknown = sorted(drop - set(present))
            if unknown:
                raise ValueError(
                    f"[msls] cannot exclude {unknown}: no such city under {meta_root} "
                    f"(available: {sorted(present.values())})"
                )
            city_list = [c for c in city_list if c.lower() not in drop]
            print(
                f"[msls] HELD OUT of training: {sorted(present[d] for d in drop)} "
                f"-> {len(city_list)} training cities"
            )

        # Unconditional, unlike exclude_cities: the evaluation's test cities must never
        # train, and no flag combination should be able to let them.
        leaked = sorted(c for c in city_list if c.lower() in MSLS_TEST_CITIES)
        if leaked:
            raise ValueError(
                f"[msls] benchmark test cities present in the training metadata tree: "
                f"{leaked}. These cities are scored by the evaluation and must never "
                f"train; {meta_root} appears to mix the test split into train_val."
            )

        index, stats = [], {"rows": 0, "missing": 0, "night": 0, "merged": 0, "sep": 0}
        for ci, city in enumerate(city_list):
            per_split = {}
            for split in splits:
                csv_path = os.path.join(meta_root, city, split, "postprocessed.csv")
                if not os.path.exists(csv_path):
                    continue
                df = _read_msls_csv(csv_path, city, split)
                if exclude_night and "night" in df.columns:
                    n0 = len(df)
                    df = df[~df["night"].astype(str).str.lower().isin(("true", "1"))]
                    stats["night"] += n0 - len(df)
                per_split[split] = df
                stats["rows"] += len(df)
            if not per_split:
                continue

            merge = _msls_should_merge(per_split) if merge_splits == "auto" else bool(merge_splits)
            stats["merged" if merge else "sep"] += 1

            groups = defaultdict(list)
            for split, df in per_split.items():
                img_dir = os.path.join(images_dir, city, split, "images")
                view = _msls_view_bin(df, view_bins)
                for key, cluster, vb in zip(df["key"], df["unique_cluster"], view):
                    p = os.path.join(img_dir, f"{key}{ext}")
                    if require_exists and not os.path.exists(p):
                        stats["missing"] += 1
                        continue
                    # split is part of the class key only when the id spaces are disjoint
                    groups[(cluster, vb) if merge else (split, cluster, vb)].append(p)

            for gi, (_, ps) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
                if len(ps) < min_img_per_place:
                    continue
                ps = sorted(ps)
                if max_img_per_place:
                    ps = ps[:max_img_per_place]
                index.append((place_offset + ci * 1_000_000 + gi, ps))

        n_imgs = sum(len(p) for _, p in index)
        print(
            f"[msls] {len(city_list)} cities, {stats['rows']} metadata rows -> "
            f"{len(index)} places / {n_imgs} images (>= {min_img_per_place}/place); "
            f"splits merged in {stats['merged']} cities, kept separate in {stats['sep']}"
        )
        if stats["missing"]:
            print(f"[msls] {stats['missing']} keys had no converted image (partial conversion?)")
        if exclude_night and stats["night"]:
            print(f"[msls] excluded {stats['night']} night images")
        if not index:
            raise RuntimeError(
                f"[msls] built an empty index. Check that {images_dir} really holds "
                f"<city>/<split>/images/<key>{ext} matching the keys in {meta_root}."
            )
        return index

    return cached_index("msls", params, _build, cache_dir, refresh)


# Two traversals' cluster centroids this close (metres) are taken to be the same place.
_MSLS_MERGE_TOL_M = 25.0


def _msls_view_bin(df, view_bins):
    """``view_direction`` -> a small integer bin, whatever its dtype.

    Released as either a compass angle or a category depending on the MSLS version, so
    numeric values are binned into ``view_bins`` sectors and anything else is used as-is.
    ``view_bins <= 1`` disables the split entirely (place = cluster only).
    """
    if "view_direction" not in df.columns or view_bins <= 1:
        return [0] * len(df)
    col = df["view_direction"]
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().mean() > 0.5:
        return ((num.fillna(0) % 360) // (360.0 / view_bins)).astype(int).tolist()
    return col.astype(str).tolist()


def _msls_should_merge(per_split):
    """Do database and query share one ``unique_cluster`` id space?

    Merging traversals is the point of MSLS (same place, different season/time), but only
    if a cluster id means the same location in both. Compares centroids of the shared ids
    instead of trusting the convention.
    """
    if len(per_split) < 2:
        return False
    cents = []
    for df in per_split.values():
        if not {"easting", "northing", "unique_cluster"} <= set(df.columns):
            return False
        cents.append(df.groupby("unique_cluster")[["easting", "northing"]].mean())
    shared = cents[0].index.intersection(cents[1].index)
    if len(shared) < 5:
        return False
    d = np.hypot(*(cents[0].loc[shared].values - cents[1].loc[shared].values).T)
    return float(np.median(d)) <= _MSLS_MERGE_TOL_M


def build_megascenes_index(
    images_dir,
    ext=".png",
    min_img_per_place=4,
    place_offset=0,
    max_img_per_place=64,
    refresh=False,
    **_,
):
    """MegaScenes: class = reconstruction, i.e. ``<AAA>/<BBB>/<Scene>/<recon>/``.

    Landmark photo collections, so classes are wildly uneven (many have 1-3 images and
    are dropped by ``min_img_per_place``); ``max_img_per_place`` keeps a huge landmark
    from dominating. Co-visibility quadruplet mining is a later refinement (needs COLMAP).
    """
    return build_dir_class_index(
        images_dir,
        "megascenes",
        ext,
        min_img_per_place,
        place_offset,
        max_img_per_place,
        refresh=refresh,
    )


# Place-id stride per scan, so a sub-classed place id still names its scan (see
# scannet_scan_of). 1000 chunks/scan is far above anything a DSLR capture produces, and
# 1006 scans * 1000 stays well inside ScanNet's 100M place-id block.
_SCANNET_CHUNKS_PER_SCAN = 1000


def scannet_scan_of(place_id, place_offset=500_000_000):
    """Recover the scan index a sub-classed ScanNet place belongs to.

    Mirrors ``sf_xl_group_of``: the scan is a pure function of the id, so a future sampler
    can keep two chunks of the *same* scan out of one sub-batch (they are the residual
    false-negative risk of sub-classing) without carrying a side-map through the cache.
    """
    return (int(place_id) - int(place_offset)) // _SCANNET_CHUNKS_PER_SCAN


def _contiguous_runs(paths, chunk, gap, min_len):
    """Split a capture-ordered path list into runs of ``chunk``, dropping ``gap`` between.

    The gap is what keeps *adjacent* classes from being consecutive frames: without it,
    the last frame of run i and the first of run i+1 are neighbouring views labelled as
    different places, which is a false negative handed straight to the loss.
    """
    stride = chunk + gap
    runs = [paths[s : s + chunk] for s in range(0, len(paths), stride)]
    return [r for r in runs if len(r) >= min_len]  # trailing short run drops out


def build_scannet_index(
    images_dir,
    ext=".png",
    min_img_per_place=4,
    place_offset=0,
    max_img_per_place=64,
    chunk=0,
    gap=0,
    refresh=False,
    **_,
):
    """ScanNet(++): class = scan/scene ``<scan_id>/dslr/``, or a *run within* one scan.

    ``chunk=0`` (default) keeps the original organization: one whole scan = one place.
    That is a coarse approximation of MegaLoc, which mines pose/depth co-visibility
    quadruplets (<10 m, <30 deg) it can only build from the poses — absent from the
    converted tree. Measured cost of the approximation: ``loss/scannet`` plateaus ~.50
    while ``gsv_cities`` reaches ~.25, because a random K-draw from a whole room can share
    no visual overlap, so the positives are not learnable from the pixels.

    ``chunk>0`` **sub-classes** each scan into contiguous runs of ``chunk`` frames
    separated by ``gap`` dropped frames, i.e. place = one sub-trajectory of the capture.
    This assumes sorted filename order tracks capture order (true for ScanNet++ DSLR:
    ``DSC000NN``) — the same assumption the ``sorted(paths)[:max]`` head-truncation
    already makes. Two effects, both wanted:

      * **positives overlap.** Frames a few apart in one capture see the same surfaces,
        which is what the co-visibility criterion was buying.
      * **the stream stops recycling.** 1006 scans is 31 batches/epoch at P=32, so ScanNet
        was re-shown ~1290x over a 40k-step run against MegaScenes' ~17. Sub-classing
        multiplies the place count by the runs per scan and lands it in the same band as
        the other streams.

    It also uses far more of the data: ``chunk/(chunk+gap)`` of every scan rather than the
    first ``max_img_per_place`` frames. ``max_img_per_place`` is ignored when chunking —
    ``chunk`` is the cap.

    Residual risk: two runs of one scan can still co-occur in a sub-batch as separate
    classes. ``gap`` makes them non-adjacent; ``scannet_scan_of`` is there if that ever
    needs enforcing with a sampler.
    """
    if not chunk:
        return build_dir_class_index(
            images_dir,
            "scannet",
            ext,
            min_img_per_place,
            place_offset,
            max_img_per_place,
            refresh=refresh,
        )
    if chunk < min_img_per_place:
        raise ValueError(
            f"scannet chunk={chunk} < min_img_per_place={min_img_per_place}: "
            f"every run would be dropped"
        )

    params = {
        "root": images_dir,
        "ext": ext,
        "min": min_img_per_place,
        "chunk": chunk,
        "gap": gap,
        "offset": place_offset,
    }

    def _build():
        scans = defaultdict(list)
        for p in _walk_images(images_dir, ext):
            scans[os.path.dirname(p)].append(p)

        index, per_scan, n_avail, dropped, clipped = [], [], 0, 0, 0
        for si, (_, ps) in enumerate(sorted(scans.items())):
            n_avail += len(ps)
            runs = _contiguous_runs(sorted(ps), chunk, gap, min_img_per_place)
            if not runs:
                dropped += 1
                continue
            if len(runs) > _SCANNET_CHUNKS_PER_SCAN:
                clipped += 1
                runs = runs[:_SCANNET_CHUNKS_PER_SCAN]
            for ci, run in enumerate(runs):
                index.append((place_offset + si * _SCANNET_CHUNKS_PER_SCAN + ci, run))
            per_scan.append(len(runs))

        n_imgs = sum(len(p) for _, p in index)
        per_scan.sort()
        mid = per_scan[len(per_scan) // 2] if per_scan else 0
        print(
            f"[scannet] {len(scans)} scans -> {len(index)} places / {n_imgs} images "
            f"(sub-classed: runs of {chunk}, gap {gap}); runs/scan min {per_scan[0] if per_scan else 0} "
            f"median {mid} max {per_scan[-1] if per_scan else 0}; "
            f"using {100 * n_imgs / max(n_avail, 1):.0f}% of the {n_avail} images on disk"
        )
        if dropped:
            print(f"[scannet] dropped {dropped} scans with no run of >= {min_img_per_place}")
        if clipped:
            print(
                f"[scannet] {clipped} scans had > {_SCANNET_CHUNKS_PER_SCAN} runs — clipped "
                f"(raise _SCANNET_CHUNKS_PER_SCAN if this is not rare)"
            )
        return index

    return cached_index("scannet", params, _build, refresh=refresh)
