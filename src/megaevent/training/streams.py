"""Six-stream v8 data mixture and training normalization statistics."""

import json
import os
import random
from dataclasses import dataclass
from typing import Callable

from . import dataset as ds


@dataclass
class StreamSpec:
    """How to find and build one MegaLoc sub-batch stream."""

    name: str
    subdirs: tuple  # candidate dirs under cfg.data_root, first match wins
    builder: Callable  # (root, cfg, spec) -> place index
    place_offset: int  # keeps place ids from colliding across streams
    subpath: str = ""  # extra path under <subdir>/jpg (e.g. SF-XL's "train")
    ext: str = ""  # per-stream image extension; "" -> cfg.image_ext
    optional: bool = False  # missing data -> warn instead of raise
    group_cycled: bool = False  # SF-XL: cycle EigenPlaces groups per epoch (see dataset.py)


def _image_root(cfg, spec):
    """I2E writes ``<out>/jpg/<relpath>``; accept the bare dir as a fallback.

    ``spec.subpath`` narrows it further — SF-XL's tree is ``jpg/{train,test,val}`` and
    only ``train`` is training data. Pointing at ``jpg/`` would walk the test/val
    database and query images too: they are a different filename format (the heading
    field is empty), so they would be silently skipped after a large wasted walk.
    """
    for sub in spec.subdirs:
        for tail in ("jpg", ""):
            base = (
                os.path.join(cfg.data_root, sub, tail) if tail else os.path.join(cfg.data_root, sub)
            )
            p = os.path.join(base, spec.subpath) if spec.subpath else base
            if os.path.isdir(p):
                return p
    return None


def _ext(cfg, spec):
    """The image extension for one stream.

    Per-stream by default, because the converted trees are mixed (``.jpg`` for
    gsvcities/sf_xl/mapillary, ``.png`` for megascenes/scannet). ``cfg.force_ext``
    overrides every stream at once — the switch to flip once the B1 re-render has made the
    whole tree PNG. ``ext`` is part of every builder's cache-key params, so changing it
    invalidates the stale ``.vpr_index_*.json`` files instead of silently reusing them.
    """
    return getattr(cfg, "force_ext", None) or spec.ext or cfg.image_ext


def _build_gsv(root, cfg, spec):
    """GSV-Cities: place id is field 1 of the filename; classes are already disjoint."""
    ext = _ext(cfg, spec)
    params = {"root": root, "ext": ext, "min": cfg.K_images, "offset": spec.place_offset}
    return ds.cached_index(
        spec.name,
        params,
        lambda: ds.build_gsv_cities_from_tree(root, ext=ext, min_img_per_place=cfg.K_images),
        refresh=cfg.refresh_index,
    )


def _build_sf_xl(mode):
    def _build(root, cfg, spec):
        # Cycling covers all N*N groups across epochs (one group per sub-batch); otherwise
        # honour the manual sfxl_groups subset.
        groups = tuple(range(cfg.sfxl_N**2)) if cfg.sfxl_cycle_groups else tuple(cfg.sfxl_groups)
        return ds.build_sf_xl_index(
            root,
            mode=mode,
            ext=_ext(cfg, spec),
            M=cfg.sfxl_M,
            N=cfg.sfxl_N,
            focal_dist=cfg.sfxl_focal_dist,
            groups=groups,
            min_images_per_class=cfg.sfxl_min_images_per_class,
            min_img_per_place=cfg.K_images,
            place_offset=spec.place_offset,
            refresh=cfg.refresh_index,
        )

    return _build


def _build_msls(root, cfg, spec):
    return ds.build_msls_index(
        root,
        meta_root=cfg.msls_meta_root,
        ext=_ext(cfg, spec),
        place_offset=spec.place_offset,
        min_img_per_place=cfg.K_images,
        # the in-domain validation cities must not train (i2eval.py)
        exclude_cities=getattr(cfg, "msls_exclude_cities", None),
        view_bins=cfg.msls_view_bins,
        merge_splits=cfg.msls_merge_splits,
        max_img_per_place=cfg.max_img_per_place,
        refresh=cfg.refresh_index,
    )


def _build_megascenes(root, cfg, spec):
    return ds.build_megascenes_index(
        root,
        ext=_ext(cfg, spec),
        place_offset=spec.place_offset,
        min_img_per_place=cfg.K_images,
        max_img_per_place=cfg.max_img_per_place,
        refresh=cfg.refresh_index,
    )


def _build_scannet(root, cfg, spec):
    return ds.build_scannet_index(
        root,
        ext=_ext(cfg, spec),
        place_offset=spec.place_offset,
        min_img_per_place=cfg.K_images,
        max_img_per_place=cfg.max_img_per_place,
        chunk=cfg.scannet_chunk,
        gap=cfg.scannet_gap,
        refresh=cfg.refresh_index,
    )


# Layout verified by src/vpr/survey_data.py against /work/qvpr/data/processed/EventGeM.
# Place-id offsets are 100M apart: far above any within-dataset id.
# NB the converted extensions differ per dataset: gsvcities/sf_xl/mapillary are .jpg,
# megascenes/scannet are .png.
REGISTRY = {
    "gsv_cities": StreamSpec("gsv_cities", ("gsvcities", "gsv-cities"), _build_gsv, 0, ext=".jpg"),
    # SF-XL: jpg/{train,test,val} — training data is train/ only.
    "sf_xl_frontal": StreamSpec(
        "sf_xl_frontal",
        ("sf_xl", "sf-xl", "sfxl"),
        _build_sf_xl("frontal"),
        100_000_000,
        subpath="train",
        ext=".jpg",
        group_cycled=True,
    ),
    "sf_xl_lateral": StreamSpec(
        "sf_xl_lateral",
        ("sf_xl", "sf-xl", "sfxl"),
        _build_sf_xl("lateral"),
        200_000_000,
        subpath="train",
        ext=".jpg",
        group_cycled=True,
    ),
    "msls": StreamSpec("msls", ("mapillary", "msls"), _build_msls, 300_000_000, ext=".jpg"),
    # class = <AAA>/<BBB>/<Scene>/<recon>/  (one reconstruction)
    "megascenes": StreamSpec(
        "megascenes", ("megascenes",), _build_megascenes, 400_000_000, ext=".png"
    ),
    # class = <scan_id>/dslr/  (one scan)
    "scannet": StreamSpec("scannet", ("scannet",), _build_scannet, 500_000_000, ext=".png"),
}

MEGALOC_STREAMS = ["sf_xl_frontal", "sf_xl_lateral", "gsv_cities", "msls", "megascenes", "scannet"]


# ---------------------------------------------------------------------------
# Normalisation stats
# ---------------------------------------------------------------------------
# The resolution the stats pass resizes to, deliberately NOT cfg.H/cfg.W. Normalisation
# constants describe the data, not the input geometry, but the cache key below covers only
# the stream set, representation and extension -- so an --img-size 322 run and a 224 run
# share one cache entry while computing different numbers from it. Sequentially that makes
# the constants depend on submission order; run the sweep concurrently and it is a race,
# landing on exactly the arm whose whole purpose is to isolate resolution. Pinning the pass
# makes every writer produce the same blob: the race becomes harmless, and an --img-size run
# differs from its baseline in one thing rather than two. 224 is what every cached entry was
# already computed at, so this is bit-identical for existing runs.
STATS_HW = (224, 224)


def resolve_stats(cfg, indices):
    """Set ``cfg.tencode_mean/std`` from the data, cached to disk.

    One model means one normalisation, so the stats are computed over a sample drawn
    across *all* active streams rather than per stream. tencode stats are nothing like
    GEPT's white-background alignment prior (measured black-bg GSV: mean
    [.298,.101,.229]), and they are baked into every checkpoint, so they must be
    settled before the transforms are built.

    Cached under ``cfg.data_root`` keyed by the stream set — recomputing a 5k-image
    pass at the start of every job in a sweep is pure waste. The pass itself runs at
    ``STATS_HW``, not at ``cfg.H/cfg.W``; see the comment on that constant.
    """
    # Key by stream set AND representation: accumulate (white-bg, ~0.9 means) and tencode
    # (black-bg, ~[.3,.1,.2]) have completely different statistics, so a stats cache that
    # ignored the rep would silently normalise one rep with the other's mean/std (PLAN 1q).
    # `v2` marks the corrected sampling (even per-stream IMAGE budgets, shuffled
    # consumption) so a blob written by the old place-capped pass is never reused, and
    # force_ext is in the key because a PNG re-render of a JPEG tree genuinely changes the
    # pixel statistics it is describing.
    rep = getattr(cfg, "representation", "tencode")
    key = "+".join(sorted(indices)) + (f"@{rep}" if rep != "tencode" else "")
    if getattr(cfg, "force_ext", None):
        key += f"@{cfg.force_ext.lstrip('.')}"
    key += "@v2"
    cache = os.path.join(cfg.data_root, "tencode_stats.json")
    if not cfg.refresh_stats and os.path.exists(cache):
        try:
            with open(cache) as f:
                blob = json.load(f)
            if key in blob:
                cfg.tencode_mean, cfg.tencode_std = blob[key]["mean"], blob[key]["std"]
                print(
                    f"[stats] cached {key}: mean={[round(m, 4) for m in cfg.tencode_mean]} "
                    f"std={[round(s, 4) for s in cfg.tencode_std]}"
                )
                return
        except (OSError, ValueError):
            pass

    # Even sample across streams, counted in IMAGES. Capping *places* evenly (what this
    # used to do) is not the same thing: a place holds ~4 images in GSV-Cities but up to
    # max_img_per_place=64 in MegaScenes/ScanNet, so equal place budgets handed those two
    # streams up to ~16x the pixel weight. And because the stats pass then stopped at
    # stats_max_images while walking a stream-ordered list, the budget was spent inside the
    # leading streams and the trailing ones could contribute ZERO pixels to constants that
    # get baked into every checkpoint. Both halves are fixed: per-stream *image* budgets
    # here, shuffled consumption inside compute_channel_stats.
    per_stream = max(1, cfg.stats_max_images // max(len(indices), 1))
    rng = random.Random(cfg.seed)
    sample, drawn = [], {}
    for name, idx in indices.items():
        # every place holds >= K_images, so this many places covers the image budget
        need = max(1, -(-per_stream // max(cfg.K_images, 1)))
        flat = [
            (pid, [p])
            for pid, ps in ds.subset_index(idx, min(need, len(idx)), seed=cfg.seed)
            for p in ps
        ]
        rng.shuffle(flat)
        sample.extend(flat[:per_stream])
        drawn[name] = min(len(flat), per_stream)
    print(
        f"[stats] sampling {sum(drawn.values())} images across {len(drawn)} stream(s): "
        + ", ".join(f"{k}={v}" for k, v in drawn.items())
    )
    mean, std = ds.compute_channel_stats(
        sample,
        hw=STATS_HW,
        max_images=cfg.stats_max_images,
        num_workers=min(cfg.n_workers, 8),
        seed=cfg.seed,
    )
    cfg.tencode_mean, cfg.tencode_std = mean, std
    print(
        f"[stats] computed {key}: mean={[round(m, 4) for m in mean]} "
        f"std={[round(s, 4) for s in std]}"
    )
    try:
        blob = {}
        if os.path.exists(cache):
            try:
                with open(cache) as f:
                    blob = json.load(f)
            except ValueError as err:
                # JSONDecodeError is a ValueError, NOT an OSError, so it used to escape the
                # handler below and kill the job -- after it had already paid for the full
                # stats pass. Losing the other keys is the cheap outcome: they are one 5k
                # image pass each to recompute.
                print(f"[stats] existing cache is unreadable ({err}) — rewriting it")
        blob[key] = {"mean": mean, "std": std}
        ds.atomic_write_json(cache, blob, indent=2)
        print(f"[stats] cached -> {cache}")
    except OSError as err:
        print(f"[stats] could not cache ({err})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def check_id_ranges(indices):
    """Measure each stream's actual place-id range and warn on cross-stream overlap.

    Overlap is harmless under the per-stream loss (labels never cross a sub-batch
    boundary) but silently poisons anything that merges streams: two colliding ids would
    make a GSV place and an SF-XL place "the same place", turning true negatives into
    false positives. This is a warning rather than an error because today's training is
    correct — but any cross-stream label use must re-mint ids first (see module docstring).
    Returns ``{name: (min_id, max_id)}`` so callers and tests can inspect the layout.
    """
    ranges = {
        name: (min(pid for pid, _ in idx), max(pid for pid, _ in idx))
        for name, idx in indices.items()
    }
    names = sorted(ranges)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            (a_lo, a_hi), (b_lo, b_hi) = ranges[a], ranges[b]
            if a_lo <= b_hi and b_lo <= a_hi:
                print(
                    f"[build_streams] WARNING: place-id ranges of {a} "
                    f"[{a_lo:,}..{a_hi:,}] and {b} [{b_lo:,}..{b_hi:,}] overlap. "
                    f"Safe for the per-stream loss; MUST be re-minted before any "
                    f"cross-stream label use."
                )
    for name in names:
        lo, hi = ranges[name]
        print(f"    [{name}] place ids [{lo:,} .. {hi:,}]")
    return ranges


def build_streams(cfg):
    """``cfg.streams`` -> ``dict[name, DataLoader]``, each yielding a P x K sub-batch."""
    names = list(cfg.streams)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown stream(s) {unknown}; known: {sorted(REGISTRY)}")

    # 1. resolve roots first, so a typo'd path fails before any expensive tree walk
    roots = {}
    for name in names:
        spec = REGISTRY[name]
        root = _image_root(cfg, spec)
        if root is None:
            msg = (
                f"stream {name!r}: none of {spec.subdirs} found under {cfg.data_root} "
                f"(run `python src/vpr/survey_data.py --root {cfg.data_root}` to see what is there)"
            )
            if spec.optional:
                print(f"[build_streams] SKIP {msg}")
                continue
            raise SystemExit(f"[build_streams] {msg}")
        roots[name] = root

    # 2. build the place indices
    indices = {}
    for name, root in roots.items():
        spec = REGISTRY[name]
        idx = spec.builder(root, cfg, spec)
        if not idx:
            raise SystemExit(f"[build_streams] stream {name!r} built an empty index from {root}")
        indices[name] = idx
    check_id_ranges(indices)

    # 3. normalisation stats — before build_transforms bakes mean/std into Normalize
    if cfg.compute_stats:
        resolve_stats(cfg, indices)

    # 4. loaders. The worker budget is shared: 6 streams x n_workers each would
    #    oversubscribe the job's CPU allocation several times over.
    per_stream_workers = max(0, cfg.n_workers // max(len(indices), 1))
    tf = ds.build_transforms(cfg, train=True)
    streams, summary = {}, []
    for name, idx in indices.items():
        spec = REGISTRY[name]
        d = ds.PlacesDataset(idx, cfg.K_images, tf)
        if spec.group_cycled and cfg.sfxl_cycle_groups:
            gpi = [
                ds.sf_xl_group_of(pid, spec.place_offset, cfg.sfxl_N) for pid, _ in d.place_index
            ]
            streams[name] = ds.make_group_cycling_loader(
                d, gpi, cfg.P_places, per_stream_workers, seed=cfg.seed
            )
            print(
                f"    [{name}] EigenPlaces group-cycling over {len(set(gpi))} groups "
                f"({len(d)} places total)"
            )
        else:
            streams[name] = ds.make_places_loader(d, cfg.P_places, per_stream_workers)
        summary.append((name, len(d), sum(len(p) for _, p in idx)))

    print(
        f"\n[build_streams] {len(streams)} stream(s), "
        f"{cfg.P_places}x{cfg.K_images}={cfg.P_places * cfg.K_images} images each "
        f"-> {len(streams) * cfg.P_places * cfg.K_images} images/step, "
        f"{per_stream_workers} worker(s)/stream"
    )
    for name, n_places, n_imgs in summary:
        print(f"    {name:<16} {n_places:>8} places  {n_imgs:>10} images")
    return streams


def stream_summary(streams):
    """Compact dict for wandb config / logging.

    ``build_streams`` prints a richer table, but that happens *before* ``wandb.init()``
    captures stdout, so it only survives in the PBS job log. This is what a finished run
    can still be audited from — hence the image counts and ``epochs_per_1k_steps``, which
    is where stream imbalance shows up (ScanNet's 1006 places recycle ~32x per 1k steps
    while MegaScenes takes ~2.4k steps to see each place once).
    """
    out = {}
    for name, loader in streams.items():
        n_batches = len(loader)
        out[name] = {
            "places": len(loader.dataset),
            "images": sum(len(ps) for _, ps in loader.dataset.place_index),
            "batches_per_epoch": n_batches,
            "epochs_per_1k_steps": round(1000 / n_batches, 2) if n_batches else None,
        }
    return out
