"""Builders for the VPR-methods-evaluation controls megaevent does not carry itself.

BoQ, QAA and SuperVLAD are implemented in ``/home/adam/repo/VPR-methods-evaluation/
vpr_models/``. They are loaded from those files **by path** rather than reimplemented here,
so a control scored by megaevent is the same network that repo evaluates.

Why by path and not ``import vpr_models``: that package's ``__init__`` imports every method
eagerly, so a single missing optional dependency breaks the import of models that do not
need it — ``supervlad`` pulls ``gdown``, which is not in this environment, and that alone
made ``import vpr_models`` fail for ``boq`` too.

**No ResizingWrapper.** The repo wraps QAA and SuperVLAD in ``ResizingWrapper(...,
"dino_v2_resize")``, which rounds each side to the nearest multiple of 14. Every model here
is evaluated at 322, and 322 = 14 x 23 exactly, so the wrapper is the identity and the
frames these models see are the ones ``megaloc_transform`` already produced. BoQ carries no
wrapper in the repo either.

Each model's descriptor width and input size are its own published evaluation configuration,
taken from ``VPR-methods-evaluation/parser.py``.
"""

import importlib.util
import os
import sys

VPRBENCH = "/home/adam/repo/VPR-methods-evaluation"

# name -> (descriptor width, square input size, provenance)
BOQ_DESC_DIM, BOQ_RESOLUTION = 12288, 322
BOQ_SRC = "amaralibey/Bag-of-Queries (DINOv2 ViT-B, 12288-d)"
QAA_DESC_DIM, QAA_RESOLUTION = 8192, 322
QAA_SRC = "xjh19972/QAA-8192 (DINOv2, 8192-d)"
SUPERVLAD_DESC_DIM, SUPERVLAD_RESOLUTION = 3072, 322
SUPERVLAD_SRC = "SuperVLAD (DINOv2 ViT-B, 3072-d)"


def _module(name):
    """One ``vpr_models/<name>.py`` loaded standalone, without its package ``__init__``."""
    path = os.path.join(VPRBENCH, "vpr_models", f"{name}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. These controls are read out of the "
            f"VPR-methods-evaluation checkout; set src/vprbench.py::VPRBENCH if it moved.")
    if VPRBENCH not in sys.path:            # some of them import siblings by absolute name
        sys.path.insert(0, VPRBENCH)
    spec = importlib.util.spec_from_file_location(f"_vprbench_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_boq():
    """BoQ on DINOv2 ViT-B — 12288-d at 322. Weights from the upstream GitHub release."""
    return _module("boq").get_boq(backbone="Dinov2",
                                  descriptors_dimension=BOQ_DESC_DIM)


def build_qaa(descriptors_dimension=QAA_DESC_DIM):
    """QAA on DINOv2 — 8192-d at 322. Weights from HuggingFace."""
    return _module("qaa").get_qaa(descriptors_dimension=descriptors_dimension)


def build_supervlad(variant="SuperVLAD"):
    """SuperVLAD on DINOv2 ViT-B — 3072-d at 322.

    Needs ``gdown`` (the weights are on Google Drive): ``pixi add gdown`` in this repo.
    ``variant="SuperVLAD-CrossImage"`` is the cross-image arm, whose descriptors depend on
    the batch the way CricaVPR's do — it must not be given an OOM batch backoff.
    """
    try:
        module = _module("supervlad")
    except ModuleNotFoundError as err:
        if "gdown" in str(err):
            raise ModuleNotFoundError(
                "SuperVLAD downloads its weights from Google Drive and needs gdown, which "
                "is not in this environment. Run `pixi add gdown` in /home/adam/repo/"
                "megaevent, then re-run.") from err
        raise
    _fetch_supervlad_weights(module, variant)
    return module.get_supervlad(variant)


def _fetch_supervlad_weights(module, variant):
    """Put the checkpoint where ``get_supervlad`` looks, so its own download never runs.

    Upstream calls ``gdown.download(..., fuzzy=True)`` to turn a Google Drive ``/view`` link
    into a download. ``fuzzy`` was removed in gdown 6, which is what this environment has, so
    that call raises ``TypeError`` before the model is ever built. Rather than patch the
    VPR-methods-evaluation checkout — which works against its own pinned gdown — the file is
    fetched here by id (accepted by every gdown version) and written to the exact relative
    path upstream checks, so its ``if not os.path.exists`` branch is skipped.
    """
    import gdown

    path = os.path.join("trained_models", "supervlad", f"{variant}.pth")
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = module.MODELS_INFO[variant][0]
    file_id = url.split("/d/")[1].split("/")[0]        # .../file/d/<id>/view
    gdown.download(id=file_id, output=path, quiet=False)
