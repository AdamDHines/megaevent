"""Driving LENS v2 descriptor extraction from megaevent's own environment.

LENS is a 111,694-parameter spiking conv net that deploys on a SynSense Speck2f, so it
needs ``sinabs`` (and, for the chip-faithful path, ``samna``) to build a ``DynapcnnNetwork``
and read its int8 layers back out. megaevent's environment has neither, and vendoring them
would pin its torch. So the forward pass runs in LENS's own pixi environment and this
module is the seam — the same shape as :mod:`src.spikevpr_bridge`::

    resolve checkpoint -> JSON job -> `pixi run --manifest-path <LENSV2>/pixi.toml -e cuda`
        -> [N, 1024] bank

Three things are decided here and only here.

**fp32 or int8, and it is not a tax.** ``load_backbone`` gives the trained fp32 weights;
``build_chip_sim`` gives what actually deploys. On Brisbane the int8 network *improves*
R@1 (sunset1 60.8 -> 67.1) and nearly doubles descriptor spike count (1011 -> 1954), so the
two are different models rather than one model measured twice. Which one produced a bank is
recorded in the job, the method's ``meta`` and its artifact tag.

**The renderer is LENS's, not** :mod:`src.npzdata`. Every other method here shares
``src/npzdata.py`` precisely so the environments cannot disagree about what a frame is, and
breaking that rule needs a reason. LENS's is that its input is ``(2, 128, 128)`` **event
counts** and its IAF threshold is calibrated against them: ``load_onoff`` resamples to
(260, 346), so reaching 128 through it resamples twice and rescales every bin, which moves
the operating point of a network whose first spiking layer fires on absolute counts. LENS
therefore decodes each ``.npz`` once, natively, with ``lens.src.i2e.load_event_frame`` —
the same call its training loader makes. The renderer differing is the point of the
comparison, not a deviation from it.

**Batch size is part of the contract.** sinabs' stateful layers make the batch dimension
visible to the network (TASKS.md: 64 and 128 are different models), so it is pinned here,
carried in the job, and folded into the tag — a database bank and a query bank extracted at
different batch sizes are not comparable.

Density note, because it is the first thing to check when a number looks wrong: LENS trains
on I2E frames at ~48 events/px on its own 128x128 grid. Tokyo 24/7 renders 37-45 there, so
it is **in distribution**; a Brisbane 50 ms slice renders 3.3, which is 15x sparser. That is
the opposite of SpikeVPR's exposure (see :class:`src.methods.SpikeVPRMethod`), and it is why
LENS's weak axis is Brisbane's cross-condition gallery rather than Tokyo's density.

See ``scripts/lens_extract.py`` for the other end.
"""

import hashlib
import json
import os
import subprocess

import numpy as np
from loguru import logger

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
EXTRACTOR = os.path.join(REPO_ROOT, "scripts", "lens_extract.py")

#: LENS's checkout. The extractor imports ``lens.*`` from here and runs under its manifest.
DEFAULT_LENS_REPO = "/home/adam/repo/LENSV2"
#: pixi environment inside that manifest. The default env lacks the GPU stack.
LENS_ENV = "cuda"

#: Named checkpoints, so a published number traces to a model rather than to a path that
#: could be replaced. Anything else is passed through as a filesystem path.
LENS_CHECKPOINTS = {
    "v2_best": os.path.join(DEFAULT_LENS_REPO, "lens", "models", "lens_v2_best.pth"),
}

#: (C, H, W). Not a choice: BACKBONE_SPEC is budgeted for the Speck2f cores at this input,
#: and the 16x8x8 descriptor geometry falls out of it.
INPUT_SHAPE = (2, 128, 128)
DESC_DIM = 1024
#: Pinned; see the module docstring. Also what every published LENS eval used.
BATCH_SIZE = 64
# Decoding, not the forward pass, is the bottleneck, and it is worth oversubscribing the
# CPU for: a Tokyo .npz carries ~0.9M events, and on an 8-core box 1024 database frames
# take 26 s at 8 workers against 14 s at 24 (39 -> 73 frames/s). eventcv's rebin is Rust
# and releases the GIL, so more workers than cores still queues useful work.
WORKERS = 24


def resolve_checkpoint(model, lens_repo=DEFAULT_LENS_REPO):
    """``path`` for a checkpoint name or path — the one place the mapping is decided."""
    path = LENS_CHECKPOINTS.get(model, model)
    if model in LENS_CHECKPOINTS and lens_repo != DEFAULT_LENS_REPO:
        path = os.path.join(lens_repo, os.path.relpath(path, DEFAULT_LENS_REPO))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no LENS checkpoint at {path!r} (from --lens-model {model!r}); pass a path, "
            f"or one of {sorted(LENS_CHECKPOINTS)}")
    return os.path.abspath(path)


def checkpoint_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def describe(path):
    """``{'ann', 'spike_threshold', 'step'}`` read straight out of the checkpoint.

    Provenance that must not be re-derived from flags. ``chip_sim.load_backbone`` trusts
    the checkpoint's own record of how it was trained, and a threshold taken from a CLI
    default silently produces a different descriptor — 1.0 starves this network outright
    (37/64 units dead, loss frozen at 1.047), and the sweep arms carry 0.35/0.5/0.75.
    Recorded in ``meta`` so a published number names the model it came from.
    """
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"{path!r} is not a LENS v2 training checkpoint (no 'model' key)")
    return {"ann": bool(state.get("ann", False)),
            "spike_threshold": float(state.get("spike_threshold", 1.0)),
            "step": int(state.get("step", -1))}


def make_job(checkpoint, out, *, quantise="chip", batch_size=BATCH_SIZE, workers=WORKERS,
             lens_repo=DEFAULT_LENS_REPO, label="", **extra):
    """The fields every job carries, plus the mode-specific ones passed through."""
    if quantise not in ("fp32", "chip"):
        raise ValueError(f"quantise must be 'fp32' or 'chip', got {quantise!r}")
    return {"checkpoint": os.path.abspath(checkpoint), "quantise": quantise,
            "out": os.path.abspath(out), "input_shape": list(INPUT_SHAPE),
            "batch_size": int(batch_size), "workers": int(workers), "label": label,
            "lens_repo": os.path.abspath(lens_repo), **extra}


def npz_job(paths, **kwargs):
    """A bank with one row per ``.npz``, in the order given — image sets."""
    return make_job(mode="npz", paths=[os.path.abspath(p) for p in paths], **kwargs)


def traverse_job(recording, sensor, dt_ms, offset_ms, hot_pixel, filter_dt_us, **kwargs):
    """A bank with one row per ``dt_ms`` slice of one recording — pooled traverses.

    ``filter_dt_us`` is eventcv's background-activity window in **microseconds**, its raw
    timestamp unit. Passing a millisecond value asks for a 1000x shorter correlation window
    and quietly discards most active pixels; the caller converts.
    """
    return make_job(mode="traverse", recording=os.path.abspath(recording),
                    sensor=list(sensor), dt_ms=int(dt_ms), offset_ms=int(offset_ms),
                    hot_pixel=bool(hot_pixel),
                    filter_dt_us=None if filter_dt_us is None else int(filter_dt_us),
                    **kwargs)


def run(job, lens_repo=DEFAULT_LENS_REPO, env=LENS_ENV, keep_bank=False):
    """Run one job in LENS's environment. -> ``[N, 1024]`` float32.

    The child's stdout is relayed line by line through ``loguru``, so its progress lands in
    the same run log as everything else. The job file is written beside the bank and kept
    if the run fails, which is usually all that is needed to see why.
    """
    out = job["out"]
    job_path = out + ".job.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(job_path, "w") as handle:
        json.dump(job, handle, indent=2)

    manifest = os.path.join(job.get("lens_repo", lens_repo), "pixi.toml")
    command = ["pixi", "run", "--manifest-path", manifest, "-e", env,
               "python", "-u", EXTRACTOR, "--job", job_path]
    logger.info(f"lens: {os.path.basename(job['checkpoint'])} ({job['quantise']}) "
                f"-> {os.path.basename(out)}")
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        # Matches the spikevpr bridge: CONDA_OVERRIDE_CUDA is what lets pixi resolve the
        # CUDA environment on this machine, and MPLCONFIGDIR keeps a read-only HOME from
        # failing the import chain.
        env={**os.environ, "CONDA_OVERRIDE_CUDA": "12", "MPLCONFIGDIR": "/tmp"})
    for line in process.stdout:
        line = line.rstrip()
        if line:
            logger.info(f"  | {line}")
    code = process.wait()
    if code != 0:
        raise RuntimeError(
            f"LENS extraction failed (exit {code}). The job is kept at {job_path}; "
            f"reproduce it with:\n  CONDA_OVERRIDE_CUDA=12 {' '.join(command)}")

    if not os.path.exists(out):
        raise RuntimeError(f"extraction reported success but wrote no bank at {out}")
    bank = np.load(out)
    # Width comes from the checkpoint, not from DESC_DIM: `--descriptor_channels 64` gives a
    # 4096-D readout on the same seven cores, and hardcoding 1024 here rejected a perfectly
    # good 4096-D bank *after* paying 24 minutes to extract it.
    expected = job.get("descriptor_dim") or DESC_DIM
    if bank.ndim != 2 or bank.shape[1] != expected:
        raise ValueError(f"{out} holds {bank.shape}; expected [N, {expected}]")
    os.remove(job_path)
    if not keep_bank:
        # src.scoring.cached_array persists the canonical copy keyed on the path manifest;
        # keeping this one too would double every bank on disk.
        os.remove(out)
    return bank.astype(np.float32, copy=False)
