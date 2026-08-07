"""Driving SpikeVPR's descriptor extraction from megaevent's own environment.

SpikeVPR needs `spikingjelly`, which megaevent's environment does not have, and the
vendored ``SpikeVPR/`` clone that does ships a CPU-only PyTorch — 4.3 frames/s against 680
on the GPU, which is the difference between twenty minutes and fifty hours for this
benchmark. So the forward pass runs in a third environment (``envs/spikevpr/``) and this
module is the seam:

    resolve model -> JSON job -> `pixi run --manifest-path envs/spikevpr` -> [N, D] bank

Everything the child needs is a plain value in the job, so it imports nothing from
megaevent except ``src/npzdata.py`` — the shared renderer, which is exactly the thing that
must not be duplicated. See ``scripts/spikevpr_extract.py`` for the other end.

The **neuron type is resolved here and only here**. ``IFNode`` and ``LIFNode`` are both
parameter-free, so loading a checkpoint under the wrong one succeeds silently and returns a
different descriptor — there is nothing downstream that could catch it. :data:`SPIKEVPR_CHECKPOINTS`
is transcribed from ``SpikeVPR/src/weights/MANIFEST.md`` and the resolved value is recorded
in the job, the method's ``meta`` and its artifact tag, so a published number can always be
traced back to the pairing that produced it.
"""

import hashlib
import json
import os
import subprocess

import numpy as np
from loguru import logger

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
EXTRACTOR = os.path.join(REPO_ROOT, "scripts", "spikevpr_extract.py")
DEFAULT_REPO = os.path.join(REPO_ROOT, "SpikeVPR")
DEFAULT_ENV = os.path.join(REPO_ROOT, "envs", "spikevpr")
# eventcv is not a dependency of the extraction environment; it is reached by sys.path, the
# way src/methods.py reaches EventVLAD's networks. Mirrors the editable path in pixi.toml.
EVENTCV_PYTHON = "/home/adam/repo/eventcv/python"

# checkpoint stem -> (weights file, MixVPR neuron type). From SpikeVPR/src/weights/MANIFEST.md.
SPIKEVPR_CHECKPOINTS = {
    "brisbane": ("sew_resnet34_brisbane.pth", "LIFNode"),
    "nsavp": ("sew_resnet34_nsavp.pth", "LIFNode"),
    "nyc": ("sew_resnet34_nyc.pth", "IFNode"),
}
# (H, W). Not really a choice: spikevpr.models.factory hardcodes MixVPR for the (512, 9, 11)
# feature map a (2, 260, 346) frame produces, so every dataset has to meet here.
GRID = (260, 346)
ENCODER = "sew_resnet34"
OUT_CHANNELS, OUT_ROWS = 512, 8         # 512 * 8 = the 4096-D descriptor every ckpt ships
BATCH_SIZE = 32                         # ~2.5 GB peak on an 8 GB card at 260x346
WORKERS = 8                             # rendering, not the forward pass, is the bottleneck


def resolve_checkpoint(model, repo=DEFAULT_REPO):
    """``(path, neuron)`` for a checkpoint stem — the one place the pairing is decided."""
    if model not in SPIKEVPR_CHECKPOINTS:
        raise ValueError(f"unknown SpikeVPR checkpoint '{model}'; choose from "
                         f"{sorted(SPIKEVPR_CHECKPOINTS)}")
    name, neuron = SPIKEVPR_CHECKPOINTS[model]
    path = os.path.join(repo, "src", "weights", name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no SpikeVPR checkpoint at {path}. Fetch the weights with "
            f"{os.path.join(repo, 'src', 'weights', 'download_weights.sh')}.")
    return path, neuron


def checkpoint_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_job(checkpoint, neuron, out, *, max_events=None, batch_size=BATCH_SIZE,
             workers=WORKERS, spikevpr_repo=DEFAULT_REPO, label="", **extra):
    """The fields every job carries, plus the mode-specific ones passed through."""
    return {"checkpoint": os.path.abspath(checkpoint), "neuron": neuron,
            "encoder": ENCODER, "out_channels": OUT_CHANNELS, "out_rows": OUT_ROWS,
            "out": os.path.abspath(out), "size": list(GRID), "max_events": max_events,
            "batch_size": int(batch_size), "workers": int(workers), "label": label,
            "spikevpr_repo": os.path.abspath(spikevpr_repo),
            "eventcv_path": EVENTCV_PYTHON, **extra}


def npz_job(paths, **kwargs):
    """A bank with one row per ``.npz``, in the order given — image sets."""
    return make_job(mode="npz", paths=[os.path.abspath(p) for p in paths], **kwargs)


def traverse_job(recording, sensor, dt_ms, offset_ms, hot_pixel, filter_dt_us, **kwargs):
    """A bank with one row per ``dt_ms`` slice of one recording — pooled traverses.

    ``filter_dt_us`` is eventcv's background-activity window in **microseconds**, its raw
    timestamp unit. Passing a millisecond value here asks for a 1000x shorter correlation
    window and quietly discards most of the active pixels; the caller converts.
    """
    return make_job(mode="traverse", recording=os.path.abspath(recording),
                    sensor=list(sensor), dt_ms=int(dt_ms), offset_ms=int(offset_ms),
                    hot_pixel=bool(hot_pixel),
                    filter_dt_us=None if filter_dt_us is None else int(filter_dt_us),
                    **kwargs)


def run(job, env_dir=DEFAULT_ENV, keep_bank=False):
    """Run one job in the extraction environment. -> ``[N, D]`` float32.

    The child's stdout is relayed line by line through ``loguru``, so its progress lands in
    the same run log as everything else rather than disappearing into a pipe. The job file
    is written beside the bank and kept if the run fails, which is usually all that is
    needed to see why.
    """
    out = job["out"]
    job_path = out + ".job.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(job_path, "w") as handle:
        json.dump(job, handle, indent=2)

    command = ["pixi", "run", "--manifest-path", os.path.join(env_dir, "pixi.toml"),
               "python", "-u", EXTRACTOR, "--job", job_path]
    logger.info(f"spikevpr: {os.path.basename(job['checkpoint'])} ({job['neuron']}) "
                f"-> {os.path.basename(out)}")
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        # CONDA_OVERRIDE_CUDA is what lets pixi resolve the CUDA environment on this
        # machine; PYTHONWARNINGS silences the SyntaxWarning storm spikingjelly emits from
        # its docstrings on first import, which would otherwise bury the progress lines.
        env={**os.environ, "CONDA_OVERRIDE_CUDA": "12", "MPLCONFIGDIR": "/tmp",
             "PYTHONWARNINGS": "ignore"})
    for line in process.stdout:
        line = line.rstrip()
        if line:
            logger.info(f"  | {line}")
    code = process.wait()
    if code != 0:
        raise RuntimeError(
            f"SpikeVPR extraction failed (exit {code}). The job is kept at {job_path}; "
            f"reproduce it with:\n  CONDA_OVERRIDE_CUDA=12 {' '.join(command)}")

    if not os.path.exists(out):
        raise RuntimeError(f"extraction reported success but wrote no bank at {out}")
    bank = np.load(out)
    os.remove(job_path)
    if not keep_bank:
        # src.scoring.cached_array persists the canonical copy keyed on the path manifest;
        # keeping this one too would double ~13 GB of banks across the full benchmark.
        os.remove(out)
    return bank
