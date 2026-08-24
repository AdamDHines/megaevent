"""Export a training checkpoint to the shipping format kept in ``ckpts/``.

    pixi run python3 scripts/export_ckpt.py \
        /media/adam/vprdatasets/megaevent/runs/b_full_P64_v4_vpr/step10000.pt

A run's ``.pt`` carries the whole trainer — ``model``, ``optimizer``, ``rng``, ``scaler``,
``eval_history``, ``best`` — because it exists to be resumed from. Three quarters of that file
is Adam moment estimates that nobody downloading a place-recognition model will ever use: the
v5 ViT-B checkpoint is 2.7 GB on disk and 915 MB of it is weights. ``ckpts/s_salad_ft4.pt``,
the one export that predates this script, is ``{model, config, meta}`` and nothing else, so
that is the format reproduced here.

**The config is trimmed, deliberately.** A saved config holds ~130 entries, including
``megaloc_weights``, ``data``, ``eval_root`` and friends — absolute paths into ``/work/qvpr``
and ``/mnt/hpccs01/home/<user>`` on the HPC. Those say nothing about the weights and should not
go up to a public model host, so the export keeps exactly the keys
``src.inference.cfg_from_ckpt`` reads back (``_RESTORE``) plus the backbone-geometry fields the
existing export carries. Anything outside that set cannot affect how the model rebuilds, which
is what ``--verify`` then proves rather than assumes.

``--verify`` (on by default) rebuilds both the source and the exported file through
``inference.load_model`` and requires **bitwise-identical descriptors** on a fixed random
batch. A config key dropped by mistake would change the architecture and fail the strict
``load_state_dict`` or move a descriptor; either way the export refuses to be written.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import inference as inf  # noqa: E402

# The keys cfg_from_ckpt reads, plus the backbone-geometry fields ckpts/s_salad_ft4.pt carries.
# "vit" must survive: apply_vit() re-derives the register-token count (0 small / 4 base) and a
# wrong count fails the strict state_dict load.
KEEP = tuple(sorted(set(inf._RESTORE) | {
    "vit", "vit_backbone", "white_frame", "backbone_img_size", "n_tokens_per_image"}))


def run_name_of(ckpt_path):
    """``.../runs/b_full_P64_v4_vpr/step10000.pt`` -> ``b_full_P64_v4``."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
    return parent[:-4] if parent.endswith("_vpr") else parent


def export(ckpt_path, out_path, verify=True, device=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise SystemExit(f"{ckpt_path}: no 'model' state dict — not a trainer checkpoint")
    saved = ck.get("config", {})
    config = {k: saved[k] for k in KEEP if k in saved}
    dropped = sorted(set(saved) - set(config))
    step = int(ck.get("step", ck.get("meta", {}).get("step", -1)))
    payload = {"model": ck["model"],
               "config": config,
               "meta": {"run_name": run_name_of(ckpt_path), "step": step}}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)

    src_gb = os.path.getsize(ckpt_path) / 2 ** 30
    out_gb = os.path.getsize(out_path) / 2 ** 30
    print(f"  {run_name_of(ckpt_path)} step {step}: {src_gb:.2f} GB -> {out_gb:.2f} GB "
          f"({len(ck['model'])} tensors, {len(config)} config keys, {len(dropped)} dropped)")

    if verify:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        a, cfg_a, step_a = inf.load_model(ckpt_path, device)
        b, cfg_b, step_b = inf.load_model(out_path, device)
        if (cfg_a.H, cfg_a.W, cfg_a.desc_dim) != (cfg_b.H, cfg_b.W, cfg_b.desc_dim):
            raise SystemExit(f"{out_path}: geometry changed on export")
        if step_a != step_b:
            raise SystemExit(f"{out_path}: step {step_b} != source {step_a}")
        torch.manual_seed(0)
        x = torch.randn(2, 3, cfg_a.H, cfg_a.W, device=device)
        with torch.no_grad():
            da, db = a(x), b(x)
        if not torch.equal(da, db):
            raise SystemExit(f"{out_path}: descriptors differ from the source checkpoint "
                             f"(max |d| {(da - db).abs().max().item():.3e})")
        print(f"    verified: identical {tuple(da.shape)} descriptors, step {step_b}")
        del a, b
        torch.cuda.empty_cache()
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt", nargs="+", help="training checkpoint(s), or LABEL=PATH to rename")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ckpts"))
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the descriptor-equality check (it needs a GPU pass)")
    cli = ap.parse_args()

    for spec in cli.ckpt:
        label, _, path = spec.rpartition("=")
        if not label:
            label = f"{run_name_of(path)}_step{torch.load(path, map_location='cpu', weights_only=False).get('step', 0)}"
        export(path, os.path.join(cli.out_dir, f"{label}.pt"), verify=not cli.no_verify)


if __name__ == "__main__":
    main()
