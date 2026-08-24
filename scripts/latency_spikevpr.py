"""SpikeVPR's single-query latency, measured inside ``envs/spikevpr``.

Not invoked by hand — ``scripts/latency.py`` launches this through
``pixi run --manifest-path envs/spikevpr`` and merges the JSON it writes, the same seam
``src/spikevpr_bridge.py`` uses for extraction. SpikeVPR needs ``spikingjelly``, which
megaevent's environment does not have, so its forward pass cannot be timed in-process
alongside the others.

Stages and protocol are ``scripts/latency.py``'s, so the row lands in that table unchanged:
one 50 ms slice, background-activity filter at 50,000 us, ON/OFF counts at 260x346 through
``src/npzdata.onoff_from_stream`` (the *shared* renderer — the whole point of the two-process
arrangement is that the two environments cannot disagree about what an input frame is),
batch 1, median of ``--repeats`` distinct slices, and a 4096-D match against the same
83,952-row gallery.

``read`` and ``render`` are measured here rather than taken from the parent because SpikeVPR's
representation is its own: a ``[2, 260, 346]`` ON/OFF count pair resized in the event domain,
not the ``[3, 346, 260]`` countmask the megaevent family renders.
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

DT_MS = 50
FILTER_DT_US = 50000
GRID = (260, 346)                   # (H, W); spikevpr.models.factory hardcodes MixVPR for it


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recording", required=True)
    ap.add_argument("--sensor", type=int, nargs=2, required=True, metavar=("W", "H"))
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--gallery", type=int, default=83952)
    # Resolved by the parent, which owns SPIKEVPR_CHECKPOINTS: IFNode and LIFNode are both
    # parameter-free, so a checkpoint loaded under the wrong one succeeds silently and returns
    # a different descriptor. Deciding it in two places is how that goes wrong.
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--neuron", required=True, choices=["IFNode", "LIFNode"])
    ap.add_argument("--encoder", default="sew_resnet34")
    ap.add_argument("--out-channels", type=int, default=512)
    ap.add_argument("--out-rows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    cli = ap.parse_args()

    try:
        import eventcv as ecv
    except ImportError:
        sys.path.insert(0, "/home/adam/repo/eventcv/python")
        import eventcv as ecv
    sys.path.insert(0, os.path.join(REPO, "SpikeVPR", "src"))

    # Only the shared renderer comes from the parent repo — `src.spikevpr_bridge` pulls in
    # loguru, which this environment does not have, and that is the point of the split.
    from src.npzdata import onoff_from_stream
    from spikevpr.models.factory import build_spikevpr

    device = torch.device("cuda")
    model = build_spikevpr(cli.encoder, out_channels=cli.out_channels,
                           out_rows=cli.out_rows, neuron_type=cli.neuron,
                           checkpoint=cli.checkpoint, device=device, eval_mode=True)
    params = sum(q.numel() for q in model.parameters())

    raw = ecv.open(cli.recording, dt_ms=DT_MS, sensor_size=tuple(cli.sensor),
                   hot_pixel_filter=True, offset=0)
    n_slices = int(raw.n_slices)
    rng = np.random.default_rng(cli.seed)
    idx = rng.choice(n_slices, 2 * cli.repeats + cli.warmup, replace=False)
    warm = idx[:cli.warmup]
    meas = idx[cli.warmup:cli.warmup + cli.repeats]

    # Stage 1: read. The BA filter is chained on the reader, as the extractor builds it.
    filtered = ecv.open(cli.recording, dt_ms=DT_MS, sensor_size=tuple(cli.sensor),
                        hot_pixel_filter=True, offset=0
                        ).background_activity_filter(FILTER_DT_US)
    for i in meas:
        filtered[int(i)]
    read_ms, streams = [], {}
    for i in meas:
        i = int(i)
        t = time.perf_counter()
        s = filtered[i]
        read_ms.append((time.perf_counter() - t) * 1e3)
        streams[i] = s

    # Stage 2: render, from streams already resident.
    render_ms, frames = [], {}
    for i, s in streams.items():
        t = time.perf_counter()
        f = onoff_from_stream(s, size=GRID)
        render_ms.append((time.perf_counter() - t) * 1e3)
        frames[i] = np.asarray(f)

    dim = cli.out_channels * cli.out_rows
    gallery = torch.nn.functional.normalize(
        torch.randn(cli.gallery, dim, device=device), dim=1)
    st = {k: [] for k in ("tensor", "encode", "match")}
    order = list(warm[:5]) + list(meas)
    for j, i in enumerate(order):
        frame = frames[int(i)] if int(i) in frames else frames[int(meas[0])]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        x = torch.from_numpy(np.ascontiguousarray(frame)).float()
        t2 = time.perf_counter()
        with torch.no_grad():
            # SpikeVPR is stateful: the spiking neurons carry membrane potential between
            # calls, so the state is reset per query exactly as the extractor does it.
            try:
                from spikingjelly.activation_based import functional
                functional.reset_net(model)
            except Exception:
                pass
            q = model(x.unsqueeze(0).to(device, non_blocking=True))
        q = torch.nn.functional.normalize(q.float().reshape(1, -1), dim=1)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        (gallery @ q.T).squeeze(1).argmax(0)
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        if j >= 5:
            st["tensor"].append((t2 - t1) * 1e3)
            st["encode"].append((t3 - t2) * 1e3)
            st["match"].append((t4 - t3) * 1e3)

    med = {k: statistics.median(v) for k, v in st.items()}
    med.update(read=statistics.median(read_ms), render=statistics.median(render_ms),
               rerank=0.0, params=params, desc_dim=dim,
               rep="onoff", label=f"SpikeVPR ({cli.encoder}+MixVPR)")
    med["compute"] = med["render"] + med["tensor"] + med["encode"] + med["match"]
    med["total"] = med["compute"] + med["read"]
    with open(cli.out_json, "w") as h:
        json.dump(med, h, indent=2)
    print(f"  SpikeVPR {params:,} params, {dim}-D, neuron {cli.neuron}")


if __name__ == "__main__":
    main()
