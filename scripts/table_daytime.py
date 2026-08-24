"""Brisbane pooled with **daytime** as the query, beside the reported sunset1 row.

    pixi run python3 scripts/table_daytime.py

The reported Brisbane-Event benchmark queries `sunset1` and pools the rest. The trainer's
in-loop real metric (`eval/score_real`) is instead the mean of four *pairwise* evaluations —
sunset1, morning, daytime, sunrise each used **as the query** against the `sunset2` reference.
Decomposing it shows the early-checkpoint advantage lives almost entirely in daytime
(+0.09..+0.14 R@1) and not at all in sunset1 (-0.006..+0.008), which is why the reported row
sees roughly half the gain wandb advertises. This script runs the benchmark the other way round.

Protocol is otherwise identical to the sunset1 row: 322x322 countmask (each RGB control at its
own published size), BA filter at 50000 us, 25 m Euclidean radius, native descriptor, **no PCA**.
The database drops sunset2 exactly as the reported row does — here that is also the only
traverse with no cached bank for any method, so every cell is a re-score rather than a re-render.

Caveat worth keeping attached to these numbers: the in-loop daytime metric scored daytime
against `sunset2` alone. This pools four *other* traverses instead, so it is not the same
measurement as wandb's — it is the reported benchmark's protocol with the query swapped.
"""

import json
import os

ROOT = "/media/adam/vprdatasets/megaevent"
DAY = f"{ROOT}/brisbane_daytime"

# label -> (pretty, daytime json, key, sunset1 R@1, sunset1 R@10) — sunset1 from tab:native.
OURS = [
    ("v5b_mloc_s500",   "ViT-B MLoc  real-best s500",  f"{DAY}/realbest/realbest_daytime.json", "r322ba50_v5b_mloc_s500",   0.846, 0.932),
    ("v5b_salad_s1000", "ViT-B SALAD real-best s1000", f"{DAY}/realbest/realbest_daytime.json", "r322ba50_v5b_salad_s1000", 0.828, 0.929),
    ("v6s_mloc_s1000",  "ViT-S MLoc  real-best s1000", f"{DAY}/realbest/realbest_daytime.json", "r322ba50_v6s_mloc_s1000",  0.863, 0.937),
    ("v6s_salad_s1000", "ViT-S SALAD real-best s1000", f"{DAY}/realbest/realbest_daytime.json", "r322ba50_v6s_salad_s1000", 0.841, 0.933),
]
SHIP = {
    "v5b_mloc_s500":   ("ViT-B MLoc  shipped s3500", f"{DAY}/v5/v5_daytime.json",     "r322ba50"),
    "v5b_salad_s1000": ("ViT-B SALAD shipped s8000", f"{DAY}/ship/ship_daytime.json", "r322ba50_b_salad_s8000"),
    "v6s_mloc_s1000":  ("ViT-S MLoc  shipped s3500", f"{DAY}/ship/ship_daytime.json", "r322ba50_s_mloc_s3500"),
    "v6s_salad_s1000": ("ViT-S SALAD shipped s2000", f"{DAY}/ship/ship_daytime.json", "r322ba50_s_salad_s2000"),
}
# baseline -> (daytime json, key, space, sunset1 R@1, sunset1 R@10)
BASE = [
    ("Event-GeM", f"{DAY}/eventgem/brisbane_event_daytime.json", "r240ba50", "native",        0.514, 0.721),
    ("+rerank",   f"{DAY}/eventgem/brisbane_event_daytime.json", "r240ba50", "native+rerank", 0.776, 0.819),
    ("MegaLoc",   f"{DAY}/megaloc/brisbane_event.json",  "r322ba50", "native", 0.782, 0.901),
    ("CricaVPR",  f"{DAY}/cricavpr/brisbane_event.json", "r224ba50", "native", 0.722, 0.868),
    ("MixVPR",    f"{DAY}/mixvpr/brisbane_event.json",   "r320ba50", "native", 0.794, 0.904),
    ("SALAD",     f"{DAY}/salad/brisbane_event.json",    "r322ba50", "native", 0.640, 0.839),
]


def cell(path, key, space="native"):
    """-> ((R@1, R@10), (n_database, scorable))."""
    with open(path) as h:
        r = json.load(h)["results"][key]
    if list(r["recall"]) not in (["native"], ["native", "native+rerank"]):
        raise SystemExit(f"{path}:{key} has spaces {list(r['recall'])} — PCA was not off")
    rec = r["recall"][space]
    return (rec["1"], rec["10"]), (r["n_database"], r["scorable"])


def main():
    shapes = set()
    print("\nBrisbane pooled, DAYTIME query — native, no PCA, 25 m")
    print(f"  {'model':30s} {'R@1':>7s} {'R@10':>7s}   {'sunset1 R@1':>11s} {'R@10':>7s}   "
          f"{'d vs shipped (daytime)':>22s}")
    for label, pretty, path, key, s_r1, s_r10 in OURS:
        (r1, r10), shape = cell(path, key)
        shapes.add(shape)
        sp, spath, skey = SHIP[label]
        (h1, h10), sshape = cell(spath, skey)
        shapes.add(sshape)
        print(f"  {pretty:30s} {r1:>7.3f} {r10:>7.3f}   {s_r1:>11.3f} {s_r10:>7.3f}   "
              f"{r1 - h1:>+10.3f} {r10 - h10:>+10.3f}")
        print(f"  {'  ' + sp:30s} {h1:>7.3f} {h10:>7.3f}")
    print()
    for name, path, key, space, s_r1, s_r10 in BASE:
        (r1, r10), shape = cell(path, key, space)
        shapes.add(shape)
        print(f"  {name:30s} {r1:>7.3f} {r10:>7.3f}   {s_r1:>11.3f} {s_r10:>7.3f}   "
              f"{'(sunset1 -> daytime ' + format(r1 - s_r1, '+.3f') + ')':>22s}")
    if len(shapes) > 1:
        raise SystemExit(f"cells disagree on the benchmark size: {shapes}")
    shape = shapes.pop()
    print(f"\n  every cell scored {shape[0]:,} database x {shape[1]:,} queries")


if __name__ == "__main__":
    main()
