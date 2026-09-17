"""Per-query spatial similarity profiles: how each method's cosine falls off with distance.

    pixi run python3 scripts/similarity_profile.py --dataset brisbane_event --pair sunset1 morning
    pixi run python3 scripts/similarity_profile.py --dataset nsavp --pair R0_FS0 R0_FA0

Recall says *whether* a method retrieves the right place; it says nothing about the shape of
the descriptor's response. This asks the mechanistic question directly: for a query frame,
how does the cosine similarity to a database frame vary with that database frame's physical
distance from the query's true location?

A place-recognising descriptor should answer with a **peak at zero distance that stands well
above the far-field background** — the co-located database frame (same place, other
condition/viewpoint) scores high, distant places score low, and the gap is what makes the
retrieval correct. A descriptor that keys on appearance instead loses the co-located frame
when the condition changes, so its near-field similarity collapses toward the background and
the peak flattens. The **within-radius decay** (0 -> 25 m) is the viewpoint-tolerance
signature: a robust descriptor stays high as the database frame drifts off the exact spot; a
brittle one falls away fast.

Everything is off cached banks on the published pairwise geometry (single reference traverse,
single query traverse). The roster is the whole field — conventional VPR *and* the
event-native methods — because the claim is about all of them.

Outputs `output/simprofile/<dataset>_<ref>_<query>.json`:
  * `curve[method]`  mean cosine per distance bin (aggregated over sampled scorable queries)
  * `background[method]`  far-field mean cosine (75-200 m), the discrimination floor
  * `examples[]`  a few individual query profiles, for the single-query view
  * `decision[method]`  per-query (true-place cos, best-distractor cos) for a subsample —
    points where true-place > best-distractor are the queries retrieval gets right
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import brisbane_pooled as bp  # noqa: E402
import pairwise_sunset_ref as psr  # noqa: E402

# Fine inside the 25 m radius (to resolve viewpoint decay), coarse outside.
BIN_EDGES = [0, 2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 75, 100, 150, 250]
FAR_LO, FAR_HI = 75.0, 200.0          # far-field background window
# A "competitor" is any database frame OUTSIDE the positive radius. With the competitor
# threshold set to the radius, "best true-place cosine > best competitor cosine" holds for a
# query iff its top-1 lies within the radius — i.e. the above-diagonal fraction of the
# decision scatter is exactly R@1. Overridden to threshold_m at call time.
DISTRACTOR_MIN = 25.0
# The field, conventional + event. Names as pairwise_sunset_ref prints them.
DEFAULT_ROSTER = ["megaevent v8-B", "MegaLoc", "MixVPR", "SALAD", "QAA", "CricaVPR",
                  "Event-GeM 0.1.0", "EventVLAD", "SpikeVPR"]
DECISION_METHODS = ["megaevent v8-B", "MegaLoc", "Event-GeM 0.1.0"]


def method_arms(dataset, roster, geom, ref, query):
    """{method: (db_desc[torch], db_xy, q_desc[torch], q_xy, chunk, db_chunk)} for one cell."""
    cfg = psr.CONFIGS[dataset]
    bank_names = [m[0] for m in psr.models_for(dataset, "bank") if m[0] in roster]
    files = {}
    for name, _, dirs, _ in psr.models_for(dataset, "bank"):
        if name not in bank_names:
            continue
        bank_dir, template = dirs[dataset]
        files[name] = {t: {name: os.path.join(bank_dir, template.format(t=t))}
                       for t in (ref, query)}
        for t in (ref, query):
            if not os.path.exists(files[name][t][name]):
                raise SystemExit(f"{name}: missing bank {files[name][t][name]}")
    arms_own = psr.own_grid_arms(dataset, geom, [ref, query])

    out = {}
    for name in roster:
        if name in bank_names:
            db_desc, db_xy, _ = bp.pool_database(files[name], geom, [ref], name)
            q_xy_full, q_cov, _, _ = geom[query]
            q_bank = np.load(files[name][query][name], mmap_mode="r")
            q_desc = torch.from_numpy(np.asarray(q_bank[q_cov]))
            q_xy = q_xy_full[q_cov]
            chunk, db_chunk = cfg["chunk"], cfg["db_chunk"]
        elif name in arms_own:
            db_desc, db_xy = arms_own[name][ref]
            q_desc, q_xy = arms_own[name][query]
            chunk, db_chunk = psr.OWN_GRID_CHUNK.get(name, (cfg["chunk"], cfg["db_chunk"]))
        else:
            raise SystemExit(f"{name}: no banks for {dataset}")
        # L2-normalise both sides so the dot product is a true cosine in [-1, 1] and the
        # curves are comparable across methods. Most VPR heads already emit unit vectors, so
        # this is a no-op for them; EventVLAD's banks are not normalised (constant norm
        # ~1.47, so ranking is preserved) and would otherwise plot off-scale.
        db_desc = torch.nn.functional.normalize(db_desc.float(), dim=1)
        q_desc = torch.nn.functional.normalize(q_desc.float(), dim=1)
        out[name] = (db_desc, np.asarray(db_xy), q_desc, np.asarray(q_xy))
    return out


def profile(db_desc, db_xy, q_desc, q_xy, q_idx, device, threshold=25.0):
    """Aggregate + per-query similarity-vs-distance for the sampled queries.

    Returns (bin_sum, bin_cnt, far_mean, near_by_q, dist_by_q, tp_by_q, distractor_by_q)
    where the *_by_q are per sampled query.
    """
    edges = np.array(BIN_EDGES)
    nb = len(edges) - 1
    bin_sum = np.zeros(nb)
    bin_cnt = np.zeros(nb, dtype=np.int64)
    far_sum = far_cnt = 0.0
    near_by_q, tp_by_q, distr_by_q = [], [], []
    db = db_desc.to(device)
    with torch.no_grad():
        for s in range(0, len(q_idx), 256):
            cols = q_idx[s:s + 256]
            qb = q_desc[cols].to(device)
            sims = (db @ qb.T).cpu().numpy()                 # [n_db, b]
            d = np.linalg.norm(db_xy[:, None, :] - q_xy[cols][None, :, :], axis=2)  # [n_db,b]
            idx = np.digitize(d.ravel(), edges) - 1
            valid = (idx >= 0) & (idx < nb)
            flat = sims.ravel()
            np.add.at(bin_sum, idx[valid], flat[valid])
            np.add.at(bin_cnt, idx[valid], 1)
            farm = (d >= FAR_LO) & (d < FAR_HI)
            far_sum += float(flat[farm.ravel()].sum())
            far_cnt += int(farm.sum())
            for j in range(len(cols)):
                near = d[:, j] <= threshold
                near_by_q.append(float(sims[near, j].mean()) if near.any() else np.nan)
                tp_by_q.append(float(sims[near, j].max()) if near.any() else np.nan)
                far_j = d[:, j] > threshold
                distr_by_q.append(float(sims[far_j, j].max()) if far_j.any() else np.nan)
            del sims
    far_mean = far_sum / max(far_cnt, 1)
    return bin_sum, bin_cnt, far_mean, np.array(near_by_q), np.array(tp_by_q), np.array(distr_by_q)


def one_query_cloud(db_desc, db_xy, q_desc, q_xy, qi, device, threshold, rng,
                    n_far=320):
    """A single query's (distance, cosine) point cloud for every database frame.

    Keeps every frame within 50 m (the region the eye needs) and a random sample beyond, so
    the scatter is legible without shipping 14k points. Also returns the winning frame (global
    argmax) and whether it lies inside the radius — that is this query's retrieval outcome.
    """
    with torch.no_grad():
        sims = (db_desc.to(device) @ q_desc[qi:qi + 1].to(device).T).cpu().numpy().ravel()
    d = np.linalg.norm(db_xy - q_xy[qi], axis=1)
    near = np.flatnonzero(d <= 50.0)
    far = np.flatnonzero(d > 50.0)
    if len(far) > n_far:
        far = rng.choice(far, size=n_far, replace=False)
    keep = np.concatenate([near, far])
    amax = int(np.argmax(sims))
    return {
        "pts": [[round(float(d[i]), 1), round(float(sims[i]), 3)] for i in keep],
        "win": [round(float(d[amax]), 1), round(float(sims[amax]), 3)],
        "win_correct": bool(d[amax] <= threshold),
        # the true place: highest-scoring frame within the radius
        "truept": ([round(float(d[near[np.argmax(sims[near])]]), 1),
                    round(float(sims[near[np.argmax(sims[near])]]), 3)]
                   if len(near) else None),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="brisbane_event", choices=sorted(psr.CONFIGS))
    ap.add_argument("--pair", nargs=2, default=["sunset1", "morning"],
                    metavar=("REF", "QUERY"))
    ap.add_argument("--methods", nargs="+", default=DEFAULT_ROSTER)
    ap.add_argument("--n-queries", type=int, default=600)
    ap.add_argument("--n-decision", type=int, default=300)
    ap.add_argument("--threshold-m", type=float, default=25.0)
    ap.add_argument("--out-json", default=None)
    cli = ap.parse_args()
    ref, query = cli.pair

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geom = psr.geometry(cli.dataset)
    arms = method_arms(cli.dataset, cli.methods, geom, ref, query)

    # Scorable query frames (a true positive exists within the radius) — the co-located
    # frame is what the near-field peak measures, so a query without one is meaningless here.
    # Use the anchor method's geometry (all bank methods share it; own-grid differ by <=2).
    anchor = "megaevent v8-B" if "megaevent v8-B" in arms else cli.methods[0]
    _, adb_xy, _, aq_xy = arms[anchor]
    has_pos = np.array([bool((np.linalg.norm(adb_xy - aq_xy[j], axis=1) <= cli.threshold_m).any())
                        for j in range(len(aq_xy))])
    scorable = np.flatnonzero(has_pos)
    rng = np.random.default_rng(0)
    q_sample = np.sort(rng.choice(scorable, size=min(cli.n_queries, len(scorable)),
                                  replace=False))
    print(f"{cli.dataset} {ref}->{query}: {len(scorable)} scorable, sampling {len(q_sample)}")

    centers = [round((BIN_EDGES[i] + BIN_EDGES[i + 1]) / 2, 2) for i in range(len(BIN_EDGES) - 1)]
    out = {"dataset": cli.dataset, "reference": ref, "query": query,
           "threshold_m": cli.threshold_m, "bin_edges": BIN_EDGES, "bin_centers": centers,
           "far_window_m": [FAR_LO, FAR_HI],
           "n_scorable": int(len(scorable)), "n_sampled": int(len(q_sample)),
           "note": "curve = mean cosine (L2-normalised descriptors) to database frames per "
                   "distance bin over sampled scorable queries; background = far-field "
                   "(75-200 m) mean cosine; decision = per-query (best cosine to a frame "
                   "within the radius, best cosine to a frame outside it) — true>competitor "
                   "iff the top-1 is correct, so the above-diagonal fraction is R@1.",
           "curve": {}, "curve_n": {}, "background": {}, "win_rate": {}, "decision": {},
           "examples": []}

    dec_cols = rng.choice(len(q_sample), size=min(cli.n_decision, len(q_sample)),
                          replace=False)
    tpd = {}   # per method: (tp_by_q, distr_by_q) aligned to q_sample (bank methods)
    for name in cli.methods:
        db_desc, db_xy, q_desc, q_xy = arms[name]
        # own-grid methods have their own (shorter) query grid; clamp the shared sample.
        qs = q_sample[q_sample < len(q_xy)]
        bsum, bcnt, farm, near_q, tp_q, distr_q = profile(
            db_desc, db_xy, q_desc, q_xy, qs, device, cli.threshold_m)
        mean = [round(float(bsum[b] / bcnt[b]), 4) if bcnt[b] else None
                for b in range(len(bcnt))]
        out["curve"][name] = mean
        out["curve_n"][name] = [int(c) for c in bcnt]
        out["background"][name] = round(float(farm), 4)
        finite = np.isfinite(tp_q) & np.isfinite(distr_q)
        win = float(np.mean(tp_q[finite] > distr_q[finite]))
        out["win_rate"][name] = round(win, 4)
        tpd[name] = (tp_q, distr_q, qs)
        dc = dec_cols[dec_cols < len(qs)]
        out["decision"][name] = [[round(float(tp_q[i]), 4), round(float(distr_q[i]), 4)]
                                 for i in dc if np.isfinite(tp_q[i]) and np.isfinite(distr_q[i])]
        peak = mean[0] if mean[0] is not None else float("nan")
        print(f"  {name:<18s} peak {peak:.3f}  far {farm:.3f}  "
              f"sep {peak - farm:+.3f}  true>competitor {win:.3f}")

    # Pick example queries that dramatise the decision: ours correct while a conventional and
    # an event baseline both fail, spread along the route. Falls back to any ours-correct query.
    base_c = "MegaLoc" if "MegaLoc" in tpd else cli.methods[1]
    base_e = "Event-GeM 0.1.0" if "Event-GeM 0.1.0" in tpd else cli.methods[-1]
    tp_o, dr_o, _ = tpd[anchor]
    tp_c, dr_c, _ = tpd[base_c]
    ours_win = tp_o > dr_o
    contrast = ours_win & (tp_c <= dr_c)
    pool = np.flatnonzero(contrast) if contrast.any() else np.flatnonzero(ours_win)
    pool = pool[np.argsort(q_sample[pool])]
    picks = [int(q_sample[pool[int(f * (len(pool) - 1))]]) for f in (0.25, 0.75)] if len(pool) else []
    print(f"  example queries: {picks} (of {len(pool)} candidates)")

    ex_rng = np.random.default_rng(1)
    for qi in picks:
        ex = {"index": int(qi), "methods": {}}
        for name in cli.methods:
            db_desc, db_xy, q_desc, q_xy = arms[name]
            if qi < len(q_xy):
                ex["methods"][name] = one_query_cloud(
                    db_desc, db_xy, q_desc, q_xy, qi, device, cli.threshold_m, ex_rng)
        out["examples"].append(ex)

    path = cli.out_json or os.path.join("output", "simprofile",
                                        f"{cli.dataset}_{ref}_{query}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(out, handle, indent=1)
    os.replace(tmp, path)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
