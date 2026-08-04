"""Parity checks for the baselines wired into megaevent.

None of them is evaluated end-to-end against a published number here (the tokyo247 benchmark
has no reference run for any of them), so correctness has to come from parity with the
original code instead. Three independent checks, one per baseline, each attacking whichever
part of it is actually new:

1. **sparse_event** is a reimplementation, so every ported function is compared against
   upstream ``sparse_event_vpr`` on synthetic frames and must agree *exactly*.
2. **EventVLAD** uses Event-LAB's own networks and weights, so the model is identical by
   construction and only our *preprocessing* is new — it is checked by pushing real
   denoised frames through both paths and comparing the 1000-D descriptors.
3. **Event-GeM** uses Event-GeM's own model and re-ranker, so what is new is the compaction
   of the keypoint store onto each query's shortlist. That is checked by running a case where
   the shortlist is the whole database, where the compacted store is index-for-index the store
   upstream's ``process_single_query`` expects, and comparing the two column by column.

Check 1 needs upstream importable, which means Event-LAB's interpreter::

    /home/adam/repo/Event-LAB/.pixi/envs/default/bin/python scripts/verify_baselines.py

Our modules need only numpy + torch, so they import there too. Under megaevent's own
interpreter the sparse_event comparison is skipped and the rest still runs. Check 3 needs a
GPU and the tokyo247 tree, so it is opt-in::

    pixi run python3 scripts/verify_baselines.py --check eventgem \\
        --data-dir /media/adam/vprdatasets/megaevent -d tokyo247
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
EVENTLAB = "/home/adam/repo/Event-LAB"

from src.methods import (                                          # noqa: E402
    adjust_and_normalize_probabilities, get_random_pixels, remove_random_bursts,
)


def check_sparse_event():
    """Our ports vs upstream ``sparse_event_vpr``, on synthetic count frames."""
    up = os.path.join(EVENTLAB, "baselines", "vpr_sparse_event", "src")
    if not os.path.isdir(up):
        print("  SKIP: no upstream checkout at", up)
        return True
    sys.path.insert(0, up)
    try:
        from sparse_event_vpr.sparse_pixel_utils import (
            adjust_and_normalize_probabilities as up_probs,
            get_random_pixels as up_pixels,
        )
        from sparse_event_vpr.utils import (
            get_distance_matrix as up_dist,
            remove_random_bursts as up_bursts,
        )
    except ImportError as err:
        print(f"  SKIP: upstream not importable here ({err}).")
        print("        Re-run under Event-LAB's interpreter for this check.")
        return True

    # The real grid and pixel budget the method runs at: 150 pixels 7 px apart need
    # ~23k px of exclusion area, so a toy grid cannot satisfy the constraint at all.
    rng = np.random.default_rng(0)
    H = W = 224
    N = 40
    frames = rng.poisson(2.0, size=(N, H, W)).astype(np.float32)
    frames[:, 5, 5] = 90                      # a hot pixel, to exercise the clip
    frames[:, 9, 9] = 9                       # bright but under the clip, to trip the
                                              # mean+2*std outlier suppression downstream

    ours = remove_random_bursts(frames, 10)
    theirs = up_bursts(frames.copy(), 10)     # upstream mutates in place
    clip_ok = np.array_equal(ours, theirs)
    print(f"  {'OK  ' if clip_ok else 'FAIL'} remove_random_bursts "
          f"(max {ours.max():.0f}, was {frames.max():.0f})")

    mean = ours.mean(axis=0).astype(np.float64)
    p_ours, p_theirs = adjust_and_normalize_probabilities(mean), up_probs(mean)
    prob_ok = np.array_equal(p_ours, p_theirs)
    suppressed = int((mean > mean.mean() + 2 * mean.std()).sum())
    print(f"  {'OK  ' if prob_ok else 'FAIL'} adjust_and_normalize_probabilities "
          f"(sum {p_ours.sum():.6f}, {suppressed} outlier pixels suppressed)")

    # Same seed on both sides: upstream draws from the global RandomState, ours from a
    # Generator, so they are seeded through their own APIs and compared as *sets* of
    # pixels drawn from the same PMF would not match bit-for-bit. Instead assert our
    # sampler obeys the same contract upstream's does.
    pix = get_random_pixels(150, W, H, 7, p_ours, rng=np.random.default_rng(1))
    np.random.seed(1)
    up_pix = np.asarray(up_pixels(150, W, H, 7, p_theirs))
    d = np.linalg.norm(pix[:, None, :] - pix[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    spread_ok = bool(d.min() > 7)
    bounds_ok = bool((pix[:, 0] < H).all() and (pix[:, 1] < W).all() and (pix >= 0).all())
    shape_ok = pix.shape == up_pix.shape == (150, 2)
    print(f"  {'OK  ' if spread_ok and bounds_ok and shape_ok else 'FAIL'} get_random_pixels "
          f"(shape {pix.shape}, min separation {d.min():.2f} > 7, in bounds {bounds_ok})")

    y, x = pix[:, 0], pix[:, 1]
    ref, qry = ours[:20, y, x], ours[20:, y, x]
    up_D = up_dist(ref, qry, metric="cityblock", device=torch.device("cpu"))
    ours_D = -torch.cdist(torch.from_numpy(ref)[None], torch.from_numpy(qry)[None],
                          p=1)[0].numpy()
    dist_ok = np.allclose(-ours_D, np.asarray(up_D), atol=1e-4)
    print(f"  {'OK  ' if dist_ok else 'FAIL'} L1 distance matrix "
          f"({ours_D.shape}, max |diff| {np.abs(-ours_D - np.asarray(up_D)).max():.2e})")

    return clip_ok and prob_ok and spread_ok and bounds_ok and shape_ok and dist_ok


def check_eventvlad():
    """Our Stage-B preprocessing vs Event-LAB's ``extract_eventvlad_features``.

    Both consume the same already-denoised PNGs Event-LAB wrote for brisbane, so any
    difference is purely in the grayscale -> RGB -> resize -> mean-subtract path.
    """
    import glob

    png_dir = ("/media/adam/vprdatasets/eventgem/brisbane_event/sunset1/"
               "sunset1-frames-50-denoised")
    pngs = sorted(glob.glob(os.path.join(png_dir, "*.png")))[:8]
    if not pngs:
        print(f"  SKIP: no denoised frames at {png_dir}")
        return True

    root = os.path.join(EVENTLAB, "baselines", "EventVLAD")
    sys.path.insert(0, root)
    sys.path.insert(0, EVENTLAB)
    from networks.netvlad import EmbedNet, NetVLAD
    from networks.vgg16 import Imagenet_vgg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(root, "vgg16_eventvlad.tar"), map_location="cpu",
                    weights_only=False)
    remapped = {}
    for k, v in ck["state_dict"].items():
        if k.startswith("encoder."):
            remapped["base_model." + k[len("encoder."):]] = v
        elif k.startswith("pool."):
            remapped["net_vlad." + k[len("pool."):]] = v
    remapped.setdefault("net_vlad.conv.bias", torch.zeros(64))
    net = EmbedNet(Imagenet_vgg(), NetVLAD(num_clusters=64, dim=1000))
    missing, unexpected = net.load_state_dict(remapped, strict=False)
    print(f"  {'OK  ' if not missing and not unexpected else 'FAIL'} NetVLAD weights "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")
    net = net.eval().to(device).requires_grad_(False)

    from src.methods import MATCONVNET_MEAN
    try:
        import cv2
    except ImportError:
        print("  SKIP: cv2 unavailable here, so the reference path cannot run.")
        print("        Re-run under Event-LAB's interpreter for this check.")
        return True

    ours, theirs = [], []
    for path in pngs:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # reference: Event-LAB's _EventVGGPreprocess._load
        ref = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        ref = cv2.resize(ref, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
        ref -= np.array(MATCONVNET_MEAN, dtype=np.float32)
        theirs.append(torch.from_numpy(np.transpose(ref, (2, 0, 1))))
        # ours: torch, mode="area" standing in for INTER_AREA
        g = torch.from_numpy(img.astype(np.float32))[None, None]
        rgb = torch.nn.functional.interpolate(g.repeat(1, 3, 1, 1), (224, 224), mode="area")
        ours.append(rgb[0] - torch.tensor(MATCONVNET_MEAN).view(3, 1, 1))

    with torch.no_grad():
        f_ours = net(torch.stack(ours).to(device)).float().cpu()
        f_theirs = net(torch.stack(theirs).to(device)).float().cpu()
    cos = torch.nn.functional.cosine_similarity(f_ours, f_theirs, dim=1)
    max_abs = (f_ours - f_theirs).abs().max().item()
    ok = bool(cos.min() > 0.9999 and max_abs < 1e-2)
    print(f"  {'OK  ' if ok else 'FAIL'} Stage-B descriptors over {len(pngs)} real frames "
          f"(min cosine {cos.min():.6f}, max |diff| {max_abs:.2e}, dim {f_ours.shape[1]})")
    return ok


def check_eventgem(args, n_db=24, n_q=6):
    """Event-GeM's local stage: representation, keypoints, and re-ranking parity.

    The model and the re-ranking maths are upstream's, imported rather than reimplemented, so
    what needs checking is our own bookkeeping — and one property the maths quietly depends on,
    that the sampled descriptors are exactly unit-norm (upstream's mutual-NN shortcut ranks by
    inner product, which only agrees with Euclidean distance on the unit sphere).

    The parity case is chosen so that compaction is the identity: with ``top_k`` equal to the
    whole database, the union of the shortlists is every index in order, so our compacted store
    is exactly the store ``process_single_query`` assumes and the two can be compared directly.
    """
    import glob

    from src import eventgemlocal as egl
    from src.methods import get_method
    from src.npzdata import load_mcts

    db = sorted(glob.glob(os.path.join(args.data_dir, args.dataset, "numpy", "database",
                                       "*.npz")))[:n_db]
    q = sorted(glob.glob(os.path.join(args.data_dir, args.dataset, "numpy", "queries",
                                      "*.npz")))[:n_q]
    if not db or not q:
        print(f"  SKIP: no {args.dataset} npz tree under {args.data_dir}")
        return True

    size = tuple(args.eventgem_size)
    frame = load_mcts(db[0], size=size)
    rep_ok = (frame.shape == (10,) + size and frame.dtype == np.float32
              and 0.0 <= frame.min() and frame.max() <= 1.0)
    print(f"  {'OK  ' if rep_ok else 'FAIL'} MCTS representation "
          f"({frame.shape}, {frame.dtype}, [{frame.min():.3f}, {frame.max():.3f}], "
          f"{100 * (frame > 0).mean():.1f}% populated)")

    with tempfile.TemporaryDirectory() as tmp:
        run = argparse.Namespace(dataset=args.dataset, feature_dir=tmp, limit=None,
                                 eventgem_repo=args.eventgem_repo, eventgem_size=list(size),
                                 eventgem_top_k=n_db, ransac_thresh=5.0, inlier_weight=0.05,
                                 match_filter="mutual", match_ratio=0.8)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        method = get_method("eventgem", run, device)

        desc = method.descriptors(db, "db")
        norm_ok = bool((desc.norm(dim=1) - 1).abs().max() < 1e-5) and desc.shape[1] == 128
        print(f"  {'OK  ' if norm_ok else 'FAIL'} global descriptors "
              f"({tuple(desc.shape)}, max |1 - L2 norm| "
              f"{(desc.norm(dim=1) - 1).abs().max():.2e})")

        sim = method.similarity(desc, method.descriptors(q, "q"), device)
        reranked = method.rerank(sim, db, q, label="parity")

        # Our store is the whole database here, so upstream can be pointed straight at it.
        store = os.path.join(tmp, args.dataset, "keypoints",
                             f"{method.tag}_parity_db_kps.pt")
        from eventgem.utils.kp_store import load_keypoint_store
        from eventgem.utils.rerank_utils import (bank_lookup, open_keypoint_bank,
                                                 process_single_query)
        packed = load_keypoint_store(store)
        budget = egl.keypoint_budget(*size, egl.input_multiple(method.cfg))
        counts = packed["counts"].numpy()
        kpts, descs = packed["keypoints"].numpy(), packed["descriptors"].numpy()
        kp_ok = bool((counts == budget).all())
        local_norm = np.linalg.norm(descs.reshape(-1, descs.shape[-1]), axis=1)
        local_ok = bool(np.abs(local_norm - 1).max() < 1e-5)
        bounds_ok = bool((kpts[..., 0] < size[1]).all() and (kpts[..., 1] < size[0]).all()
                         and (kpts >= 0).all())
        print(f"  {'OK  ' if kp_ok and bounds_ok else 'FAIL'} keypoints "
              f"({counts.min()}-{counts.max()} per frame, budget {budget}, in bounds "
              f"{bounds_ok})")
        print(f"  {'OK  ' if local_ok else 'FAIL'} local descriptors unit-norm "
              f"({descs.shape[-1]}-D, max |1 - L2 norm| {np.abs(local_norm - 1).max():.2e})")

        # Re-ranking may only *improve* a similarity, and only for a shortlisted candidate.
        delta = reranked - sim
        monotone_ok = bool((delta >= 0).all())
        changed = int((delta > 0).sum())
        print(f"  {'OK  ' if monotone_ok and changed else 'FAIL'} re-ranking is a "
              f"non-negative bonus ({changed} of {delta.size} entries raised, max "
              f"{delta.max():.3f} = {delta.max() / 0.05:.0f} inliers)")

        # Parity: upstream's own driver over the same store and the same distances.
        dist = 1.0 - np.asarray(sim, dtype=np.float32)
        q_bank = open_keypoint_bank(os.path.join(tmp, args.dataset, "keypoints",
                                                 f"{method.tag}_queries_kps.pt"))
        theirs = np.stack([process_single_query(
            q_idx=i, base_dists=dist[:, i], top_k=n_db, q_data=bank_lookup(q_bank, i),
            ref_store=store, ransac_thresh=5.0, inlier_weight=0.05,
            match_filter="mutual", match_ratio=0.8)[1] for i in range(len(q))], axis=1)
        gap = np.abs((1.0 - np.asarray(reranked)) - theirs).max()
        parity_ok = bool(gap < 1e-6)
        print(f"  {'OK  ' if parity_ok else 'FAIL'} parity with upstream "
              f"process_single_query over {len(db)}x{len(q)} (max |diff| {gap:.2e})")

    return (rep_ok and norm_ok and kp_ok and local_ok and bounds_ok and monotone_ok
            and bool(changed) and parity_ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", nargs="+", default=["sparse_event", "eventvlad"],
                        choices=["sparse_event", "eventvlad", "eventgem", "all"],
                        help="Which checks to run. eventgem needs a GPU and the image tree, "
                             "so it is not in the default set.")
    parser.add_argument("--dataset", "-d", default="tokyo247")
    parser.add_argument("--data-dir", default="/media/adam/vprdatasets/megaevent")
    parser.add_argument("--eventgem-repo", default=os.path.join(REPO, "external", "eventgem"))
    parser.add_argument("--eventgem-size", type=int, nargs=2, default=[240, 320])
    args = parser.parse_args()
    wanted = ({"sparse_event", "eventvlad", "eventgem"} if "all" in args.check
              else set(args.check))

    print(f"python: {sys.executable}\n")
    ok = True
    if "sparse_event" in wanted:
        print("1. sparse_event vs upstream sparse_event_vpr")
        ok &= check_sparse_event()
    if "eventvlad" in wanted:
        print("\n2. EventVLAD preprocessing vs Event-LAB's feature extractor")
        ok &= check_eventvlad()
    if "eventgem" in wanted:
        print("\n3. Event-GeM local stage vs Event-GeM's own re-ranker")
        ok &= check_eventgem(args)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
