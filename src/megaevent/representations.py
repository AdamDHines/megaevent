"""Checkpoint-compatible event representations; event IO is handled by EventCV."""

import numpy as np

VALID = ("tencode", "accumulate", "polmask", "countmask")


def _prep(x, y, t, p, H, W):
    """Common front-matter: int coords, float time, {0,1} polarity, in-bounds mask."""
    x = np.asarray(x).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    t = np.asarray(t).astype(np.float64)
    p = np.asarray(p)
    pol = (p > 0).astype(np.float32)  # {0,1}; handles both {0,1} and {-1,1} inputs
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    return x[valid], y[valid], t[valid], pol[valid]


def tencode_numpy(x, y, t, p, H, W, white_frame=False):
    """R=+pol, G=recency (newest darker), B=-pol; last-event-wins. CHW uint8 [3,H,W].

    Mirrors ``i2e_infer.tencode_numpy`` exactly (default black background).
    """
    base = 255.0 if white_frame else 0.0
    ten = np.full((3, H, W), base, dtype=np.float32)
    xv, yv, tv, pol = _prep(x, y, t, p, H, W)
    if xv.size == 0:
        return ten.astype(np.uint8)
    order = np.argsort(tv)  # last-event-wins: later writes overwrite
    xv, yv, tv, pol = xv[order], yv[order], tv[order], pol[order]
    tn = (tv - tv[0]) / (tv[-1] - tv[0]) if tv[-1] != tv[0] else np.zeros_like(tv, np.float32)
    ten[0, yv, xv] = 255.0 * pol
    ten[1, yv, xv] = 255.0 * (1.0 - tn)
    ten[2, yv, xv] = 255.0 * (1.0 - pol)
    return np.clip(ten, 0, 255).astype(np.uint8)


def polmask_numpy(x, y, t, p, H, W, white_frame=False):
    """tencode's polarity R/B, but G = binary occupancy mask (no time). CHW uint8.

    One-factor swap from :func:`tencode_numpy`: only the green channel changes, from
    recency to a 255/base event-occupancy mask, to test whether the time channel is the
    thing that hurts sim-to-real transfer.
    """
    base = 255.0 if white_frame else 0.0
    ten = np.full((3, H, W), base, dtype=np.float32)
    xv, yv, tv, pol = _prep(x, y, t, p, H, W)
    if xv.size == 0:
        return ten.astype(np.uint8)
    order = np.argsort(tv)
    xv, yv, pol = xv[order], yv[order], pol[order]
    ten[0, yv, xv] = 255.0 * pol
    ten[1, yv, xv] = 255.0  # green = "an event occurred here" (binary mask)
    ten[2, yv, xv] = 255.0 * (1.0 - pol)
    return np.clip(ten, 0, 255).astype(np.uint8)


def accumulate_numpy(x, y, t, p, H, W, pct=99.0, white_frame=True):
    """GEPT-native white-bg red/blue count frame; G = inverse activity mask. CHW uint8.

    Ported verbatim from ``src/utils.py::accumulate_to_rgb`` (the frames the checkpoints
    were aligned on), only transposed to CHW. ``white_frame`` is accepted for a uniform
    signature but is intrinsically true here — the representation is defined on a white
    background (base means ~0.90-0.97, see ``src/config.py``). ``t`` is unused: this
    representation carries no time.
    """
    xv, yv, _tv, pol = _prep(x, y, t, p, H, W)
    pos = np.zeros((H, W), dtype=np.float32)
    neg = np.zeros((H, W), dtype=np.float32)
    if xv.size:
        posmask = pol > 0
        np.add.at(pos, (yv[posmask], xv[posmask]), 1)
        np.add.at(neg, (yv[~posmask], xv[~posmask]), 1)

    def norm_pct(a):
        if a.max() == 0:
            return a
        thr = np.percentile(a[a > 0], pct) if np.any(a > 0) else 1.0
        if thr <= 0:
            thr = float(a.max())
        return np.clip(a, 0, thr) / thr

    pos_n, neg_n = norm_pct(pos), norm_pct(neg)
    dominate_pos = pos_n >= neg_n
    inten_pos = pos_n * dominate_pos
    inten_neg = neg_n * (~dominate_pos)

    R = np.ones((H, W), dtype=np.float32)
    G = np.ones((H, W), dtype=np.float32)
    B = np.ones((H, W), dtype=np.float32)
    G -= inten_pos
    B -= inten_pos
    R -= inten_neg
    G -= inten_neg
    img = np.stack([np.clip(R, 0, 1), np.clip(G, 0, 1), np.clip(B, 0, 1)], axis=0)
    return (img * 255).astype(np.uint8)


def countmask_numpy(x, y, t, p, H, W, pct=99.0, white_frame=False):
    """GEPT paper Sec. 3.2 'Event Accumulation' (Eq. 2), verbatim. CHW uint8.

    Independent per-polarity count channels Mr, Mb, jointly percentile-normalized
    by a single alpha_n taken across *both* channels' pixel values (not per-channel,
    and not winner-take-all like ``accumulate_numpy``). G is a plain binary activity
    mask -- 1 wherever an event of either polarity fell, else 0 (not the inverse
    count-graded mask ``accumulate_numpy`` uses). No time.
    """
    xv, yv, _tv, pol = _prep(x, y, t, p, H, W)
    Mr = np.zeros((H, W), dtype=np.float32)
    Mb = np.zeros((H, W), dtype=np.float32)
    if xv.size:
        posmask = pol > 0
        np.add.at(Mr, (yv[posmask], xv[posmask]), 1)
        np.add.at(Mb, (yv[~posmask], xv[~posmask]), 1)

    combined = np.concatenate([Mr.ravel(), Mb.ravel()])
    nz = combined[combined > 0]
    alpha = float(np.percentile(nz, pct)) if nz.size else 1.0
    if alpha <= 0:
        alpha = float(combined.max()) if combined.max() > 0 else 1.0

    R = np.clip(Mr, 0, alpha) / alpha
    B = np.clip(Mb, 0, alpha) / alpha
    G = ((Mr + Mb) > 0).astype(np.float32)

    if white_frame:
        R, G, B = 1.0 - R, 1.0 - G, 1.0 - B

    img = np.stack([R, G, B], axis=0)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


RENDERERS = {
    "tencode": tencode_numpy,
    "accumulate": accumulate_numpy,
    "polmask": polmask_numpy,
    "countmask": countmask_numpy,
}


def render(name, x, y, t, p, H, W, white_frame=None):
    """Dispatch to a renderer by name -> CHW uint8 [3,H,W]."""
    if name not in RENDERERS:
        raise ValueError(f"unknown representation {name!r}; known: {sorted(RENDERERS)}")
    fn = RENDERERS[name]
    if white_frame is None:
        return fn(x, y, t, p, H, W)
    return fn(x, y, t, p, H, W, white_frame=white_frame)


def render_events(name, ev4, H, W, white_frame=None):
    """Render an ``(N,4)`` ``[x,y,t,p]`` event array (eventcv layout) -> CHW uint8."""
    ev4 = np.asarray(ev4)
    return render(name, ev4[:, 0], ev4[:, 1], ev4[:, 2], ev4[:, 3], H, W, white_frame)
