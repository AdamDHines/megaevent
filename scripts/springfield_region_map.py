"""Folium map of Springfield's mean R@1 per query spot: one circle per spot, viridis.

Run from the repo root::

    pixi run python3 scripts/springfield_region_map.py
    pixi run python3 scripts/springfield_region_map.py --tiles imagery

One number per physical query location, on a continuous scale, from the ViT-B run's own
per-slice ranking. Per-slice top-1 hits come from the diag dump (``springfield_diag.py
--stage dump``: top-1 within the radius). The slices that count are the paper cell's: sweeps
in ``--sweeps`` (default ``day dawn`` — night is an appearance failure for every model) and
camera within ``--exclude-reversals-deg`` of the route direction (default 135, so the
reversed passes — 180 degrees from the forward arm and at least 90 from every arm — do not
count). That is ``springfield_baselines.exclude_reversals``'s rule on the same orient ``psi``,
and with the defaults it is the 5,557-slice cell behind the paper table's 0.4925.

Regions are the results JSON's spot clusters. Each is one circle of fixed screen size at the
spot's mean query position, filled with the viridis colour of its mean R@1 and labelled with
the value — no per-pass tracks, no slice points. Basemaps and keys are in
``scripts/springfield_basemap.py``: the default ``esri-gray`` draws streets with no labels and
needs no key; CARTO's no-label layer needs ``--carto-key`` (a keyless CARTO tile is an
"API KEY REQUIRED" placeholder, which is what folium's ``CartoDB positron`` shows today).

Writes ``maps/region_map_r1_<filter>.pdf`` (vector overlays over one raster basemap — editable),
``.png``, ``.html`` and ``.json`` beside the results JSON.
"""

import argparse
import json
import os
import sys

import folium
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42        # TrueType: text stays text in the PDF
matplotlib.rcParams["ps.fonttype"] = 42
import numpy as np
from branca.colormap import LinearColormap
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src.traversegps import EARTH_RADIUS_M                      # noqa: E402
from springfield_basemap import (attribution, draw_basemap, folium_tiles,  # noqa: E402
                                 metre_scale, resolve_key, scale_bar, to_mercator)
from springfield_failure_map import DEFAULT_RESULTS, DEFAULT_ROOT, track  # noqa: E402

DEFAULT_DIAG = "output/springfield_diag"
VIRIDIS = matplotlib.colormaps["viridis"]
def enu_to_latlon(xy, lat0, lon0):
    """Inverse of ``src.traversegps.local_enu`` — the equirectangular projection the eval uses."""
    xy = np.asarray(xy, dtype=np.float64)
    lat = lat0 + np.degrees(xy[:, 1] / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(xy[:, 0] / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    return np.column_stack([lat, lon])


def colour(r1):
    return to_hex(VIRIDIS(float(np.clip(r1, 0.0, 1.0))))


def text_colour(fill_hex):
    """Black on the light (yellow/green) end of viridis, white on the dark end."""
    r, g, b = to_rgb(fill_hex)
    return "#111111" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else "#ffffff"


def load_run(cli):
    with open(cli.results) as f:
        res = json.load(f)
    tag = res["tag"]
    d = np.load(os.path.join(cli.diag_dir, f"dump_{tag}.npz"), allow_pickle=False)
    o = np.load(os.path.join(cli.diag_dir, f"orient_{tag}.npz"), allow_pickle=False)
    with open(os.path.join(cli.diag_dir, f"orient_{tag}.json")) as f:
        lat0, lon0 = json.load(f)["__origin__"]
    sids = [str(s) for s in d["sids"]]
    sweeps = np.array([str(s) for s in d["sweeps"]])
    psi = np.concatenate([o[f"{s}_psi"] for s in sids])
    if len(psi) != len(d["q_sid"]):
        raise SystemExit(f"orient ({len(psi)}) and dump ({len(d['q_sid'])}) slice counts differ")
    q_xy, db_xy = d["q_xy"].astype(np.float64), d["db_xy"].astype(np.float64)
    top_d = np.linalg.norm(db_xy[d["top_i"][:10]] - q_xy[None], axis=2)      # [10, n_q]
    spot_of = {r["session"]: r["spot"] for r in res["per_session"]}
    return {"res": res, "tag": tag, "sids": sids, "sweeps": sweeps, "q_sid": d["q_sid"],
            "q_sweep": sweeps[d["q_sid"]], "q_xy": q_xy, "db_xy": db_xy,
            "db_sid": d["db_sid"].astype(str),
            "spot": np.array([spot_of[sids[i]] for i in d["q_sid"]]),
            "hit1": top_d[0] <= res["threshold_m"],
            "hit10": (top_d <= res["threshold_m"]).any(axis=0),
            "psi": psi, "origin": (float(lat0), float(lon0))}


def select(run, cli):
    keep = np.isin(run["q_sweep"], cli.sweeps)
    if cli.exclude_reversals_deg > 0:
        keep &= np.abs(run["psi"]) < cli.exclude_reversals_deg
    return keep


def summarise(run, keep):
    rows = []
    for spot in sorted(set(run["spot"])):
        m = keep & (run["spot"] == spot)
        if not m.any():
            continue
        passes = sorted(set(run["q_sid"][m].tolist()))
        rows.append({"spot": spot, "n_slices": int(m.sum()), "n_passes": len(passes),
                     "r1": float(run["hit1"][m].mean()), "r10": float(run["hit10"][m].mean()),
                     # The spot's position is physical: every slice of its passes, filtered or not.
                     "centre_xy": run["q_xy"][run["spot"] == spot].mean(axis=0).round(1).tolist(),
                     "passes": [{"session": run["sids"][i], "sweep": str(run["sweeps"][i]),
                                 "n_slices": int((m & (run["q_sid"] == i)).sum()),
                                 "r1": round(float(run["hit1"][m & (run["q_sid"] == i)].mean()), 4)}
                                for i in passes]})
    return rows


def caption(run, keep, cli):
    res = run["res"]
    rev = (f"reversals (|&psi;| &ge; {cli.exclude_reversals_deg:g}&deg;) excluded"
           if cli.exclude_reversals_deg > 0 else "reversals included")
    return (f"{os.path.basename(res['checkpoint']).replace('.pt', '')} (ViT-B), "
            f"{' + '.join(cli.sweeps)}, {rev}; {int(keep.sum())} slices over "
            f"{len(set(run['q_sid'][keep].tolist()))} passes; overall R@1 "
            f"{run['hit1'][keep].mean():.3f}")


def write_html(run, keep, rows, cli, out):
    lat0, lon0 = run["origin"]
    res = run["res"]
    centres = enu_to_latlon(np.array([r["centre_xy"] for r in rows]), lat0, lon0)
    key = resolve_key(cli.basemap, {"carto": cli.carto_key, "stadia": cli.stadia_key})
    tiles, attr = folium_tiles(cli.basemap, key)
    m = folium.Map(location=centres.mean(axis=0).tolist(), zoom_start=16, tiles=tiles,
                   attr=attr, control_scale=True)
    if tiles is None:
        from springfield_basemap import fetch_osm_features, folium_streets
        ll = enu_to_latlon(run["db_xy"], lat0, lon0)
        folium_streets(m, fetch_osm_features(ll[:, 0].min() - 0.001, ll[:, 0].max() + 0.001,
                                             ll[:, 1].min() - 0.001, ll[:, 1].max() + 0.001))
    if not cli.no_db_track:
        for sid in res["database"]:
            t = track(os.path.join(cli.root, "database", sid), every=cli.db_every)
            folium.PolyLine(t.tolist(), color="#b5b5b5", weight=2, opacity=0.7,
                            tooltip=f"database {res['database'][sid]['arm']} ({sid})").add_to(m)

    for r, (lat, lon) in zip(rows, centres):
        c = colour(r["r1"])
        lines = "".join(f"<tr><td>{p['session']}</td><td>{p['sweep']}</td>"
                        f"<td style='text-align:right'>{p['n_slices']}</td>"
                        f"<td style='text-align:right'>{p['r1']:.2f}</td></tr>"
                        for p in r["passes"])
        popup = folium.Popup(
            f"<b>{r['spot']}</b> &nbsp; mean R@1 <b>{r['r1']:.3f}</b> &nbsp; R@10 {r['r10']:.3f}"
            f"<br>{r['n_slices']} slices over {r['n_passes']} passes"
            f"<table style='font-size:11px;margin-top:4px'><tr><th>pass</th><th>sweep</th>"
            f"<th>slices</th><th>R@1</th></tr>{lines}</table>", max_width=360)
        folium.CircleMarker([float(lat), float(lon)], radius=cli.radius_px, color="#222222",
                            weight=1, fill=True, fill_color=c, fill_opacity=0.92, popup=popup,
                            tooltip=f"{r['spot']}  R@1 {r['r1']:.2f}  n={r['n_slices']}"
                            ).add_to(m)
        size = 2 * cli.radius_px
        folium.Marker([float(lat), float(lon)], icon=folium.DivIcon(
            icon_size=(size, size), icon_anchor=(size // 2, size // 2),
            html=f"<div style='width:{size}px;height:{size}px;display:flex;align-items:center;"
                 f"justify-content:center;font:bold 12px sans-serif;color:{text_colour(c)};"
                 f"pointer-events:none'>{r['r1']:.2f}</div>")).add_to(m)
        folium.Marker([float(lat), float(lon)], icon=folium.DivIcon(
            icon_size=(90, 16), icon_anchor=(45, -cli.radius_px - 2),
            html=f"<div style='width:90px;text-align:center;font:11px sans-serif;color:#222;"
                 f"text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 4px #fff;pointer-events:none'>"
                 f"{r['spot']}</div>")).add_to(m)

    LinearColormap([colour(v) for v in np.linspace(0, 1, 16)], vmin=0.0, vmax=1.0,
                   caption=f"mean R@1 within {res['threshold_m']:g} m").add_to(m)
    legend = f"""
    <div style="position: fixed; bottom: 18px; left: 18px; z-index: 9999;
                background: rgba(255,255,255,.93); padding: 10px 14px; font: 12px
                sans-serif; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.3)">
      <b>mean R@1 per query spot</b><br>{caption(run, keep, cli)}
      {'' if cli.no_db_track else '<br><span style="color:#b5b5b5">&#9632;</span> database route'}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.save(out)


def write_figure(run, keep, rows, cli, stem):
    """The same map as a vector PDF (and PNG) over a raster basemap, in Mercator metres."""
    lat0, lon0 = run["origin"]
    k = 1.0 / metre_scale(lat0)
    db = to_mercator(*enu_to_latlon(run["db_xy"], lat0, lon0).T)
    centres = to_mercator(*enu_to_latlon(np.array([r["centre_xy"] for r in rows]), lat0, lon0).T)
    pad = cli.pad_m * k
    bounds = (db[:, 0].min() - pad, db[:, 0].max() + pad, db[:, 1].min() - pad, db[:, 1].max() + pad)
    fig, ax = plt.subplots(figsize=(cli.width_in, cli.width_in * (bounds[3] - bounds[2])
                                    / (bounds[1] - bounds[0])), constrained_layout=True)
    key = resolve_key(cli.basemap, {"carto": cli.carto_key, "stadia": cli.stadia_key})
    pt_per_m = cli.width_in * 72 / (bounds[1] - bounds[0])
    draw_basemap(ax, cli.basemap, bounds, cli.zoom, key, alpha=cli.basemap_alpha, lat=lat0,
                 pt_per_merc_m=pt_per_m)
    if not cli.no_db_track:
        for sid in np.unique(run["db_sid"]):          # one line per session, never joined
            idx = np.flatnonzero(run["db_sid"] == sid)[::2]
            ax.plot(db[idx, 0], db[idx, 1], color="#8a8a8a", lw=1.2, alpha=0.8, zorder=2)
    for r, (cx, cy) in zip(rows, centres):
        c = colour(r["r1"])
        ax.add_patch(Circle((cx, cy), cli.radius_m * k, facecolor=c, edgecolor="#222222",
                            lw=0.8, zorder=4))
        ax.text(cx, cy, f"{r['r1']:.2f}", ha="center", va="center", fontsize=9,
                fontweight="bold", color=text_colour(c), zorder=5)
        ax.text(cx, cy - (cli.radius_m + 5) * k, r["spot"], ha="center", va="top", fontsize=7.5,
                color="#222222", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    res = run["res"]
    rev = (f"reversals (|ψ| ≥ {cli.exclude_reversals_deg:g}°) excluded"
           if cli.exclude_reversals_deg > 0 else "reversals included")
    ax.set_title(f"mean R@1 per query spot — ViT-B "
                 f"({os.path.basename(res['checkpoint']).replace('.pt', '')}); "
                 f"{' + '.join(cli.sweeps)} queries, {rev}\n"
                 f"{int(keep.sum())} slices over {len(set(run['q_sid'][keep].tolist()))} "
                 f"passes; overall R@1 {run['hit1'][keep].mean():.3f}", fontsize=8.5, loc="left")
    sm = plt.cm.ScalarMappable(cmap=VIRIDIS, norm=matplotlib.colors.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02,
                 label=f"mean R@1 within {res['threshold_m']:g} m")
    scale_bar(ax, lat0, 100.0)
    attribution(ax, cli.basemap)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=cli.dpi, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help="the springfield_full results JSON of the run to map (its tag names "
                         "the diag dump and orient files)")
    ap.add_argument("--diag-dir", default=DEFAULT_DIAG)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--sweeps", nargs="+", default=["day", "dawn"],
                    choices=("day", "dawn", "night"))
    ap.add_argument("--exclude-reversals-deg", type=float, default=135.0,
                    help="drop query slices whose camera is at least this far off the route "
                         "direction (the paper cell's rule); 0 keeps every slice")
    ap.add_argument("--basemap", default="osm-vector",
                    help="osm-vector (streets drawn as vectors, no names, no key) | esri-gray | "
                         "esri-imagery | osm | carto-nolabels/-light/-dark/-voyager (CARTO key) | "
                         "stadia-<style>")
    ap.add_argument("--carto-key", default=None, help="or set CARTO_KEY")
    ap.add_argument("--stadia-key", default=None, help="or set STADIA_KEY")
    ap.add_argument("--zoom", type=int, default=18, help="tile zoom for the PDF/PNG basemap")
    ap.add_argument("--basemap-alpha", type=float, default=1.0)
    ap.add_argument("--pad-m", type=float, default=50.0)
    ap.add_argument("--width-in", type=float, default=9.0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--radius-px", type=int, default=26, help="circle radius on screen")
    ap.add_argument("--radius-m", type=float, default=22.0, help="circle radius in the PDF/PNG")
    ap.add_argument("--no-db-track", action="store_true", help="omit the grey database route")
    ap.add_argument("--db-every", type=int, default=10)
    ap.add_argument("--out", default=None,
                    help="output stem (default maps/region_map_r1_<filter> beside the results "
                         "JSON); .pdf, .png, .html and .json are appended")
    cli = ap.parse_args()

    run = load_run(cli)
    res = run["res"]
    filter_tag = "baoff" if res["filter_dt_us"] is None else f"ba{res['filter_dt_us'] // 1000}"
    stem = cli.out or os.path.join(os.path.dirname(cli.results), "maps",
                                   f"region_map_r1_{filter_tag}")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    print(f"{os.path.basename(res['checkpoint'])} step {res['step']}: {res['tag']}")

    keep = select(run, cli)
    rows = summarise(run, keep)
    n_sweep = int(np.isin(run["q_sweep"], cli.sweeps).sum())
    print(f"queries {' + '.join(cli.sweeps)}: {n_sweep} slices, {int(keep.sum())} after "
          f"{'excluding reversals' if cli.exclude_reversals_deg > 0 else 'no exclusion'}; "
          f"overall R@1 {run['hit1'][keep].mean():.4f}  R@10 {run['hit10'][keep].mean():.4f}")
    print(f"  {'spot':8s} {'passes':>6s} {'slices':>7s} {'R@1':>6s} {'R@10':>6s}")
    for r in rows:
        print(f"  {r['spot']:8s} {r['n_passes']:6d} {r['n_slices']:7d} {r['r1']:6.3f} {r['r10']:6.3f}")

    write_html(run, keep, rows, cli, stem + ".html")
    write_figure(run, keep, rows, cli, stem)
    with open(stem + ".json", "w") as f:
        json.dump({"tag": res["tag"], "checkpoint": res["checkpoint"], "step": res["step"],
                   "threshold_m": res["threshold_m"],
                   "selection": {"sweeps": cli.sweeps,
                                 "exclude_reversals_deg": cli.exclude_reversals_deg,
                                 "n_slices_in_sweeps": n_sweep, "n_selected": int(keep.sum()),
                                 "r1": float(run["hit1"][keep].mean()),
                                 "r10": float(run["hit10"][keep].mean())},
                   "origin_latlon": run["origin"], "spots": rows}, f, indent=2)
    print(f"-> {stem}.pdf  .png  .html  .json")


if __name__ == "__main__":
    main()
