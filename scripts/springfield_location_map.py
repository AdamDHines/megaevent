"""Springfield location map: database arms with travel arrows, query passes with direction
arrows, the excluded (reversed) passes greyed — as an editable PDF, a PNG and a folium HTML.

Run from the repo root::

    pixi run python3 scripts/springfield_location_map.py                        # Esri gray
    pixi run python3 scripts/springfield_location_map.py --basemap carto-nolabels \\
        --carto-key $CARTO_KEY

What is drawn. The seven database sessions, one line each, coloured by arm (forward, left,
right camera mounts; the four short xA/xB passes in the same colours) with arrowheads along
the walk showing the direction of travel — the camera faces the arrow (forward), 90° left of
it (left) or 90° right (right). The three full passes share one footpath, so they are drawn
offset a few metres to each side of it purely for legibility (``--arm-offset-m``). Query
passes are the day and dawn sessions only (night is omitted), each as its GPS track with an
arrowhead at the end pointing the way it was walked, which is the way the camera faced.
Slices the paper cell excludes — camera at least ``--exclude-reversals-deg`` off the route
(the reversed passes, ``springfield_baselines.exclude_reversals``) — are grey; kept slices
carry their sweep's colour. The same per-slice ``psi`` the exclusion uses decides the colour.

Basemaps and keys: see ``scripts/springfield_basemap.py``. The PDF keeps every overlay as a
vector path with TrueType text, so it opens editable in Illustrator or Inkscape; the basemap is
one raster image underneath it.
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
from folium import plugins
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from springfield_basemap import (attribution, draw_basemap, folium_tiles,  # noqa: E402
                                 metre_scale, resolve_key, scale_bar, to_mercator)
from springfield_failure_map import DEFAULT_RESULTS, DEFAULT_ROOT  # noqa: E402
from springfield_full import DB_ARMS  # noqa: E402
from springfield_region_map import DEFAULT_DIAG, enu_to_latlon  # noqa: E402

ARM_COLOUR = {"forward": "#3572b0", "left": "#2a9d4e", "right": "#d43d3d"}
SWEEP_COLOUR = {"day": "#f28e2b", "dawn": "#8e5bd6", "night": "#222222"}
EXCLUDED = "#9c9c9c"


def load(cli):
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
        raise SystemExit("orient and dump slice counts differ")
    spot_of = {r["session"]: r["spot"] for r in res["per_session"]}
    r1_of = {r["session"]: r["r1_slice"] for r in res["per_session"]}
    return {"res": res, "tag": tag, "sids": sids, "sweeps": sweeps, "q_sid": d["q_sid"],
            "q_xy": d["q_xy"].astype(np.float64), "psi": psi, "spot_of": spot_of, "r1_of": r1_of,
            "db_xy": d["db_xy"].astype(np.float64), "db_sid": d["db_sid"].astype(str),
            "db_bear": d["db_bear"].astype(np.float64), "origin": (float(lat0), float(lon0))}


def merc_of_enu(run, xy):
    ll = enu_to_latlon(xy, *run["origin"])
    return to_mercator(ll[:, 0], ll[:, 1])


def runs_of(mask):
    """``[(start, stop, value)]`` maximal runs of a boolean array."""
    out, start = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            out.append((start, i, bool(mask[start])))
            start = i
    return out


def db_passes(run, cli):
    """Per database session: Mercator polyline (offset by arm), arm, facing."""
    k = 1.0 / metre_scale(run["origin"][0])          # ground m -> Mercator m
    out = []
    for sid, (arm, facing) in DB_ARMS.items():
        rows = np.flatnonzero(run["db_sid"] == sid)
        if not rows.size:
            continue
        pts = merc_of_enu(run, run["db_xy"][rows])
        side = {"forward": 0.0, "left": +1.0, "right": -1.0}[facing]
        if side and cli.arm_offset_m > 0:
            # Offset to the walker's left/right of the travel direction.
            bear = np.radians(run["db_bear"][rows])
            normal = np.column_stack([-np.cos(bear), np.sin(bear)])   # left of travel (E,N)
            pts = pts + side * cli.arm_offset_m * k * normal
        out.append({"session": sid, "arm": arm, "facing": facing, "pts": pts,
                    "n": int(rows.size)})
    return out


def query_passes(run, cli):
    out = []
    for i, sid in enumerate(run["sids"]):
        sweep = str(run["sweeps"][i])
        if sweep not in cli.sweeps:
            continue
        rows = np.flatnonzero(run["q_sid"] == i)
        pts = merc_of_enu(run, run["q_xy"][rows])
        kept = np.abs(run["psi"][rows]) < cli.exclude_reversals_deg \
            if cli.exclude_reversals_deg > 0 else np.ones(rows.size, bool)
        net = pts[-1] - pts[0]
        out.append({"session": sid, "sweep": sweep, "spot": run["spot_of"][sid],
                    "pts": pts, "kept": kept, "kept_frac": float(kept.mean()),
                    "median_abs_psi": float(np.median(np.abs(run["psi"][rows]))),
                    "spin": bool(np.linalg.norm(net) < 5.0 / metre_scale(run["origin"][0])),
                    "r1": run["r1_of"][sid], "n": int(rows.size)})
    return out


def arrowheads_along(ax, pts, colour, every_m, k, size=9, zorder=5):
    """Arrowheads every ``every_m`` ground metres along a polyline, pointing along it."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    step = every_m * k
    for target in np.arange(step / 2, cum[-1], step):
        j = int(np.searchsorted(cum, target))
        if j <= 0 or j >= len(pts):
            continue
        a, b = pts[j - 1], pts[j]
        if np.linalg.norm(b - a) < 1e-6:
            continue
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=size, color=colour,
                                     lw=0, shrinkA=0, shrinkB=0, zorder=zorder))


def end_arrow(ax, pts, colour, k, size=10, zorder=8):
    """One arrowhead at the end of a pass, along its final ~6 m of travel."""
    if len(pts) < 2:
        return
    back = np.linalg.norm(pts - pts[-1], axis=1)
    j = int(np.flatnonzero(back > 6.0 * k)[-1]) if (back > 6.0 * k).any() else 0
    ax.add_patch(FancyArrowPatch(pts[j], pts[-1], arrowstyle="-|>", mutation_scale=size,
                                 color=colour, lw=0, shrinkA=0, shrinkB=0, zorder=zorder))


def write_figure(run, dbs, qs, cli, stem):
    lat0 = run["origin"][0]
    k = 1.0 / metre_scale(lat0)
    allpts = np.concatenate([p["pts"] for p in dbs] + [q["pts"] for q in qs])
    pad = cli.pad_m * k
    bounds = (allpts[:, 0].min() - pad, allpts[:, 0].max() + pad,
              allpts[:, 1].min() - pad, allpts[:, 1].max() + pad)
    fig, ax = plt.subplots(figsize=(cli.width_in, cli.width_in * (bounds[3] - bounds[2])
                                    / (bounds[1] - bounds[0])), constrained_layout=True)
    key = resolve_key(cli.basemap, {"carto": cli.carto_key, "stadia": cli.stadia_key})
    pt_per_m = cli.width_in * 72 / (bounds[1] - bounds[0])
    draw_basemap(ax, cli.basemap, bounds, cli.zoom, key, alpha=cli.basemap_alpha, lat=lat0,
                 pt_per_merc_m=pt_per_m)

    for p in dbs:
        c = ARM_COLOUR[p["facing"]]
        ax.plot(p["pts"][:, 0], p["pts"][:, 1], color=c, lw=1.4, alpha=0.9, zorder=3,
                solid_capstyle="round")
        arrowheads_along(ax, p["pts"], c, cli.db_arrow_every_m, k)

    for q in qs:
        pts, kept = q["pts"], q["kept"]
        c_keep = SWEEP_COLOUR[q["sweep"]]
        for a, b, is_kept in runs_of(kept):
            seg = pts[max(a - 1, 0):b]
            ax.plot(seg[:, 0], seg[:, 1], color=c_keep if is_kept else EXCLUDED,
                    lw=2.0 if is_kept else 1.4, ls="-" if is_kept else (0, (3, 2)),
                    alpha=0.95, zorder=6 if is_kept else 5, solid_capstyle="round")
        head = c_keep if q["kept_frac"] >= 0.5 else EXCLUDED
        if q["spin"]:
            ax.plot(pts[-1, 0], pts[-1, 1], "o", ms=4, mfc="white", mec=head, mew=1.2, zorder=8)
        else:
            end_arrow(ax, pts, head, k)

    # Spot labels at each cluster's centre.
    for spot in sorted({q["spot"] for q in qs}):
        c = np.concatenate([q["pts"] for q in qs if q["spot"] == spot]).mean(axis=0)
        ax.text(c[0], c[1] + 30 * k, spot, ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#222222", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

    n_keep = sum(int(q["kept"].sum()) for q in qs)
    n_all = sum(q["n"] for q in qs)
    handles = [Line2D([], [], color=ARM_COLOUR[f], lw=2, label=f"database, camera {f}")
               for f in ("forward", "left", "right")]
    handles += [Line2D([], [], color=SWEEP_COLOUR[s], lw=2.2, label=f"query pass, {s}")
                for s in cli.sweeps]
    if cli.exclude_reversals_deg > 0:
        handles.append(Line2D([], [], color=EXCLUDED, lw=1.6, ls=(0, (3, 2)),
                              label=f"excluded: camera ≥ {cli.exclude_reversals_deg:g}° off route"))
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.92,
              title="arrows point the way each pass was walked", title_fontsize=7.5)
    ax.set_title(f"Springfield capture — database arms and {' + '.join(cli.sweeps)} query passes; "
                 f"{n_keep} of {n_all} query slices kept", fontsize=9, loc="left")
    scale_bar(ax, lat0, 100.0)
    attribution(ax, cli.basemap)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=cli.dpi, facecolor="white")
    plt.close(fig)


def write_html(run, dbs, qs, cli, out):
    lat0, lon0 = run["origin"]
    key = resolve_key(cli.basemap, {"carto": cli.carto_key, "stadia": cli.stadia_key})
    tiles, attr = folium_tiles(cli.basemap, key)
    centre = enu_to_latlon(run["q_xy"], lat0, lon0).mean(axis=0)
    m = folium.Map(location=centre.tolist(), zoom_start=16, tiles=tiles, attr=attr,
                   control_scale=True)
    if tiles is None:
        from springfield_basemap import fetch_osm_features, folium_streets
        ll = enu_to_latlon(run["db_xy"], lat0, lon0)
        folium_streets(m, fetch_osm_features(ll[:, 0].min() - 0.001, ll[:, 0].max() + 0.001,
                                             ll[:, 1].min() - 0.001, ll[:, 1].max() + 0.001))

    def latlon_of_merc(pts):
        from springfield_basemap import R
        lon = np.degrees(pts[:, 0] / R)
        lat = np.degrees(2 * np.arctan(np.exp(pts[:, 1] / R)) - np.pi / 2)
        return np.column_stack([lat, lon]).tolist()

    for p in dbs:
        c = ARM_COLOUR[p["facing"]]
        line = folium.PolyLine(latlon_of_merc(p["pts"][::3]), color=c, weight=3, opacity=0.85,
                               tooltip=f"database {p['arm']} ({p['session']}), camera {p['facing']}")
        line.add_to(m)
        plugins.PolyLineTextPath(line, "  ►  ", repeat=True, offset=5,
                                 attributes={"fill": c, "font-size": "13"}).add_to(m)
    for q in qs:
        c_keep = SWEEP_COLOUR[q["sweep"]]
        head = c_keep if q["kept_frac"] >= 0.5 else EXCLUDED
        for a, b, is_kept in runs_of(q["kept"]):
            seg = q["pts"][max(a - 1, 0):b]
            folium.PolyLine(latlon_of_merc(seg), color=c_keep if is_kept else EXCLUDED,
                            weight=4 if is_kept else 3, opacity=0.9,
                            dash_array=None if is_kept else "6",
                            tooltip=f"{q['session']} {q['sweep']} {q['spot']}  "
                                    f"{'kept' if is_kept else 'excluded'}").add_to(m)
        end = latlon_of_merc(q["pts"][-2:])
        arrow = folium.PolyLine(end, color=head, weight=1, opacity=0.0)
        arrow.add_to(m)
        plugins.PolyLineTextPath(arrow, "►", repeat=False, center=True,
                                 attributes={"fill": head, "font-size": "18"}).add_to(m)
        folium.CircleMarker(end[-1], radius=5, color=head, fill=True, fill_opacity=0.0,
                            opacity=0.0, popup=folium.Popup(
                                f"<b>{q['session']}</b> {q['sweep']} {q['spot']}<br>"
                                f"kept {q['kept'].sum()} of {q['n']} slices "
                                f"(median |ψ| {q['median_abs_psi']:.0f}°)<br>"
                                f"pass R@1 {q['r1'] if q['r1'] is not None else float('nan'):.2f}",
                                max_width=280)).add_to(m)
    legend = f"""
    <div style="position: fixed; bottom: 18px; left: 18px; z-index: 9999;
                background: rgba(255,255,255,.93); padding: 10px 14px; font: 12px
                sans-serif; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.3)">
      <b>database arms</b> (arrows = walking direction; camera faces the arrow / 90° left / 90° right)<br>
      <span style="color:{ARM_COLOUR['forward']}">&#9644;</span> forward &nbsp;
      <span style="color:{ARM_COLOUR['left']}">&#9644;</span> left &nbsp;
      <span style="color:{ARM_COLOUR['right']}">&#9644;</span> right<br>
      <b>query passes</b> (arrow at the end of each walk)<br>
      {' &nbsp; '.join(f'<span style="color:{SWEEP_COLOUR[s]}">&#9644;</span> {s}' for s in cli.sweeps)}
      &nbsp; <span style="color:{EXCLUDED}">&#9644;</span> excluded (camera ≥ {cli.exclude_reversals_deg:g}° off route)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.save(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--diag-dir", default=DEFAULT_DIAG)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--sweeps", nargs="+", default=["day", "dawn"], choices=("day", "dawn", "night"))
    ap.add_argument("--exclude-reversals-deg", type=float, default=135.0)
    ap.add_argument("--basemap", default="osm-vector",
                    help="osm-vector (streets drawn as vectors, no names, no key) | esri-gray | "
                         "esri-imagery | osm | carto-nolabels/-light/-dark/-voyager (CARTO key) | "
                         "stadia-<style>")
    ap.add_argument("--carto-key", default=None, help="or set CARTO_KEY")
    ap.add_argument("--stadia-key", default=None, help="or set STADIA_KEY")
    ap.add_argument("--zoom", type=int, default=18, help="tile zoom for the PDF/PNG basemap")
    ap.add_argument("--basemap-alpha", type=float, default=1.0)
    ap.add_argument("--arm-offset-m", type=float, default=3.0,
                    help="draw the left/right arms this far to each side of the footpath")
    ap.add_argument("--db-arrow-every-m", type=float, default=70.0)
    ap.add_argument("--pad-m", type=float, default=50.0)
    ap.add_argument("--width-in", type=float, default=11.0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", default=None,
                    help="output stem (default maps/location_map_<filter>); .pdf .png .html")
    cli = ap.parse_args()

    run = load(cli)
    res = run["res"]
    filter_tag = "baoff" if res["filter_dt_us"] is None else f"ba{res['filter_dt_us'] // 1000}"
    stem = cli.out or os.path.join(os.path.dirname(cli.results), "maps",
                                   f"location_map_{filter_tag}")
    os.makedirs(os.path.dirname(stem), exist_ok=True)

    dbs = db_passes(run, cli)
    qs = query_passes(run, cli)
    # Are the three full passes walked the same way? Compare left/right travel bearings with
    # the nearest forward slice's — the caption depends on it.
    fwd = [p for p in dbs if p["arm"] == "forward"][0]
    fwd_rows = np.flatnonzero(run["db_sid"] == fwd["session"])
    for p in dbs:
        if p["arm"] in ("left", "right"):
            rows = np.flatnonzero(run["db_sid"] == p["session"])[::50]
            near = np.argmin(np.linalg.norm(run["db_xy"][rows][:, None] -
                                            run["db_xy"][fwd_rows][None], axis=2), axis=1)
            diff = (run["db_bear"][rows] - run["db_bear"][fwd_rows][near] + 180) % 360 - 180
            print(f"  {p['arm']} pass vs forward pass travel bearing: median |Δ| "
                  f"{np.median(np.abs(diff)):.0f}°")
    n_keep = sum(int(q["kept"].sum()) for q in qs)
    print(f"{len(dbs)} database passes; {len(qs)} {' + '.join(cli.sweeps)} query passes, "
          f"{n_keep} of {sum(q['n'] for q in qs)} slices kept; "
          f"{sum(1 for q in qs if q['kept_frac'] < 0.5)} passes mostly excluded")
    write_figure(run, dbs, qs, cli, stem)
    write_html(run, dbs, qs, cli, stem + ".html")
    with open(stem + ".json", "w") as f:
        json.dump({"tag": res["tag"], "basemap": cli.basemap, "sweeps": cli.sweeps,
                   "exclude_reversals_deg": cli.exclude_reversals_deg,
                   "database": [{k: v for k, v in p.items() if k != "pts"} for p in dbs],
                   "queries": [{**{k: v for k, v in q.items() if k not in ("pts", "kept")},
                                "n_kept": int(q["kept"].sum())} for q in qs]}, f, indent=2)
    print(f"-> {stem}.pdf  .png  .html  .json")


if __name__ == "__main__":
    main()
