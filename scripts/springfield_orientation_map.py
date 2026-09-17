"""Re-render ``map_orientation.html`` — the capture's orientation map — with the excluded
queries marked or removed, on a basemap that does not watermark, as HTML, PDF and PNG.

Run from the repo root::

    pixi run python3 scripts/springfield_orientation_map.py                       # streets drawn as vectors
    pixi run python3 scripts/springfield_orientation_map.py --carto-key $CARTO_KEY  # the original CARTO tiles

The original (``<sessions>/map_orientation.html``, 2026-08-30) has no surviving generator, so
this reads the map itself: every query arrow (session, position, the bearing the rig faced,
its region colour), every database trident (position, travel direction) and every route dot,
straight out of the HTML. Nothing about *where* things are or which way they point is
recomputed — the map is the original's geometry, re-rendered.

What changes. (1) Tiles: the original used CARTO ``light_nolabels``, which now needs an API
key (``carto.com/basemaps/apikey``) and otherwise stamps "API KEY REQUIRED" on every tile;
with ``--carto-key`` the same tiles come back clean, without it the streets are drawn as
vectors from OpenStreetMap (no names, no key). (2) The excluded queries: night sessions, and
day/dawn sessions whose slices are mostly reversals (camera >= ``--exclude-reversals-deg`` off
the route, the paper cell's rule from the diag ``orient`` bundle) are drawn grey in
``map_orientation.*`` and dropped altogether in ``map_orientation_kept.*``. (3) A PDF with
every glyph as a vector path and TrueType text, so it edits in Illustrator/Inkscape.
The region circles — one translucent circle per spot in the spot's colour, around its arrows —
are the original's too, read back with their centres and radii.
"""

import argparse
import html as html_mod
import json
import os
import re
import sys

import folium
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from springfield_basemap import (attribution, draw_basemap, fetch_osm_features,  # noqa: E402
                                 folium_streets, folium_tiles, metre_scale, resolve_key,
                                 scale_bar, to_mercator)
from springfield_failure_map import DEFAULT_RESULTS, DEFAULT_ROOT  # noqa: E402
from springfield_region_map import DEFAULT_DIAG  # noqa: E402

DEFAULT_ORIGINAL = os.path.join(DEFAULT_ROOT, "map_orientation.html")
BLUE, PAPER, GREY = "#2a78d6", "#fcfcfb", "#9c9c9c"
TRIDENT_D = ("M0,7 L0,-7 M0,-7 L-3,-3.5 M0,-7 L3,-3.5 M0,0 L-7,0 M-7,0 L-3.5,-3 M-7,0 L-3.5,3 "
             "M0,0 L7,0 M7,0 L3.5,-3 M7,0 L3.5,3")
ARROW_D = "M0,-9 L5.5,7 L0,3.5 L-5.5,7 Z"


# ---------------------------------------------------------------------------
# 1. The original map, read back
# ---------------------------------------------------------------------------
def read_original(path):
    """``{"arrows": [...], "tridents": [...], "dots": [[lat, lon], ...], "bounds": ...}``."""
    s = open(path).read()
    arrows, tridents, dots = [], [], []
    pattern = (r'var marker_([0-9a-f]+) = L\.marker\(\s*\[([-0-9.]+), ([-0-9.]+)\],.*?\)'
               r'\.addTo\((\w+)\);(.*?)(?=var marker_|var circle_marker_|$)')
    for m in re.finditer(pattern, s, re.S):
        _, lat, lon, _, body = m.groups()
        rot = re.search(r'rotate\(([-0-9.]+)\)', body)
        fill = re.search(r'Z\\" fill=\\"(#[0-9a-f]{6})', body)
        tip = re.search(r'bindTooltip\(\s*`(.*?)`', body, re.S)
        tip = re.sub(r"<[^>]+>", "", re.sub(r"\s+", " ", html_mod.unescape(tip.group(1)))).strip() \
            if tip else ""
        if fill:
            sid = re.search(r"(\d{8}T\d{6}Z)", tip)
            arrows.append({"session": sid.group(1) if sid else None, "lat": float(lat),
                           "lon": float(lon), "facing": float(rot.group(1)), "colour": fill.group(1)})
        else:
            tridents.append({"lat": float(lat), "lon": float(lon), "travel": float(rot.group(1))})
    for m in re.finditer(r'L\.circleMarker\(\s*\[([-0-9.]+), ([-0-9.]+)\]', s):
        dots.append([float(m.group(1)), float(m.group(2))])
    circles = []
    for m in re.finditer(r'L\.circle\(\s*\[([-0-9.]+), ([-0-9.]+)\],\s*\{(.*?)\}\s*\)', s, re.S):
        opts = m.group(3)
        get = lambda key, default: (re.search(rf'"{key}": ([^,\n]+)', opts) or [None, default])[1]
        circles.append({"lat": float(m.group(1)), "lon": float(m.group(2)),
                        "radius_m": float(get("radius", "30")),
                        "colour": get("fillColor", '"#888888"').strip('"'),
                        "fill_opacity": float(get("fillOpacity", "0.15")),
                        "weight": float(get("weight", "2"))})
    fb = re.search(r'fitBounds\(\s*(\[\[.*?\]\])', s, re.S)
    bounds = json.loads(fb.group(1)) if fb else None
    if not arrows or not tridents or not dots:
        raise SystemExit(f"{path}: could not read the map back (arrows {len(arrows)}, "
                         f"tridents {len(tridents)}, dots {len(dots)})")
    return {"arrows": arrows, "tridents": tridents, "dots": dots, "circles": circles,
            "bounds": bounds}


def exclusion(cli, sessions):
    """``{session: (sweep, kept_frac, spot)}`` from the results JSON + orient psi."""
    with open(cli.results) as f:
        res = json.load(f)
    o = np.load(os.path.join(cli.diag_dir, f"orient_{res['tag']}.npz"), allow_pickle=False)
    out = {}
    for r in res["per_session"]:
        sid = r["session"]
        psi = o[f"{sid}_psi"]
        kept = float((np.abs(psi) < cli.exclude_reversals_deg).mean()) \
            if cli.exclude_reversals_deg > 0 else 1.0
        out[sid] = {"sweep": r["sweep"], "kept_frac": kept, "spot": r["spot"],
                    "r1": r["r1_slice"], "median_abs_psi": float(np.median(np.abs(psi)))}
    missing = [s for s in sessions if s not in out]
    if missing:
        raise SystemExit(f"sessions in the map but not in the results JSON: {missing[:5]}")
    return out


def is_kept(info, cli):
    return info["sweep"] in cli.sweeps and info["kept_frac"] >= 0.5


# ---------------------------------------------------------------------------
# 2. HTML — the original's own construction
# ---------------------------------------------------------------------------
def trident_svg(rot, colour=BLUE):
    return (f'<svg width="26" height="26" viewBox="-13 -13 26 26"><g transform="rotate({rot:g})" '
            f'fill="none" stroke-linecap="round"><path d="{TRIDENT_D}" stroke="{PAPER}" '
            f'stroke-width="3.6"/><path d="{TRIDENT_D}" stroke="{colour}" stroke-width="1.8"/>'
            f'</g></svg>')


def arrow_svg(rot, colour, muted=False):
    op = ' fill-opacity="0.55"' if muted else ""
    return (f'<svg width="22" height="22" viewBox="-11 -11 22 22"><g transform="rotate({rot:g})">'
            f'<path d="{ARROW_D}" fill="{colour}" stroke="{PAPER}" stroke-width="1.3"{op}/></g></svg>')


def write_html(orig, info, cli, out, drop_excluded):
    key = resolve_key(cli.basemap, {"carto": cli.carto_key, "stadia": cli.stadia_key})
    tiles, attr = folium_tiles(cli.basemap, key)
    m = folium.Map(location=[0, 0], zoom_start=1, tiles=tiles, attr=attr, control_scale=True)
    if tiles is None:
        b = orig["bounds"]
        folium_streets(m, fetch_osm_features(b[0][0] - 0.001, b[1][0] + 0.001,
                                             b[0][1] - 0.001, b[1][1] + 0.001))
    db = folium.FeatureGroup(name="database (7 sessions, averaged)").add_to(m)
    for lat, lon in orig["dots"]:
        folium.CircleMarker([lat, lon], radius=2.5, color=PAPER, weight=1, fill=True,
                            fill_color=BLUE, fill_opacity=1).add_to(db)
    for t in orig["tridents"]:
        folium.Marker([t["lat"], t["lon"]], tooltip=f"travel direction {t['travel']:.0f}°",
                      icon=folium.DivIcon(html=trident_svg(t["travel"]), icon_size=(26, 26),
                                          icon_anchor=(13, 13), class_name="empty")).add_to(db)
    qg = folium.FeatureGroup(name="queries").add_to(m)
    for c in orig["circles"]:                     # the original's region circles, as they were
        folium.Circle([c["lat"], c["lon"]], radius=c["radius_m"], color=c["colour"],
                      weight=c["weight"], opacity=1.0, fill=True, fill_color=c["colour"],
                      fill_opacity=c["fill_opacity"]).add_to(qg)
    n_keep = n_drop = 0
    for a in orig["arrows"]:
        inf = info[a["session"]]
        kept = is_kept(inf, cli)
        if not kept and drop_excluded:
            n_drop += 1
            continue
        n_keep += kept
        n_drop += not kept
        why = ("" if kept else
               (" — excluded: night" if inf["sweep"] == "night" else
                f" — excluded: reversed ({inf['kept_frac']:.0%} of slices within "
                f"{cli.exclude_reversals_deg:g}°)"))
        folium.Marker([a["lat"], a["lon"]],
                      tooltip=f"{a['session']} — facing {a['facing']:.0f}° · {inf['spot']} "
                              f"{inf['sweep']}{why}",
                      icon=folium.DivIcon(html=arrow_svg(a["facing"], a["colour"] if kept else GREY,
                                                         muted=not kept),
                                          icon_size=(22, 22), icon_anchor=(11, 11),
                                          class_name="empty")).add_to(qg)
    folium.LayerControl().add_to(m)
    m.fit_bounds(orig["bounds"])
    excl = (f"{n_drop} excluded queries removed (night, and passes reversed "
            f"≥ {cli.exclude_reversals_deg:g}° off route)" if drop_excluded else
            f"{n_drop} excluded queries in grey: night, and passes reversed "
            f"≥ {cli.exclude_reversals_deg:g}° off route")
    legend = f"""
    <div style="position:fixed;bottom:48px;left:12px;z-index:9999;background:{PAPER};
                padding:10px 14px;border-radius:6px;border:1px solid rgba(11,11,11,0.10);
                font:12px system-ui,sans-serif;box-shadow:0 1px 4px rgba(11,11,11,0.15)">
      <div style="margin:2px 0;color:#0b0b0b"><span style="display:inline-block;width:7px;
        height:7px;border-radius:50%;background:{BLUE};vertical-align:middle"></span>
        database — 7 sessions · 3.7 km mapped; tridents show travel direction (forward/left/right)</div>
      <div style="margin:2px 0;color:#0b0b0b">7 query regions, colour-coded — {n_keep} arrows point
        where the rig faced ({' + '.join(cli.sweeps)})</div>
      <div style="margin:2px 0;color:#555">{excl}</div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.save(out)
    return n_keep, n_drop


# ---------------------------------------------------------------------------
# 3. PDF / PNG — the same glyphs as vector paths
# ---------------------------------------------------------------------------
def svg_path(d):
    """A tiny SVG path (M/L/Z only) -> matplotlib Path, y flipped to point up."""
    verts, codes = [], []
    for cmd, x, y in re.findall(r"([MLZ])\s*([-0-9.]*),?([-0-9.]*)", d):
        if cmd == "Z":
            codes.append(Path.CLOSEPOLY); verts.append(verts[-1])
        else:
            codes.append(Path.MOVETO if cmd == "M" else Path.LINETO)
            verts.append((float(x), -float(y)))
    return Path(verts, codes)


TRIDENT_PATH = svg_path(TRIDENT_D)
ARROW_PATH = svg_path(ARROW_D)


def glyph(path, bearing_deg):
    """The glyph rotated clockwise by a compass bearing (matplotlib rotates anticlockwise)."""
    return MarkerStyle(path, transform=Affine2D().rotate_deg(-bearing_deg))


def write_figure(orig, info, cli, stem, drop_excluded):
    b = orig["bounds"]
    lat0 = (b[0][0] + b[1][0]) / 2
    k = 1.0 / metre_scale(lat0)
    corners = to_mercator([b[0][0], b[1][0]], [b[0][1], b[1][1]])
    pad = cli.pad_m * k
    bounds = (corners[:, 0].min() - pad, corners[:, 0].max() + pad,
              corners[:, 1].min() - pad, corners[:, 1].max() + pad)
    fig, ax = plt.subplots(figsize=(cli.width_in, cli.width_in * (bounds[3] - bounds[2])
                                    / (bounds[1] - bounds[0])), constrained_layout=True)
    key = resolve_key(cli.basemap, {"carto": cli.carto_key, "stadia": cli.stadia_key})
    draw_basemap(ax, cli.basemap, bounds, cli.zoom, key, lat=lat0,
                 pt_per_merc_m=cli.width_in * 72 / (bounds[1] - bounds[0]))

    dots = to_mercator(*np.array(orig["dots"]).T)
    ax.plot(dots[:, 0], dots[:, 1], "o", ms=3.2, mfc=BLUE, mec=PAPER, mew=0.6, ls="none",
            zorder=3)
    from matplotlib.patches import Circle
    for c in orig["circles"]:                     # the original's region circles, as they were
        x, y = to_mercator([c["lat"]], [c["lon"]])[0]
        ax.add_patch(Circle((x, y), c["radius_m"] * k, facecolor=c["colour"],
                            edgecolor=c["colour"], lw=c["weight"] * 0.5,
                            alpha=None, zorder=2))
        ax.patches[-1].set_facecolor(matplotlib.colors.to_rgba(c["colour"], c["fill_opacity"]))
    for t in orig["tridents"]:
        x, y = to_mercator([t["lat"]], [t["lon"]])[0]
        ax.plot([x], [y], marker=glyph(TRIDENT_PATH, t["travel"]), ms=cli.glyph_pt,
                mec=BLUE, mew=cli.stroke_pt, mfc="none", ls="none", zorder=4)
    n_keep = n_drop = 0
    for a in orig["arrows"]:
        inf = info[a["session"]]
        kept = is_kept(inf, cli)
        if not kept and drop_excluded:
            n_drop += 1
            continue
        n_keep += kept
        n_drop += not kept
        x, y = to_mercator([a["lat"]], [a["lon"]])[0]
        col = a["colour"] if kept else GREY
        ax.plot([x], [y], marker=glyph(ARROW_PATH, a["facing"]), ms=cli.glyph_pt * 0.95,
                mfc=col, mec=col, mew=cli.stroke_pt, ls="none",
                alpha=1.0 if kept else 0.7, zorder=6 if kept else 5)

    spots = sorted({(v["spot"], a["colour"]) for a in orig["arrows"]
                    for v in [info[a["session"]]]})
    handles = [Line2D([], [], marker="o", ms=4, mfc=BLUE, mec=PAPER, ls="none",
                      label="database route (7 sessions, averaged); tridents = travel direction")]
    handles += [Line2D([], [], marker=glyph(ARROW_PATH, 0), ms=9, mfc=c, mec=c, mew=cli.stroke_pt,
                       ls="none", label=f"{s} query arrows (rig facing); circle = region")
                for s, c in spots]
    if not drop_excluded:
        handles.append(Line2D([], [], marker=glyph(ARROW_PATH, 0), ms=9, mfc=GREY, mec=GREY,
                              mew=cli.stroke_pt, ls="none", alpha=0.7,
                              label=f"excluded: night, or reversed ≥ {cli.exclude_reversals_deg:g}°"))
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.92)
    ax.set_title(f"Springfield capture — orientation map; {n_keep} query arrows kept "
                 f"({' + '.join(cli.sweeps)}), {n_drop} excluded"
                 f"{' and removed' if drop_excluded else ' in grey'}", fontsize=9, loc="left")
    scale_bar(ax, lat0, 100.0)
    attribution(ax, cli.basemap)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=cli.dpi, facecolor="white")
    plt.close(fig)


LAYERS = ("arrows", "dots", "circles", "tridents", "basemap")


def frame(orig, cli):
    """The figure and axes every layer shares: same bounds, same size, transparent, no chrome."""
    b = orig["bounds"]
    lat0 = (b[0][0] + b[1][0]) / 2
    k = 1.0 / metre_scale(lat0)
    corners = to_mercator([b[0][0], b[1][0]], [b[0][1], b[1][1]])
    pad = cli.pad_m * k
    bounds = (corners[:, 0].min() - pad, corners[:, 0].max() + pad,
              corners[:, 1].min() - pad, corners[:, 1].max() + pad)
    fig = plt.figure(figsize=(cli.width_in, cli.width_in * (bounds[3] - bounds[2])
                              / (bounds[1] - bounds[0])))
    ax = fig.add_axes([0, 0, 1, 1])           # the whole canvas: layers register pixel for pixel
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig, ax, bounds, lat0, k


def write_layer(orig, info, cli, stem, layer):
    """One layer by itself — arrows, dots, circles, tridents or basemap — as PDF + PNG."""
    fig, ax, bounds, lat0, k = frame(orig, cli)
    if layer == "basemap":
        key = resolve_key(cli.basemap, {"carto": cli.carto_key, "stadia": cli.stadia_key})
        draw_basemap(ax, cli.basemap, bounds, cli.zoom, key, lat=lat0,
                     pt_per_merc_m=cli.width_in * 72 / (bounds[1] - bounds[0]))
        ax.axis("off")
    elif layer == "dots":
        dots = to_mercator(*np.array(orig["dots"]).T)
        ax.plot(dots[:, 0], dots[:, 1], "o", ms=3.2, mfc=BLUE, mec=PAPER, mew=0.6, ls="none")
    elif layer == "circles":
        from matplotlib.patches import Circle
        for c in orig["circles"]:
            x, y = to_mercator([c["lat"]], [c["lon"]])[0]
            ax.add_patch(Circle((x, y), c["radius_m"] * k, edgecolor=c["colour"],
                                facecolor=matplotlib.colors.to_rgba(c["colour"], c["fill_opacity"]),
                                lw=c["weight"] * 0.5))
    elif layer == "tridents":
        for t in orig["tridents"]:
            x, y = to_mercator([t["lat"]], [t["lon"]])[0]
            ax.plot([x], [y], marker=glyph(TRIDENT_PATH, t["travel"]), ms=cli.glyph_pt,
                    mec=BLUE, mew=cli.stroke_pt, mfc="none", ls="none")
    elif layer == "arrows":
        n = 0
        for a in orig["arrows"]:
            if not is_kept(info[a["session"]], cli):
                continue
            n += 1
            x, y = to_mercator([a["lat"]], [a["lon"]])[0]
            ax.plot([x], [y], marker=glyph(ARROW_PATH, a["facing"]), ms=cli.glyph_pt * 0.95,
                    mfc=a["colour"], mec=a["colour"], mew=cli.stroke_pt, ls="none")
    fig.savefig(stem + ".pdf", transparent=True)
    fig.savefig(stem + ".png", dpi=cli.dpi, transparent=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--original", default=DEFAULT_ORIGINAL,
                    help="the map_orientation.html to re-render")
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--diag-dir", default=DEFAULT_DIAG)
    ap.add_argument("--sweeps", nargs="+", default=["day", "dawn"], choices=("day", "dawn", "night"))
    ap.add_argument("--exclude-reversals-deg", type=float, default=135.0)
    ap.add_argument("--basemap", default=None,
                    help="carto-nolabels (the original's tiles; needs --carto-key/CARTO_KEY) | "
                         "osm-vector (streets drawn, no names, no key; the default without a "
                         "key) | esri-imagery | osm | stadia-<style>")
    ap.add_argument("--carto-key", default=None, help="or set CARTO_KEY")
    ap.add_argument("--stadia-key", default=None, help="or set STADIA_KEY")
    ap.add_argument("--zoom", type=int, default=17, help="tile zoom for a raster PDF basemap")
    ap.add_argument("--glyph-pt", type=float, default=11.0, help="glyph size in points")
    ap.add_argument("--stroke-pt", type=float, default=0.1,
                    help="stroke width of every arrow and trident in the PDF/PNG (no halos)")
    ap.add_argument("--keep-excluded", action="store_true",
                    help="also write map_orientation.* with the excluded arrows in grey")
    ap.add_argument("--layers", action="store_true",
                    help="also write each layer by itself (arrows, dots, circles, tridents, "
                         "basemap) as transparent PDF + PNG in the same frame, under layers/")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--pad-m", type=float, default=40.0)
    ap.add_argument("--width-in", type=float, default=11.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(DEFAULT_RESULTS), "maps"))
    cli = ap.parse_args()
    if cli.basemap is None:
        cli.basemap = ("carto-nolabels" if (cli.carto_key or os.environ.get("CARTO_KEY"))
                       else "osm-vector")
        if cli.basemap == "osm-vector":
            print("no CARTO key — drawing streets as vectors (pass --carto-key for the "
                  "original tiles)")

    orig = read_original(cli.original)
    info = exclusion(cli, [a["session"] for a in orig["arrows"]])
    print(f"{cli.original}: {len(orig['arrows'])} query arrows, {len(orig['tridents'])} "
          f"tridents, {len(orig['dots'])} route dots, {len(orig['circles'])} region circles")
    os.makedirs(cli.out_dir, exist_ok=True)
    with open(os.path.join(cli.out_dir, "map_orientation_geometry.json"), "w") as f:
        json.dump({"source": os.path.abspath(cli.original), **orig,
                   "sessions": {a["session"]: {**info[a["session"]],
                                               "kept": is_kept(info[a["session"]], cli)}
                                for a in orig["arrows"]}}, f, indent=1)
    print(f"  {len(orig['circles'])} region circles: "
          + "  ".join(f"{c['colour']} r={c['radius_m']:.0f} m" for c in orig["circles"]))
    if cli.layers:
        ldir = os.path.join(cli.out_dir, "layers")
        os.makedirs(ldir, exist_ok=True)
        for layer in LAYERS:
            write_layer(orig, info, cli, os.path.join(ldir, f"map_orientation_{layer}"), layer)
            print(f"  layer {layer:9s} -> {ldir}/map_orientation_{layer}.pdf .png")
    variants = [("map_orientation_kept", True)]
    if cli.keep_excluded:
        variants.append(("map_orientation", False))
    for name, drop in variants:
        stem = os.path.join(cli.out_dir, name)
        n_keep, n_drop = write_html(orig, info, cli, stem + ".html", drop)
        write_figure(orig, info, cli, stem, drop)
        print(f"  {name}: {n_keep} kept arrows, {n_drop} excluded "
              f"({'removed' if drop else 'grey'}) -> {stem}.html .pdf .png")


if __name__ == "__main__":
    main()
