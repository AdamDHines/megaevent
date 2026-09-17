"""Slippy-map basemaps for the Springfield figures: tiles fetched once, drawn under matplotlib
vector overlays (PDF/PNG), and named for folium (HTML).

Providers — and what each one costs::

    osm-vector       streets drawn HERE from OpenStreetMap road geometry (Overpass API):
                     no names, no key, no raster — the PDF stays fully vector and editable.
                     The default.
    esri-gray        Esri Light Gray Canvas: streets WITH names at the zooms it serves (<= 16
                     here), no key
    esri-imagery     Esri World Imagery: satellite, no labels, no key
    osm              OpenStreetMap standard: streets WITH names, no key (a User-Agent is sent)
    carto-nolabels   CARTO light, no labels     } KEY REQUIRED (since 2026): request one at
    carto-light      CARTO light with labels    } https://carto.com/basemaps/apikey — email,
    carto-dark       CARTO dark, no labels      } domain, purpose; no account; the key is
    carto-voyager    CARTO voyager with labels  } emailed back. Pass --carto-key or CARTO_KEY.
    stadia-<style>   Stadia Maps (stamen_toner_lite, stamen_toner_lines, alidade_smooth, ...):
                     key from https://client.stadiamaps.com (free tier), --stadia-key/STADIA_KEY

Without its key a CARTO tile is a grey "API KEY REQUIRED" placeholder — every tile, all
zooms — which is what a folium map built on ``CartoDB positron`` shows today. The helper
refuses a keyed provider without a key rather than draw that.

Tiles are cached under ``<cache>/<provider>/<z>/<x>/<y>.<ext>``; a figure re-render costs no
network. Everything is composed in Web Mercator metres (EPSG:3857), the tiles' own frame, so
vector overlays sit exactly on the raster; locally the projection is conformal, so shapes and
bearings are right and only the scale bar needs the ``1/cos(lat)`` factor.
"""

import hashlib
import io
import json
import math
import os
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

R = 6378137.0                       # WGS84 semi-major axis, the Web Mercator sphere
WORLD = 2 * math.pi * R
USER_AGENT = "megaevent-figures/1.0 (research use; https://github.com/)"
CARTO_ATTR = ("© OpenStreetMap contributors © CARTO")
ESRI_ATTR = "Tiles © Esri"
OSM_ATTR = "© OpenStreetMap contributors"

PROVIDERS = {
    "osm-vector": (None, OSM_ATTR, None, False),
    "esri-gray": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                  "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}", ESRI_ATTR, None, False),
    "esri-imagery": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
                     "MapServer/tile/{z}/{y}/{x}", ESRI_ATTR, None, False),
    "osm": ("https://tile.openstreetmap.org/{z}/{x}/{y}.png", OSM_ATTR, None, True),
    "carto-nolabels": ("https://basemaps.cartocdn.com/rastertiles/light_nolabels/{z}/{x}/{y}.png"
                       "?key={key}", CARTO_ATTR, "carto", False),
    "carto-light": ("https://basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png"
                    "?key={key}", CARTO_ATTR, "carto", True),
    "carto-dark": ("https://basemaps.cartocdn.com/rastertiles/dark_nolabels/{z}/{x}/{y}.png"
                   "?key={key}", CARTO_ATTR, "carto", False),
    "carto-voyager": ("https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
                      "?key={key}", CARTO_ATTR, "carto", True),
}
# Where a service stops serving real tiles for this area — beyond it every tile is a "Map data
# not yet available" placeholder, which a figure must never carry. Others reach 19+.
MAX_ZOOM = {"esri-gray": 16}
KEY_ENV = {"carto": "CARTO_KEY", "stadia": "STADIA_KEY"}
KEY_HELP = {"carto": "request one at https://carto.com/basemaps/apikey (emailed back, no "
                     "account) and pass --carto-key or set CARTO_KEY",
            "stadia": "create one at https://client.stadiamaps.com (free tier) and pass "
                      "--stadia-key or set STADIA_KEY"}


def provider_spec(name):
    """``(url template, attribution, key kind, has labels)`` — ``stadia-<style>`` is open-ended."""
    if name in PROVIDERS:
        return PROVIDERS[name]
    if name.startswith("stadia-"):
        style = name[len("stadia-"):]
        return (f"https://tiles.stadiamaps.com/tiles/{style}/{{z}}/{{x}}/{{y}}.png?api_key={{key}}",
                "© Stadia Maps © Stamen Design © OpenMapTiles © OpenStreetMap contributors",
                "stadia", "lines" not in style and "background" not in style)
    raise SystemExit(f"unknown basemap {name!r}; choose from {', '.join(PROVIDERS)} or stadia-<style>")


def resolve_key(name, keys):
    """The key a provider needs, from ``keys`` (a dict by kind) or the environment; refuses
    a keyed provider without one rather than draw placeholder tiles."""
    _, _, kind, _ = provider_spec(name)
    if kind is None:
        return None
    key = (keys or {}).get(kind) or os.environ.get(KEY_ENV[kind])
    if not key:
        raise SystemExit(f"basemap {name} needs a {kind} API key — {KEY_HELP[kind]}")
    return key


def tile_url(name, z, x, y, key=None):
    url, _, _, _ = provider_spec(name)
    return url.format(z=z, x=x, y=y, key=key or "")


def folium_tiles(name, key=None):
    """``(url template, attribution)`` for ``folium.Map(tiles=..., attr=...)``; the vector
    provider has no tiles — pass ``tiles=None`` and call :func:`folium_streets`."""
    url, attr, _, _ = provider_spec(name)
    if url is None:
        return None, attr
    return url.replace("{key}", key or ""), attr


# ---------------------------------------------------------------------------
# Vector streets from OpenStreetMap (Overpass)
# ---------------------------------------------------------------------------
OVERPASS = "https://overpass-api.de/api/interpreter"
# (line width in ground metres, fill, casing) per highway class; unlisted classes use "other".
ROAD_STYLE = {
    "motorway": (14.0, "#ffffff", "#c4c2bc"), "trunk": (14.0, "#ffffff", "#c4c2bc"),
    "primary": (12.0, "#ffffff", "#c4c2bc"), "secondary": (10.0, "#ffffff", "#c8c6c0"),
    "tertiary": (9.0, "#ffffff", "#cbc9c3"), "residential": (7.0, "#ffffff", "#cfcdc7"),
    "unclassified": (7.0, "#ffffff", "#cfcdc7"), "living_street": (6.0, "#ffffff", "#cfcdc7"),
    "service": (4.0, "#ffffff", "#d6d4ce"), "footway": (2.2, "#e2e0da", None),
    "path": (2.2, "#e2e0da", None), "cycleway": (2.2, "#e2e0da", None),
    "pedestrian": (3.0, "#e2e0da", None), "track": (2.2, "#e2e0da", None),
    "steps": (2.2, "#e2e0da", None), "other": (3.0, "#e8e6e0", None),
}
BACKGROUND = "#f5f4f1"
BUILDING = "#e4e2dd"


def fetch_osm_features(lat_min, lat_max, lon_min, lon_max, cache=None):
    """Roads and buildings inside the box, from Overpass, cached as JSON by box.

    -> ``{"roads": [(highway, [[lat, lon], ...]), ...], "buildings": [[[lat, lon], ...], ...]}``
    """
    cache = cache or os.path.join(os.path.expanduser("~"), ".cache", "megaevent_tiles")
    box = f"{lat_min:.5f},{lon_min:.5f},{lat_max:.5f},{lon_max:.5f}"
    path = os.path.join(cache, "osm-vector", hashlib.md5(box.encode()).hexdigest() + ".json")
    if not os.path.exists(path):
        query = (f"[out:json][timeout:90];(way[\"highway\"]({box});"
                 f"way[\"building\"]({box}););out geom;")
        req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": query}).encode(),
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode())
        roads, buildings = [], []
        for el in raw.get("elements", []):
            if el.get("type") != "way" or "geometry" not in el:
                continue
            coords = [[g["lat"], g["lon"]] for g in el["geometry"]]
            tags = el.get("tags", {})
            if "highway" in tags:
                roads.append((tags["highway"], coords))
            elif "building" in tags:
                buildings.append(coords)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"box": box, "roads": roads, "buildings": buildings}, f)
        print(f"  overpass: {len(roads)} roads, {len(buildings)} buildings for {box}")
    with open(path) as f:
        return json.load(f)


def draw_vector_streets(ax, bounds_m, lat, pt_per_merc_m, cache=None, buildings=True):
    """Paint the box background, buildings and roads as vector artists in Mercator metres."""
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.patches import Rectangle

    k = 1.0 / metre_scale(lat)
    xmin, xmax, ymin, ymax = bounds_m
    lat_min = math.degrees(2 * math.atan(math.exp(ymin / R)) - math.pi / 2)
    lat_max = math.degrees(2 * math.atan(math.exp(ymax / R)) - math.pi / 2)
    lon_min, lon_max = math.degrees(xmin / R), math.degrees(xmax / R)
    feats = fetch_osm_features(lat_min, lat_max, lon_min, lon_max, cache)
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor=BACKGROUND,
                           edgecolor="none", zorder=0))
    if buildings and feats["buildings"]:
        polys = [to_mercator(*np.array(b).T) for b in feats["buildings"] if len(b) >= 3]
        ax.add_collection(PolyCollection(polys, facecolors=BUILDING, edgecolors="none",
                                         zorder=0.2))
    for pass_no in (0, 1):                      # casings first, then fills, so joins are clean
        for cls_order in (("motorway", "trunk", "primary", "secondary", "tertiary",
                           "residential", "unclassified", "living_street", "service",
                           "pedestrian", "other", "footway", "path", "cycleway", "track",
                           "steps"),):
            for cls in cls_order:
                segs = [to_mercator(*np.array(c).T) for h, c in feats["roads"]
                        if (h if h in ROAD_STYLE else "other") == cls]
                if not segs:
                    continue
                width_m, fill, casing = ROAD_STYLE[cls]
                lw = width_m * k * pt_per_merc_m
                if pass_no == 0 and casing:
                    ax.add_collection(LineCollection(segs, colors=casing, linewidths=lw + 1.2,
                                                     capstyle="round", joinstyle="round",
                                                     zorder=0.4))
                elif pass_no == 1:
                    ax.add_collection(LineCollection(segs, colors=fill, linewidths=lw,
                                                     capstyle="round", joinstyle="round",
                                                     zorder=0.6))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return feats


def folium_streets(m, feats):
    """The same roads and buildings as folium layers over a blank map."""
    import folium

    for b in feats["buildings"]:
        folium.Polygon(b, color=BUILDING, weight=0, fill=True, fill_color=BUILDING,
                       fill_opacity=1.0).add_to(m)
    for h, coords in feats["roads"]:
        width_m, fill, casing = ROAD_STYLE.get(h, ROAD_STYLE["other"])
        if casing:
            folium.PolyLine(coords, color=casing, weight=max(2, width_m * 0.9) + 2,
                            opacity=1.0).add_to(m)
    for h, coords in feats["roads"]:
        width_m, fill, casing = ROAD_STYLE.get(h, ROAD_STYLE["other"])
        folium.PolyLine(coords, color=fill, weight=max(2, width_m * 0.9), opacity=1.0).add_to(m)
    m.get_root().html.add_child(folium.Element(
        f"<style>.leaflet-container {{ background: {BACKGROUND}; }}</style>"))


# ---------------------------------------------------------------------------
# Web Mercator
# ---------------------------------------------------------------------------
def to_mercator(lat, lon):
    """Degrees -> EPSG:3857 metres, ``[N, 2]``."""
    lat = np.clip(np.asarray(lat, dtype=np.float64), -85.05, 85.05)
    lon = np.asarray(lon, dtype=np.float64)
    x = R * np.radians(lon)
    y = R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return np.column_stack([x, y])


def metre_scale(lat):
    """Ground metres per Mercator metre at this latitude."""
    return math.cos(math.radians(lat))


def _tile_of(xm, ym, z):
    n = 2 ** z
    tx = (xm + WORLD / 2) / WORLD * n
    ty = (WORLD / 2 - ym) / WORLD * n
    return tx, ty


def fetch_tile(name, z, x, y, key, cache):
    """One tile as an RGB(A) array, from the cache or the network; ``None`` if unavailable."""
    ext = "png"
    path = os.path.join(cache, name, str(z), str(x), f"{y}.{ext}")
    if not os.path.exists(path):
        req = urllib.request.Request(tile_url(name, z, x, y, key),
                                     headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
        except Exception as exc:                     # noqa: BLE001 — a hole in the basemap
            print(f"  tile {name} {z}/{x}/{y}: {exc}")
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.open(io.BytesIO(data)).convert("RGBA").save(path)
    return np.asarray(Image.open(path).convert("RGBA"))


def basemap_image(name, bounds_m, zoom, key=None, cache=None):
    """``(image, extent)`` covering ``bounds_m = (xmin, xmax, ymin, ymax)`` in Mercator metres.

    ``extent`` is the imshow extent of the composed image (tile-aligned, so slightly larger than
    ``bounds_m``), with y increasing upward.
    """
    cache = cache or os.path.join(os.path.expanduser("~"), ".cache", "megaevent_tiles")
    cap = MAX_ZOOM.get(name, 19)
    if zoom > cap:
        print(f"  basemap {name}: zoom {zoom} not served here, using {cap}")
        zoom = cap
    xmin, xmax, ymin, ymax = bounds_m
    tx0, ty1 = _tile_of(xmin, ymin, zoom)
    tx1, ty0 = _tile_of(xmax, ymax, zoom)
    x0, x1 = int(math.floor(tx0)), int(math.floor(tx1))
    y0, y1 = int(math.floor(ty0)), int(math.floor(ty1))
    n = 2 ** zoom
    size = WORLD / n
    canvas = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256, 4), dtype=np.uint8)
    canvas[..., :3] = 235                 # a neutral fill where a tile is missing
    canvas[..., 3] = 255
    missing = 0
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            tile = fetch_tile(name, zoom, x, y, key, cache)
            if tile is None:
                missing += 1
                continue
            if tile.shape[:2] != (256, 256):
                tile = np.asarray(Image.fromarray(tile).resize((256, 256)))
            canvas[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = tile
    extent = (-WORLD / 2 + x0 * size, -WORLD / 2 + (x1 + 1) * size,
              WORLD / 2 - (y1 + 1) * size, WORLD / 2 - y0 * size)
    if missing:
        print(f"  basemap {name} z{zoom}: {missing} of {(x1 - x0 + 1) * (y1 - y0 + 1)} tiles missing")
    return canvas, extent


def draw_basemap(ax, name, bounds_m, zoom, key=None, cache=None, alpha=1.0, lat=None,
                 pt_per_merc_m=None):
    """Paint the basemap under ``ax`` (Mercator metres) and clip the axes to ``bounds_m``.

    ``osm-vector`` needs ``lat`` (for road widths) and ``pt_per_merc_m`` (points per Mercator
    metre of the axes, so widths are drawn to scale); returns the features it drew.
    """
    if name == "osm-vector":
        return draw_vector_streets(ax, bounds_m, lat, pt_per_merc_m, cache)
    image, extent = basemap_image(name, bounds_m, zoom, key, cache)
    ax.imshow(image, extent=extent, origin="upper", interpolation="bilinear", zorder=0,
              alpha=alpha)
    ax.set_xlim(bounds_m[0], bounds_m[1])
    ax.set_ylim(bounds_m[2], bounds_m[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return extent


def scale_bar(ax, lat, length_m=100.0, loc=(0.04, 0.05), colour="#222222"):
    """A ground-metre scale bar, corrected for Mercator's latitude scale."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    merc_len = length_m / metre_scale(lat)
    bx = x0 + loc[0] * (x1 - x0)
    by = y0 + loc[1] * (y1 - y0)
    ax.plot([bx, bx + merc_len], [by, by], color=colour, lw=2.2, solid_capstyle="butt", zorder=20)
    ax.text(bx + merc_len / 2, by + 0.012 * (y1 - y0), f"{length_m:g} m", ha="center",
            va="bottom", fontsize=7.5, color=colour, zorder=20)


def attribution(ax, name, colour="#444444"):
    _, attr, _, _ = provider_spec(name)
    ax.text(0.995, 0.006, attr, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5,
            color=colour, zorder=20,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.7))


def placeholder_md5():
    """md5 of CARTO's keyless "API KEY REQUIRED" tile, for tests."""
    return hashlib.md5(b"").hexdigest()
