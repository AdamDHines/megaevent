"""Folium map of Springfield query passes coloured by their evaluation outcome.

Run from the repo root::

    CONDA_OVERRIDE_CUDA=12 pixi run python3 scripts/springfield_failure_map.py --sweep dawn

Reads a ``results_*.json`` written by :mod:`scripts.springfield_full` and draws, over the
grey database tracks, every query session's GPS track coloured by its within-visit R@1
(green >= 0.5, orange 0.05-0.5, red < 0.05), with the per-session numbers in the popup.
The point is triage: *where* the failures are — the reverse-direction passes fail on the
same footpath the with-route passes succeed on, while a hard spot is red in every
direction.
"""

import argparse
import glob
import json
import os

import folium
import numpy as np

DEFAULT_RESULTS = ("output/springfield_full/"
                   "results_b_v8_accum_s750_s750_3f4eb54780_accumulate_r322_hp1_baoff.json")
DEFAULT_ROOT = "/media/adam/vprdatasets/megaevent/springfield/sessions"


def track(sdir, every=5):
    rows = np.genfromtxt(os.path.join(sdir, "derived", "gps_interp.csv"),
                         delimiter=",", names=True, dtype=np.float64)
    return np.column_stack([rows["lat"], rows["lon"]])[::every]


def colour(r1):
    if r1 is None:
        return "#888888"
    return "#2a9d4e" if r1 >= 0.5 else "#e8a13a" if r1 >= 0.05 else "#d43d3d"


def fmt(v, spec="{}"):
    return spec.format(v) if v is not None else "—"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--sweep", default="dawn", choices=("day", "dawn", "night"))
    ap.add_argument("--out", default=None,
                    help="output html (default: maps/failure_map_<sweep>_<filter>.html "
                         "beside the results json)")
    cli = ap.parse_args()

    with open(cli.results) as f:
        res = json.load(f)
    rows = [r for r in res["per_session"] if r["sweep"] == cli.sweep]
    if not rows:
        raise SystemExit(f"no {cli.sweep} sessions in {cli.results}")
    filter_tag = "baoff" if res["filter_dt_us"] is None else f"ba{res['filter_dt_us'] // 1000}"
    out = cli.out or os.path.join(os.path.dirname(cli.results), "maps",
                                  f"failure_map_{cli.sweep}_{filter_tag}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Database tracks: light grey context under everything.
    db_tracks = {sid: track(os.path.join(cli.root, "database", sid), every=10)
                 for sid in res["database"]}
    centre = np.concatenate(list(db_tracks.values())).mean(axis=0)
    m = folium.Map(location=centre.tolist(), zoom_start=16, tiles="OpenStreetMap")
    for sid, t in db_tracks.items():
        folium.PolyLine(t.tolist(), color="#9a9a9a", weight=2, opacity=0.6,
                        tooltip=f"database {res['database'][sid]['arm']} ({sid})"
                        ).add_to(m)

    # Query passes, coloured by outcome; spot labels at cluster centroids.
    spot_pts = {}
    for r in sorted(rows, key=lambda r: r["session"]):
        sdir = os.path.join(cli.root, f"query_{cli.sweep}", r["session"])
        t = track(sdir, every=2)
        spot_pts.setdefault(r["spot"], []).append(t.mean(axis=0))
        c = colour(r["r1_slice"])
        popup = folium.Popup(
            f"<b>{r['session']}</b> ({r['spot']})<br>"
            f"R@1 {fmt(r['r1_slice'], '{:.2f}')} &nbsp; "
            f"R@10 {fmt(r['r10_slice'], '{:.2f}')}<br>"
            f"heading offset {fmt(r['heading_offset_deg'], '{:.0f}&deg;')}<br>"
            f"top-1 median {fmt(r['top1_median_m'], '{:.0f} m')}<br>"
            f"path {r['path_m']:.0f} m, net {r['net_disp_m']:.0f} m<br>"
            f"{'clock: ' + r['clock_flag'] if r['clock_flag'] else ''}",
            max_width=320)
        folium.PolyLine(t.tolist(), color=c, weight=5, opacity=0.9,
                        tooltip=f"{r['session']}  R@1 {fmt(r['r1_slice'], '{:.2f}')}"
                        ).add_to(m)
        folium.CircleMarker(t[0].tolist(), radius=6, color=c, fill=True,
                            fill_opacity=1.0, popup=popup,
                            tooltip=f"{r['session']} start").add_to(m)
    for spot, pts in spot_pts.items():
        folium.Marker(np.mean(pts, axis=0).tolist(),
                      icon=folium.DivIcon(html=f"<div style='font: bold 13px sans-serif;"
                                               f"color:#222; text-shadow: 0 0 3px #fff,"
                                               f"0 0 3px #fff'>{spot}</div>")).add_to(m)

    legend = f"""
    <div style="position: fixed; bottom: 18px; left: 18px; z-index: 9999;
                background: rgba(255,255,255,.92); padding: 10px 14px; font: 12px
                sans-serif; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.3)">
      <b>{cli.sweep} query passes &times; pooled DB ({filter_tag})</b><br>
      within-visit R@1 @ {res['threshold_m']:g} m:<br>
      <span style="color:#2a9d4e">&#9632;</span> &ge; 0.50 &nbsp;
      <span style="color:#e8a13a">&#9632;</span> 0.05&ndash;0.50 &nbsp;
      <span style="color:#d43d3d">&#9632;</span> &lt; 0.05<br>
      <span style="color:#9a9a9a">&#9632;</span> database tracks (7 sessions)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.save(out)
    n_fail = sum(1 for r in rows if (r["r1_slice"] or 0) < 0.05)
    print(f"{len(rows)} {cli.sweep} sessions ({n_fail} failing < 0.05 R@1) -> {out}")


if __name__ == "__main__":
    main()
