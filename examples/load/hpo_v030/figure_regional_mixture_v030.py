"""Map of the bottom-up MLpol mixture: the three heaviest experts of each region.

For each of the 12 regions, reruns the regional MLpol of the paper's bottom-up
aggregate (GraphToolbox Aggregation, one-day blocks, 46 experts) on
./load_sweep_v030.npz, fills the region with the color of its heaviest expert,
and lists its three heaviest experts with their year-mean weights. Region
boundaries: ./geo/regions.geojson (france-geojson, simplified regions).

Usage: python figure_regional_mixture_v030.py [OUTPUT.pdf]
"""
import os
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from operator_colors import operator_color
from graphtoolbox.aggregation import Aggregation

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "regional_mixture_v030.pdf"))
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
INK, MUTED, EDGE = "#0b0b0b", "#52514e", "#ffffff"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 7, "legend.fontsize": 6.3, "pdf.fonttype": 42})
# Node order of the datasets (regions sorted by their RTE name) and the matching geojson names.
REGIONS = ["Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté", "Bretagne", "Centre-Val de Loire", "Grand Est",
           "Hauts-de-France", "Île-de-France", "Normandie", "Nouvelle-Aquitaine", "Occitanie",
           "Pays de la Loire", "Provence-Alpes-Côte d'Azur"]
# Label anchors (lon, lat) where the region centroid leaves the text cramped; others use the centroid.
ANCHOR = {"Hauts-de-France": (2.75, 50.0), "Normandie": (0.2, 49.1), "Bretagne": (-3.0, 48.15),
          "Provence-Alpes-Côte d'Azur": (6.3, 43.85), "Centre-Val de Loire": (1.45, 47.6),
          "Bourgogne-Franche-Comté": (5.0, 47.15),
          "Pays de la Loire": (-1.65, 47.0)}
# Île-de-France is too small for its label, which sits north-east of the map with a leader line.
OUTSIDE = {"Île-de-France": (7.0, 50.75)}

with np.load(os.path.join(HERE, "load_sweep_v030.npz")) as z:
    arr = {k: z[k] for k in z.files}
Yn = arr["target_nodes"].astype(float)
ops = sorted(k for k in arr if k not in {"target", "target_nodes"} and not k.startswith("aggregate__"))
top3 = {}
for k, name in enumerate(REGIONS):
    X = np.column_stack([arr[o][k] for o in ops]).astype(float)
    w = Aggregation(model="MLpol", loss="square").run(X, Yn[k], block_size=48).weights_.mean(axis=0)
    top3[name] = [(ops[i], w[i]) for i in np.argsort(-w)[:3]]

leaders = sorted({v[0][0] for v in top3.values()}, key=lambda o: (-sum(v[0][0] == o for v in top3.values()), o))
color = {o: operator_color(o) for o in leaders}
tint = lambda c, a=0.45: tuple(1 - a * (1 - np.array(to_rgb(c))))
HALO = [pe.withStroke(linewidth=1.4, foreground=(1, 1, 1, 0.75))]
FS, LINE = 5.0, 6.0  # font size and line height (pt)


def label(x, y, rows, title=None):
    """Block of lines centered on (x, y): optional title, then one line per expert with its
    name left-aligned and its weight right-aligned; the heaviest expert in bold."""
    lines = ([(title, None, "normal")] if title else []) + [
        (o.replace("Conv", ""), f"{100 * w:.0f}%", "bold" if i == 0 else "normal") for i, (o, w) in enumerate(rows)]
    size = lambda t, w: TextPath((0, 0), t, size=FS, prop=FontProperties(
        family=plt.rcParams["font.serif"], weight=w)).get_extents().width
    width = (max(size(n, w) for n, p, w in lines if p) + 3.0
             + max(size(p, w) for n, p, w in lines if p))  # points
    top = (len(lines) - 1) * LINE / 2
    for k, (name, pct, weight) in enumerate(lines):
        dy = top - k * LINE
        if pct is None:
            ax.annotate(name, (x, y), xytext=(0, dy), textcoords="offset points", ha="center", va="center",
                        fontsize=FS, color=INK, path_effects=HALO)
            continue
        ax.annotate(name, (x, y), xytext=(-width / 2, dy), textcoords="offset points", ha="left", va="center",
                    fontsize=FS, color=INK, fontweight=weight, path_effects=HALO)
        ax.annotate(pct, (x, y), xytext=(width / 2, dy), textcoords="offset points", ha="right", va="center",
                    fontsize=FS, color=INK, fontweight=weight, path_effects=HALO)


geo = gpd.read_file(os.path.join(HERE, "geo", "regions.geojson"))
fig, ax = plt.subplots(figsize=(3.45, 3.55))
for _, r in geo.iterrows():
    if r.nom not in top3:
        gpd.GeoSeries([r.geometry]).plot(ax=ax, color="#eeeeec", edgecolor=EDGE, linewidth=0.6)
        continue
    gpd.GeoSeries([r.geometry]).plot(ax=ax, color=tint(color[top3[r.nom][0][0]]), edgecolor=EDGE, linewidth=0.8)
    c = r.geometry.representative_point()
    if r.nom in OUTSIDE:
        x, y = OUTSIDE[r.nom]
        ax.annotate("", xy=(c.x, c.y), xytext=(x - 1.35, y - 0.55),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=0))
        ax.plot(c.x, c.y, "o", ms=1.6, color=MUTED)
        label(x, y, top3[r.nom], title="Île-de-France")
        continue
    x, y = ANCHOR.get(r.nom, (c.x, c.y))
    label(x, y, top3[r.nom])
ax.set_xlim(-5.3, 8.4); ax.set_ylim(42.2, 51.2)
ax.set_aspect(1 / np.cos(np.deg2rad(46.5)))
ax.axis("off")
ax.legend(handles=[Patch(facecolor=tint(color[o]), edgecolor="none", label=o.replace("Conv", "")) for o in leaders],
          title="Heaviest expert", title_fontsize=6.3, frameon=False, loc="lower left",
          bbox_to_anchor=(-0.02, -0.02), handlelength=1.0)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
print("written", OUT); [print(k, [(o, round(100 * w)) for o, w in v]) for k, v in top3.items()]
