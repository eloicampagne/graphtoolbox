"""GatedGraphConv against pooled XGBoost on French load, 2019 test set: regional accuracy and error compensation.

One map of France per month: each region is filled with the RMSE of GatedGraphConv (the run of the load table,
    ./load_sweep_v030.npz) divided by that of pooled XGBoost (seed 42, regional forecasts cached by
    calendar_diagnosis.py) over the half-hours of the month. Blue: the graph model is more accurate.
The json also holds, for pooled XGBoost, GatedGraphConv and the average of its five seeds
(preds/GatedGraphConv_seed*.pt), the national RMSE and sqrt(sum of the regional mean squared errors), the national
RMSE if the regional errors were uncorrelated. Region boundaries: ./geo/regions.geojson.

Usage: python figure_regional_vs_xgb_v030.py [OUTPUT.pdf]     # also writes figure_regional_vs_xgb_v030.json
"""
import json
import os
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "regional_vs_xgb_v030.pdf"))
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 7, "pdf.fonttype": 42})
REGIONS = ["Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté", "Bretagne", "Centre-Val de Loire", "Grand Est",
           "Hauts-de-France", "Île-de-France", "Normandie", "Nouvelle-Aquitaine", "Occitanie",
           "Pays de la Loire", "Provence-Alpes-Côte d'Azur"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

with np.load(os.path.join(HERE, "load_sweep_v030.npz")) as z:
    Y, G = z["target_nodes"].astype(float), z["GatedGraphConv"].astype(float)
X = np.load(os.path.join(HERE, "calendar_diagnosis_xgb.npy"))
S = [torch.load(os.path.join(HERE, "preds", f"GatedGraphConv_seed{s}.pt"), weights_only=False).numpy().astype(float)
     for s in (42, 0, 1, 2, 3)]
dates = pd.date_range("2019-01-01", periods=Y.shape[1], freq="30min")
rm = lambda P, k, m: np.sqrt(np.mean((P[k, dates.month == m] - Y[k, dates.month == m]) ** 2))
ratio = np.array([[rm(G, k, m) / rm(X, k, m) for m in range(1, 13)] for k in range(len(REGIONS))])


def split(P):
    """National MSE = sum of regional MSEs + cross-region terms, and the mean correlation of regional errors."""
    E = P - Y
    C = E @ E.T / E.shape[1]
    corr = np.corrcoef(E)[~np.eye(len(E), dtype=bool)].mean()
    return {"regional": float(np.trace(C)), "cross": float(C.sum() - np.trace(C)),
            "national_rmse": float(np.sqrt(C.sum())), "mean_error_correlation": float(corr),
            "regional_rmse": np.sqrt(np.diag(C)).tolist()}


models = {"XGBoost (pooled)": X, "GatedGraphConv": G, "GatedGraphConv,\nfive seeds averaged": np.mean(S, 0)}
dec = {k.replace("\n", " "): split(P) for k, P in models.items()}
worse = ratio >= 1
res = {"ratio": ratio.round(3).tolist(), "cells_gnn_lower": int((ratio < 1).sum()),
       "cells_gnn_not_lower_by_month": dict(zip([m[:3] for m in MONTHS], worse.sum(0).tolist())),
       "decomposition": dec}
json.dump(res, open(os.path.join(HERE, "figure_regional_vs_xgb_v030.json"), "w"), indent=1)
print("cells where GatedGraphConv is more accurate:", res["cells_gnn_lower"], "of", ratio.size)
print("cells where it is not, by month:", res["cells_gnn_not_lower_by_month"])
for k, d in dec.items():
    print(f"{k:<40} national {d['national_rmse']:.0f} MW, regional {d['regional'] / 1e3:.0f}k, cross {d['cross'] / 1e3:.0f}k "
          f"({100 * d['cross'] / (d['regional'] + d['cross']):.0f}%), mean correlation {d['mean_error_correlation']:.2f}")

geo = gpd.read_file(os.path.join(HERE, "geo", "regions.geojson"))
geo = geo[geo["nom"].isin(REGIONS)].to_crs(2154).set_index("nom").loc[REGIONS]
# Diverging bins around 1: blue where the graph model is more accurate, orange where XGBoost is, gray in between.
bounds = [0.4, 0.55, 0.7, 0.85, 0.95, 1.05, 1.2, 1.4]
cmap = ListedColormap(["#0d3b73", "#2a6fbf", "#79a9de", "#c3d8f0", "#e4e3df", "#f4b98f", "#d9622b"])
norm = BoundaryNorm(bounds, cmap.N)

fig = plt.figure(figsize=(7.16, 2.75))
gs = fig.add_gridspec(2, 6, wspace=0.03, hspace=0.12, left=0.005, right=0.995, top=0.93, bottom=0.16)
for m in range(12):
    ax = fig.add_subplot(gs[m // 6, m % 6])
    geo.assign(r=ratio[:, m]).plot(column="r", cmap=cmap, norm=norm, ax=ax, edgecolor="white", linewidth=0.35)
    ax.set_title(MONTHS[m], fontsize=7, color=INK, pad=1.5)
    ax.set_axis_off(); ax.set_aspect("equal")
cax = fig.add_axes([0.25, 0.075, 0.5, 0.035])
ticks = sorted(bounds + [1.0])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, orientation="horizontal",
                  spacing="uniform", ticks=ticks)
cb.ax.set_xticklabels([f"{b:g}" for b in ticks], fontsize=6.3, color=MUTED)
# Dashed mark at a ratio of 1, where both models have the same RMSE: bins are evenly spaced on the bar.
k = np.searchsorted(bounds, 1.0) - 1
one = (k + (1.0 - bounds[k]) / (bounds[k + 1] - bounds[k])) / (len(bounds) - 1)
cb.ax.plot([one, one], [0, 1], transform=cb.ax.transAxes, color=INK, lw=0.8, ls=(0, (2, 1.5)))
cb.outline.set_linewidth(0.4); cb.ax.tick_params(length=2, width=0.4, color=MUTED)
cb.set_label("RMSE of GatedGraphConv / RMSE of pooled XGBoost, by region and month", fontsize=6.6, color=INK,
             labelpad=2)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=220)
print("written", OUT)
