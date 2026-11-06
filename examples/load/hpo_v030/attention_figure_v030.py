"""Attention figure of the IEEE Access version.

Spatial graph (attention_v030.py GATConv space, influence_v030.py GATConv space):
within the neighbours j != i of a receiving region i, the attention share of j is
alpha_ij / sum_k alpha_ik and its influence share is I_ij / sum_k I_ik, where I_ij is
the occlusion influence of j on the forecast of i (MW). Both are compared with the
uniform share 1/m_i, m_i the number of neighbours, through m_i * share - 1.
Top row: (a) seasonal maps, one arrow per directed edge, colored by the attention
deviation and with a width proportional to the influence in MW.
Bottom row: (b) year-mean attention deviation against influence deviation, one point
per edge entering a region with several neighbours; (c) DTW graph (attention_v030.py
GATConv): year-mean deviation d_i * alpha_ij - 1 of every edge, receiver x sender.

Usage: python attention_figure_v030.py [OUTPUT.pdf]
"""
import json
import os
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD = os.path.dirname(HERE)
sys.path.insert(0, LOAD)
import config as C  # noqa: E402

OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "attention_v030.pdf"))
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
SPACE = sys.argv[2] if len(sys.argv) > 2 else "GATv2Conv"   # model of the spatial-graph panels
TAG = ("_space" if SPACE == "GATConv" else f"_{SPACE}_space") + (sys.argv[3] if len(sys.argv) > 3 else "")   # e.g. _seed0
INK, MUTED, LAND, SINGLE = "#0b0b0b", "#52514e", "#e4e3de", "#a8a7a1"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7.5, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
                     "pdf.fonttype": 42})
diverging = LinearSegmentedColormap.from_list("bgr", ["#1c5cab", "#86b6ef", "#c9c8c3", "#ef9a99", "#c0302f"])
diverging_cells = LinearSegmentedColormap.from_list("bgr2", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#c0302f"])
DISPLAY = {"Auvergne_Rhone_Alpes": "Auvergne-Rhône-Alpes", "Bourgogne_Franche_Comte": "Bourgogne-Franche-Comté",
           "Bretagne": "Bretagne", "Centre_Val_de_Loire": "Centre-Val de Loire", "Grand_Est": "Grand Est",
           "Hauts_de_France": "Hauts-de-France", "Ile_de_France": "Île-de-France", "Normandie": "Normandie",
           "Nouvelle_Aquitaine": "Nouvelle-Aquitaine", "Occitanie": "Occitanie",
           "Pays_de_la_Loire": "Pays de la Loire", "Provence_Alpes_Cote_d_Azur": "Provence-Alpes-Côte d'Azur"}
SEASONS = [("Winter (Dec–Feb)", (12, 1, 2)), ("Spring (Mar–May)", (3, 4, 5)),
           ("Summer (Jun–Aug)", (6, 7, 8)), ("Autumn (Sep–Nov)", (9, 10, 11))]

# --- spatial graph -----------------------------------------------------------------
# Attention and influence are averaged over three trainings (seeds 42, 0, 1).
SEEDS = {42: "", 0: "_seed0", 1: "_seed1"}
zs = {seed: np.load(os.path.join(HERE, f"attention_v030{TAG}{sfx}.npz")) for seed, sfx in SEEDS.items()}
Is = {seed: np.load(os.path.join(HERE, f"influence_v030{TAG}{sfx}.npz"), allow_pickle=True)["infl"]
      for seed, sfx in SEEDS.items()}
z = zs[42]
assert all(np.array_equal(v["edge_index"], z["edge_index"]) for v in zs.values())
A = np.mean([v["alpha"].mean(axis=1) for v in zs.values()], axis=0)   # [days, edges], layers and seeds averaged
src, dst = z["edge_index"]
n = int(z["edge_index"].max()) + 1
I = np.mean(list(Is.values()), axis=0)                         # [days, recv, send] MW, seeds averaged
dates = pd.date_range(C.data_kwargs["day_inf_test"], periods=A.shape[0], freq="D")
E = np.where(src != dst)[0]                                   # edges between distinct regions
m = np.array([np.sum((dst[E] == dst[e])) for e in E])         # neighbours of the receiver
multi = m > 1


def deviations(a, infl):
    """Attention and influence deviations (%) from the uniform share, per edge of E."""
    an = a[E] / np.array([a[E][dst[E] == dst[e]].sum() for e in E])
    inf = infl[dst[E], src[E]]
    sh = inf / np.array([inf[dst[E] == dst[e]].sum() for e in E])
    return 100 * (m * an - 1), 100 * (m * sh - 1), inf


season = [deviations(A[dates.month.isin(ms)].mean(0), I[dates.month.isin(ms)].mean(0)) for _, ms in SEASONS]
att_y, inf_y, mw_y = deviations(A.mean(0), I.mean(0))
I_mean = I.mean(0)
adj = np.zeros((n, n), bool); adj[dst[E], src[E]] = True
two_hop = ((adj.astype(int) @ adj.astype(int)) > 0) & ~adj & ~np.eye(n, dtype=bool)
stats = {"edges_distinct": int(len(E)), "receivers_several_neighbours": int(len(set(dst[E][multi]))),
         "edges_into_several": int(multi.sum()),
         "self_loop_attention_mean": float(A.mean(0)[src == dst].mean()),
         "influence_edge_MW_min_median_max": [float(mw_y.min()), float(np.median(mw_y)), float(mw_y.max())],
         "influence_two_hop_MW_median": float(np.median(I_mean[two_hop])),
         "influence_self_MW_median": float(np.median(np.diag(I_mean))),
         "att_dev_abs_median_year": float(np.median(np.abs(att_y[multi]))),
         "inf_dev_abs_median_year": float(np.median(np.abs(inf_y[multi]))),
         "spearman_year": float(spearmanr(att_y[multi], inf_y[multi])[0]),
         "spearman_season": [float(spearmanr(a[multi], b[multi])[0]) for a, b, _ in season]}

by_seed = {}
for seed in SEEDS:
    a_s, i_s, _ = deviations(zs[seed]["alpha"].mean(axis=(0, 1)), Is[seed].mean(0))
    by_seed[seed] = (a_s, i_s)
att_seeds = np.stack([v[0] for v in by_seed.values()])        # [seeds, edges of E]
inf_seeds = np.stack([v[1] for v in by_seed.values()])
pairs = [(0, 1), (0, 2), (1, 2)]
stats["spearman_year_by_seed"] = {s_: float(spearmanr(a[multi], b[multi])[0]) for s_, (a, b) in by_seed.items()}
stats["spearman_seed_mean"] = float(spearmanr(att_seeds.mean(0)[multi], inf_seeds.mean(0)[multi])[0])
stats["attention_across_seeds"] = [float(spearmanr(att_seeds[a][multi], att_seeds[b][multi])[0]) for a, b in pairs]
stats["influence_across_seeds"] = [float(spearmanr(inf_seeds[a][multi], inf_seeds[b][multi])[0]) for a, b in pairs]

# --- DTW graph: static attention ------------------------------------------------------
zd = np.load(os.path.join(HERE, "attention_v030.npz"))
Ad = zd["alpha"].mean(axis=1)
sd, dd = zd["edge_index"]
deg = np.bincount(dd, minlength=n)
reld = 100 * (Ad * deg[dd] - 1)
W = np.full((Ad.shape[0], n, n), np.nan); W[:, dd, sd] = Ad
full = [i for i in range(n) if deg[i] == n]
stats["dtw_max_spread_full_receivers"] = float(np.nanmax(np.nanmax(W[:, full], 1) - np.nanmin(W[:, full], 1)))
zv = np.load(os.path.join(HERE, "attention_v030_GATv2Conv.npz"))   # GATv2 on the DTW graph, panel (c)
sd, dd = zv["edge_index"]
reld = 100 * (zv["alpha"].mean(axis=1) * np.bincount(dd, minlength=n)[dd] - 1)
year_d = reld.mean(0)
offd = sd != dd
fit = np.array([year_d[offd & (sd == j)].mean() for j in range(n)])[sd[offd]]
stats["dtw_gatv2_sender_share"] = float(1 - ((year_d[offd] - fit) ** 2).sum() / ((year_d[offd] - year_d[offd].mean()) ** 2).sum())
stats["dtw_year_abs_dev_median"] = float(np.median(np.abs(year_d[sd != dd])))
stats["dtw_year_abs_dev_max"] = float(np.abs(year_d[sd != dd]).max())
json.dump(stats, open(os.path.join(HERE, f"attention_stats_v030{TAG}.json"), "w"), indent=1)

# --- figure -------------------------------------------------------------------------
names = [DISPLAY[r] for r in C.df_pos["REGION"].tolist()[:n]]
geo = gpd.read_file(os.path.join(HERE, "geo", "regions.geojson"))
geo = geo[geo.nom != "Corse"].copy()
geo["node"] = geo.nom.str.replace("’", "'").map({v: i for i, v in enumerate(names)})
assert geo["node"].notna().all()
pts = geo.to_crs(2154).representative_point().to_crs(4326)
xy = np.zeros((n, 2))
for k, i in enumerate(geo["node"].astype(int)):
    xy[i] = pts.iloc[k].x, pts.iloc[k].y
lim = max(np.abs(a[multi]).max() for a, _, _ in season)
norm = TwoSlopeNorm(0, -lim, lim)
width = lambda mw: 0.35 + 2.4 * mw / I_mean[adj].max()

fig = plt.figure(figsize=(7.1, 5.3))
gs = fig.add_gridspec(2, 6, height_ratios=[0.75, 1.3], width_ratios=[1, 1, 1, 1, 0.10, 0.045], hspace=0.30, wspace=0.04)
maps = []
for k, (label, _) in enumerate(SEASONS):
    ax = fig.add_subplot(gs[0, k]); maps.append(ax)
    geo.plot(ax=ax, color=LAND, edgecolor="white", linewidth=0.5)
    att, _, mw = season[k]
    for q, e in enumerate(E):
        col = diverging(norm(att[q])) if multi[q] else SINGLE
        ax.add_patch(FancyArrowPatch(tuple(xy[src[e]]), tuple(xy[dst[e]]), connectionstyle="arc3,rad=0.22",
                                     arrowstyle="-|>", mutation_scale=4.5, lw=width(mw[q]), color=col,
                                     shrinkA=2.5, shrinkB=2.5, zorder=3))
    ax.scatter(xy[:, 0], xy[:, 1], s=4, color=INK, zorder=4)
    ax.set_title(label, fontsize=6.8)
    ax.set_xlim(-5.2, 8.3); ax.set_ylim(42.2, 51.2); ax.set_aspect(1 / np.cos(np.deg2rad(46.5))); ax.axis("off")
maps[0].annotate(f"(a) Attention (color) and influence (width) of each edge, {SPACE} on the spatial graph, mean of three seeds", xy=(0, 1),
                 xycoords="axes fraction", xytext=(0, 13), textcoords="offset points", fontsize=7.5, ha="left", va="bottom")
handles = [Line2D([], [], color=MUTED, lw=width(v), label=f"{v:.0f} MW") for v in (40, 80, 120)]
handles.append(Line2D([], [], color=SINGLE, lw=1.2, label="single neighbor"))
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.47, maps[0].get_position().y0 + 0.005), ncol=4,
           frameon=False, fontsize=6.0, handlelength=2.0, title="Influence (width)", title_fontsize=6.0)
cax = fig.add_subplot(gs[0, 5])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=diverging), cax=cax)
cb.set_label("Attention deviation (%)", fontsize=6.3); cb.outline.set_linewidth(0.4)

bottom = gs[1, :].subgridspec(1, 2, wspace=0.75, width_ratios=[0.85, 1])
ax = fig.add_subplot(bottom[0])
r = max(np.abs(att_seeds[:, multi]).max(), np.abs(inf_seeds[:, multi]).max()) * 1.1
ax.axhline(0, color="#e1e0d9", lw=0.6, zorder=0); ax.axvline(0, color="#e1e0d9", lw=0.6, zorder=0)
ax.plot([-r, r], [-r, r], color="#c3c2b7", lw=0.8, ls="--", zorder=1)
am, im_ = att_seeds.mean(0)[multi], inf_seeds.mean(0)[multi]
ax.errorbar(am, im_, xerr=[am - att_seeds.min(0)[multi], att_seeds.max(0)[multi] - am],
            yerr=[im_ - inf_seeds.min(0)[multi], inf_seeds.max(0)[multi] - im_],
            fmt="none", ecolor="#9fc4ef", elinewidth=0.6, capsize=0, zorder=2)
ax.scatter(am, im_, s=13, color="#2a78d6", edgecolor="white", linewidth=0.4, zorder=3)
ax.text(0.03, 0.97, f"$\\rho={stats['spearman_seed_mean']:.2f}$", transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_aspect("equal")
ax.set_xlabel("Attention deviation (%)"); ax.set_ylabel("Influence deviation (%)")
ax.set_title("(b) Year-mean deviations, mean of three seeds", fontsize=7.5)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
for s_ in ("left", "bottom"):
    ax.spines[s_].set_color("#c3c2b7"); ax.spines[s_].set_linewidth(0.6)
ax.tick_params(length=2, color="#c3c2b7")

dev = np.full((n, n), np.nan); dev[dd, sd] = year_d
lim_d = np.abs(year_d[sd != dd]).max()
ax2 = fig.add_subplot(bottom[1])
im = ax2.imshow(dev, cmap=diverging_cells, norm=TwoSlopeNorm(0, -lim_d, lim_d))
ax2.set_xticks(range(n), names, rotation=45, ha="right", rotation_mode="anchor"); ax2.set_yticks(range(n), names)
ax2.set_xlabel("Sending region"); ax2.set_ylabel("Receiving region")
ax2.set_title("(c) Year-mean attention deviation, GATv2Conv on the DTW graph", fontsize=7.5)
cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03); cb.set_label("Deviation from 1/in-degree (%)"); cb.outline.set_linewidth(0.4)
ax2.tick_params(length=0)
for s_ in ax2.spines.values():
    s_.set_visible(False)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
print(json.dumps(stats, indent=1))
