"""Figures of the IEEE Access version: per-epoch training time and the load sweep.

Reads ./timing/timing_per_epoch.csv (timing_bench.py under 0.3.0),
./load_sweep_v030.npz and ./stats_v030.json; writes PDFs to the paper's figures/.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graphtoolbox.evaluation import bootstrap_metric

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "pdf.fonttype": 42,
})

# Figure 1: per-epoch training time, dot plot on a log axis.
t = pd.read_csv(os.path.join(HERE, "timing", "timing_per_epoch.csv"))
t = t.sort_values("load_s_per_epoch", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(3.45, 2.3))
yy = np.arange(len(t))
for i, r in t.iterrows():
    ax.plot([r.load_s_per_epoch * 1e3, r.netload_s_per_epoch * 1e3], [i, i], color=GRID, lw=2, zorder=1)
ax.scatter(t.load_s_per_epoch * 1e3, yy, s=18, color=BLUE, zorder=3, label="Load (125 features, 1090 days)",
           edgecolor="white", linewidth=0.6)
ax.scatter(t.netload_s_per_epoch * 1e3, yy, s=18, marker="s", color=ORANGE, zorder=3,
           label="Net-load (175 features, 1461 days)", edgecolor="white", linewidth=0.6)
ax.set_xscale("log")
ticks = [100, 200, 500, 1000, 2000, 5000]
ax.set_xticks(ticks, [str(v) for v in ticks])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlim(90, 5000)
ax.set_yticks(yy, t.conv)
ax.set_xlabel("Training time per epoch (ms, log scale)")
ax.xaxis.grid(True, which="both", color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper right", handletextpad=0.3)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "timing_per_epoch_v030.pdf"))
plt.close(fig)

# Figure 2: national RMSE of the 46 operators of the load sweep.
stats = json.load(open(os.path.join(HERE, "stats_v030.json")))
with np.load(os.path.join(HERE, "load_sweep_v030.npz")) as z:
    arr = {k: z[k] for k in z.files}
y = arr["target"]
ops = [k for k in arr if k not in {"target", "target_nodes"} and not k.startswith("aggregate__")]
nat = {k: arr[k].sum(axis=0) for k in ops}
bs = {k: bootstrap_metric(nat[k], y, metric="rmse", n_boot=2000, block_len=48, seed=0) for k in ops}
order = sorted(ops, key=lambda k: bs[k].point)
mcs = set(stats["mcs_members"])
fig, ax = plt.subplots(figsize=(7.1, 2.55))
x = np.arange(len(order))
for inside, face, lab in [(True, BLUE, f"In the 90% Model Confidence Set ({len(mcs)})"),
                          (False, "white", f"Outside it ({len(order) - len(mcs)})")]:
    idx = [i for i, k in enumerate(order) if (k in mcs) == inside]
    ax.errorbar(x[idx], [bs[order[i]].point for i in idx], yerr=[bs[order[i]].se for i in idx],
                fmt="o", ms=3.6, mfc=face, mec=BLUE, mew=0.9, ecolor=BLUE, elinewidth=0.7,
                capsize=0, label=lab, zorder=3)
# Aggregates are labelled at the right end, references at the left end, away from the points.
refs = [(stats["table"]["MLpol (top-level)"]["rmse"], "MLpol (top-level)", "right", "top"),
        (stats["table"]["Uniform average"]["rmse"], "Uniform average", "right", "bottom"),
        (json.load(open(os.path.join(HERE, "nongraph_baselines.json")))["XGBoost (b-global)"]["rmse_mean"],
         "XGBoost (pooled)", "left", "bottom"),
        (1062, "TAM (regional)", "left", "bottom"), (1200, "GAM (regional)", "left", "top")]
for v, lab, side, va in refs:
    agg = "MLpol" in lab or "Uniform" in lab
    ax.axhline(v, color=ORANGE if agg else MUTED, lw=0.8, ls="-" if agg else "--", zorder=2)
    xt = len(order) - 0.4 if side == "right" else -0.5
    ax.text(xt, v + (5 if va == "bottom" else -5), f"{lab} ({v:.0f})", ha=side, va=va, fontsize=6, color=INK)
ax.set_xticks(x, order, rotation=90)
ax.set_xlim(-0.8, len(order) - 0.2)
ax.set_ylim(bottom=755)
ax.set_ylabel("National RMSE (MW)")
ax.yaxis.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.3)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "load_sweep_v030.pdf"))
print("written to", OUT)
