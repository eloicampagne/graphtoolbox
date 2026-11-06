"""Contributions of the feature groups of the additive graph model to the net-load forecast.

Reads the outputs of additive_ale_v030.py (default ./additive_v030_leads):
national forecast over the week of 2019 with the largest standard deviation of observed wind, decomposed into the contributions of the
feature groups, each centered on its test mean (the means and the intercept form the
baseline). The ALE importances of additive_ale_v030.py are reported in the text only.

Usage: python ale_figure_v030.py [RESULTS_DIR] [OUTPUT.pdf]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "additive_v030_leads")
OUT = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "ale_v030.pdf"))
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
INK, MUTED, GRID, BLUE, BAND = "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6", "#b7d3f6"
MAIN = os.path.join(HERE, "netload_v030", "leads_seed42")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "pdf.fonttype": 42, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6})
# Group names and colors; the hues are distinct from the operator colors of Figures 3 and 4.
GROUPS = {"netload_lags": ("Net-load, day before", "#77766d"),
          "temp_lags": ("Temperature", "#eb6834"),
          "wind_leads": ("Wind", "#2a78d6"),
          "cloud_leads": ("Cloud cover", "#eda100"),
          "exogenous": ("Other covariates", "#1baf7a"),
          "calendar_dummies": ("Calendar", "#a39fd8")}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2, color="#c3c2b7")


# Week with the largest standard deviation of observed national wind in 2019 (4-10 February).
c = np.load(os.path.join(SRC, "group_contributions.npz"))
T = len(c["prediction"])
dates = pd.date_range("2019-01-01", periods=T, freq="30min")
wind = pd.Series(torch.load(os.path.join(MAIN, "target__Wind_power.pt"), weights_only=False).numpy()
                 .astype(float).sum(axis=0)[:T], index=dates)
full = wind[wind.index >= "2019-01-07"]
week = full.groupby(full.index.to_period("W-SUN")).std().idxmax()
sel = (dates >= week.start_time) & (dates < week.start_time + pd.Timedelta(days=7))
order = [g for g in GROUPS if f"contrib:{g}" in c.files]
dev = {g: (c[f"contrib:{g}"] - c[f"contrib:{g}"].mean())[sel] * 1e-3 for g in order}
baseline = float(c["intercept"]) + sum(float(c[f"contrib:{g}"].mean()) for g in order)

fig, ax = plt.subplots(figsize=(7.16, 2.1))
t = dates[sel]
pos, neg = np.zeros(sel.sum()), np.zeros(sel.sum())
for g in order:
    v = dev[g]
    up, down = np.clip(v, 0, None), np.clip(v, None, 0)
    ax.fill_between(t, pos, pos + up, color=GROUPS[g][1], lw=0, label=GROUPS[g][0])
    ax.fill_between(t, neg, neg + down, color=GROUPS[g][1], lw=0)
    pos, neg = pos + up, neg + down
ax.plot(t, (c["prediction"][sel] - baseline) * 1e-3, color=INK, lw=0.9, label="Forecast")
ax.plot(t, (c["target"][sel] - baseline) * 1e-3, color=INK, lw=0.7, ls=(0, (1, 1.2)), label="Observed")
ax.axhline(0, color=MUTED, lw=0.5)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
ax.set_xlabel(f"{week.start_time:%-d}–{(week.start_time + pd.Timedelta(days=6)):%-d %B %Y}", fontsize=6.3)
ax.set_ylabel(f"From the baseline of {baseline * 1e-3:.1f} GW (GW)", fontsize=6.8)
ax.legend(frameon=False, fontsize=6.3, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0),
          handlelength=1.4)
style(ax)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
print("week:", week, "baseline:", round(baseline))
