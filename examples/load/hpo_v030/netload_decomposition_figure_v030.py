"""Net-load decomposition figure: what the weather trajectories of the forecast day change.

(a) RMSE of each component (load, wind, solar) for the 10 operators of the decomposed
sweep, without (./netload_v030/seed42) and with (./netload_v030/leads_seed42) the
trajectories; (b) national wind production over the week of 2019 with the largest
standard deviation of observed wind, against the decomposed forecasts of the best
operator with and without the trajectories; (c) monthly national net-load RMSE of the
best direct and decomposed forecasts against the regional GAM and the LSTM (mean of the
monthly RMSEs of its five seeds).

Usage: python netload_decomposition_figure_v030.py [OUTPUT.pdf]
"""
import json
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
EXAMPLES = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "netload_decomposition_v030.pdf"))
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
# Colors of Figures 3 and 4 are reserved for the operators shown there; this figure, whose only
# operator is FastRGCNConv, uses ink for the model with trajectories and greys elsewhere.
WITH, REF, AREA, GRID = "#0b0b0b", "#77766d", "#dcdbd3", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "pdf.fonttype": 42, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6})

ld = lambda run, name: torch.load(os.path.join(HERE, "netload_v030", run, f"{name}.pt"),
                                  weights_only=False).numpy().astype(float).sum(axis=0)
stats = {run: json.load(open(os.path.join(HERE, f"netload_stats_v030{suffix}.json")))
         for run, suffix in (("seed42", ""), ("leads_seed42", "_leads"))}
best_d = stats["leads_seed42"]["summary"]["best_direct"]
best_c = stats["leads_seed42"]["summary"]["best_decomposed"]
y = ld("leads_seed42", "target__NetLoad")
T = len(y)
dates = pd.date_range("2019-01-01", periods=T, freq="30min")


def decomposed(run, name):
    return (ld(run, f"{name}__Load") - ld(run, f"{name}__Wind_power") - ld(run, f"{name}__Solar_power"))[:T]


fig = plt.figure(figsize=(7.16, 2.05))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.0, 1.35], wspace=0.32)

# (a) component RMSE without and with the trajectories
ax = fig.add_subplot(gs[0])
comps = [("Load", "Load"), ("Wind_power", "Wind"), ("Solar_power", "Solar")]
rng = np.random.default_rng(0)
for run, color, dx in (("seed42", REF, -0.17), ("leads_seed42", WITH, 0.17)):
    cr = stats[run]["component_rmse"]
    for k, (comp, _) in enumerate(comps):
        vals = np.array([cr[conv][comp] for conv in sorted(cr)])
        ax.scatter(k + dx + rng.uniform(-0.06, 0.06, len(vals)), vals, s=9, color=color, lw=0.4,
                   edgecolor="white", zorder=3)
ax.set_xticks(range(3), [c[1] for c in comps])
ax.set_xlim(-0.55, 2.55)
ax.set_ylim(0, 1900)
ax.set_ylabel("Component RMSE (MW)")
ax.set_title("(a) Error per component", fontsize=7.5, loc="left")

# (b) week of 2019 with the largest standard deviation of observed national wind
ax = fig.add_subplot(gs[1])
wind = pd.Series(ld("leads_seed42", "target__Wind_power")[:T], index=dates)
full_weeks = wind[wind.index >= "2019-01-07"]
weekly_std = full_weeks.groupby(full_weeks.index.to_period("W-SUN")).std()
week = weekly_std.idxmax()
sel = (dates >= week.start_time) & (dates < week.start_time + pd.Timedelta(days=7))
ax.fill_between(dates[sel], 0, wind[sel] * 1e-3, color=AREA, lw=0, label="Observed")
ax.plot(dates[sel], ld("seed42", f"{best_c}__Wind_power")[:T][sel] * 1e-3, color=REF, lw=1.0, ls=(0, (3, 1.5)),
        label="Without trajectories")
ax.plot(dates[sel], ld("leads_seed42", f"{best_c}__Wind_power")[:T][sel] * 1e-3, color=WITH, lw=1.0,
        label="With trajectories")
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
ax.set_xlabel(f"{week.start_time:%B %Y}")
ax.set_ylabel("Wind production (GW)")
ax.set_ylim(0, None)
ax.set_title("(b) National wind production", fontsize=7.5, loc="left")
ax.legend(frameon=False, fontsize=6.0, loc="upper left", handlelength=1.6, borderaxespad=0.1)

# (c) monthly net-load RMSE
ax = fig.add_subplot(gs[2])
monthly = lambda p: pd.Series((p[:T] - y) ** 2, index=dates).groupby(dates.month).mean() ** 0.5
gam = pd.read_csv(os.path.join(EXAMPLES, "netload", "results_gam_regional.csv"))["national_gam_pred"].to_numpy()
lstm = np.mean([monthly(np.load(os.path.join(EXAMPLES, "results_lstm_seeds", f"netload_s{s}.npy")))
                for s in (42, 0, 1, 2, 3)], axis=0)
series = [("GAM (regional)", monthly(gam), REF, "-", "s"), ("LSTM", lstm, REF, ":", "^"),
          (f"Direct {best_d}", monthly(ld("leads_seed42", best_d)), WITH, (0, (3, 1.5)), "o"),
          (f"Decomposed {best_c}", monthly(decomposed("leads_seed42", best_c)), WITH, "-", "o")]
months = np.arange(1, 13)
for label, values, color, ls, mk in series:
    ax.plot(months, np.asarray(values), color=color, lw=1.0, ls=ls, marker=mk, ms=2.6,
            mfc="white" if ls != "-" or color == REF else color, label=label)
ax.set_xticks(months, list("JFMAMJJASOND"))
ax.set_ylim(0, None)
ax.set_ylabel("Net-load RMSE (MW)")
ax.set_title("(c) Monthly net-load error", fontsize=7.5, loc="left")
ax.legend(frameon=False, fontsize=5.6, loc="lower left", handlelength=1.4, borderaxespad=0.1)

for ax in fig.axes:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, color="#c3c2b7")
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
print("week:", week, "best:", best_d, best_c)
for label, values, *_ in series:
    print(f"{label:<26}", np.round(np.asarray(values)).astype(int))
