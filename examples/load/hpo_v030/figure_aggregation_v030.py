"""Expert-aggregation figure: MLpol weights over 2019 and the cumulative error they produce.

Reads ./load_sweep_v030.npz, reruns the top-level MLpol of the paper (GraphToolbox
Aggregation, one-day blocks) to recover its weights, and writes a PDF.

Usage: python figure_aggregation_v030.py [OUTPUT.pdf]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from operator_colors import operator_color
from graphtoolbox.aggregation import Aggregation

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "aggregation_v030.pdf"))
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
OTHER, INK, MUTED, GRID = "#c3c2b7", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
    "ytick.color": MUTED, "pdf.fonttype": 42,
})

with np.load(os.path.join(HERE, "load_sweep_v030.npz")) as z:
    arr = {k: z[k] for k in z.files}
y = arr["target"].astype(float)
ops = sorted(k for k in arr if k not in {"target", "target_nodes"} and not k.startswith("aggregate__"))
X = np.column_stack([arr[k].sum(axis=0) for k in ops]).astype(float)
agg = Aggregation(model="MLpol", loss="square").run(X, y, block_size=48)
assert np.abs(agg.prediction_ - arr["aggregate__mlpol_top"]).max() < 1.0

days = pd.date_range("2019-01-01", periods=len(y) // 48, freq="D")
W = agg.weights_[::48][:len(days)]
order = np.argsort(-W.mean(axis=0))
top = order[:5]
rmse = lambda p: np.sqrt(np.mean((p - y) ** 2))
best = min(range(len(ops)), key=lambda i: rmse(X[:, i]))

fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 3.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
stack = [W[:, i] for i in top] + [W[:, order[5:]].sum(axis=1)]
labels = [ops[i] for i in top] + [f"{len(ops) - 5} other operators"]
a.stackplot(days, stack, colors=[operator_color(ops[i]) for i in top] + [OTHER], labels=labels, edgecolor="white", linewidth=0.25)
a.set_ylim(0, 1)
a.set_ylabel("(a) MLpol weight")
a.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handlelength=1.0,
         handletextpad=0.4, columnspacing=0.9)

n_days = len(days)
cum = lambda p: np.sqrt(np.cumsum(((p - y) ** 2)[: n_days * 48].reshape(n_days, 48).sum(axis=1))
                        / (48 * np.arange(1, n_days + 1)))
b.plot(days, cum(agg.prediction_), color=INK, lw=1.2, label=f"MLpol (top-level), {rmse(agg.prediction_):.0f} MW")
b.plot(days, cum(X.mean(axis=1)), color=MUTED, lw=1.0, ls="--", label=f"Uniform average, {rmse(X.mean(axis=1)):.0f} MW")
b.plot(days, cum(X[:, best]), color=operator_color(ops[best]) if best in top else MUTED, lw=1.2,
       label=f"{ops[best]} (best single), {rmse(X[:, best]):.0f} MW")
b.set_ylabel("(b) RMSE since 1 Jan. (MW)")
# The first days, forecast from uniform weights around the New Year holidays, are clipped.
b.set_ylim(650, 1250)
b.legend(frameon=False, loc="upper right", handlelength=1.6, handletextpad=0.4)
b.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
b.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
for ax in (a, b):
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(days[0], days[-1])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
print("written", OUT, "| top:", labels[:5], "| best:", ops[best])
