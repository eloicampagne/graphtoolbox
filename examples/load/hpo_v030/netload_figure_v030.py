"""Net-load figure, all forecasts with the weather trajectories of the forecast day (./netload_v030/leads_seed42).

(a) national RMSE of the direct and decomposed forecasts of the 10 operators, against pooled XGBoost (mean RMSE
of its three seeds, netload_nongraph.py), TAM and the regional GAM, the references of the load figure; (b) national net-load RMSE by half-hour of the day for the
best direct and decomposed forecasts, the regional GAM and the LSTM; (c) monthly RMSE of the same four.
Replaces netload_decomposition_figure_v030.py, whose panels compared runs without and with trajectories.
The script fails if two texts of the figure overlap.

Usage: python netload_figure_v030.py [OUTPUT.pdf]
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "netload_v030.pdf"))
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
WITH, REF, GRID = "#0b0b0b", "#77766d", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "pdf.fonttype": 42, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6})
run = "leads_seed42"
ld = lambda name: torch.load(os.path.join(HERE, "netload_v030", run, f"{name}.pt"), weights_only=False).numpy().astype(float).sum(axis=0)
stats = json.load(open(os.path.join(HERE, "netload_stats_v030_leads.json")))
y = ld("target__NetLoad"); T = len(y)
dates = pd.date_range("2019-01-01", periods=T, freq="30min")
dec = lambda op: (ld(f"{op}__Load") - ld(f"{op}__Wind_power") - ld(f"{op}__Solar_power"))[:T]
rmse = lambda p: float(np.sqrt(np.mean((p[:T] - y) ** 2)))
ops = sorted(stats["component_rmse"])
rows = sorted(((op, rmse(ld(op)), rmse(dec(op))) for op in ops), key=lambda r: r[1])
gam = pd.read_csv(os.path.join(EX, "netload", "results_gam_regional.csv"))["national_gam_pred"].to_numpy()
lstm = [np.load(os.path.join(EX, "results_lstm_seeds", f"netload_s{s}.npy")) for s in (42, 0, 1, 2, 3)]
best = stats["summary"]["best_direct"]
for op, d, c in rows: print(f"{op:<20}{d:7.0f}{c:7.0f}")
print("GAM", round(rmse(gam)), "LSTM mean", round(np.mean([rmse(l) for l in lstm])))

fig = plt.figure(figsize=(7.16, 2.15))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.5, 1.35], wspace=0.34)
# (a) paired direct -> decomposed per operator
ax = fig.add_subplot(gs[0])
for k, (op, d, c) in enumerate(rows):
    ax.plot([d, c], [k, k], color="#c3c2b7", lw=0.8, zorder=1)
    ax.scatter([d], [k], s=10, facecolor="white", edgecolor=WITH, lw=0.7, zorder=3)
    ax.scatter([c], [k], s=10, color=WITH, zorder=3)
tam = np.load(os.path.join(EX, "netload", "tam_static_forecasts.npz"))["test"]
xgb = json.load(open(os.path.join(HERE, "netload_nongraph.json")))["XGBoost (b-global)"]["rmse_mean"]
refs = [(rmse(tam), "TAM", -0.8), (rmse(gam), "GAM", -1.8), (xgb, "XGBoost (pooled)", -2.8)]
print("references", [(n, round(v)) for v, n, _ in refs])
ref_lines = []
for v, _, row in refs:  # each line starts at the height of its own label
    ref_lines.append(ax.vlines(v, row + 0.25, len(rows) - 0.5, color=REF, lw=0.8, ls="--"))
ax.set_yticks(range(len(rows)), [r[0].replace("Conv", "") for r in rows], fontsize=5.8)
ax.set_ylim(len(rows) - 0.5, -3.6)
ax.set_xlim(1100, 1830)
ax.set_xticks([1200, 1400, 1600])
# Direct labels on the first row and on the reference lines, above the data.
lab = dict(fontsize=5.6, va="bottom", clip_on=False)
ax.text(rows[0][2] + 20, -0.8, "decomposed", ha="right", **lab)
ax.text(rows[0][1] - 20, -0.8, "direct", ha="left", **lab)
ref_labels = []
for v, name, row in refs:  # staircase: each label ends just left of the top of its line, above the lower lines
    ref_labels.append(ax.text(v - 6, row + 0.25, name, ha="right", color=REF, **lab))
ax.set_xlabel("Net-load RMSE (MW)")
ax.set_title("(a) Direct and decomposed, per operator", fontsize=7.5, loc="left")
# (b) RMSE by half-hour of the day
ax = fig.add_subplot(gs[1])
hh = (dates.hour * 2 + dates.minute // 30)[:T]
by = lambda p: pd.Series((p[:T] - y) ** 2).groupby(np.asarray(hh)).mean() ** 0.5
x = np.arange(48) / 2
series = [("GAM (regional)", by(gam), REF, "-"), ("LSTM", np.mean([by(l) for l in lstm], axis=0), REF, ":"),
          (f"Direct {best}", by(ld(best)), WITH, (0, (3, 1.5))), (f"Decomposed {best}", by(dec(best)), WITH, "-")]
for label, v, col, ls in series: ax.plot(x, np.asarray(v), color=col, lw=1.0, ls=ls, label=label)
ax.set_xticks([0, 6, 12, 18, 24], ["0h", "6h", "12h", "18h", "24h"]); ax.set_ylim(0, None)
ax.set_ylabel("Net-load RMSE (MW)"); ax.set_xlabel("Time of day")
ax.set_title("(b) Error by time of day", fontsize=7.5, loc="left")
ax.legend(frameon=False, fontsize=5.6, loc="lower right", handlelength=1.6, borderaxespad=0.1)
# (c) monthly
ax = fig.add_subplot(gs[2])
monthly = lambda p: pd.Series((p[:T] - y) ** 2, index=dates).groupby(dates.month).mean() ** 0.5
ms = [("GAM (regional)", monthly(gam), REF, "-"), ("LSTM", np.mean([monthly(l) for l in lstm], axis=0), REF, ":"),
      (f"Direct {best}", monthly(ld(best)), WITH, (0, (3, 1.5))), (f"Decomposed {best}", monthly(dec(best)), WITH, "-")]
m = np.arange(1, 13)
for label, v, col, ls in ms:  # line styles of panel (b), no markers
    ax.plot(m, np.asarray(v), color=col, lw=1.0, ls=ls)
ax.set_xticks(m, list("JFMAMJJASOND")); ax.set_ylim(0, None); ax.set_ylabel("Net-load RMSE (MW)")
ax.set_title("(c) Monthly net-load error", fontsize=7.5, loc="left")
for a in fig.axes:
    for s in ("top", "right"): a.spines[s].set_visible(False)
    (a.xaxis if a is fig.axes[0] else a.yaxis).grid(True, color=GRID, lw=0.5)
    a.set_axisbelow(True); a.tick_params(length=2, color="#c3c2b7")
fig.canvas.draw()
r = fig.canvas.get_renderer()
ticks, shown = set(), []
for ax_ in fig.axes:
    for axis in (ax_.xaxis, ax_.yaxis):
        lo, hi = sorted(axis.get_view_interval())
        for tick in axis.get_major_ticks():
            ticks.add(id(tick.label1))
            if lo - 1e-9 <= tick.get_loc() <= hi + 1e-9 and tick.label1.get_visible():
                shown.append(tick.label1)
# Tick labels outside the view interval are not drawn, so only the visible ones are checked.
texts = shown + [t for t in fig.findobj(matplotlib.text.Text)
                 if id(t) not in ticks and t.get_visible() and t.get_text().strip()]
# 2 px margin between texts, the entries of one legend being stacked by matplotlib itself.
group = {id(t): k for k, ax_ in enumerate(fig.axes) if ax_.get_legend() for t in ax_.get_legend().get_texts()}
boxes = [(t.get_text(), group.get(id(t)), t.get_window_extent(r).padded(2)) for t in texts]
clash = [(a, b) for i, (a, ga, ba) in enumerate(boxes) for b, gb, bb in boxes[i + 1:]
         if ba.overlaps(bb) and (ga is None or ga != gb)]
legends = [ax_.get_legend() for ax_ in fig.axes if ax_.get_legend()]
for leg in legends:
    lb = leg.get_window_extent(r)
    for line in leg.axes.get_lines():
        xy = leg.axes.transData.transform(line.get_xydata())
        if any(lb.contains(*p) for p in xy):
            clash.append(("legend", line.get_label()))
# No reference line may cross a text of panel (a).
ax_a = fig.axes[0]
for coll in ref_lines:
    (x0, y0), (x1, y1) = coll.get_segments()[0]
    seg = matplotlib.transforms.Bbox(ax_a.transData.transform([[x0, y0], [x1, y1]])).padded(1)
    for t in [t for t in ax_a.texts] + ax_a.get_yticklabels():
        if t.get_text().strip() and t.get_window_extent(r).overlaps(seg):
            clash.append(("reference line", t.get_text()))
assert not clash, clash
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02, dpi=220)
v = {l: np.asarray(s) for l, s, *_ in series}
print("by-hour RMSE, direct/decomposed/GAM at 0h,6h,12h,18h:", [[int(v[k][i]) for i in (0, 12, 24, 36)] for k in v])
