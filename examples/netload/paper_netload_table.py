"""Produce the exact net-load numbers for the ICTAI paper table (mirror of tab:load).

Reads the cached convolution sweep, temporal cells and target, and reports, on the
national net-load series (sum over the 12 regions, 2019 test set):
  * top-N static convolutions by RMSE, with RMSE and sMAPE block-bootstrap SEs;
  * the three expert-aggregation rows (uniform, MLpol bottom-up, MLpol top-level);
  * MCS (alpha=0.10) count and Holm-Diebold-Mariano vs best count;
  * paired DM of the mixture vs the best single convolution;
  * a 1-day persistence reference and the recurrent temporal cells (text summary).
Net-load crosses zero, so RMSE is the headline and sMAPE replaces plain MAPE.
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from opera import Mixture
from graphtoolbox.evaluation import (bootstrap_metric, model_confidence_set,
                                     pairwise_dm, diebold_mariano)

SWEEP = os.path.join(os.path.dirname(__file__), "results_all_convolutions_netload")
TEMP = os.path.join(os.path.dirname(__file__), "results_temporal_netload")
STEP = 48
N_BOOT, BLOCK, SEED = 2000, 48, 0
rng = np.random.default_rng(SEED)


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def smape(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return 100.0 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8))


def _blocks(T):
    nb = int(np.ceil(T / BLOCK))
    starts = np.arange(0, T - BLOCK + 1)
    return nb, starts


def boot_se(fn, p, y):
    """Moving-block bootstrap SE for a generic scalar metric fn(p, y)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    T = len(y)
    nb, starts = _blocks(T)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, nb)])[:T]
        vals[b] = fn(p[idx], y[idx])
    return float(np.std(vals, ddof=1))


# Load cached predictions.
skip = {"target.pt", "target_nodes.pt"}
preds = {}
for path in sorted(glob.glob(os.path.join(SWEEP, "*.pt"))):
    fn = os.path.basename(path)
    if fn in skip:
        continue
    preds[fn[:-3]] = torch.load(path, weights_only=False)
y = torch.load(os.path.join(SWEEP, "target.pt"), weights_only=False).numpy().astype(float)
ynodes = torch.load(os.path.join(SWEEP, "target_nodes.pt"), weights_only=False)
T = len(y)
nat = {n: p.sum(dim=0).numpy().astype(float) for n, p in preds.items()}

# Top ten convolutions by RMSE.
rows = []
for n, p in nat.items():
    r = bootstrap_metric(p, y, metric="rmse", n_boot=N_BOOT, block_len=BLOCK, seed=SEED)
    rows.append((n, r.point, r.se, smape(p, y), boot_se(smape, p, y)))
rows.sort(key=lambda x: x[1])
print("=== Static GNN sweep (default configs), national net-load, top-12 by RMSE ===")
print(f'{"Model":<20} {"RMSE (MW)":>16} {"sMAPE (%)":>16}')
for n, rp, rs, sp, ss in rows[:12]:
    print(f"{n:<20} ${rp:.0f} \\pm {rs:.0f}$   ${sp:.2f} \\pm {ss:.2f}$")

# Uniform, top-level MLpol, and bottom-up MLpol aggregation.
experts_df = pd.DataFrame({n: p for n, p in nat.items()})
bad = experts_df.columns[~np.isfinite(experts_df.values).all(axis=0)].tolist()
experts_df = experts_df.drop(columns=bad)
uniform = experts_df.values.mean(axis=1)


def mlpol(experts, target):
    ne = experts.shape[1]
    out = np.empty(len(target))
    with np.errstate(all="ignore"):
        out[:STEP] = experts.iloc[:STEP].values @ (np.ones(ne) / ne)
    mix = Mixture(y=target[:STEP], experts=experts.iloc[:STEP], model="MLpol", loss_type="mse")
    for s in range(STEP, len(target), STEP):
        e = min(s + STEP, len(target))
        out[s:e] = np.ravel(mix.predict(new_experts=experts.iloc[s:e]))[:e - s]
        mix.update(new_experts=experts.iloc[s:e], new_y=target[s:e])
    return out


top_level = mlpol(experts_df, y)
valid = [n for n in nat if n not in bad]
bu = np.zeros(T)
for i in range(ynodes.shape[0]):
    yn = ynodes[i].numpy().astype(float)
    ndf = pd.DataFrame({n: preds[n][i].numpy().astype(float) for n in valid})
    ndf = ndf.drop(columns=ndf.columns[~np.isfinite(ndf.values).all(axis=0)].tolist())
    bu += mlpol(ndf, yn)

print("\n=== Expert aggregation ===")
for label, p in [("Uniform average", uniform), ("MLpol (bottom-up)", bu), ("MLpol (top-level)", top_level)]:
    r = bootstrap_metric(p, y, metric="rmse", n_boot=N_BOOT, block_len=BLOCK, seed=SEED)
    print(f"{label:<20} ${r.point:.0f} \\pm {r.se:.0f}$   ${smape(p, y):.2f} \\pm {boot_se(smape, p, y):.2f}$")

# Significance tests.
mcs = model_confidence_set(nat, y, loss="squared", alpha=0.10, n_boot=N_BOOT, block_len=BLOCK, seed=SEED)
best = min(nat, key=lambda n: rmse(nat[n], y))
dm = pairwise_dm(nat, y, loss="squared", h=STEP, correction="holm")
nsig = int((dm["pvalue_adjusted"][best].drop(best) < 0.05).sum())
print("\n=== Significance ===")
print(f"Best single conv: {best} ({rmse(nat[best], y):.0f} MW)")
print(f"MCS (alpha=0.10): {len(mcs.included)} of {len(nat)} retained")
print(f"Holm-DM vs best: {nsig} of {len(nat)-1} significantly worse")
for label, p in [("MLpol (top-level)", top_level), ("MLpol (bottom-up)", bu)]:
    rr = diebold_mariano(p, nat[best], y, loss="squared", h=STEP, names=(label, best))
    print(f"Paired DM {label} vs {best}: p={rr.pvalue:.4f} ({'better' if rr.better==label else 'not better'})")

# Persistence and temporal-cell baselines.
pers = y.copy()
pers[STEP:] = y[:-STEP]
print("\n=== References ===")
print(f"Persistence (1 day): RMSE={rmse(pers[STEP:], y[STEP:]):.0f} MW  sMAPE={smape(pers[STEP:], y[STEP:]):.2f}%")
if os.path.isdir(TEMP):
    yt = torch.load(os.path.join(TEMP, "target.pt"), weights_only=False)
    print("Temporal cells (recurrent regime):")
    trows = []
    for f in glob.glob(os.path.join(TEMP, "*.pt")):
        n = os.path.basename(f)[:-3]
        if n == "target":
            continue
        pt = torch.load(f, weights_only=False).sum(dim=0).numpy().astype(float)
        trows.append((n, rmse(pt, yt.numpy()), smape(pt, yt.numpy())))
    for n, r, s in sorted(trows, key=lambda x: x[1]):
        print(f"  {n:<10} RMSE={r:.0f} MW  sMAPE={s:.2f}%")
