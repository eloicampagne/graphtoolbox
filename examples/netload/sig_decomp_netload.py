"""Significance of the decomposed GNN pipeline against the operational GAM.

Rebuilds the ten decomposed net-load experts and their MLpol aggregation from
cache, recomputes the per-tod GAM forecast, and reports block-bootstrap CIs, a
Model Confidence Set over all of them, and a paired Diebold-Mariano test of the
decomposed aggregation against the GAM.
"""
import os
import numpy as np
import pandas as pd
import torch
from opera import Mixture
from pygam import s, f, l
from graphtoolbox.evaluation import (bootstrap_metric, model_confidence_set,
                                     diebold_mariano)
from graphtoolbox.models.baseline import gam_forecast_by_slot

import netload_pipeline as P
import baselines_netload as B

HERE = P.HERE
DIR = os.path.join(HERE, "results_decomp_sweep_netload")
STEP, N_BOOT, BLOCK, SEED = 48, 2000, 48, 0
rng = np.random.default_rng(SEED)

y = torch.load(os.path.join(HERE, "results_all_convolutions_netload/target.pt"), weights_only=False).numpy()
ynodes = torch.load(os.path.join(HERE, "results_all_convolutions_netload/target_nodes.pt"), weights_only=False)
T = len(y)


def smape(p, yy=None):
    yy = y if yy is None else yy
    p, yy = np.asarray(p, float)[:T], np.asarray(yy, float)
    return 100.0 * np.mean(2 * np.abs(p - yy) / (np.abs(p) + np.abs(yy) + 1e-8))


def rmse(p): return float(np.sqrt(np.mean((np.asarray(p, float)[:T] - y) ** 2)))


def boot_smape_se(p):
    p = np.asarray(p, float)[:T]
    nb = int(np.ceil(T / BLOCK)); starts = np.arange(0, T - BLOCK + 1)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([np.arange(s0, s0 + BLOCK) for s0 in rng.choice(starts, nb)])[:T]
        pp, yy = p[idx], y[idx]
        vals[b] = 100 * np.mean(2 * np.abs(pp - yy) / (np.abs(pp) + np.abs(yy) + 1e-8))
    return float(np.std(vals, ddof=1))


# Decomposed experts and aggregation.
import glob
convs = sorted(set(os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(DIR, "*__Load.pt"))))
net_nodes = {}
for c in convs:
    L_ = torch.load(os.path.join(DIR, f"{c}__Load.pt"), weights_only=False)
    W_ = torch.load(os.path.join(DIR, f"{c}__Wind_power.pt"), weights_only=False)
    S_ = torch.load(os.path.join(DIR, f"{c}__Solar_power.pt"), weights_only=False)
    net_nodes[c] = (L_ - W_ - S_).cpu().numpy()
nat = {c: net_nodes[c].sum(0)[:T] for c in convs}


def mlpol(edf, target):
    ne = edf.shape[1]; out = np.empty(len(target))
    with np.errstate(all="ignore"):
        out[:STEP] = edf.iloc[:STEP].values @ (np.ones(ne) / ne)
    mix = Mixture(y=target[:STEP], experts=edf.iloc[:STEP], model="MLpol", loss_type="mse")
    for s0 in range(STEP, len(target), STEP):
        e = min(s0 + STEP, len(target))
        out[s0:e] = np.ravel(mix.predict(new_experts=edf.iloc[s0:e]))[:e - s0]
        mix.update(new_experts=edf.iloc[s0:e], new_y=target[s0:e])
    return out


edf = pd.DataFrame(nat)
top = mlpol(edf, y)
bu = np.zeros(T)
for i in range(ynodes.shape[0]):
    yn = ynodes[i].numpy().astype(float)
    ndf = pd.DataFrame({c: net_nodes[c][i][:T] for c in convs})
    bu += mlpol(ndf, yn)

# Per-slot GAM forecast from baselines_netload.
nat_df = B.build_national().dropna(subset=["lag336"])
train = nat_df.loc["2014-01-01":"2018-12-31"]
test = nat_df.loc["2019-01-01":].iloc[:T]
gorder = ["temperature", "temp950", "temp990", "toy", "wind_pw", "nebu_pw",
          "day_type_week", "day_type_jf", "period_holiday", "period_hour_changed", "year"]
gi = {c: i for i, c in enumerate(gorder)}
formula = (s(gi["temperature"]) + s(gi["temp950"]) + s(gi["temp990"]) + s(gi["toy"], by=gi["day_type_week"])
           + f(gi["day_type_week"]) + f(gi["day_type_jf"]) + f(gi["period_holiday"]) + f(gi["period_hour_changed"])
           + s(gi["toy"], basis="cp") + s(gi["wind_pw"]) + s(gi["toy"], by=gi["wind_pw"], basis="cp")
           + s(gi["nebu_pw"]) + s(gi["toy"], by=gi["nebu_pw"], basis="cp") + l(gi["year"]))
gam_pred = gam_forecast_by_slot(
    train, test, gorder, "NetLoad", "tod", formula)

# Significance tests.
print("% decomposed rows with block-bootstrap SE (block=48, 2000 resamples)")
for name, p in [("Best decomp (GatedGraphConv)", nat["GatedGraphConv"]),
                ("MLpol top-level (decomp)", top), ("MLpol bottom-up (decomp)", bu),
                ("GAM", gam_pred)]:
    r = bootstrap_metric(p[:T], y, metric="rmse", n_boot=N_BOOT, block_len=BLOCK, seed=SEED)
    print(f"{name:<30} sMAPE ${smape(p):.2f} \\pm {boot_smape_se(p):.2f}$  RMSE ${r.point:.0f} \\pm {r.se:.0f}$")

# MCS over decomposed convs + GAM + decomposed aggregation.
pool = {c: nat[c] for c in convs}
pool["GAM"] = gam_pred
pool["MLpol-decomp-BU"] = bu
mcs = model_confidence_set(pool, y, loss="squared", alpha=0.10, n_boot=N_BOOT, block_len=BLOCK, seed=SEED)
print(f"\n90% MCS over {len(pool)} models: {len(mcs.included)} retained")
print("  members:", ", ".join(sorted(mcs.included)))
print(f"  GAM in MCS: {'GAM' in mcs.included} ; decomposed aggregation in MCS: {'MLpol-decomp-BU' in mcs.included}")

# Paired DM: decomposed aggregation vs GAM.
for name, p in [("MLpol bottom-up (decomp)", bu), ("MLpol top-level (decomp)", top),
                ("Best decomp (GatedGraphConv)", nat["GatedGraphConv"])]:
    rr = diebold_mariano(p[:T], gam_pred, y, loss="squared", h=STEP, names=(name, "GAM"))
    verdict = f"{rr.better} better (p={rr.pvalue:.4f})" if rr.pvalue < 0.05 else f"not significant (p={rr.pvalue:.4f})"
    print(f"Paired DM {name} vs GAM: {verdict}")
