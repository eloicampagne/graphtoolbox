"""Recompute all net-load table numbers at the standardized config (h64/l1/300ep).

Reads the direct and decomposed sweeps (same ten convolutions, same config) and
produces, on the 2019 national net-load: per-convolution RMSE and sMAPE with
block-bootstrap SE, MLpol aggregation for both the direct and decomposed
experts, per-component RMSE for the decomposed models, the per-tod GAM forecast,
and the MCS / paired-DM significance of the decomposed pipeline against the GAM.
"""
import os
import glob
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
DIRECT = os.path.join(HERE, "results_std_direct_netload")
DECOMP = os.path.join(HERE, "results_std_decomp_netload")
CT = os.path.join(HERE, "results_decomp_netload")   # component ground-truth targets
STEP, N_BOOT, BLOCK, SEED = 48, 2000, 48, 0
rng = np.random.default_rng(SEED)

y = torch.load(os.path.join(HERE, "results_all_convolutions_netload/target.pt"), weights_only=False).numpy()
ynodes = torch.load(os.path.join(HERE, "results_all_convolutions_netload/target_nodes.pt"), weights_only=False)
T = len(y)


def rmse(p): return float(np.sqrt(np.mean((np.asarray(p, float)[:T] - y) ** 2)))
def smape(p):
    p = np.asarray(p, float)[:T]
    return 100.0 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8))
def se_rmse(p):
    return bootstrap_metric(np.asarray(p, float)[:T], y, metric="rmse", n_boot=N_BOOT, block_len=BLOCK, seed=SEED).se
def se_smape(p):
    p = np.asarray(p, float)[:T]
    nb = int(np.ceil(T / BLOCK)); st = np.arange(0, T - BLOCK + 1); v = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([np.arange(s0, s0 + BLOCK) for s0 in rng.choice(st, nb)])[:T]
        pp, yy = p[idx], y[idx]; v[b] = 100 * np.mean(2 * np.abs(pp - yy) / (np.abs(pp) + np.abs(yy) + 1e-8))
    return float(np.std(v, ddof=1))


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


def bottom_up(node_preds):  # dict conv -> [nodes,T]
    bu = np.zeros(T)
    for i in range(ynodes.shape[0]):
        yn = ynodes[i].numpy().astype(float)
        ndf = pd.DataFrame({c: node_preds[c][i][:T] for c in node_preds})
        bu += mlpol(ndf, yn)
    return bu


def block(title, nat, node_preds):
    print(f"\n=== {title} ===")
    for c in sorted(nat, key=lambda k: rmse(nat[k])):
        print(f"  {c:<18} sMAPE ${smape(nat[c]):.2f} \\pm {se_smape(nat[c]):.2f}$  RMSE ${rmse(nat[c]):.0f} \\pm {se_rmse(nat[c]):.0f}$")
    edf = pd.DataFrame(nat)
    uni = edf.values.mean(axis=1); top = mlpol(edf, y); bu = bottom_up(node_preds)
    for name, p in [("Uniform average", uni), ("MLpol (top-level)", top), ("MLpol (bottom-up)", bu)]:
        print(f"  {name:<18} sMAPE ${smape(p):.2f} \\pm {se_smape(p):.2f}$  RMSE ${rmse(p):.0f} \\pm {se_rmse(p):.0f}$")
    return {"uni": uni, "top": top, "bu": bu}


# Direct forecasts.
d_nodes = {os.path.basename(f)[:-3]: torch.load(f, weights_only=False).numpy()
           for f in glob.glob(os.path.join(DIRECT, "*.pt")) if os.path.basename(f) != "target.pt"}
d_nat = {c: d_nodes[c].sum(0)[:T] for c in d_nodes}
direct_agg = block("DIRECT GNN (h64/l1/300ep)", d_nat, d_nodes)

# Decomposed forecasts.
convs = sorted(os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(DECOMP, "*__Load.pt")))
dec_nodes = {}
for c in convs:
    L_ = torch.load(os.path.join(DECOMP, f"{c}__Load.pt"), weights_only=False).numpy()
    W_ = torch.load(os.path.join(DECOMP, f"{c}__Wind_power.pt"), weights_only=False).numpy()
    S_ = torch.load(os.path.join(DECOMP, f"{c}__Solar_power.pt"), weights_only=False).numpy()
    dec_nodes[c] = L_ - W_ - S_
dec_nat = {c: dec_nodes[c].sum(0)[:T] for c in convs}
dec_agg = block("DECOMPOSED GNN (h64/l1/300ep)", dec_nat, dec_nodes)

# Per-component RMSE.
ct = {comp: torch.load(os.path.join(CT, f"{comp}_target.pt"), weights_only=False).sum(0).numpy()
      for comp in ("Load", "Wind_power", "Solar_power")}
print("\n=== PER-COMPONENT national RMSE (MW) ===")
rows = []
for c in convs:
    pr = {comp: torch.load(os.path.join(DECOMP, f"{c}__{comp}.pt"), weights_only=False).sum(0).numpy() for comp in ct}
    rows.append((c, np.sqrt(np.mean((pr["Load"]-ct["Load"])**2)), np.sqrt(np.mean((pr["Wind_power"]-ct["Wind_power"])**2)),
                 np.sqrt(np.mean((pr["Solar_power"]-ct["Solar_power"])**2)), rmse(dec_nat[c])))
for c, ld, wd, sl, nt in sorted(rows, key=lambda x: x[4]):
    print(f"  {c:<18} Load {ld:.0f}  Wind {wd:.0f}  Solar {sl:.0f}  Net {nt:.0f}")

# GAM and significance tests.
nat_df = B.build_national().dropna(subset=["lag336"])
tr = nat_df.loc["2014-01-01":"2018-12-31"]; te = nat_df.loc["2019-01-01":].iloc[:T]
go = ["temperature", "temp950", "temp990", "toy", "wind_pw", "nebu_pw", "day_type_week", "day_type_jf",
      "period_holiday", "period_hour_changed", "year"]
gi = {c: i for i, c in enumerate(go)}
formula = (s(gi["temperature"]) + s(gi["temp950"]) + s(gi["temp990"]) + s(gi["toy"], by=gi["day_type_week"])
           + f(gi["day_type_week"]) + f(gi["day_type_jf"]) + f(gi["period_holiday"]) + f(gi["period_hour_changed"])
           + s(gi["toy"], basis="cp") + s(gi["wind_pw"]) + s(gi["toy"], by=gi["wind_pw"], basis="cp")
           + s(gi["nebu_pw"]) + s(gi["toy"], by=gi["nebu_pw"], basis="cp") + l(gi["year"]))
gam = gam_forecast_by_slot(tr, te, go, "NetLoad", "tod", formula)

pool = {c: dec_nat[c] for c in convs}; pool["GAM"] = gam; pool["MLpol-decomp-BU"] = dec_agg["bu"]
mcs = model_confidence_set(pool, y, loss="squared", alpha=0.10, n_boot=N_BOOT, block_len=BLOCK, seed=SEED)
print(f"\n=== SIGNIFICANCE ===\nGAM: sMAPE {smape(gam):.2f}  RMSE {rmse(gam):.0f}")
print(f"90% MCS over {len(pool)} models: {len(mcs.included)} retained; GAM in: {'GAM' in mcs.included}; decomp-agg in: {'MLpol-decomp-BU' in mcs.included}")
for nm, p in [("decomp MLpol-BU", dec_agg["bu"]), ("decomp MLpol-top", dec_agg["top"]), ("direct MLpol-BU", direct_agg["bu"])]:
    r = diebold_mariano(p[:T], gam, y, loss="squared", h=STEP, names=(nm, "GAM"))
    print(f"Paired DM {nm} vs GAM: p={r.pvalue:.4f} ({'better '+r.better if r.pvalue<0.05 else 'ns'})")
