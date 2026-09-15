"""Significance of the net-load leaders: decomposed GNN, GAMs, LSTM.

Once the covariate-aware LSTM and the regional GAM join the comparison, the top
of the net-load table is a cluster of models within a few tens of megawatts of
each other. This script asks whether any of them is actually ahead: it rebuilds
the decomposed MLpol aggregate and the national per-tod GAM, reads the regional
GAM and the seeded LSTM forecasts, then runs a Model Confidence Set and paired
Diebold-Mariano tests over the cluster.
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from opera import Mixture
from pygam import s, f, l
from graphtoolbox.evaluation import model_confidence_set, diebold_mariano
from graphtoolbox.models.baseline import gam_forecast_by_slot
import netload_pipeline as P
import baselines_netload as B

HERE = P.HERE
DECOMP = os.path.join(HERE, "results_std_decomp_netload")
LSTM_DIR = os.path.join(os.path.dirname(HERE), "results_lstm_seeds")
STEP, N_BOOT, BLOCK, SEED = 48, 2000, 48, 0

y = torch.load(os.path.join(HERE, "results_all_convolutions_netload/target.pt"), weights_only=False).numpy()
ynodes = torch.load(os.path.join(HERE, "results_all_convolutions_netload/target_nodes.pt"), weights_only=False)
T = len(y)


def rmse(p):
    return float(np.sqrt(np.mean((np.asarray(p, float)[:T] - y) ** 2)))


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


# Decomposed GNN experts and bottom-up MLpol aggregate.
convs = sorted(os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(DECOMP, "*__Load.pt")))
dec_nodes = {}
for c in convs:
    L_ = torch.load(os.path.join(DECOMP, f"{c}__Load.pt"), weights_only=False).numpy()
    W_ = torch.load(os.path.join(DECOMP, f"{c}__Wind_power.pt"), weights_only=False).numpy()
    S_ = torch.load(os.path.join(DECOMP, f"{c}__Solar_power.pt"), weights_only=False).numpy()
    dec_nodes[c] = L_ - W_ - S_
dec_nat = {c: dec_nodes[c].sum(0)[:T] for c in convs}
bu = np.zeros(T)
for i in range(ynodes.shape[0]):
    yn = ynodes[i].numpy().astype(float)
    bu += mlpol(pd.DataFrame({c: dec_nodes[c][i][:T] for c in convs}), yn)
best_conv = min(convs, key=lambda c: rmse(dec_nat[c]))

# National per-slot GAM.
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

# Regional GAM and seeded LSTM.
pool = {"GNN-decomp-MLpol": bu, f"GNN-decomp-{best_conv}": dec_nat[best_conv], "GAM-national": gam}
reg_csv = os.path.join(HERE, "results_gam_regional.csv")
if os.path.exists(reg_csv):
    pool["GAM-regional"] = pd.read_csv(reg_csv)["national_gam_pred"].values[:T]
lstm_files = sorted(glob.glob(os.path.join(LSTM_DIR, "netload_s*.npy")))
lstm = [np.load(f)[:T] for f in lstm_files]
if lstm:
    for f, p in zip(lstm_files, lstm):
        pool[f"LSTM-{os.path.basename(f)[8:-4]}"] = p
    pool["LSTM-seedmean"] = np.mean(lstm, axis=0)

print("=== Net-load leaders, 2019 national RMSE (MW) ===")
for k in sorted(pool, key=lambda k: rmse(pool[k])):
    print(f"  {k:<24} {rmse(pool[k]):.0f}")
if lstm:
    r = np.array([rmse(p) for p in lstm])
    print(f"\nLSTM over {len(lstm)} seeds: mean {r.mean():.0f} ± {r.std(ddof=1):.0f}  [{r.min():.0f}, {r.max():.0f}]")

mcs = model_confidence_set(pool, y, loss="squared", alpha=0.10, n_boot=N_BOOT, block_len=BLOCK, seed=SEED)
print(f"\n90% MCS over {len(pool)} models: {len(mcs.included)} retained")
print("  retained:", sorted(mcs.included))

print("\n=== Paired Diebold-Mariano (Holm not applied; raw p) ===")
pairs = [("GNN-decomp-MLpol", "GAM-national"), ("GNN-decomp-MLpol", "GAM-regional"),
         ("GNN-decomp-MLpol", "LSTM-seedmean"), ("GAM-regional", "LSTM-seedmean"),
         ("GAM-regional", "GAM-national")]
for a, b in pairs:
    if a in pool and b in pool:
        r = diebold_mariano(pool[a][:T], pool[b][:T], y, loss="squared", h=STEP, names=(a, b))
        verdict = f"better: {r.better}" if r.pvalue < 0.05 else "ns (tie)"
        print(f"  {a:<20} vs {b:<16} p={r.pvalue:.4f}  {verdict}")
