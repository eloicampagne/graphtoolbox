"""Every net-load number of the paper, recomputed from the v0.3.0 sweeps (./netload_v030/seed42).

Direct and decomposed tables with block-bootstrap standard errors (RMSE and
sMAPE, 2000 resamples of one-day blocks, seed 0), the uniform and MLpol
aggregates (GraphToolbox Aggregation, as for load), the paired Diebold-Mariano
test between the best decomposed and the best direct forecasts, the component
errors, and the reference rows read from the repository. Writes
./netload_stats_v030.json. With the argument "leads", reads ./netload_v030/leads_seed42
(models that see the wind and cloud-cover trajectories of the forecast day) and writes
./netload_stats_v030_leads.json.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from graphtoolbox.aggregation import Aggregation
from graphtoolbox.evaluation import bootstrap_metric, diebold_mariano
from graphtoolbox.evaluation.significance import _moving_block_indices

HERE = os.path.dirname(os.path.abspath(__file__))
NETLOAD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "netload")
import sys
RUN = sys.argv[1] if len(sys.argv) > 1 else ""
SRC = os.path.join(HERE, "netload_v030", ("leads_" if RUN == "leads" else "") + "seed42")
CONVS = ["RGATConv", "GatedGraphConv", "SuperGATConv", "FiLMConv", "PANConv", "GMMConv",
         "GCN2Conv", "GATConv", "FastRGCNConv", "ResGatedGraphConv"]
COMPS = ["Load", "Wind_power", "Solar_power"]
N_BOOT = 2000

ld = lambda name: torch.load(os.path.join(SRC, f"{name}.pt"), weights_only=False).numpy().astype(float)
Yn = ld("target__NetLoad")
y = Yn.sum(axis=0)
T = len(y)


def smape(p, t):
    return float(100 * np.mean(2 * np.abs(p - t) / (np.abs(p) + np.abs(t) + 1e-8)))


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def smape_se(p, t):
    rng = np.random.default_rng(0)
    s = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = _moving_block_indices(T, 48, rng)
        s[b] = smape(p[idx], t[idx])
    return float(s.std(ddof=1))


def row(p):
    return {"smape": smape(p, y), "smape_se": smape_se(p, y), "rmse": rmse(p, y),
            "rmse_se": float(bootstrap_metric(p, y, metric="rmse", n_boot=N_BOOT, block_len=48, seed=0).se)}


def aggregates(nodes):
    names = sorted(nodes)
    nat = np.column_stack([nodes[n].sum(axis=0) for n in names])
    out = {"Uniform average": nat.mean(axis=1),
           "MLpol (top-level)": Aggregation(model="MLpol", loss="square").run(nat, y, block_size=48).prediction_}
    bottom = np.zeros(T)
    for k in range(Yn.shape[0]):
        ex = np.column_stack([nodes[n][k] for n in names])
        bottom += Aggregation(model="MLpol", loss="square").run(ex, Yn[k], block_size=48).prediction_
    out["MLpol (bottom-up)"] = bottom
    return out


direct = {c: ld(c)[:, :T] for c in CONVS}
comp = {c: {k: ld(f"{c}__{k}")[:, :T] for k in COMPS} for c in CONVS}
decomp = {c: comp[c]["Load"] - comp[c]["Wind_power"] - comp[c]["Solar_power"] for c in CONVS}
comp_target = {k: ld(f"target__{k}")[:, :T].sum(axis=0) for k in COMPS}
assert np.allclose(comp_target["Load"] - comp_target["Wind_power"] - comp_target["Solar_power"], y, atol=1.0), \
    "component targets do not recombine into the net-load target"

res = {"direct": {c: row(direct[c].sum(axis=0)) for c in CONVS},
       "decomposed": {c: row(decomp[c].sum(axis=0)) for c in CONVS}}
agg_d, agg_c = aggregates(direct), aggregates(decomp)
res["direct_aggregates"] = {k: row(v) for k, v in agg_d.items()}
res["decomposed_aggregates"] = {k: row(v) for k, v in agg_c.items()}

best_d = min(CONVS, key=lambda c: res["direct"][c]["rmse"])
best_c = min(CONVS, key=lambda c: res["decomposed"][c]["rmse"])
gain = {c: res["direct"][c]["rmse"] - res["decomposed"][c]["rmse"] for c in CONVS}
dm = diebold_mariano(decomp[best_c].sum(axis=0), direct[best_d].sum(axis=0), y, loss="squared", h=48)
comp_rmse = {c: {k: rmse(comp[c][k].sum(axis=0), comp_target[k]) for k in COMPS} for c in CONVS}
wind_share = [comp_rmse[c]["Wind_power"] ** 2 / sum(v ** 2 for v in comp_rmse[c].values()) for c in CONVS]

refs = {}
b = pd.read_csv(os.path.join(NETLOAD, "results_baselines_netload.csv")).set_index("model")
e = pd.read_csv(os.path.join(NETLOAD, "results_baselines_extra_netload.csv")).set_index("model")
for name in ["XGBoost", "GAM", "Chronos-Bolt"]:
    refs[name] = {"smape": float(b.loc[name, "smape"]), "rmse": float(b.loc[name, "rmse"])}
for name in ["Chronos-2", "ARIMA-X"]:
    refs[name] = {"smape": float(e.loc[name, "smape"]), "rmse": float(e.loc[name, "rmse"])}
lstm = pd.read_csv(os.path.join(os.path.dirname(NETLOAD), "results_lstm_seeds", "netload_summary.csv"))
refs["LSTM (5-seed mean)"] = {"smape": float(lstm.pct.mean()), "rmse": float(lstm.rmse.mean())}
gr = pd.read_csv(os.path.join(NETLOAD, "results_gam_regional.csv"))["national_gam_pred"].to_numpy()
refs["GAM (regional)"] = {"smape": smape(gr[:T], y), "rmse": rmse(gr[:T], y), "n": len(gr)}
refs["Persistence (1 day)"] = {"smape": smape(y[:-48], y[48:]), "rmse": rmse(y[:-48], y[48:]), "note": "on days 2-365"}

bottom_c = res["decomposed_aggregates"]["MLpol (bottom-up)"]["rmse"]
# Paired tests of the best GNN forecasts against the strongest references whose forecasts
# are stored: the regional GAM and the five LSTM seeds.
lstm_dir = os.path.join(os.path.dirname(NETLOAD), "results_lstm_seeds")
lstm_preds = {sd: np.load(os.path.join(lstm_dir, f"netload_s{sd}.npy"))[:T] for sd in (42, 0, 1, 2, 3)}
gnn_best = {"decomposed MLpol (bottom-up)": agg_c["MLpol (bottom-up)"],
            f"decomposed {best_c}": decomp[best_c].sum(axis=0)}
dm_refs = {}
for gname, gp in gnn_best.items():
    t = diebold_mariano(gp, gr[:T], y, loss="squared", h=48)
    dm_refs[gname] = {"GAM (regional)": {"p": float(t.pvalue), "better": t.better}}
    tl = [diebold_mariano(gp, lp, y, loss="squared", h=48) for lp in lstm_preds.values()]
    dm_refs[gname]["LSTM seeds"] = {"p": [float(x.pvalue) for x in tl], "better": [x.better for x in tl]}
res["summary"] = {
    "direct_smape_band": [min(r["smape"] for r in res["direct"].values()), max(r["smape"] for r in res["direct"].values())],
    "best_direct": best_d, "best_decomposed": best_c,
    "decomposition_improves": sum(g > 0 for g in gain.values()), "mean_rmse_gain": float(np.mean(list(gain.values()))),
    "dm_best_decomposed_vs_best_direct_p": float(dm.pvalue), "dm_better": dm.better,
    "dm_vs_references": dm_refs,
    "decomposed_bottom_minus_refs": {k: bottom_c - refs[k]["rmse"] for k in ["GAM", "GAM (regional)", "LSTM (5-seed mean)"]},
    "component_rmse_ranges": {k: [min(comp_rmse[c][k] for c in CONVS), max(comp_rmse[c][k] for c in CONVS)] for k in COMPS},
    "wind_share_of_squared_component_rmse": [float(min(wind_share)), float(np.mean(wind_share)), float(max(wind_share))],
    "spearman_wind_vs_recombined": float(spearmanr([comp_rmse[c]["Wind_power"] for c in CONVS],
                                                   [res["decomposed"][c]["rmse"] for c in CONVS]).correlation),
}
res["references"] = refs
res["component_rmse"] = comp_rmse
json.dump(res, open(os.path.join(HERE, "netload_stats_v030" + ("_leads" if RUN == "leads" else "") + ".json"), "w"), indent=1)
print(json.dumps(res["summary"], indent=1))
for blk in ["direct", "direct_aggregates", "decomposed", "decomposed_aggregates"]:
    print(f"--- {blk}")
    for k, r in sorted(res[blk].items(), key=lambda kv: kv[1]["rmse"]):
        print(f"{k:<20} {r['smape']:.2f} ± {r['smape_se']:.2f}   {r['rmse']:.0f} ± {r['rmse_se']:.0f}")
print("--- references"); print(json.dumps(refs, indent=1))
