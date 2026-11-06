"""Rows of the reconciled forecasts for the load and net-load tables: same metrics and block bootstrap as
recompute_load_stats.py (MAPE, RMSE) and netload_stats_v030.py (sMAPE, RMSE), 2000 resamples of one-day blocks.

Reads mint_tam_forecasts.npz, mint_xgb_forecasts.npz and mint_mlp_forecasts.npz (written by mint_tam.py) and writes mint_table_rows.json.
"""
import json
import os

import numpy as np
from graphtoolbox.evaluation import bootstrap_metric
from graphtoolbox.evaluation.significance import _moving_block_indices
from graphtoolbox.models.baseline import mape

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 2000
rmse = lambda p, y: float(np.sqrt(np.mean((p - y) ** 2)))
smape = lambda p, y: float(100 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8)))


def smape_se(p, y):
    rng = np.random.default_rng(0)
    return float(np.std([smape(p[i], y[i]) for i in (_moving_block_indices(len(y), 48, rng) for _ in range(N_BOOT))],
                        ddof=1))


tam, xgb, mlp = (np.load(os.path.join(HERE, f)) for f in ("mint_tam_forecasts.npz", "mint_xgb_forecasts.npz",
                                                       "mint_mlp_forecasts.npz"))
load = "load GatedGraphConv"
y = tam[f"{load}__target"]
assert np.allclose(y, xgb[f"{load}__target"])
rows = {"load": {}, "netload": {}}
for name, p in [("internal national XGBoost, diagonal", tam[f"{load}__trainer_wls_xgb_top"]),
                ("TAM, diagonal", tam[f"{load}__mint_wls"]), ("TAM, full", tam[f"{load}__mint_shrink"]),
                ("pooled XGBoost, diagonal", xgb[f"{load}__mint_wls"]), ("pooled XGBoost, full", xgb[f"{load}__mint_shrink"])]:
    rows["load"][name] = {"mape": float(mape(p, y)), "rmse": rmse(p, y),
                          "mape_se": float(bootstrap_metric(p, y, metric="mape", n_boot=N_BOOT, block_len=48, seed=0).se),
                          "rmse_se": float(bootstrap_metric(p, y, metric="rmse", n_boot=N_BOOT, block_len=48, seed=0).se)}
for (top, z), kind in [(t, k) for t in (("TAM", tam), ("MLP", mlp)) for k in ("direct", "decomposed")]:
    key = f"netload {kind} FastRGCNConv"
    yn = z[f"{key}__target"]
    for w, lab in (("wls", "diagonal"), ("shrink", "full")):
        p = z[f"{key}__mint_{w}"]
        rows["netload"][f"{kind}, {top}, {lab}"] = {
            "smape": smape(p, yn), "smape_se": smape_se(p, yn), "rmse": rmse(p, yn),
            "rmse_se": float(bootstrap_metric(p, yn, metric="rmse", n_boot=N_BOOT, block_len=48, seed=0).se)}
json.dump(rows, open(os.path.join(HERE, "mint_table_rows.json"), "w"), indent=1)
for task, r in rows.items():
    for n, v in r.items():
        m = "mape" if task == "load" else "smape"
        print(f"{task:<8} {n:<40} {v[m]:.2f} ± {v[m + '_se']:.2f}   {v['rmse']:.0f} ± {v['rmse_se']:.0f}")
