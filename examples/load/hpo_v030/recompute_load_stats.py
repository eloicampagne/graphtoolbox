"""Rebuild the load archive with the v0.3.0 configurations and recompute the paper's load statistics.

Reads ../forecasts_paper/load_sweep.npz, replaces the forecasts of the operators
given on the command line by ./preds/<conv>_v030_unrec.pt, recomputes the
uniform and MLpol aggregates as package_paper_forecasts.add_load_aggregates
does, and writes ./load_sweep_v030.npz and ./stats_v030.json. The committed
archive is left untouched. With --check, the aggregates of the committed
archive are first recomputed and compared with the stored ones.

With --suffix S, the replacing forecasts are ./preds/<conv>_S.pt (default v030_unrec, the
configurations of the hyperparameter searches). The paper uses --suffix default_unrec: the
three operators at the common configuration of the sweep, as every other operator.

Usage: python recompute_load_stats.py LEConv GATConv APPNP [--suffix S] [--check] [--n-boot 2000]
"""
import json
import os
import sys

import numpy as np
import torch

from graphtoolbox.aggregation import Aggregation
from graphtoolbox.evaluation import bootstrap_metric, diebold_mariano, model_confidence_set, pairwise_dm
from graphtoolbox.models.baseline import mape, rmse

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "forecasts_paper", "load_sweep.npz")
PREFIX = "aggregate__"
LOAD_TABLE = ["LEConv", "GatedGraphConv", "GATv2Conv", "ARMAConv", "ResGatedGraphConv",
              "SGConv", "RGATConv", "RGCNConv", "GCNConv", "GATConv"]
SUFFIX = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else "v030_unrec"
args = [a for a in sys.argv[1:] if not a.startswith("--") and a != SUFFIX]
n_boot = int(sys.argv[sys.argv.index("--n-boot") + 1]) if "--n-boot" in sys.argv else 2000
replace = [a for a in args if not a.isdigit()]


def aggregate(nodes, target, target_nodes):
    names = sorted(nodes)
    national = np.column_stack([nodes[n].sum(axis=0) for n in names])
    out = {f"{PREFIX}uniform": national.mean(axis=1).astype(np.float32),
           f"{PREFIX}mlpol_top": Aggregation(model="MLpol", loss="square").run(
               national, target, block_size=48).prediction_.astype(np.float32)}
    bottom = np.zeros(len(target), dtype=np.float32)
    for k in range(target_nodes.shape[0]):
        experts = np.column_stack([nodes[n][k] for n in names])
        bottom += Aggregation(model="MLpol", loss="square").run(
            experts, target_nodes[k], block_size=48).prediction_.astype(np.float32)
    out[f"{PREFIX}mlpol_bottom"] = bottom
    return out


with np.load(SRC) as z:
    arrays = {k: z[k] for k in z.files}
target, target_nodes = arrays["target"], arrays["target_nodes"]
# ECConv is an alias of NNConv in PyTorch Geometric: its forecasts duplicate those of NNConv.
ALIASES = {"ECConv"}
nodes = {k: v for k, v in arrays.items() if k not in {"target", "target_nodes"} | ALIASES and not k.startswith(PREFIX)}

if "--check" in sys.argv:
    again = aggregate(nodes, target, target_nodes)
    for k, v in again.items():
        print(f"check {k}: max |recomputed - stored| = {np.abs(v - arrays[k]).max():.3f} MW, "
              f"RMSE {rmse(v, target):.1f} vs {rmse(arrays[k], target):.1f}")

old_rmse = {n: rmse(nodes[n].sum(axis=0), target) for n in replace}
for n in replace:
    nodes[n] = torch.load(os.path.join(HERE, "preds", f"{n}_{SUFFIX}.pt"), weights_only=False).numpy().astype(np.float32)
new = dict(nodes, target=target, target_nodes=target_nodes, **aggregate(nodes, target, target_nodes))
np.savez_compressed(os.path.join(HERE, "load_sweep_v030.npz"), **new)

national = {n: p.sum(axis=0) for n, p in nodes.items()}
labels = {"Uniform average": new[f"{PREFIX}uniform"], "MLpol (bottom-up)": new[f"{PREFIX}mlpol_bottom"],
          "MLpol (top-level)": new[f"{PREFIX}mlpol_top"]}
rows = {}
for name, pred in [(n, national[n]) for n in LOAD_TABLE] + list(labels.items()):
    b = bootstrap_metric(pred, target, metric="rmse", n_boot=n_boot, block_len=48, seed=0)
    pb = bootstrap_metric(pred, target, metric="mape", n_boot=n_boot, block_len=48, seed=0)
    rows[name] = {"mape": float(mape(pred, target)), "rmse": float(rmse(pred, target)), "rmse_se": float(b.se),
                  "mape_se": float(pb.se)}
ranked = sorted(national, key=lambda n: rmse(national[n], target))
mapes = sorted(float(mape(national[n], target)) for n in national)
mcs = model_confidence_set(national, target, loss="squared", alpha=0.10, n_boot=n_boot, block_len=48, seed=0)
best = ranked[0]
dm_table = pairwise_dm({n: national[n] for n in LOAD_TABLE}, target, loss="squared", h=48, correction="holm")
best_in_table = min(LOAD_TABLE, key=lambda n: rmse(national[n], target))
dm_mlpol = diebold_mariano(new[f"{PREFIX}mlpol_top"], national[best], target, loss="squared", h=48,
                           names=("MLpol (top-level)", best))
stats = {
    "replaced": {n: {"old_rmse": old_rmse[n], "new_rmse": rows[n]["rmse"] if n in rows else rmse(national[n], target)}
                 for n in replace},
    "table": rows,
    "sweep_size": len(national),
    "sweep_mape_min_median_max": [mapes[0], float(np.median(mapes)), mapes[-1]],
    "best_single": best, "best_single_rmse": rmse(national[best], target),
    "top10_by_rmse": ranked[:10],
    "mcs_retained": len(mcs.included), "mcs_members": sorted(mcs.included),
    "table_in_mcs": [n for n in LOAD_TABLE if n in mcs.included],
    "holm_dm_separates_from_best_in_table": [n for n in LOAD_TABLE if n != best_in_table
                                             and dm_table["pvalue_adjusted"].loc[n, best_in_table] < 0.05],
    "mlpol_top_vs_best_dm_pvalue": float(dm_mlpol.pvalue), "mlpol_top_vs_best_better": dm_mlpol.better,
    "mlpol_top_minus_best_rmse": rmse(new[f"{PREFIX}mlpol_top"], target) - rmse(national[best], target),
}
json.dump(stats, open(os.path.join(HERE, "stats_v030.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in stats.items() if k not in {"mcs_members"}}, indent=1, default=str))
