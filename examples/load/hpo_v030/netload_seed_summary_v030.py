"""Seed study of the leading net-load models (weather trajectories, seeds 0-4).

Reads the leaders from ./netload_stats_v030_leads.json and the forecasts of
./netload_v030/leads_seed{0..4}; reports the national RMSE of each seed, its mean and
standard deviation, the reported run (seed 42), and the paired gap between the direct
and decomposed forecasts of each seed. Writes ./netload_seeds_v030_leads.json.
"""
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
summary = json.load(open(os.path.join(HERE, "netload_stats_v030_leads.json")))["summary"]
direct, decomp = summary["best_direct"], summary["best_decomposed"]
ld = lambda run, name: torch.load(os.path.join(HERE, "netload_v030", run, f"{name}.pt"),
                                  weights_only=False).numpy().astype(float).sum(axis=0)
y = ld("leads_seed42", "target__NetLoad")
T = len(y)
rmse = lambda p: float(np.sqrt(np.mean((p[:T] - y) ** 2)))


def forecasts(run):
    d = ld(run, direct)[:T]
    c = (ld(run, f"{decomp}__Load") - ld(run, f"{decomp}__Wind_power") - ld(run, f"{decomp}__Solar_power"))[:T]
    return d, c


rows = {s: [rmse(f) for f in forecasts(f"leads_seed{s}")] for s in SEEDS}
reported = [rmse(f) for f in forecasts("leads_seed42")]
d_seeds, c_seeds = np.array([r[0] for r in rows.values()]), np.array([r[1] for r in rows.values()])
out = {"direct": {"model": direct, "rmse_by_seed": d_seeds.tolist(), "mean": float(d_seeds.mean()),
                  "std": float(d_seeds.std(ddof=1)), "reported": reported[0]},
       "decomposed": {"model": decomp, "rmse_by_seed": c_seeds.tolist(), "mean": float(c_seeds.mean()),
                      "std": float(c_seeds.std(ddof=1)), "reported": reported[1]},
       "paired_gap_mean": float((d_seeds - c_seeds).mean()), "paired_gap_std": float((d_seeds - c_seeds).std(ddof=1))}
json.dump(out, open(os.path.join(HERE, "netload_seeds_v030_leads.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
