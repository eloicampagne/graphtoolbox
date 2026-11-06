"""Cross-seed consistency of the year-mean attention and influence deviations on the spatial graph.

For each model (GATConv, GATv2Conv) and seed (42, 0, 1), computes the deviation
m_i s_ij - 1 of every edge entering a region with several neighbors, for the
attention and for the occlusion influence, then the Spearman correlation of each
quantity between seeds and between the two quantities within a seed.
Writes ./attention_seed_consistency.json.
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from graphtoolbox.interpretability import neighbourhood_deviation  # noqa: E402

SEEDS = {42: "", 0: "_seed0", 1: "_seed1"}
out = {}
for model, prefix in (("GATConv", "_space"), ("GATv2Conv", "_GATv2Conv_space")):
    att, inf, rmse = {}, {}, {}
    for seed, sfx in SEEDS.items():
        z = np.load(os.path.join(HERE, f"attention_v030{prefix}{sfx}.npz"))
        ei = z["edge_index"]
        I = np.load(os.path.join(HERE, f"influence_v030{prefix}{sfx}.npz"), allow_pickle=True)["infl"].mean(0)
        a = neighbourhood_deviation(z["alpha"].mean(axis=(0, 1)), ei)
        b = neighbourhood_deviation(I[ei[1], ei[0]], ei)
        m = np.array([((ei[1] == d) & (ei[0] != ei[1])).sum() for d in ei[1]])
        keep = (ei[0] != ei[1]) & (m > 1)
        order = np.lexsort((ei[0][keep], ei[1][keep]))
        att[seed], inf[seed] = a[keep][order], b[keep][order]
    pairs = list(itertools.combinations(SEEDS, 2))
    out[model] = {
        "attention_vs_influence": {s: float(spearmanr(att[s], inf[s])[0]) for s in SEEDS},
        "attention_across_seeds": {f"{a}-{b}": float(spearmanr(att[a], att[b])[0]) for a, b in pairs},
        "influence_across_seeds": {f"{a}-{b}": float(spearmanr(inf[a], inf[b])[0]) for a, b in pairs},
        "influence_seed_mean_vs_attention_seed_mean": float(spearmanr(np.mean([att[s] for s in SEEDS], 0),
                                                                      np.mean([inf[s] for s in SEEDS], 0))[0]),
    }
    print(model, json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 2)
                             for k, v in out[model].items()}))
json.dump(out, open(os.path.join(HERE, "attention_seed_consistency.json"), "w"), indent=1)
