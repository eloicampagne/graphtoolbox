"""Sender share of the attention deviations, and relation to the DTW weights, for each attention variant."""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)) + "/..")
from graphtoolbox.interpretability import neighbourhood_deviation, sender_variance_share  # noqa: E402

W = np.loadtxt(os.path.join(os.path.dirname(os.path.dirname(HERE)), "graph_representations", "dtw", "W.txt"))[:, :12]
out = {}
for tag in ["", "_prior", "_GATv2Conv", "_GATv2Conv_prior", "_space", "_GATv2Conv_space"]:
    f = os.path.join(HERE, f"attention_v030{tag}.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    alpha, ei = z["alpha"].mean(axis=1), z["edge_index"]
    dev_year = neighbourhood_deviation(alpha.mean(0), ei, exclude_self=False)   # d_i * alpha_ij - 1
    off = ei[0] != ei[1]
    daily = [sender_variance_share(neighbourhood_deviation(a, ei, exclude_self=False)[off], ei[:, off]) for a in alpha]
    r = {"sender_share_year": sender_variance_share(dev_year[off], ei[:, off]),
         "sender_share_daily_median": float(np.median(daily)),
         "abs_dev_median_pct": float(100 * np.median(np.abs(dev_year[off]))),
         "abs_dev_max_pct": float(100 * np.abs(dev_year[off]).max())}
    if "space" not in tag:
        r["spearman_dev_vs_dtw_weight"] = float(spearmanr(dev_year[off], W[ei[1][off], ei[0][off]])[0])
    out[tag or "_GATConv"] = r
    print(tag or "_GATConv", {k: round(v, 3) for k, v in r.items()})
json.dump(out, open(os.path.join(HERE, "attention_variants_stats.json"), "w"), indent=1)
