"""Cost of the same code path at two graph scales (Table 3): parameter count and inference latency per forecast window.

GraphSAGE, GAT and APPNP (K = 1, alpha = 0.9) at one common configuration, hidden width 64 and two layers, on the
12 French regions (load inputs and DTW graph of the sweep, 187 features) and on the 500 WEAVE-UK feeders (50 features,
correlation graph). The WEAVE-UK files (train_weave_500.csv, test_weave_500.csv and graph_representations/) are read
from a folder Weave beside the repository. Neither quantity depends on the weights, so the models are not trained.
Latency: CPU, eval mode, one test window, 20 warm-up calls, then the mean over 200 calls.

Usage: python footprint_v030.py     # writes footprint_v030.csv
"""
import os
import time

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(os.cpu_count())
from graphtoolbox.data.dataset import DataClass, GraphDataset
from graphtoolbox.models.gnn import myGNN
from torch_geometric.nn.conv import APPNP, GATConv, SAGEConv

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD = os.path.dirname(HERE)
GTB = os.path.dirname(os.path.dirname(LOAD))
WEAVE = os.path.join(os.path.dirname(GTB), "Weave")
H, WIDTH, LAYERS = 48, 64, 2
ARCHS = [("GraphSAGE", SAGEConv, {}), ("GAT", GATConv, {}), ("APPNP", APPNP, {"K": 1, "alpha": 0.9})]


def build(path_train, path_test, data_kwargs, dataset_kwargs, folder, graphs):
    data = DataClass(path_train=path_train, path_test=path_test, folder_config=folder, data_kwargs=data_kwargs)
    common = dict(dataset_kwargs=dataset_kwargs, out_channels=H, graph_folder=graphs)
    tr = GraphDataset(data=data, period="train", **common)
    te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat, scalers_target=tr.scalers_target, **common)
    te._set_adj_matrix(adj_matrix=dataset_kwargs["adj_matrix"])
    return te


france_data = {
    "node_var": "Region",
    "features_to_lag": {"load": (1, 48), "temp": (-47, -1)},
    "dummies": ["Instant", "RegionInt", "JourSemaine", "DayType", "offset"],
    "day_inf_train": "2015-01-01", "day_sup_train": "2018-01-01",
    "day_inf_val": "2018-01-01", "day_sup_val": "2018-12-31",
    "day_inf_test": "2019-01-01", "day_sup_test": "2019-12-31",
    "computed_features": {
        "heat_deg": lambda df: (15 - df["temp"]).clip(lower=0),
        "cool_deg": lambda df: (df["temp"] - 15).clip(lower=0),
        "temp_sq": lambda df: df["temp"] ** 2,
        "is_summer": lambda df: (pd.to_datetime(df["date"]).dt.month == 7).astype(float)
                                + (pd.to_datetime(df["date"]).dt.month == 8).astype(float),
    },
}
france_ds = {
    "batch_size": 32, "adj_matrix": "dtw", "target_base": "load",
    "features_base": ["temp", "nebu", "wind", "tempMax", "tempMin", "Posan", "Instant", "RegionInt", "JourSemaine",
                      "JourFerie", "offset", "DayType", "Weekend", "temp_liss_fort", "temp_liss_faible",
                      "heat_deg", "cool_deg", "temp_sq", "is_summer"]
                     + [f"load_l{t}" for t in range(1, 49)] + [f"temp_l{t}" for t in range(-47, 0)],
}
weave_data = {
    "node_var": "id_unique", "dummies": [],
    "day_inf_train": "2024-02-14", "day_sup_train": "2024-02-23",
    "day_inf_val": "2024-02-23", "day_sup_val": "2024-02-24",
    "day_inf_test": "2024-02-25", "day_sup_test": "2024-02-28",
}
weave_ds = {"batch_size": 16, "adj_matrix": "correlation", "target_base": "consumption",
            "features_base": ["instant", "weekday"] + [f"consumption_l{t}" for t in range(1, 49)]}

os.chdir(LOAD)
sets = {
    "France (12 nodes)": build(os.path.join(LOAD, "train.csv"), os.path.join(LOAD, "test.csv"), france_data,
                               france_ds, LOAD, os.path.join(GTB, "examples", "graph_representations")),
    "WEAVE (500 nodes)": build(os.path.join(WEAVE, "train_weave_500.csv"), os.path.join(WEAVE, "test_weave_500.csv"),
                               weave_data, weave_ds, WEAVE, os.path.join(WEAVE, "graph_representations")),
}
rows = []
for tag, te in sets.items():
    g = te[0]
    for name, conv, kw in ARCHS:
        torch.manual_seed(0)
        model = myGNN(in_channels=te.num_node_features, num_layers=LAYERS, hidden_channels=WIDTH, out_channels=H,
                      conv_class=conv, conv_kwargs=kw).eval()
        with torch.no_grad():
            for _ in range(20):
                model(g.x, g.edge_index)
            t0 = time.perf_counter()
            for _ in range(200):
                model(g.x, g.edge_index)
            ms = (time.perf_counter() - t0) / 200 * 1e3
        rows.append({"model": name, "dataset": tag, "n_nodes": g.x.shape[0], "n_edges": g.edge_index.shape[1],
                     "in_channels": te.num_node_features, "params_k": round(sum(p.numel() for p in model.parameters()) / 1e3, 1),
                     "infer_ms": round(ms, 2)})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "footprint_v030.csv"), index=False)
