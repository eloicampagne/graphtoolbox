"""Pairwise influence of the attention models of attention_v030.py, by input occlusion.

Uses graphtoolbox.interpretability.occlusion_influence: for each test day and each
region j, the input features of j are replaced by their test-year mean and the model
is run again. The influence of j on region i is the
mean absolute change of the 48 half-hourly forecasts of i, in MW. It counts every
path from j to i through the two layers, so it does not depend on how the attention
weights are read. Loads the checkpoint trained by attention_v030.py.
Writes ./influence_v030[_GATv2Conv].npz with infl [days, receiver, sender] (MW).

Usage: python influence_v030.py [GATConv|GATv2Conv]
"""
import os
import sys

import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import GATConv, GATv2Conv

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD = os.path.dirname(HERE)
sys.path.insert(0, LOAD)
os.chdir(LOAD)
import config as C  # noqa: E402
CONV = {"GATConv": GATConv, "GATv2Conv": GATv2Conv}[sys.argv[1] if len(sys.argv) > 1 else "GATConv"]
GRAPH = sys.argv[2] if len(sys.argv) > 2 else "dtw"
PRIOR = "prior" in sys.argv[3:]     # graph weights as edge attributes
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 42
SUFFIX = ("" if CONV is GATConv else "_" + CONV.__name__) + ("" if GRAPH == "dtw" else "_" + GRAPH) + ("_prior" if PRIOR else "") + ("" if SEED == 42 else f"_seed{SEED}")
from graphtoolbox.data.dataset import DataClass, GraphDataset  # noqa: E402
from graphtoolbox.models import myGNN  # noqa: E402
from graphtoolbox.training import Trainer  # noqa: E402
from graphtoolbox.interpretability import occlusion_influence, target_unit_scale  # noqa: E402

data = DataClass(path_train="train.csv", path_test="test.csv", data_kwargs=C.data_kwargs, folder_config=LOAD)
common = dict(graph_folder=os.path.join(os.path.dirname(LOAD), "graph_representations"),
              dataset_kwargs=C.dataset_kwargs, out_channels=48)
train = GraphDataset(data=data, period="train", **common)
val = GraphDataset(data=data, period="val", scalers_feat=train.scalers_feat, scalers_target=train.scalers_target, **common)
test = GraphDataset(data=data, period="test", scalers_feat=train.scalers_feat, scalers_target=train.scalers_target, **common)
for ds in (train, val, test):
    ds._set_adj_matrix(adj_matrix=GRAPH)

seed_everything(SEED)
model = myGNN(in_channels=train.num_node_features, num_layers=2, hidden_channels=64, out_channels=48,
              conv_class=CONV, heads=1, edge_weight_as_attr=PRIOR)
trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test, batch_size=32,
                  model_kwargs={"lr": 1e-3, "num_epochs": 100}, reconcile=False, lam_reg=0)
pred, target = trainer.train(force_training=False, patience=30, plot_loss=False,
                             saving_directory=os.path.join(HERE, "checkpoints_attention", CONV.__name__ + "_" + GRAPH + ("_prior" if PRIOR else "") + ("" if SEED == 42 else f"_seed{SEED}")))[:2]
p, y = pred.sum(0).cpu().numpy(), target.sum(0).cpu().numpy()
print(f"{CONV.__name__}: national RMSE {np.sqrt(np.mean((p - y) ** 2)):.0f} MW (checkpoint)", flush=True)

infl = occlusion_influence(model, test, node_scale=target_unit_scale(train.scalers_target, trainer.nodes))
np.savez(os.path.join(HERE, f"influence_v030{SUFFIX}.npz"), infl=infl, nodes=np.array(trainer.nodes))
n = infl.shape[1]
off = ~np.eye(n, dtype=bool)
print(f"self influence median {np.median(infl.mean(0)[~off]):.0f} MW, cross influence median {np.median(infl.mean(0)[off]):.0f} MW")
