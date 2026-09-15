"""Fair-config test comparison: retrain the table convolutions at the
validation-selected config (hidden=64, 1 layer, shallow as the spectral gap of
the DTW graph predicts) and report national net-load RMSE / sMAPE on the 2019
test year, to check whether a fairly-configured GNN beats the operational GAM.
"""
import os
import gc
import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import (RGATConv, GatedGraphConv, SuperGATConv, FiLMConv,
                                     PANConv, GMMConv, GCN2Conv, GATConv, FastRGCNConv,
                                     ResGatedGraphConv)
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer
import netload_pipeline as P

HIDDEN, LAYERS, EPOCHS, ADJ = 64, 1, 300, "dtw"
OUT = os.path.join(os.path.dirname(__file__), "results_std_direct_netload")
os.makedirs(OUT, exist_ok=True)

CONVS = {
    "RGATConv": (RGATConv, {"heads": 2}), "GatedGraphConv": (GatedGraphConv, {}),
    "SuperGATConv": (SuperGATConv, {}), "FiLMConv": (FiLMConv, {}), "PANConv": (PANConv, {}),
    "GMMConv": (GMMConv, {}), "GCN2Conv": (GCN2Conv, {}), "GATConv": (GATConv, {"heads": 2}),
    "FastRGCNConv": (FastRGCNConv, {}), "ResGatedGraphConv": (ResGatedGraphConv, {}),
}


def smape(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return 100.0 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8))


def main():
    print(f"Fair config: hidden={HIDDEN}, layers={LAYERS}, epochs={EPOCHS}, graph={ADJ}")
    train, val, test, nodes = P.build_datasets()
    for ds in (train, val, test):
        ds._set_adj_matrix(adj_matrix=ADJ)
    rows = []
    for name, (cls, kw) in CONVS.items():
        path = os.path.join(OUT, f"{name}.pt")
        if os.path.exists(path):
            pred = torch.load(path, weights_only=False)
            tgt = torch.load(os.path.join(OUT, "target.pt"), weights_only=False)
        else:
            print(f"\nTraining {name}...")
            seed_everything(42); gc.collect()
            try:
                model = myGNN(in_channels=train.num_node_features, num_layers=LAYERS,
                              hidden_channels=HIDDEN, out_channels=P.OUT_CHANNELS,
                              conv_class=cls, conv_kwargs=kw)
                tr = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                             batch_size=16, model_kwargs={"lr": 1e-3, "num_epochs": EPOCHS},
                             reconcile=True, top_level_model="xgb", lam_reg=0)
                res = tr.train(force_training=True, save=False, patience=40)
                pred, target = res[0], res[1]
                torch.save(pred, path)
                tgt = target.sum(dim=0).cpu(); torch.save(tgt, os.path.join(OUT, "target.pt"))
            except Exception as e:
                print(f"  failed: {str(e).splitlines()[0]}"); continue
        pn, yv = pred.sum(dim=0).cpu().numpy(), tgt.numpy()
        r = float(np.sqrt(np.mean((pn - yv) ** 2)))
        rows.append((name, smape(pn, yv), r))
        print(f"  {name:<18} sMAPE={smape(pn, yv):.2f}%  RMSE={r:.0f} MW")
    rows.sort(key=lambda x: x[2])
    print(f'\n{"Conv (h64,l1)":<20}{"sMAPE":>8}{"RMSE":>10}')
    for n, s, r in rows:
        print(f"{n:<20}{s:>7.2f}%{r:>8.0f}")
    print(f"\nBest fair-config GNN: {rows[0][0]} {rows[0][2]:.0f} MW / {rows[0][1]:.2f}%")
    print("Reference (test): GAM 1763 MW/2.92% ; default sweep best 2000 MW/3.23%")


if __name__ == "__main__":
    main()
