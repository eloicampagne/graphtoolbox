"""Fair-configuration net-load check: retrain the strongest convolutions at a
configuration comparable to the engineered GAM (hidden=256, 300 epochs), instead
of the small default sweep config (hidden=64, 100 epochs), and report national
RMSE / sMAPE against the same 2019 target. This tests whether a fairly-configured
GNN matches the operational GAM (1763 MW / 2.92%).
"""
import os
import gc
import numpy as np
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import RGATConv, GATConv, LEConv, SAGEConv, GatedGraphConv, FiLMConv

from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer
import netload_pipeline as P

HIDDEN = 256
NUM_LAYERS = 2
NUM_EPOCHS = 300
ADJ = "dtw"
OUT_DIR = os.path.join(os.path.dirname(__file__), "results_fair_config_netload")
os.makedirs(OUT_DIR, exist_ok=True)

CONVS = {
    "LEConv": (LEConv, {}),
    "SAGEConv": (SAGEConv, {}),
    "GATConv": (GATConv, {"heads": 2}),
    "RGATConv": (RGATConv, {"heads": 2}),
    "GatedGraphConv": (GatedGraphConv, {}),
    "FiLMConv": (FiLMConv, {}),
}


def smape(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return 100.0 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8))


def main():
    print(f"Fair config: hidden={HIDDEN}, layers={NUM_LAYERS}, epochs={NUM_EPOCHS}, graph={ADJ}")
    train, val, test, nodes = P.build_datasets()
    for ds in (train, val, test):
        ds._set_adj_matrix(adj_matrix=ADJ)

    rows = []
    for name, (cls, kw) in CONVS.items():
        path = os.path.join(OUT_DIR, f"{name}.pt")
        if os.path.exists(path):
            pred = torch.load(path, weights_only=False)
            tgt = torch.load(os.path.join(OUT_DIR, "target.pt"), weights_only=False)
        else:
            print(f"\nTraining {name} (hidden={HIDDEN}, {NUM_EPOCHS} ep)...")
            seed_everything(42); gc.collect()
            model = myGNN(in_channels=train.num_node_features, num_layers=NUM_LAYERS,
                          hidden_channels=HIDDEN, out_channels=P.OUT_CHANNELS,
                          conv_class=cls, conv_kwargs=kw)
            trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                              batch_size=16, model_kwargs={"lr": 1e-3, "num_epochs": NUM_EPOCHS},
                              reconcile=True, top_level_model="xgb", lam_reg=0)
            res = trainer.train(force_training=False, patience=50)
            pred, target = res[0], res[1]
            torch.save(pred, path)
            tgt = target.sum(dim=0).cpu()
            torch.save(tgt, os.path.join(OUT_DIR, "target.pt"))
        pn = pred.sum(dim=0).cpu().numpy()
        yv = tgt.numpy()
        r = float(np.sqrt(np.mean((pn - yv) ** 2)))
        rows.append((name, smape(pn, yv), r))
        print(f"  {name:<16} sMAPE={smape(pn, yv):.2f}%  RMSE={r:.0f} MW")

    print(f'\n{"Conv (hidden=256, 300ep)":<26} {"sMAPE":>8} {"RMSE":>10}')
    print("-" * 46)
    for name, sm, r in sorted(rows, key=lambda x: x[2]):
        print(f"{name:<26} {sm:>7.2f}% {r:>8.0f} MW")
    print("\nReference: GAM 2.92% / 1763 MW ; default-config sweep best 3.23% / 2000 MW")


if __name__ == "__main__":
    main()
