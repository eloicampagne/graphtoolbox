"""Mini configuration search for the net-load GNN, on validation (2018).

Grids hidden_channels in {32,64,128} x num_layers in {1,2,3} (lr, batch, graph
fixed), evaluated on the VALIDATION national net-load RMSE so the test year stays
untouched, averaged over a few representative convolutions. Larger hidden width
does not help on this 12-node graph, so the point is to pick a sensible small
config rather than to maximise capacity.

Validation predictions are obtained by passing the validation dataset in the
Trainer's test slot, so the reconciliation is fit on train and applied to val.
"""
import os
import gc
import itertools
import json
import numpy as np
from torch_geometric import seed_everything
from torch_geometric.nn.conv import LEConv, GATConv, SAGEConv

from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer
import netload_pipeline as P

HIDDENS = [32, 64, 128]
LAYERS = [1, 2, 3]
EPOCHS = 150
PATIENCE = 30
ADJ = "dtw"
CONVS = {"LEConv": (LEConv, {}), "GATConv": (GATConv, {"heads": 2}), "SAGEConv": (SAGEConv, {})}
OUT = os.path.join(os.path.dirname(__file__), "results_mini_optim_netload")
os.makedirs(OUT, exist_ok=True)


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def main():
    train, val, test, nodes = P.build_datasets()
    for ds in (train, val, test):
        ds._set_adj_matrix(adj_matrix=ADJ)

    cache_path = os.path.join(OUT, "results.json")
    results = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    for conv_name, (cls, kw) in CONVS.items():
        for hidden, layers in itertools.product(HIDDENS, LAYERS):
            key = f"{conv_name}|h{hidden}|l{layers}"
            if key in results:
                continue
            print(f"\n[{key}] training (val-selection)...")
            seed_everything(42); gc.collect()
            try:
                model = myGNN(in_channels=train.num_node_features, num_layers=layers,
                              hidden_channels=hidden, out_channels=P.OUT_CHANNELS,
                              conv_class=cls, conv_kwargs=kw)
                # Validation set placed in the test slot -> reconciled val predictions.
                trainer = Trainer(model=model, dataset_train=train, dataset_val=val,
                                  dataset_test=val, batch_size=16,
                                  model_kwargs={"lr": 1e-3, "num_epochs": EPOCHS},
                                  reconcile=True, top_level_model="xgb", lam_reg=0)
                res = trainer.train(force_training=True, save=False, patience=PATIENCE)
                pred, target = res[0], res[1]
                vr = rmse(pred.sum(dim=0).cpu().numpy(), target.sum(dim=0).cpu().numpy())
                results[key] = vr
                print(f"  val national RMSE = {vr:.0f} MW")
                json.dump(results, open(cache_path, "w"), indent=1)
            except Exception as e:
                print(f"  failed: {str(e).splitlines()[0]}")
            gc.collect()

    # Aggregate: mean val RMSE per (hidden, layers) across convolutions.
    print("\n=== Mean validation RMSE (MW) per config, across convolutions ===")
    print(f'{"hidden/layers":<16}' + "".join(f"l{l:<7}" for l in LAYERS))
    grid = {}
    for hidden in HIDDENS:
        row = f"h{hidden:<14}"
        for layers in LAYERS:
            vals = [results[f"{c}|h{hidden}|l{layers}"] for c in CONVS
                    if f"{c}|h{hidden}|l{layers}" in results]
            m = np.mean(vals) if vals else float("nan")
            grid[(hidden, layers)] = m
            row += f"{m:<8.0f}"
        print(row)
    best = min((k for k in grid if not np.isnan(grid[k])), key=lambda k: grid[k])
    print(f"\nBest config by mean validation RMSE: hidden={best[0]}, layers={best[1]} "
          f"({grid[best]:.0f} MW)")
    print("Reference (test): GAM 1763 MW ; default sweep (h64,l2) test-best 2000 MW")


if __name__ == "__main__":
    main()
