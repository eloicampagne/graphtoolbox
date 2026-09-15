"""Optuna hyperparameter optimization on net-load (mirror of load cell 18).

Runs the GraphToolbox Optimizer for the same three convolutions the load example
tuned (APPNP, GATConv, LEConv). Each run writes ./results_optim_{Conv}/myGNN_best.json,
which the convolution sweep in the notebook picks up automatically (multi-config).

Usage:
    python optimize_netload.py --trials 200 --epochs 100
    python optimize_netload.py --convs LEConv --trials 50   # cheaper subset
"""
import argparse
import os

from torch_geometric.nn.conv import APPNP, GATConv, LEConv

from graphtoolbox.models import myGNN
from graphtoolbox.optim import Optimizer

import netload_pipeline as P

optim_base = {"num_layers": (1, 5), "hidden_channels": (32, 512), "lr": (1e-5, 1e-1)}
_with_heads = optim_base | {"heads": (1, 5)}

OPTIM_KWARGS = {
    "APPNP":   optim_base | {"K": (1, 10), "alpha": (0.5, 1.0)},
    "GATConv": _with_heads,
    "LEConv":  optim_base,
}
CONV_CLASSES = {"APPNP": APPNP, "GATConv": GATConv, "LEConv": LEConv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--convs", nargs="+", default=["APPNP", "GATConv", "LEConv"])
    ap.add_argument("--force", action="store_true", help="re-run even if _best.json exists")
    args = ap.parse_args()

    os.chdir(str(P.HERE))  # Optimizer writes results_optim_* relative to cwd
    print("Building net-load datasets...")
    train, val, test, nodes = P.build_datasets()

    for conv_name in args.convs:
        best_json = f"./results_optim_{conv_name}/myGNN_best.json"
        if os.path.exists(best_json) and not args.force:
            print(f"[skip] {conv_name}: {best_json} already exists")
            continue
        print(f"\n=== Optimizing {conv_name}  ({args.trials} trials x {args.epochs} epochs) ===")
        optimizer = Optimizer(model=myGNN, dataset_train=train, dataset_val=val,
                              conv_class=CONV_CLASSES[conv_name], num_epochs=args.epochs,
                              optim_kwargs=OPTIM_KWARGS[conv_name])
        optimizer.optimize(n_trials=args.trials)
        print(f"=== Done {conv_name} -> {best_json} ===")


if __name__ == "__main__":
    main()
