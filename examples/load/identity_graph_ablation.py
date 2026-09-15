"""Identity-graph ablation used in the GraphToolbox paper.

The ten table operators are trained at one common configuration on either the
DTW graph or an identity graph.  Regional predictions, the national target, and
the summary table are cached in ``results_identity_graph_ablation``.

Run from any directory with::

    python examples/load/identity_graph_ablation.py
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import (
    ARMAConv, GATConv, GATv2Conv, GCNConv, GatedGraphConv, LEConv, RGATConv,
    RGCNConv, ResGatedGraphConv, SGConv,
)

from graphtoolbox.data import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.models.baseline import rmse
from graphtoolbox.training import Trainer, set_device


HERE = Path(__file__).resolve().parent
GRAPH_FOLDER = HERE.parent / "graph_representations"
OUTPUT = HERE / "results_identity_graph_ablation"
CHECKPOINTS = OUTPUT / "checkpoints"
OUTPUT.mkdir(exist_ok=True)

CONVOLUTIONS = {
    "LEConv": LEConv,
    "GatedGraphConv": GatedGraphConv,
    "GATv2Conv": GATv2Conv,
    "ResGatedGraphConv": ResGatedGraphConv,
    "ARMAConv": ARMAConv,
    "SGConv": SGConv,
    "RGATConv": RGATConv,
    "RGCNConv": RGCNConv,
    "GATConv": GATConv,
    "GCNConv": GCNConv,
}

DATA_KWARGS = {
    "node_var": "Region",
    "features_to_lag": {"load": (1, 48), "temp": (-47, -1)},
    "dummies": ["Instant", "RegionInt", "JourSemaine", "DayType", "offset"],
    "day_inf_train": "2015-01-01", "day_sup_train": "2018-01-01",
    "day_inf_val": "2018-01-01", "day_sup_val": "2018-12-31",
    "day_inf_test": "2019-01-01", "day_sup_test": "2019-12-31",
    "computed_features": {
        "heat_deg": lambda frame: (15 - frame["temp"]).clip(lower=0),
        "cool_deg": lambda frame: (frame["temp"] - 15).clip(lower=0),
        "temp_sq": lambda frame: frame["temp"] ** 2,
        "is_summer": lambda frame: (
            (pd.to_datetime(frame["date"]).dt.month == 7).astype(float)
            + (pd.to_datetime(frame["date"]).dt.month == 8).astype(float)
        ),
    },
}
DATASET_KWARGS = {
    "batch_size": 32,
    "adj_matrix": "dtw",
    "features_base": [
        "temp", "nebu", "wind", "tempMax", "tempMin", "Posan", "Instant",
        "RegionInt", "JourSemaine", "JourFerie", "offset", "DayType",
        "Weekend", "temp_liss_fort", "temp_liss_faible", "heat_deg",
        "cool_deg", "temp_sq", "is_summer",
    ] + [f"load_l{lag}" for lag in range(1, 49)]
      + [f"temp_l{lag}" for lag in range(-47, 0)],
    "target_base": "load",
}


def build_datasets():
    data = DataClass(path_train=HERE / "train.csv", path_test=HERE / "test.csv",
                     data_kwargs=DATA_KWARGS, folder_config=HERE)
    common = dict(graph_folder=GRAPH_FOLDER, dataset_kwargs=DATASET_KWARGS,
                  out_channels=48)
    train = GraphDataset(data=data, period="train", **common)
    validation = GraphDataset(
        data=data, period="val", scalers_feat=train.scalers_feat,
        scalers_target=train.scalers_target, **common)
    test = GraphDataset(
        data=data, period="test", scalers_feat=train.scalers_feat,
        scalers_target=train.scalers_target, **common)
    return train, validation, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    target_path = OUTPUT / "target.pt"
    prediction_paths = [
        OUTPUT / f"{name}__{graph}.pt"
        for graph in ("dtw", "eye")
        for name in CONVOLUTIONS
    ]
    retrain = args.force or not target_path.exists() or not all(
        path.exists() for path in prediction_paths
    )
    if retrain:
        missing_inputs = [
            path for path in (HERE / "train.csv", HERE / "test.csv")
            if not path.exists()
        ]
        if missing_inputs:
            missing = ", ".join(str(path) for path in missing_inputs)
            raise FileNotFoundError(
                f"Retraining requires the processed data files: {missing}"
            )
        set_device("cpu")
        train, validation, test = build_datasets()
    else:
        train = validation = test = None

    rows = []
    for graph in ("dtw", "eye"):
        if retrain:
            for dataset in (train, validation, test):
                dataset._set_adj_matrix(adj_matrix=graph)
        for name, convolution in CONVOLUTIONS.items():
            prediction_path = OUTPUT / f"{name}__{graph}.pt"
            if prediction_path.exists() and not args.force:
                prediction = torch.load(
                    prediction_path, weights_only=False, map_location="cpu")
                target = torch.load(target_path, weights_only=False,
                                    map_location="cpu")
            else:  # ``retrain`` is true if any cache is absent.
                seed_everything(args.seed)
                gc.collect()
                model = myGNN(
                    in_channels=train.num_node_features, num_layers=2,
                    hidden_channels=64, out_channels=48,
                    conv_class=convolution, conv_kwargs=None, heads=4)
                trainer = Trainer(
                    model=model, dataset_train=train, dataset_val=validation,
                    dataset_test=test, batch_size=32,
                    model_kwargs={"lr": 1e-3, "num_epochs": args.epochs},
                    reconcile=False, lam_reg=0,
                    saving_directory=CHECKPOINTS / f"{name}_{graph}")
                result = trainer.train(force_training=args.force, save=False,
                                       patience=30)
                prediction, regional_target = result[0], result[1]
                prediction = prediction.cpu()
                target = regional_target.sum(dim=0).cpu()
                torch.save(prediction, prediction_path)
                if not target_path.exists():
                    torch.save(target, target_path)
            score = rmse(prediction.sum(dim=0).numpy(), target.numpy())
            rows.append({"operator": name, "graph": graph, "seed": args.seed,
                         "rmse_mw": score})
            print(f"{name:<20} {graph:<4} RMSE={score:8.1f} MW", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    paired = summary.pivot(index="operator", columns="graph", values="rmse_mw")
    paired["gain_mw"] = paired["eye"] - paired["dtw"]
    paired.to_csv(OUTPUT / "paired_summary.csv")
    print(f"\nDTW mean {paired['dtw'].mean():.1f} MW; identity mean "
          f"{paired['eye'].mean():.1f} MW; "
          f"DTW improves {(paired['gain_mw'] > 0).sum()}/{len(paired)} operators")


if __name__ == "__main__":
    main()
