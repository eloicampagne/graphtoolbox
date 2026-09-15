"""AdditiveGraphModel + ALE feature importance on net-load (mirror of load cells 20/24).

Trains an AdditiveGraphModel whose contiguous feature groups tile the net-load
node-feature vector (NetLoad lags / temperature lags / exogenous / calendar
dummies), then computes per-feature ALE importances. Caches the national
prediction and writes the importance table to CSV.
"""
import os
import re
import gc
import argparse

import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import GCNConv

from graphtoolbox.models import AdditiveGraphModel
from graphtoolbox.training import Trainer, set_device
from graphtoolbox.interpretability import (
    aggregate_ale_importance,
    compute_feature_importances_from_ALE,
    plot_ale_group_importance,
)

import netload_pipeline as P

OUT_DIR = os.path.join(os.path.dirname(__file__), "results_additive_netload")
os.makedirs(OUT_DIR, exist_ok=True)
DUMMIES = ["tod", "day_type_week"]
AGM_ADJ = "dtw"


def _agm_kind(f):
    if re.fullmatch(r"NetLoad_l\d+", f):        return "netload_lags"
    if re.fullmatch(r"temperature_l-?\d+", f):  return "temp_lags"
    return "exogenous"


def contiguous_groups(features):
    runs = []
    for f in features:
        k = _agm_kind(f)
        if runs and runs[-1][0] == k:
            runs[-1][1].append(f)
        else:
            runs.append([k, [f]])
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    set_device("cpu")
    print("Building net-load datasets...")
    train, val, test, nodes = P.build_datasets()
    for ds in (train, val, test):
        ds._set_adj_matrix(adj_matrix=AGM_ADJ)

    runs = contiguous_groups(train.features)
    n_exo = sum(1 for k, _ in runs if k == "exogenous")
    groups, e = {}, 0
    for k, fs in runs:
        if k == "exogenous":
            e += 1
            name = "exogenous" if n_exo == 1 else ("exogenous" if e == 1 else "calendar_dummies")
        else:
            name = k
        groups[name] = groups.get(name, 0) + len(fs)
    assert sum(groups.values()) == train.num_node_features, "groups must tile all features"
    print("Feature groups (contiguous):", groups, "| total =", train.num_node_features)

    seed_everything(42); gc.collect()
    model = AdditiveGraphModel(feature_group_dims=groups, num_layers=2, hidden_channels=64,
                               out_channels=P.OUT_CHANNELS, conv_class=GCNConv, conv_kwargs={})
    trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                      batch_size=32, model_kwargs={"lr": 1e-3, "num_epochs": args.epochs},
                      reconcile=True, top_level_model="xgb", lam_reg=0)
    res = trainer.train(force_training=False, patience=30, save=True)
    pred, target = res[0], res[1]
    group_outputs = res[4] if len(res) >= 5 else None
    torch.save(pred, os.path.join(OUT_DIR, "AdditiveGraphModel.pt"))
    torch.save(target.sum(dim=0).cpu(), os.path.join(OUT_DIR, "target.pt"))
    m = P.metrics_dict(pred.sum(dim=0).cpu(), target.sum(dim=0).cpu())
    print(f"\nAdditiveGraphModel national RMSE={m['rmse']:.0f} MW  MAPE={m['mape']:.2f}%")

    # ALE feature importance.
    runs = contiguous_groups(train.features)
    n_exo = sum(1 for k, _ in runs if k == "exogenous")
    feature_groups, e = {}, 0
    for k, fs in runs:
        if k == "exogenous":
            e += 1
            nm = "exogenous" if n_exo == 1 else ("exogenous" if e == 1 else "calendar_dummies")
            feature_groups[nm] = DUMMIES if nm == "calendar_dummies" else fs
        else:
            feature_groups[k] = fs
    train.dataset_kwargs["feature_groups"] = feature_groups

    print("Computing ALE importances...")
    importance_ale = compute_feature_importances_from_ALE(
        group_outputs, train.data if hasattr(train, "data") else None, train, test,
        n_bins=20, mode="avg48", period=48, method="rms",
    ) if group_outputs is not None else None

    if importance_ale is not None:
        csv_path = os.path.join(OUT_DIR, "ale_importance.csv")
        importance_ale.to_csv(csv_path, index=False)
        grouped = aggregate_ale_importance(importance_ale)
        grouped.to_csv(os.path.join(OUT_DIR, "ale_group_importance.csv"), index=False)
        plot_ale_group_importance(
            importance_ale,
            output_path=os.path.join(OUT_DIR, "ale_group_importance.pdf"),
        )
        print("\nALE importance by feature group (%):")
        for row in grouped.itertuples(index=False):
            print(f"  {row.group:<18} {100 * row.share:6.1f}%")
        print(f"\nSaved -> {csv_path}")
    else:
        print("group_outputs unavailable; ALE skipped.")


if __name__ == "__main__":
    main()
