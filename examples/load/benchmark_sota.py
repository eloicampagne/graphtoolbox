"""Benchmark static and recurrent graph models on regional load forecasts."""
import argparse
import csv
import gc
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric import seed_everything
from torch_geometric.nn.conv import (
    GCNConv, ChebConv, GATConv, GATv2Conv, TransformerConv, APPNP, TAGConv,
    SAGEConv, LEConv,
)

from graphtoolbox.data.dataset import DataClass, GraphDataset
from graphtoolbox.models.gnn import myGNN
from graphtoolbox.models.temporal import TemporalGNN
from graphtoolbox.training.trainer import Trainer

HERE = Path(__file__).resolve().parent
OUT_CHANNELS = 48
HIDDEN = 364           # aligned with example.ipynb (config.py)
NUM_LAYERS = 3
ADJ = "dtw"            # graph used in the notebook


# Data (mirrors example.ipynb)
def build_datasets():
    data_kwargs = {
        "node_var": "Region",
        "features_to_lag": {"load": (1, 48), "temp": (-47, -1)},
        "dummies": ["Instant", "RegionInt", "JourSemaine", "DayType", "offset"],
        "day_inf_train": "2015-01-01", "day_sup_train": "2018-01-01",
        "day_inf_val": "2018-01-01", "day_sup_val": "2018-12-31",
        "day_inf_test": "2019-01-01", "day_sup_test": "2019-12-31",
    }
    dataset_kwargs = {
        "batch_size": 32,
        "adj_matrix": ADJ,
        "features_base": (
            ["temp", "nebu", "wind", "tempMax", "tempMin", "Posan", "Instant",
             "RegionInt", "JourSemaine", "JourFerie", "offset", "DayType",
             "Weekend", "temp_liss_fort", "temp_liss_faible"]
            + [f"load_l{t}" for t in range(1, 49)]
            + [f"temp_l{t}" for t in range(-47, 0)]
        ),
        "target_base": "load",
    }
    data = DataClass(path_train=str(HERE / "train.csv"),
                     path_test=str(HERE / "test.csv"),
                     data_kwargs=data_kwargs, folder_config=str(HERE))
    common = dict(graph_folder=str(HERE.parent / "graph_representations"),
                  dataset_kwargs=dataset_kwargs, out_channels=OUT_CHANNELS)
    train = GraphDataset(data=data, period="train", **common)
    val = GraphDataset(data=data, period="val", scalers_feat=train.scalers_feat,
                       scalers_target=train.scalers_target, **common)
    test = GraphDataset(data=data, period="test", scalers_feat=train.scalers_feat,
                        scalers_target=train.scalers_target, **common)
    nodes = [str(n) for n in data.nodes]
    return train, val, test, nodes


# Metrics: national (aggregated) + per region
def _m(p, t):
    p, t = np.asarray(p, float).ravel(), np.asarray(t, float).ravel()
    mask = np.abs(t) > 1e-6
    return {"rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "mae": float(np.mean(np.abs(p - t))),
            "mape": float(np.mean(np.abs((p[mask] - t[mask]) / t[mask])) * 100)}


def _national_ci(p_nat, t_nat, n_boot=2000, block_len=48, level=0.95):
    """Moving-block bootstrap CI and standard error for the national RMSE and MAPE.

    Uses the GraphToolbox significance module so the interval respects the
    temporal dependence of the errors (block = 48 half-hours = one day).
    """
    try:
        from graphtoolbox.evaluation import bootstrap_metric
    except Exception:
        return {}
    out = {}
    for metric in ("rmse", "mape"):
        r = bootstrap_metric(p_nat, t_nat, metric=metric, n_boot=n_boot,
                             block_len=block_len, level=level, seed=0)
        out[f"{metric}_lo"] = r.ci_low
        out[f"{metric}_hi"] = r.ci_high
        out[f"{metric}_se"] = r.se
    return out


def metrics(pred, target, nodes, with_ci=True):
    pred = np.asarray(pred, float)      # [num_nodes, T]
    target = np.asarray(target, float)
    if pred.shape[0] != len(nodes) and pred.shape[-1] == len(nodes):
        pred, target = pred.T, target.T
    p_nat, t_nat = pred.sum(0), target.sum(0)
    national = _m(p_nat, t_nat)
    if with_ci:
        national.update(_national_ci(p_nat, t_nat))
    regional = {nodes[i]: _m(pred[i], target[i]) for i in range(pred.shape[0])}
    return national, regional, p_nat, t_nat


# Train + evaluate through the GraphToolbox harness
def _train_eval(model, train, val, test, epochs, nodes):
    for ds in (train, val, test):
        ds._set_adj_matrix(adj_matrix=ADJ)
    trainer = Trainer(model=model, dataset_train=train, dataset_val=val,
                      dataset_test=test, batch_size=16,
                      model_kwargs={"lr": 1e-3, "num_epochs": epochs},
                      reconcile=True, top_level_model="xgb", lam_reg=0)
    pred, target, *_ = trainer.train(plot_loss=False, force_training=True,
                                     save=False, patience=max(20, epochs // 4))
    return metrics(pred, target, nodes)


def run_static(name, conv_class, conv_kwargs, train, val, test, epochs, nodes):
    seed_everything(42); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = myGNN(in_channels=train.num_node_features, num_layers=NUM_LAYERS,
                  hidden_channels=HIDDEN, out_channels=OUT_CHANNELS,
                  conv_class=conv_class, conv_kwargs=conv_kwargs or {})
    return _train_eval(model, train, val, test, epochs, nodes)


def run_temporal_sota(name, cell_class, cell_kwargs, train, val, test,
                      epochs, nodes, hidden=128):
    seed_everything(42); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = TemporalGNN.from_features(
        train.features, cell_class=cell_class, hidden_channels=hidden,
        out_channels=OUT_CHANNELS, cell_kwargs=cell_kwargs)
    return _train_eval(model, train, val, test, epochs, nodes)


STATIC_MODELS = {
    "GCN (Kipf+17)": (GCNConv, {}),
    "ChebConv (Defferrard+16)": (ChebConv, {"K": 3}),
    "TAGConv (Du+17)": (TAGConv, {"K": 3}),
    "APPNP (Gasteiger+18)": (APPNP, {"K": 2, "alpha": 0.9}),
    "GraphSAGE (Hamilton+17)": (SAGEConv, {}),
    "GAT (Velickovic+18)": (GATConv, {"heads": 2}),
    "GATv2 (Brody+21)": (GATv2Conv, {"heads": 2}),
    "TransformerConv (Shi+20)": (TransformerConv, {"heads": 2}),
    "LEConv": (LEConv, {}),
}


def temporal_cells():
    try:
        from torch_geometric_temporal.nn.recurrent import GConvGRU, DCRNN, TGCN, A3TGCN
    except Exception:
        warnings.warn("torch_geometric_temporal not installed; temporal baselines "
                      "skipped. `pip install torch-geometric-temporal`.")
        return {}
    # name -> (fair conv class factory kwargs, sota cell class, sota cell kwargs, windowed)
    return {
        "GConvGRU (Seo+18)": (GConvGRU, {"K": 2}, False),
        "DCRNN (Li+18)": (DCRNN, {"K": 2}, False),
        "TGCN (Zhao+19)": (TGCN, {}, False),
        "A3TGCN (Bai+20)": (A3TGCN, {"periods": 48}, True),
    }


def fair_temporal_conv(cell_class, cell_kwargs, windowed):
    """Wrap a temporal cell as a one-shot (x, edge_index) conv for `fair` mode."""
    class _Cell(nn.Module):
        def __init__(self, in_channels, out_channels, **kw):
            super().__init__()
            self.cell = cell_class(in_channels, out_channels,
                                   **({"periods": 1} if windowed else cell_kwargs))

        def forward(self, x, edge_index, edge_weight=None, **kw):
            z = x.unsqueeze(-1) if windowed else x
            h = self.cell(z, edge_index, edge_weight)
            return h[0] if isinstance(h, tuple) else h
    return _Cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--mode", choices=["fair", "sota", "both"], default="both")
    ap.add_argument("--no-static", action="store_true", help="skip static models")
    args = ap.parse_args()

    from graphtoolbox.training.trainer import DEVICE
    print(f"Compute device (GraphToolbox Trainer): {DEVICE}  "
          f"[cuda={torch.cuda.is_available()}, mps={torch.backends.mps.is_available()}]")

    train, val, test, nodes = build_datasets()
    nat_rows, reg_rows = [], []
    national_series = {}      # "mode:model" -> national forecast [T]
    national_target = {}      # single national ground-truth series

    def record(name, mode, nat, reg, p_nat=None, t_nat=None):
        ci = (f"  95% CI=[{nat['rmse_lo']:.0f}, {nat['rmse_hi']:.0f}]"
              if "rmse_lo" in nat else "")
        print(f"  [{mode:4s}] {name:26s}  national RMSE={nat['rmse']:8.1f}{ci}  "
              f"MAPE={nat['mape']:5.2f}%")
        nat_rows.append({"model": name, "mode": mode, **nat})
        for region, rm in reg.items():
            reg_rows.append({"model": name, "mode": mode, "region": region, **rm})
        if p_nat is not None:
            national_series[f"{mode}:{name}"] = np.asarray(p_nat, float)
            if t_nat is not None and "target" not in national_target:
                national_target["target"] = np.asarray(t_nat, float)

    if not args.no_static:
        print("Static graph models (one-shot):")
        for name, (cls, kw) in STATIC_MODELS.items():
            try:
                nat, reg, p_nat, t_nat = run_static(name, cls, kw, train, val, test, args.epochs, nodes)
                record(name, "fair", nat, reg, p_nat, t_nat)
            except Exception as exc:
                print(f"  [fair] {name:26s}  FAILED: {exc}")

    cells = temporal_cells()
    if args.mode in ("fair", "both"):
        print("Temporal models, one-shot (equal footing):")
        for name, (cell, kw, windowed) in cells.items():
            try:
                conv = fair_temporal_conv(cell, kw, windowed)
                nat, reg, p_nat, t_nat = run_static(name, conv, {}, train, val, test, args.epochs, nodes)
                record(name, "fair", nat, reg, p_nat, t_nat)
            except Exception as exc:
                print(f"  [fair] {name:26s}  FAILED: {exc}")
    if args.mode in ("sota", "both"):
        print("Temporal models, SOTA regime (recurrent over the lag window):")
        for name, (cell, kw, windowed) in cells.items():
            try:
                nat, reg, p_nat, t_nat = run_temporal_sota(name, cell, kw, train, val,
                                                           test, args.epochs, nodes)
                record(name, "sota", nat, reg, p_nat, t_nat)
            except Exception as exc:
                print(f"  [sota] {name:26s}  FAILED: {exc}")

    if nat_rows:
        nat_rows.sort(key=lambda r: r["rmse"])
        _nat_fields = ["model", "mode", "rmse", "rmse_lo", "rmse_hi", "rmse_se",
                       "mae", "mape", "mape_lo", "mape_hi", "mape_se"]
        with open(HERE / "benchmark_sota.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_nat_fields, extrasaction="ignore")
            w.writeheader(); w.writerows(nat_rows)
        with open(HERE / "benchmark_sota_regional.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "mode", "region", "rmse", "mae", "mape"])
            w.writeheader(); w.writerows(reg_rows)
        if national_series:
            np.savez(HERE / "benchmark_sota_national.npz",
                     target=national_target.get("target"),
                     **{k.replace(":", "__"): v for k, v in national_series.items()})
        print("\nNational ranking (with 95% CIs) -> benchmark_sota.csv ; regional -> "
              "benchmark_sota_regional.csv ; national forecast series -> "
              "benchmark_sota_national.npz")


if __name__ == "__main__":
    main()
