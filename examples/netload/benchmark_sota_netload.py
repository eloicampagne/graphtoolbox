"""Benchmark of GraphToolbox GNN models against SOTA spatio-temporal GNNs, on net-load.

Net-load counterpart of ``examples/load/benchmark_sota.py``. Same DTW graph,
same reconciliation, same fixed architecture across baselines (the net-load
direct model: hidden=256, 2 layers). Net-load crosses zero, so MAPE is masked
away from zero and RMSE / nRMSE are the headline.

Two modes (``--mode``): ``fair`` (every cell does one spatio-temporal update on
the lag-augmented features), ``sota`` (recurrent cells read the 48 NetLoad lags
as a sequence), ``both``. Per-model results are cached to
``benchmark_sota_netload.csv`` so reruns resume.

Run from examples/netload/:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python benchmark_sota_netload.py --mode both --epochs 500
"""
import argparse
import csv
import gc
import warnings

import numpy as np
import torch
from torch import nn
from torch_geometric import seed_everything
from torch_geometric.nn.conv import (GCNConv, ChebConv, GATConv, GATv2Conv,
                                     TransformerConv, APPNP, TAGConv, SAGEConv, LEConv)

from graphtoolbox.models.gnn import myGNN
from graphtoolbox.models.temporal import TemporalGNN
from graphtoolbox.training.trainer import Trainer

import netload_pipeline as P

HERE = P.HERE
OUT_CHANNELS = P.OUT_CHANNELS
HIDDEN = 256
NUM_LAYERS = 2
ADJ = P.ADJ
CSV_PATH = HERE / "benchmark_sota_netload.csv"
REG_CSV_PATH = HERE / "benchmark_sota_netload_regional.csv"
NPZ_PATH = HERE / "benchmark_sota_netload_national.npz"


def _national_ci(p_nat, t_nat, n_boot=2000, block_len=48, level=0.95):
    try:
        from graphtoolbox.evaluation import bootstrap_metric
    except Exception:
        return {}
    out = {}
    for metric in ("rmse", "mape"):
        r = bootstrap_metric(p_nat, t_nat, metric=metric, n_boot=n_boot,
                             block_len=block_len, level=level, seed=0)
        out[f"{metric}_lo"], out[f"{metric}_hi"], out[f"{metric}_se"] = r.ci_low, r.ci_high, r.se
    return out


def metrics(pred, target, nodes, with_ci=True):
    pred = np.asarray(pred, float)
    target = np.asarray(target, float)
    if pred.shape[0] != len(nodes) and pred.shape[-1] == len(nodes):
        pred, target = pred.T, target.T
    p_nat, t_nat = pred.sum(0), target.sum(0)
    national = P.metrics_dict(p_nat, t_nat)
    if with_ci:
        national.update(_national_ci(p_nat, t_nat))
    regional = {nodes[i]: P.metrics_dict(pred[i], target[i]) for i in range(pred.shape[0])}
    return national, regional, p_nat, t_nat


def _train_eval(model, train, val, test, epochs, nodes):
    for ds in (train, val, test):
        ds._set_adj_matrix(adj_matrix=ADJ)
    trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                      batch_size=16, model_kwargs={"lr": 1e-3, "num_epochs": epochs},
                      reconcile=True, top_level_model="xgb", lam_reg=0)
    pred, target, *_ = trainer.train(plot_loss=False, force_training=True, save=False,
                                     patience=max(20, epochs // 4))
    return metrics(pred, target, nodes)


def run_static(name, conv_class, conv_kwargs, train, val, test, epochs, nodes):
    seed_everything(42); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = myGNN(in_channels=train.num_node_features, num_layers=NUM_LAYERS,
                  hidden_channels=HIDDEN, out_channels=OUT_CHANNELS,
                  conv_class=conv_class, conv_kwargs=conv_kwargs or {})
    return _train_eval(model, train, val, test, epochs, nodes)


def run_temporal_sota(name, cell_class, cell_kwargs, train, val, test, epochs, nodes, hidden=128):
    seed_everything(42); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = TemporalGNN.from_features(train.features, cell_class=cell_class,
                                      hidden_channels=hidden, out_channels=OUT_CHANNELS,
                                      target="NetLoad", cell_kwargs=cell_kwargs)
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
        warnings.warn("torch_geometric_temporal not installed; temporal baselines skipped.")
        return {}
    return {
        "GConvGRU (Seo+18)": (GConvGRU, {"K": 2}, False),
        "DCRNN (Li+18)": (DCRNN, {"K": 2}, False),
        "TGCN (Zhao+19)": (TGCN, {}, False),
        "A3TGCN (Bai+20)": (A3TGCN, {"periods": 48}, True),
    }


def fair_temporal_conv(cell_class, cell_kwargs, windowed):
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


def _load_done():
    done = set()
    rows = []
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            for r in csv.DictReader(f):
                done.add((r["model"], r["mode"]))
                rows.append(r)
    return done, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--mode", choices=["fair", "sota", "both"], default="both")
    ap.add_argument("--no-static", action="store_true")
    args = ap.parse_args()

    from graphtoolbox.training.trainer import DEVICE
    print(f"Compute device: {DEVICE}  [cuda={torch.cuda.is_available()}, mps={torch.backends.mps.is_available()}]")

    train, val, test, nodes = P.build_datasets()
    done, nat_rows = _load_done()
    reg_rows, national_series, national_target = [], {}, {}

    _nat_fields = ["model", "mode", "rmse", "rmse_lo", "rmse_hi", "rmse_se",
                   "mae", "mape", "mape_lo", "mape_hi", "mape_se"]

    def _flush():
        nat_rows.sort(key=lambda r: float(r["rmse"]))
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_nat_fields, extrasaction="ignore")
            w.writeheader(); w.writerows(nat_rows)

    def record(name, mode, nat, reg, p_nat=None, t_nat=None):
        ci = f"  95% CI=[{nat['rmse_lo']:.0f}, {nat['rmse_hi']:.0f}]" if "rmse_lo" in nat else ""
        print(f"  [{mode:4s}] {name:26s}  national RMSE={nat['rmse']:8.1f}{ci}  MAPE={nat['mape']:5.2f}%")
        nat_rows.append({"model": name, "mode": mode, **nat})
        for region, rm in reg.items():
            reg_rows.append({"model": name, "mode": mode, "region": region, **rm})
        if p_nat is not None:
            national_series[f"{mode}:{name}"] = np.asarray(p_nat, float)
            if t_nat is not None and "target" not in national_target:
                national_target["target"] = np.asarray(t_nat, float)
        _flush()

    def _todo(name, mode):
        if (name, mode) in done:
            print(f"  [{mode:4s}] {name:26s}  cached, skipped")
            return False
        return True

    if not args.no_static:
        print("Static graph models (one-shot):")
        for name, (cls, kw) in STATIC_MODELS.items():
            if not _todo(name, "fair"):
                continue
            try:
                nat, reg, p_nat, t_nat = run_static(name, cls, kw, train, val, test, args.epochs, nodes)
                record(name, "fair", nat, reg, p_nat, t_nat)
            except Exception as exc:
                print(f"  [fair] {name:26s}  FAILED: {exc}")

    cells = temporal_cells()
    if args.mode in ("fair", "both"):
        print("Temporal models, one-shot (equal footing):")
        for name, (cell, kw, windowed) in cells.items():
            if not _todo(name, "fair"):
                continue
            try:
                conv = fair_temporal_conv(cell, kw, windowed)
                nat, reg, p_nat, t_nat = run_static(name, conv, {}, train, val, test, args.epochs, nodes)
                record(name, "fair", nat, reg, p_nat, t_nat)
            except Exception as exc:
                print(f"  [fair] {name:26s}  FAILED: {exc}")
    if args.mode in ("sota", "both"):
        print("Temporal models, SOTA regime (recurrent over the lag window):")
        for name, (cell, kw, windowed) in cells.items():
            if not _todo(name, "sota"):
                continue
            try:
                nat, reg, p_nat, t_nat = run_temporal_sota(name, cell, kw, train, val, test, args.epochs, nodes)
                record(name, "sota", nat, reg, p_nat, t_nat)
            except Exception as exc:
                print(f"  [sota] {name:26s}  FAILED: {exc}")

    if reg_rows:
        with open(REG_CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "mode", "region", "rmse", "mae", "mape"])
            w.writeheader(); w.writerows(reg_rows)
    if national_series:
        np.savez(NPZ_PATH, target=national_target.get("target"),
                 **{k.replace(":", "__"): v for k, v in national_series.items()})
    print(f"\nNational ranking -> {CSV_PATH.name} ; regional -> {REG_CSV_PATH.name} ; series -> {NPZ_PATH.name}")


if __name__ == "__main__":
    main()
