"""Temporal graph cells on net-load (mirror of examples/load cell 34).

Trains the four PyG-Temporal recurrent cells (GConvGRU, DCRNN, TGCN, A3TGCN) in
their SOTA regime: the 48 ``NetLoad_l*`` lags are read as a genuine 48-step
sequence (ConvAdapterTemporal / TemporalGNN.from_features with target='NetLoad'),
every other column kept as static context. Predictions are cached to
``results_temporal_netload/{cell}.pt`` so the notebook can reload them and merge
them into the MLpol ensemble.

On the 12-node graph CPU beats MPS by ~5x for recurrent cells, so we force CPU.
"""
import os
import gc

import torch
from torch_geometric import seed_everything

from graphtoolbox.models import TemporalGNN
from graphtoolbox.training import set_device, Trainer

import netload_pipeline as P

set_device("cpu")

TEMPORAL_HIDDEN = 128
NUM_EPOCHS = 100
PATIENCE = 30
BATCH_SIZE = 16
OUT_DIR = os.path.join(os.path.dirname(__file__), "results_temporal_netload")
os.makedirs(OUT_DIR, exist_ok=True)


def temporal_cells():
    from torch_geometric_temporal.nn.recurrent import GConvGRU, DCRNN, TGCN, A3TGCN
    return {
        "GConvGRU": (GConvGRU, {"K": 2}),
        "DCRNN":    (DCRNN,    {"K": 2}),
        "TGCN":     (TGCN,     {}),
        "A3TGCN":   (A3TGCN,   {}),   # periods auto-set to the lag length
    }


def main():
    print("Building net-load datasets...")
    train, val, test, nodes = P.build_datasets()
    for ds in (train, val, test):
        ds._set_adj_matrix(adj_matrix=P.ADJ)

    target_national = None
    _target_path = os.path.join(OUT_DIR, "target.pt")

    results = {}
    for name, (cell, ckw) in temporal_cells().items():
        pred_path = os.path.join(OUT_DIR, f"{name}.pt")
        if os.path.exists(pred_path):
            pred = torch.load(pred_path, weights_only=False)
            results[name] = pred
            if target_national is None and os.path.exists(_target_path):
                target_national = torch.load(_target_path, weights_only=False)
            print(f"[cache] {name:<10} loaded")
            continue

        print(f"\nTraining {name} (temporal)...")
        seed_everything(42)
        gc.collect()
        try:
            model = TemporalGNN.from_features(
                train.features, cell_class=cell, hidden_channels=TEMPORAL_HIDDEN,
                out_channels=P.OUT_CHANNELS, target="NetLoad", cell_kwargs=ckw or None)
            trainer = Trainer(model=model, dataset_train=train, dataset_val=val,
                              dataset_test=test, batch_size=BATCH_SIZE,
                              model_kwargs={"lr": 1e-3, "num_epochs": NUM_EPOCHS},
                              reconcile=True, top_level_model="xgb", lam_reg=0)
            result = trainer.train(plot_loss=False, force_training=False, save=True,
                                   patience=PATIENCE)
            pred, target = result[0], result[1]
            torch.save(pred, pred_path)
            results[name] = pred
            if target_national is None:
                target_national = target.sum(dim=0).cpu()
                torch.save(target_national, _target_path)
            m = P.metrics_dict(pred.sum(dim=0).cpu(), target_national)
            print(f"  RMSE={m['rmse']:.0f} MW  MAPE={m['mape']:.2f}%")
        except Exception as exc:
            print(f"  Failed: {str(exc).splitlines()[0] if str(exc).strip() else type(exc).__name__}")
        gc.collect()

    if results and target_national is not None:
        print(f'\n{"Model":<12} {"RMSE":>10} {"MAPE":>8}')
        print("-" * 32)
        ranked = sorted(results.items(),
                        key=lambda kv: P.metrics_dict(kv[1].sum(dim=0).cpu(), target_national)["rmse"])
        for name, pred in ranked:
            m = P.metrics_dict(pred.sum(dim=0).cpu(), target_national)
            print(f"{name:<12} {m['rmse']:>8.0f} MW {m['mape']:>6.2f}%")
    print("\nDone. Cached ->", OUT_DIR)


if __name__ == "__main__":
    main()
