"""Reconcile every operator of the load sweep with MinT, as in the paper's reconciliation paragraph.

Each operator is reloaded from its checkpoint (never retrained) with
reconcile=True and the gradient-boosting top-level model; its unreconciled
forecast is recomputed from the same weights and compared with the committed
archive, which checks that the checkpoint is the one behind the paper's row.
The three operators re-searched under 0.3.0 are read from ./checkpoints/<conv>, and
the same operators at the common configuration from ./checkpoints/<conv>_default;
their old configurations are also reconciled so that the paper's numbers can be
checked. Writes ./mint_all.npz and prints one line per operator.
"""
import inspect
import json
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric import seed_everything
from torch_geometric.nn import APPNP
from torch_geometric.nn import conv as pyg_conv

from graphtoolbox.data import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD = os.path.dirname(HERE)
os.chdir(LOAD)
data_kwargs = {
    'node_var': 'Region',
    'features_to_lag': {'load': (1, 48), 'temp': (-47, -1)},
    'dummies': ['Instant', 'RegionInt', 'JourSemaine', 'DayType', 'offset'],
    'day_inf_train': '2015-01-01', 'day_sup_train': '2018-01-01',
    'day_inf_val': '2018-01-01', 'day_sup_val': '2018-12-31',
    'day_inf_test': '2019-01-01', 'day_sup_test': '2019-12-31',
    'computed_features': {
        'heat_deg':  lambda df: (15 - df['temp']).clip(lower=0),
        'cool_deg':  lambda df: (df['temp'] - 15).clip(lower=0),
        'temp_sq':   lambda df: df['temp'] ** 2,
        'is_summer': lambda df: (pd.to_datetime(df['date']).dt.month == 7).astype(float)
                                + (pd.to_datetime(df['date']).dt.month == 8).astype(float),
    },
}
dataset_kwargs = {
    'batch_size': 32, 'adj_matrix': 'dtw',
    'features_base': ['temp', 'nebu', 'wind', 'tempMax', 'tempMin', 'Posan', 'Instant', 'RegionInt',
                      'JourSemaine', 'JourFerie', 'offset', 'DayType', 'Weekend', 'temp_liss_fort',
                      'temp_liss_faible', 'heat_deg', 'cool_deg', 'temp_sq', 'is_summer']
                     + [f'load_l{t}' for t in range(1, 49)] + [f'temp_l{t}' for t in range(-47, 0)],
    'target_base': 'load',
}
LOOP_PARAMS = {'num_layers': 2, 'hidden_channels': 64, 'lr': 1e-3, 'batch_size': 32,
               'adj_matrix': 'dtw', 'heads': 4, 'num_epochs': 300, 'patience': 30}
SWEEP_LEVEL = {'num_layers', 'hidden_channels', 'lr', 'batch_size', 'adj_matrix', 'heads'}
GNN_CTOR = {p.name for p in inspect.signature(myGNN.__init__).parameters.values()
            if p.name not in ('self', 'in_channels', 'out_channels', 'conv_class', 'conv_kwargs')
            and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}

with np.load(os.path.join(LOAD, "forecasts_paper", "load_sweep.npz")) as z:
    archive = {k: z[k] for k in z.files}
target = archive["target"]
ops = sorted(k for k in archive if k not in {"target", "target_nodes"} and not k.startswith("aggregate__"))
NEW = {"LEConv", "GATConv", "APPNP"}
jobs = [(op, os.path.join(HERE, "old_configs", f"{op}.json") if op in NEW else None, None, "old" if op in NEW else "same")
        for op in ops]
jobs += [(op, os.path.join(HERE, f"results_optim_{op}", "myGNN_best.json"),
          os.path.join(HERE, "checkpoints", op), "new") for op in sorted(NEW)]
# The same three operators at the common configuration of the sweep (sweep_row.py --dir checkpoints/<op>_default).
jobs += [(op, os.path.join(HERE, "old_configs", "default.json"), os.path.join(HERE, "checkpoints", f"{op}_default"),
          "default") for op in sorted(NEW)]

data = DataClass(path_train='./train.csv', path_test='./test.csv', data_kwargs=data_kwargs, folder_config='.')
common = dict(graph_folder='../graph_representations', dataset_kwargs=dataset_kwargs, out_channels=48)
train = GraphDataset(data=data, period='train', **common)
val = GraphDataset(data=data, period='val', scalers_feat=train.scalers_feat, scalers_target=train.scalers_target, **common)
test = GraphDataset(data=data, period='test', scalers_feat=train.scalers_feat, scalers_target=train.scalers_target, **common)


def rmse(p):
    return float(np.sqrt(np.mean((p.sum(0) - target) ** 2)))


out, report = {}, {}
ONLY = set(os.environ.get("MINT_ONLY", "").split(",")) - {""}
for op, best_path, ckpt, tag in jobs:
    if ONLY and op not in ONLY:
        continue
    best = json.load(open(best_path))['params'] if best_path else {}
    params = {k: type(LOOP_PARAMS[k])(best.get(k, LOOP_PARAMS[k])) for k in LOOP_PARAMS}
    model_kw = {k: v for k, v in best.items() if k in GNN_CTOR and k not in SWEEP_LEVEL}
    conv_kw = {k: v for k, v in best.items() if k not in GNN_CTOR and k not in SWEEP_LEVEL}
    cls = APPNP if op == 'APPNP' else getattr(pyg_conv, op)
    for ds in (train, val, test):
        ds._set_adj_matrix(adj_matrix=params['adj_matrix'])
    seed_everything(42)
    model = myGNN(in_channels=train.num_node_features, num_layers=params['num_layers'],
                  hidden_channels=params['hidden_channels'], out_channels=48, conv_class=cls,
                  conv_kwargs=conv_kw or None, heads=params['heads'], **model_kw)
    # The Trainer appends the heads (for attention convolutions) and the conv_kwargs to the
    # directory it is given; base is what it receives, ckpt is where the weights sit.
    base = ckpt or (f"./checkpoints/{cls.__name__}_{params['adj_matrix']}/batch{params['batch_size']}"
                    f"_hidden{model.hidden_channels}_layers{model.num_layers}_epochs300")
    ckpt = base
    try:
        if 'heads' in inspect.signature(cls).parameters:
            ckpt += f"_heads{getattr(model, 'heads', 0)}"
    except (ValueError, TypeError):
        pass
    ckpt += "".join(f"_{k}{v}" for k, v in (getattr(model, 'conv_kwargs', None) or {}).items())
    if not (os.path.isdir(ckpt) and os.listdir(ckpt)):
        print(f"{op:<20} {tag:<4} NO CHECKPOINT at {ckpt}", flush=True)
        continue
    trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                      batch_size=params['batch_size'],
                      model_kwargs={'lr': params['lr'], 'num_epochs': 300},
                      reconcile=True, top_level_model='xgb', lam_reg=0)
    rec = trainer.train(force_training=False, saving_directory=base, patience=30,
                        plot_loss=False)[0].detach().cpu().numpy()
    trainer.reconcile = False
    unrec = trainer._predict(trainer.test_loader, split='test')[1].detach().cpu().numpy()
    key = {"old": f"{op}__old", "default": f"{op}__default"}.get(tag, op)
    out[f"rec__{key}"], out[f"unrec__{key}"] = rec.astype(np.float32), unrec.astype(np.float32)
    ref = rmse(archive[op]) if tag in ("same", "old") else float("nan")
    report[key] = {"unrec": rmse(unrec), "rec": rmse(rec), "archive": ref}
    print(f"{op:<20} {tag:<4} unrec {rmse(unrec):7.1f}  archive {ref:7.1f}  rec {rmse(rec):7.1f}", flush=True)
    suffix = "_" + "_".join(sorted(ONLY)) if ONLY else ""
    np.savez_compressed(os.path.join(HERE, f"mint_all{suffix}.npz"), target=target, **out)
    json.dump(report, open(os.path.join(HERE, f"mint_all{suffix}.json"), "w"), indent=1)
