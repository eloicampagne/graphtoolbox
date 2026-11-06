"""Train (or reload) one row of the load sweep exactly as example.ipynb does.

The configuration is read from a myGNN_best.json; missing keys fall back to the
notebook's LOOP_PARAMS. Run with the working directory examples/load so that the
Trainer's ./checkpoints path is the notebook's.

Usage: python sweep_row.py CONV path/to/myGNN_best.json OUT.pt [--reconcile] [--force] [--dir CKPT_DIR] [--seed S]

--dir trains into CKPT_DIR (from scratch) instead of the notebook's ./checkpoints,
whose paths omit the learning rate and could hold weights of another run.
"""
import inspect
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric import seed_everything
from torch_geometric.nn import conv as pyg_conv
from torch_geometric.nn import APPNP

from graphtoolbox.data import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer

conv_name, best_path, out_path = sys.argv[1:4]
reconcile = '--reconcile' in sys.argv
force = '--force' in sys.argv
seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 42
ckpt_dir = os.path.abspath(sys.argv[sys.argv.index('--dir') + 1]) if '--dir' in sys.argv else None
LOAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

best = json.load(open(best_path))['params']
params = {k: type(LOOP_PARAMS[k])(best.get(k, LOOP_PARAMS[k])) for k in LOOP_PARAMS}
params['num_epochs'], params['patience'] = LOOP_PARAMS['num_epochs'], LOOP_PARAMS['patience']
model_kw = {k: v for k, v in best.items() if k in GNN_CTOR and k not in SWEEP_LEVEL}
conv_kw = {k: v for k, v in best.items() if k not in GNN_CTOR and k not in SWEEP_LEVEL}
cls = APPNP if conv_name == 'APPNP' else getattr(pyg_conv, conv_name)
print(conv_name, params, model_kw, conv_kw, 'reconcile' if reconcile else 'unreconciled', flush=True)

data = DataClass(path_train='./train.csv', path_test='./test.csv', data_kwargs=data_kwargs, folder_config='.')
common = dict(graph_folder='../graph_representations', dataset_kwargs=dataset_kwargs, out_channels=48)
train = GraphDataset(data=data, period='train', **common)
val = GraphDataset(data=data, period='val', scalers_feat=train.scalers_feat,
                   scalers_target=train.scalers_target, **common)
test = GraphDataset(data=data, period='test', scalers_feat=train.scalers_feat,
                    scalers_target=train.scalers_target, **common)
for ds in (train, val, test):
    ds._set_adj_matrix(adj_matrix=params['adj_matrix'])

seed_everything(seed)
model = myGNN(in_channels=train.num_node_features, num_layers=params['num_layers'],
              hidden_channels=params['hidden_channels'], out_channels=48, conv_class=cls,
              conv_kwargs=conv_kw or None, heads=params['heads'], **model_kw)
trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                  batch_size=params['batch_size'],
                  model_kwargs={'lr': params['lr'], 'num_epochs': params['num_epochs']},
                  reconcile=reconcile, top_level_model='xgb', lam_reg=0)
train_kw = {'saving_directory': ckpt_dir, 'force_training': True} if ckpt_dir else {'force_training': force}
pred, target = trainer.train(patience=params['patience'], plot_loss=False, **train_kw)[:2]
pred = pred.detach().cpu()
torch.save(pred, out_path)
y = target.sum(dim=0).cpu().numpy()
p = pred.sum(dim=0).numpy()
print(f"{conv_name}: national RMSE {np.sqrt(np.mean((p - y) ** 2)):.1f} MW, "
      f"MAPE {100 * np.mean(np.abs(p - y) / np.abs(y)):.2f}%", flush=True)
