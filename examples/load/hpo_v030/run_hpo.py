"""Rerun, under GraphToolbox 0.3.0, the hyperparameter searches of the load sweep.

The searches behind ../results_optim_{LEConv,GATConv,APPNP} ran under 0.1.0-0.2.0,
whose Optimizer trained and validated each epoch on a single batch. This script
repeats them with the search space of example.ipynb (200 trials, at most 100
epochs, MedianPruner) on full epochs, seeded, and writes its results under
./results_optim_<conv>/ so the original files stay untouched.

The candidate graphs are those that existed when the original searches ran
(./graphs links to them; the later 'usefulness' graph is excluded).

Usage: python run_hpo.py LEConv   (run from this folder)
"""
import os
import sys
import time

import pandas as pd
import torch
from torch_geometric.nn.conv import LEConv, GATConv
from torch_geometric.nn import APPNP

from graphtoolbox.data import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.optim import Optimizer
from graphtoolbox.training import set_device

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD = os.path.dirname(HERE)
os.chdir(HERE)
set_device(os.environ.get("HPO_DEVICE", "cpu"))
torch.set_num_threads(int(os.environ.get("HPO_THREADS", 4)))

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
    'batch_size': 32,
    'adj_matrix': 'dtw',
    'features_base': ['temp', 'nebu', 'wind', 'tempMax', 'tempMin', 'Posan', 'Instant', 'RegionInt',
                      'JourSemaine', 'JourFerie', 'offset', 'DayType', 'Weekend', 'temp_liss_fort',
                      'temp_liss_faible', 'heat_deg', 'cool_deg', 'temp_sq', 'is_summer']
                     + [f'load_l{t}' for t in range(1, 49)] + [f'temp_l{t}' for t in range(-47, 0)],
    'target_base': 'load',
}

optim_base = {'num_layers': (1, 5), 'hidden_channels': (32, 512), 'lr': (1e-5, 1e-1)}
SPACES = {
    'LEConv': (LEConv, optim_base),
    'GATConv': (GATConv, optim_base | {'heads': (1, 5)}),
    'APPNP': (APPNP, optim_base | {'K': (1, 10)} | {'alpha': (0.5, 1.0)}),
}

conv_name = sys.argv[1]
conv_class, space = SPACES[conv_name]

data = DataClass(path_train=os.path.join(LOAD, 'train.csv'), path_test=os.path.join(LOAD, 'test.csv'),
                 data_kwargs=data_kwargs, folder_config=LOAD)
graphs = os.path.join(HERE, 'graphs')
train = GraphDataset(data=data, period='train', graph_folder=graphs,
                     dataset_kwargs=dataset_kwargs, out_channels=48)
val = GraphDataset(data=data, period='val', scalers_feat=train.scalers_feat,
                   scalers_target=train.scalers_target, graph_folder=graphs,
                   dataset_kwargs=dataset_kwargs, out_channels=48)

optimizer = Optimizer(model=myGNN, dataset_train=train, dataset_val=val, conv_class=conv_class,
                      num_epochs=int(os.environ.get('HPO_EPOCHS', 100)), optim_kwargs=space, seed=42)
t0 = time.perf_counter()
optimizer.optimize(n_trials=int(os.environ.get('HPO_TRIALS', 200)), timeout=None)
print(f"{conv_name}: {time.perf_counter() - t0:.0f} s")
