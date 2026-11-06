"""Non-graph baselines of the companion study (INFORMS) on the inputs of the load sweep.

The companion study fits a shared MLP, a per-node MLP and XGBoost (pooled over the nodes, b-global, and per
node, b-local) on the French regions with the 2015-2017 / 2018 / 2019 split of the sweep, but without the inputs
that the sweep adds (temperature of the forecast day, degree days, squared temperature, summer flag, region
indicators). This script reruns the same procedures (search spaces, number of trials, seeds; functions
run_mlps and run_xgb of nongraph_models.py) on the day-aligned samples of
sweep_row.py, so that the baselines and the graph models read the same inputs.

    python nongraph_baselines.py     # writes nongraph_baselines.npz and nongraph_baselines.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(LOAD))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(LOAD)

from graphtoolbox.data import DataClass, GraphDataset  # noqa: E402

# Inputs and split of sweep_row.py.
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
data = DataClass(path_train='./train.csv', path_test='./test.csv', data_kwargs=data_kwargs, folder_config='.')
common = dict(graph_folder='../graph_representations', dataset_kwargs=dataset_kwargs, out_channels=48)
train = GraphDataset(data=data, period='train', **common)
val = GraphDataset(data=data, period='val', scalers_feat=train.scalers_feat,
                   scalers_target=train.scalers_target, **common)
test = GraphDataset(data=data, period='test', scalers_feat=train.scalers_feat,
                    scalers_target=train.scalers_target, **common)


def stack(ds):
    """Scaled features and raw targets of every day, as in the snapshots of the companion benchmark."""
    X = np.stack([d.x.numpy() for d in ds.pyg_data])
    Y = np.stack([d.y.numpy() for d in ds.pyg_data])
    return X.astype(np.float32), Y.astype(np.float64)


tr, va, te = stack(train), stack(val), stack(test)
print('samples', len(tr[0]), len(va[0]), len(te[0]), 'features', tr[0].shape[2], flush=True)

sweep = np.load(os.path.join(HERE, 'load_sweep_v030.npz'))
y = sweep['target']
assert np.max(np.abs(te[1].sum(1).reshape(-1) - y)) < 1e-3, 'test target differs from the sweep'

# The companion code changes the working directory on import; outputs use absolute paths.
import nongraph_models as B  # noqa: E402

out = B.run_mlps(tr, va, te) | B.run_xgb(tr, va, te)
rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
mape = lambda p: float(100 * np.mean(np.abs(p - y) / np.abs(y)))
res, arrays = {}, {}
for name, o in out.items():
    nat = np.stack([B.national(p) for p in o['preds']])            # seeds, T
    r = [rmse(p) for p in nat]
    res[name] = {'cfg': o['cfg'], 'rmse': r, 'rmse_mean': float(np.mean(r)),
                 'rmse_sd': float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
                 'mape_mean': float(np.mean([mape(p) for p in nat])), 'rmse_seed_mean_forecast': rmse(nat.mean(0))}
    arrays[name] = nat
    print(f"{name}: RMSE {res[name]['rmse_mean']:.0f} ± {res[name]['rmse_sd']:.0f} MW, "
          f"MAPE {res[name]['mape_mean']:.2f}%, seed-mean forecast {res[name]['rmse_seed_mean_forecast']:.0f} MW",
          flush=True)
json.dump(res, open(os.path.join(HERE, 'nongraph_baselines.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(HERE, 'nongraph_baselines.npz'), **arrays)
