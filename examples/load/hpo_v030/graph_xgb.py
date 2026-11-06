"""Pooled XGBoost with neighbor aggregates of its inputs, French load.

Each region receives, besides its own 187 inputs (those of the graph models), the weighted mean over the other
regions of every continuous input (weather, its trajectory over the forecast day, degree days, load lags):
weights of the DTW graph of the sweep (diagonal removed, rows normalized), or uniform over the 11 other regions
as a control. The search (ten configurations scored on 2018) and the per-half-hour models are those of
benchmark_rfrance.run_xgb (b-global), with one seed, since the seeds of pooled XGBoost differ by about 1 MW.

    python graph_xgb.py     # writes graph_xgb.json and graph_xgb.npz
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(LOAD)))
os.chdir(LOAD)
from graphtoolbox.data import DataClass, GraphDataset  # noqa: E402
from graphtoolbox.evaluation.significance import diebold_mariano  # noqa: E402

H = 48
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
common = dict(graph_folder='../graph_representations', dataset_kwargs=dataset_kwargs, out_channels=H)
train = GraphDataset(data=data, period='train', **common)
val = GraphDataset(data=data, period='val', scalers_feat=train.scalers_feat,
                   scalers_target=train.scalers_target, **common)
test = GraphDataset(data=data, period='test', scalers_feat=train.scalers_feat,
                    scalers_target=train.scalers_target, **common)


def stack(ds):
    return (np.stack([d.x.numpy() for d in ds.pyg_data]).astype(np.float32),
            np.stack([d.y.numpy() for d in ds.pyg_data]).astype(np.float64))


sets = [stack(ds) for ds in (train, val, test)]
DUMMY = ('Instant_', 'RegionInt_', 'JourSemaine_', 'DayType_', 'offset_')
cont = [i for i, f in enumerate(train.features) if not f.startswith(DUMMY)]
N = train.num_nodes
A = np.loadtxt('../graph_representations/dtw/W.txt')[:, :N]
A = A - np.diag(np.diag(A))
graphs = {'dtw': A / A.sum(1, keepdims=True),
          'uniform': (np.ones((N, N)) - np.eye(N)) / (N - 1)}


def xgb_global(tr, va, te, seed=42):
    """b-global branch of benchmark_rfrance.run_xgb: same search space and trials."""
    import xgboost as xgb
    (Xtr, Ytr), (Xva, Yva), (Xte, _) = tr, va, te
    F = Xtr.shape[2]
    mu = Ytr.mean(axis=(0, 2), keepdims=True); sd = Ytr.std(axis=(0, 2), keepdims=True) + 1e-9
    Xs, Ys = Xtr.reshape(-1, F), ((Ytr - mu) / sd).reshape(-1, H)
    Xv, Yv = Xva.reshape(-1, F), ((Yva - mu) / sd).reshape(-1, H)
    space = lambda rng: {
        "learning_rate": float(rng.choice([0.03, 0.05, 0.1, 0.2])),
        "max_depth": int(rng.choice([4, 6, 8, 10])),
        "min_child_weight": float(rng.choice([1.0, 3.0, 6.0, 10.0])),
        "subsample": float(rng.choice([0.7, 0.85, 1.0])),
        "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
        "reg_alpha": float(rng.choice([0.0, 1e-4, 1e-3, 1e-2])),
        "reg_lambda": float(rng.choice([0.1, 1.0, 5.0, 10.0])),
        "n_estimators": int(rng.choice([120, 180, 260, 340]))}
    rng = np.random.default_rng(42); best, cfg_best = np.inf, None
    for _ in range(10):
        cfg = space(rng)
        m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42, **cfg)
        m.fit(Xs, Ys[:, H // 2]); v = float(np.sqrt(np.mean((m.predict(Xv) - Yv[:, H // 2]) ** 2)))
        if v < best:
            best, cfg_best = v, cfg
    P = np.zeros((Xte.shape[0] * N, H))
    for h in range(H):
        m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=seed, **cfg_best)
        m.fit(Xs, Ys[:, h]); P[:, h] = m.predict(Xte.reshape(-1, F))
    return cfg_best, (P.reshape(Xte.shape[0], N, H) * sd + mu).sum(1).reshape(-1)


y = sets[2][1].sum(1).reshape(-1)
sweep = np.load(os.path.join(HERE, 'load_sweep_v030.npz'))
assert np.max(np.abs(y - sweep['target'])) < 1e-3
base = np.load(os.path.join(HERE, 'nongraph_baselines.npz'))['XGBoost (b-global)'][0]
rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
res, out = {'own inputs (seed 42)': rmse(base), 'continuous inputs': len(cont)}, {}
for g, W in graphs.items():
    aug = [(np.concatenate([X, np.einsum('ij,sjf->sif', W, X[:, :, cont])], axis=2), Y) for X, Y in sets]
    cfg, p = xgb_global(*aug)
    d = diebold_mariano(p, base, y, loss='squared', h=H)
    res[g] = {'cfg': cfg, 'rmse': rmse(p), 'dm_p_vs_own_inputs': float(d.pvalue)}
    out[g] = p
    print(f'pooled XGBoost + {g} neighbor means: {rmse(p):.0f} MW against {rmse(base):.0f} MW '
          f'with its own inputs (p = {d.pvalue:.1e})', flush=True)
json.dump(res, open(os.path.join(HERE, 'graph_xgb.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(HERE, 'graph_xgb.npz'), **out)
