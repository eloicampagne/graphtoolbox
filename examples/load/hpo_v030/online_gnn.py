"""Daily online update of GatedGraphConv over the test year with RollingTrainer(update="replay"), French load.

The model starts from the checkpoints of the five-seed study (run_seeds.sh: trained on 2015-2017, early stopping
on 2018). Each day D, it forecasts the 48 half-hours of D, then takes `steps` Adam steps (learning rate `lr`) on
mini-batches of 32 days: the 7 most recent days, D included, and 25 days drawn from all earlier days. The scalers
and the DTW graph stay those of the initial training set.

Selection: (lr, steps) is chosen on 2018, rolled from the checkpoint of seed 42 (the test stream is then 2018).
Test: each seed first rolls through 2018 without scoring (warmup_on_val), then through 2019, so that at the first
2019 origin the model has seen the data the test TAM was fitted on.

    python online_gnn.py select
    python online_gnn.py test
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric import seed_everything
from torch_geometric.nn.conv import GatedGraphConv

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(LOAD)))
os.chdir(LOAD)
from graphtoolbox.data import DataClass, GraphDataset  # noqa: E402
from graphtoolbox.models import myGNN  # noqa: E402
from graphtoolbox.training import RollingTrainer, set_device  # noqa: E402

set_device('cpu')
torch.set_num_threads(4)
SEEDS = [42, 0, 1, 2, 3]
GRID = list(itertools.product((3e-5, 1e-4, 3e-4), (10, 30)))
dataset_kwargs = {
    'batch_size': 32, 'adj_matrix': 'dtw',
    'features_base': ['temp', 'nebu', 'wind', 'tempMax', 'tempMin', 'Posan', 'Instant', 'RegionInt',
                      'JourSemaine', 'JourFerie', 'offset', 'DayType', 'Weekend', 'temp_liss_fort',
                      'temp_liss_faible', 'heat_deg', 'cool_deg', 'temp_sq', 'is_summer']
                     + [f'load_l{t}' for t in range(1, 49)] + [f'temp_l{t}' for t in range(-47, 0)],
    'target_base': 'load',
}


def split_2018():
    """train.csv cut at 2018-01-01 into two disjoint files, cached next to this script."""
    out = os.path.join(HERE, 'cache_online')
    paths = os.path.join(out, 'train_to2017.csv'), os.path.join(out, 'stream_2018.csv')
    if not all(os.path.exists(p) for p in paths):
        os.makedirs(out, exist_ok=True)
        df = pd.read_csv('./train.csv')
        before = pd.to_datetime(df['date']) < '2018-01-01'
        df[before].to_csv(paths[0], index=False)
        df[~before].to_csv(paths[1], index=False)
    return paths


def datasets(test_year):
    """Datasets of sweep_row.py; the test stream is 2019, or 2018 for the selection."""
    data_kwargs = {
        'node_var': 'Region',
        'features_to_lag': {'load': (1, 48), 'temp': (-47, -1)},
        'dummies': ['Instant', 'RegionInt', 'JourSemaine', 'DayType', 'offset'],
        'day_inf_train': '2015-01-01', 'day_sup_train': '2018-01-01',
        'day_inf_val': '2018-01-01', 'day_sup_val': '2018-12-31',
        'day_inf_test': f'{test_year}-01-01', 'day_sup_test': f'{test_year}-12-31',
        'computed_features': {
            'heat_deg':  lambda df: (15 - df['temp']).clip(lower=0),
            'cool_deg':  lambda df: (df['temp'] - 15).clip(lower=0),
            'temp_sq':   lambda df: df['temp'] ** 2,
            'is_summer': lambda df: (pd.to_datetime(df['date']).dt.month == 7).astype(float)
                                    + (pd.to_datetime(df['date']).dt.month == 8).astype(float),
        },
    }
    path_train, path_test = './train.csv', './test.csv'
    if test_year == 2018:
        # DataClass reads the test period from path_test and lags the concatenation of both files, so the
        # 2018 stream must come from a file disjoint from the training file. Validation is unused then
        # (no warm-up); it spans 2017 so that every dummy category is present.
        path_train, path_test = split_2018()
        data_kwargs.update({'day_inf_val': '2017-01-01', 'day_sup_val': '2018-01-01'})
    data = DataClass(path_train=path_train, path_test=path_test, data_kwargs=data_kwargs, folder_config='.')
    common = dict(graph_folder='../graph_representations', dataset_kwargs=dataset_kwargs, out_channels=48)
    train = GraphDataset(data=data, period='train', **common)
    val = GraphDataset(data=data, period='val', scalers_feat=train.scalers_feat,
                       scalers_target=train.scalers_target, **common)
    test = GraphDataset(data=data, period='test', scalers_feat=train.scalers_feat,
                        scalers_target=train.scalers_target, **common)
    for ds in (train, val, test):
        ds._set_adj_matrix(adj_matrix='dtw')
    return train, val, test


def load_model(train, seed):
    seed_everything(seed)
    model = myGNN(in_channels=train.num_node_features, num_layers=2, hidden_channels=64, out_channels=48,
                  conv_class=GatedGraphConv, heads=4)
    d = os.path.join(HERE, 'checkpoints', f'GatedGraphConv_seed{seed}')
    ckpt = [f for f in os.listdir(d) if f.endswith('.params')]
    assert len(ckpt) == 1, ckpt
    model.load_state_dict(torch.load(os.path.join(d, ckpt[0]), map_location='cpu'))
    return model


def roll(train, val, test, seed, lr, steps, warmup):
    """National forecast and target of the test stream."""
    out = RollingTrainer(train, val, test, model_class=myGNN, model_kwargs={}, window_size=48, step_size=48,
                         batch_size=32, reconcile=False, trainer_kwargs={'seed': seed}, update='replay',
                         initial_model=load_model(train, seed), replay_steps=steps, replay_lr=lr,
                         replay_recent=7, warmup_on_val=warmup).run()
    pred = torch.cat([r['preds'] for r in out], dim=1).sum(0).numpy()
    y = torch.cat([r['targets'] for r in out], dim=1).sum(0).numpy()
    return pred, y


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def select():
    train, val, test = datasets(2018)
    res = {}
    for lr, steps in [(0.0, 0)] + GRID:
        p, y = roll(train, val, test, 42, lr, steps, warmup=False)
        res[f'{lr}|{steps}'] = rmse(p, y)
        print(f'2018, seed 42, lr={lr:g} steps={steps}: {res[f"{lr}|{steps}"]:.0f} MW', flush=True)
    json.dump(res, open(os.path.join(HERE, 'online_gnn_select.json'), 'w'), indent=1)


def test():
    sel = json.load(open(os.path.join(HERE, 'online_gnn_select.json')))
    key = min((k for k in sel if k != '0.0|0'), key=sel.get)
    lr, steps = float(key.split('|')[0]), int(key.split('|')[1])
    train, val, test = datasets(2019)
    sweep = np.load(os.path.join(HERE, 'load_sweep_v030.npz'))
    out, res = {}, {'selected': key, 'static': [], 'online': []}
    for s in SEEDS:
        static, y = roll(train, val, test, s, 0.0, 0, warmup=False)
        assert np.max(np.abs(y - sweep['target'])) < 1e-3
        stored = torch.load(os.path.join(HERE, 'preds', f'GatedGraphConv_seed{s}.pt'), weights_only=False).numpy().sum(0)
        online, _ = roll(train, val, test, s, lr, steps, warmup=True)
        res['static'].append(rmse(static, y))
        res['online'].append(rmse(online, y))
        out[f'static_seed{s}'], out[f'online_seed{s}'] = static, online
        print(f'seed {s}: static {rmse(static, y):.0f} MW (stored run {rmse(stored, y):.0f}), '
              f'online {rmse(online, y):.0f} MW', flush=True)
    for k in ('static', 'online'):
        print(f'{k}: {np.mean(res[k]):.0f} ± {np.std(res[k], ddof=1):.0f} MW')
    json.dump(res, open(os.path.join(HERE, 'online_gnn.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(HERE, 'online_gnn.npz'), **out)


if __name__ == '__main__':
    {'select': select, 'test': test}[sys.argv[1]]()
