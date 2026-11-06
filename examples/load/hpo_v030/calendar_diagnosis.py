"""Where pooled XGBoost beats GatedGraphConv on French load, 2019: errors by calendar, temperature and region.

GatedGraphConv: the five runs of the seed study (preds/GatedGraphConv_seed*.pt), each scored separately and the
scores averaged. XGBoost: pooled over the regions, the configuration of nongraph_baselines.py, seed 42, refitted
here to keep its regional forecasts. Each breakdown reports the national RMSE of both models within a category
and the share of the gap in squared error (GNN minus XGBoost, summed over the test year) that the category holds.

    python calendar_diagnosis.py [FIGURE.png]   # writes calendar_diagnosis.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LOAD = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(LOAD)))
os.chdir(LOAD)
from graphtoolbox.data import DataClass, GraphDataset  # noqa: E402

H, SEEDS = 48, [42, 0, 1, 2, 3]
FIG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'calendar_diagnosis.png')
CACHE = os.path.join(HERE, 'calendar_diagnosis_xgb.npy')
src = open(os.path.join(HERE, 'graph_xgb.py')).read()
exec(src[src.index('data_kwargs = {'):src.index('data = DataClass')])  # inputs of the sweep
data = DataClass(path_train='./train.csv', path_test='./test.csv', data_kwargs=data_kwargs, folder_config='.')
common = dict(graph_folder='../graph_representations', dataset_kwargs=dataset_kwargs, out_channels=H)
train = GraphDataset(data=data, period='train', **common)
test = GraphDataset(data=data, period='test', scalers_feat=train.scalers_feat,
                    scalers_target=train.scalers_target, **common)
stack = lambda ds: (np.stack([d.x.numpy() for d in ds.pyg_data]).astype(np.float32),
                    np.stack([d.y.numpy() for d in ds.pyg_data]).astype(np.float64))
(Xtr, Ytr), (Xte, Yte) = stack(train), stack(test)
N = train.num_nodes
Y = Yte.transpose(1, 0, 2).reshape(N, -1)                      # regions x half-hours
y = Y.sum(0)

if not os.path.exists(CACHE):
    import xgboost as xgb
    cfg = json.load(open(os.path.join(HERE, 'nongraph_baselines.json')))['XGBoost (b-global)']['cfg']
    F = Xtr.shape[2]
    mu = Ytr.mean(axis=(0, 2), keepdims=True); sd = Ytr.std(axis=(0, 2), keepdims=True) + 1e-9
    Ys = ((Ytr - mu) / sd).reshape(-1, H)
    P = np.zeros((Xte.shape[0] * N, H))
    for h in range(H):
        m = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist', random_state=42, **cfg)
        m.fit(Xtr.reshape(-1, F), Ys[:, h]); P[:, h] = m.predict(Xte.reshape(-1, F))
    np.save(CACHE, (P.reshape(Xte.shape[0], N, H) * sd + mu).transpose(1, 0, 2).reshape(N, -1))
XG = np.load(CACHE)
GN = [torch.load(os.path.join(HERE, 'preds', f'GatedGraphConv_seed{s}.pt'), weights_only=False).numpy() for s in SEEDS]
rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
print(f'national RMSE: XGBoost {rmse(XG.sum(0) - y):.1f} MW, GatedGraphConv '
      f'{np.mean([rmse(g.sum(0) - y) for g in GN]):.1f} MW (mean of five runs)', flush=True)

# Calendar of each test half-hour, from the regional rows of the sweep.
cols = ['date', 'Region', 'load', 'temp', 'JourSemaine', 'JourFerie', 'DayType', 'Weekend']
df = pd.read_csv('./test.csv', usecols=cols)
df['date'] = pd.to_datetime(df['date'])
df = df[(df['date'] >= '2019-01-01') & (df['date'] < '2019-12-31')].sort_values(['date', 'Region'])
cal = df.groupby('date').agg(load=('load', 'sum'), temp=('temp', 'mean'), dow=('JourSemaine', 'first'),
                             holiday=('JourFerie', 'first'), daytype=('DayType', 'first')).reset_index()
assert len(cal) == len(y) and np.max(np.abs(cal['load'].to_numpy() - y)) < 1e-3, 'calendar misaligned'
day = cal['date'].dt.normalize()
cal['halfhour'] = cal['date'].dt.hour * 2 + cal['date'].dt.minute // 30
cal['month'] = cal['date'].dt.month
daily_temp = cal.groupby(day)['temp'].transform('mean')
cal['temp_decile'] = pd.qcut(daily_temp, 10, labels=False)
hol = cal.groupby(day)['holiday'].transform('max').to_numpy()
prev_hol = pd.Series(hol, index=cal.index).groupby(day).first().shift(1).reindex(day).to_numpy()
next_hol = pd.Series(hol, index=cal.index).groupby(day).first().shift(-1).reindex(day).to_numpy()
cal['kind'] = np.select([hol == 1, (prev_hol == 1) | (next_hol == 1), cal['dow'].isin([5, 6])],
                        ['holiday', 'next to a holiday', 'weekend'], 'working day')
# The DayType codes of the RTE data mark special days; 0 is the ordinary day.
cal['special'] = np.where(cal['daytype'] == 0, 'ordinary', 'special DayType')

eg = np.stack([g.sum(0) - y for g in GN])                     # seeds x half-hours
ex = XG.sum(0) - y
gap_total = float(np.mean(np.mean(eg ** 2, 0) - ex ** 2) * len(y))


def breakdown(key):
    rows = []
    for v, idx in cal.groupby(key).groups.items():
        i = np.asarray(list(idx))
        g = [rmse(e[i]) for e in eg]
        gap = float(np.sum(np.mean(eg[:, i] ** 2, 0) - ex[i] ** 2))
        rows.append({key: v if not isinstance(v, (np.integer, np.floating)) else v.item(), 'n_halfhours': len(i),
                     'gnn_rmse': float(np.mean(g)), 'gnn_rmse_sd': float(np.std(g, ddof=1)),
                     'xgb_rmse': rmse(ex[i]), 'gnn_bias': float(eg[:, i].mean()), 'xgb_bias': float(ex[i].mean()),
                     'share_of_gap': gap / gap_total})
    return rows


res = {'national': {'xgb': rmse(ex), 'gnn_mean_of_runs': float(np.mean([rmse(e) for e in eg])),
                    'gap_in_squared_error': gap_total}}
for key in ('halfhour', 'month', 'dow', 'kind', 'special', 'temp_decile'):
    res[key] = breakdown(key)
    print(f'\n{key}:')
    for r in res[key]:
        print(f"  {str(r[key]):<18} n={r['n_halfhours']:5d}  GNN {r['gnn_rmse']:6.0f} ± {r['gnn_rmse_sd']:3.0f}  "
              f"XGB {r['xgb_rmse']:6.0f}  bias GNN {r['gnn_bias']:+6.0f} XGB {r['xgb_bias']:+6.0f}  "
              f"share of gap {100 * r['share_of_gap']:+5.1f}%")
regions = sorted(df['Region'].unique())
res['region'] = [{'region': r, 'gnn_rmse': float(np.mean([rmse(g[k] - Y[k]) for g in GN])),
                  'xgb_rmse': rmse(XG[k] - Y[k]), 'mean_load': float(Y[k].mean())} for k, r in enumerate(regions)]
print('\nregion:')
for r in res['region']:
    print(f"  {r['region']:<28} GNN {r['gnn_rmse']:5.0f}  XGB {r['xgb_rmse']:5.0f}  ({r['gnn_rmse'] / r['xgb_rmse']:.2f})")
json.dump(res, open(os.path.join(HERE, 'calendar_diagnosis.json'), 'w'), indent=1)

# Figure: national RMSE of both models by half-hour, month, kind of day, temperature decile, and region.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#28251d', '#77756b', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.labelcolor': INK, 'text.color': INK})
fig, axes = plt.subplots(1, 5, figsize=(15, 3.1), gridspec_kw={'width_ratios': [2.2, 1.6, 1.1, 1.4, 2.0]})
panels = [('halfhour', 'Half-hour of the forecast day', lambda v: f'{v // 2:02d}h' if v % 6 == 0 else ''),
          ('month', 'Month', lambda v: 'JFMAMJJASOND'[v - 1]),
          ('kind', 'Kind of day', str), ('temp_decile', 'Decile of daily temperature', lambda v: str(v + 1))]
for ax, (key, title, lab) in zip(axes, panels):
    rows = res[key]
    if key == 'kind':
        order = ['working day', 'weekend', 'next to a holiday', 'holiday']
        rows = sorted(rows, key=lambda r: order.index(r[key]))
    x = np.arange(len(rows))
    gn, sd, xg = (np.array([r[k] for r in rows]) for k in ('gnn_rmse', 'gnn_rmse_sd', 'xgb_rmse'))
    if key in ('halfhour', 'temp_decile'):
        ax.fill_between(x, gn - sd, gn + sd, color=BLUE, alpha=0.15, lw=0)
        ax.plot(x, gn, color=BLUE, lw=2, label='GatedGraphConv (5 runs)')
        ax.plot(x, xg, color=ORANGE, lw=2, label='XGBoost pooled')
    else:
        ax.bar(x - 0.2, gn, 0.38, yerr=sd, color=BLUE, error_kw={'lw': 0.8, 'ecolor': INK},
               label='GatedGraphConv (5 runs)')
        ax.bar(x + 0.2, xg, 0.38, color=ORANGE, label='XGBoost pooled')
    ax.set_xticks(x, [lab(r[key]) for r in rows], rotation=0 if key != 'kind' else 25,
                  ha='center' if key != 'kind' else 'right')
    ax.set_title(title, fontsize=8.5, loc='left')
    for k, r in enumerate(rows):
        if key in ('kind',):
            ax.text(k, max(gn[k] + sd[k], xg[k]) + 15, f"{100 * r['share_of_gap']:.0f}%", ha='center', fontsize=7,
                    color=MUTED)
reg = sorted(res['region'], key=lambda r: -r['mean_load'])
x = np.arange(len(reg))
axes[4].bar(x - 0.2, [r['gnn_rmse'] for r in reg], 0.38, color=BLUE)
axes[4].bar(x + 0.2, [r['xgb_rmse'] for r in reg], 0.38, color=ORANGE)
axes[4].set_xticks(x, [r['region'].replace('_', ' ')[:12] for r in reg], rotation=60, ha='right', fontsize=6.5)
axes[4].set_title('Region (own RMSE)', fontsize=8.5, loc='left')
for ax in axes:
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel('RMSE (MW)')
axes[0].legend(frameon=False, loc='upper left')
fig.tight_layout()
fig.savefig(FIG, dpi=160)
print('figure', FIG)
