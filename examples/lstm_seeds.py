"""Seed study of the LSTM baseline at the GNN information set, load and net-load.

The seq2seq LSTM is the strongest covariate-aware baseline, so it is reported
like the GNNs: five seeds, mean and standard deviation, at hist_len=48 so that it
sees exactly the information the GNNs see (the previous day's forty-eight
half-hours plus the covariates), rather than a longer window. Predictions are
saved so the Diebold-Mariano tests can be run without refitting.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from graphtoolbox.models import baseline as CB

SEEDS = [42, 0, 1, 2, 3]
HIST = 48                      # matches the GNN lag window (previous day)
OUT = os.path.join(ROOT, "results_lstm_seeds")
os.makedirs(OUT, exist_ok=True)


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def smape(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return 100.0 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8))


def mape(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return 100.0 * np.mean(np.abs((p - y) / np.clip(np.abs(y), 1e-6, None)))


def run(tag):
    if tag == "netload":
        d = os.path.join(ROOT, "netload"); sys.path.insert(0, d); os.chdir(d)
        from baselines_netload import build_national, FEATURES
        nat = build_national().dropna(subset=["lag336"]).reset_index().rename(columns={"date": "timestamp"})
        tgt = torch.load(os.path.join(d, "results_all_convolutions_netload/target.pt"), weights_only=False).numpy()
        train = nat[nat["timestamp"] < "2019-01-01"]
        t0 = int(train.index.max()) + 1
        target_col, pct, pctname = "NetLoad", smape, "sMAPE"
    else:
        d = os.path.join(ROOT, "load"); sys.path.insert(0, d); os.chdir(d)
        from baselines_extra_load import build_national, FEATURES
        nat = build_national().dropna(subset=["lag336"]).reset_index().rename(columns={"date": "timestamp"})
        tgt = torch.load(os.path.join(d, "results_all_convolutions/target.pt"), weights_only=False).numpy()
        train = nat[(nat["timestamp"] >= "2015-01-01") & (nat["timestamp"] < "2019-01-01")]
        t0 = int(nat[nat["timestamp"] >= "2019-01-01"].index.min())
        target_col, pct, pctname = "load", mape, "MAPE"
    test = nat.iloc[t0:t0 + len(tgt)]

    rows = []
    for s in SEEDS:
        f = os.path.join(OUT, f"{tag}_s{s}.npy")
        if os.path.exists(f):
            p = np.load(f)
        else:
            p = CB.lstm_forecast(train, test, FEATURES, target_col, step=48, hist_len=HIST, seed=s)
            np.save(f, p)
        rows.append({"seed": s, "rmse": rmse(p, tgt), "pct": pct(p, tgt)})
        print(f"  {tag} seed {s}: {pctname} {rows[-1]['pct']:.2f}%  RMSE {rows[-1]['rmse']:.0f} MW", flush=True)
    r = np.array([x["rmse"] for x in rows]); q = np.array([x["pct"] for x in rows])
    print(f"{tag.upper():8} LSTM (hist={HIST}, {len(SEEDS)} seeds)  "
          f"{pctname} {q.mean():.2f} ± {q.std(ddof=1):.2f}  |  "
          f"RMSE {r.mean():.0f} ± {r.std(ddof=1):.0f}  [{r.min():.0f}, {r.max():.0f}]", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{tag}_summary.csv"), index=False)
    os.chdir(ROOT)


if __name__ == "__main__":
    for tag in (sys.argv[1:] or ["load", "netload"]):
        print(f"=== {tag} ===", flush=True)
        run(tag)
