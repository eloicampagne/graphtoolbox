"""Covariate-aware national baselines for net-load: ARIMA-X, LSTM, Chronos-2.

Complements baselines_netload.py (XGBoost, GAM, Chronos-Bolt). Same national
series (sum over regions), same 2019 test window as the GNN target.pt, same
features as XGBoost, scored with RMSE and sMAPE.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(__file__)
from graphtoolbox.models import baseline as CB
from baselines_netload import build_national, FEATURES, smape, rmse

STEP = 48


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    nat = build_national().dropna(subset=["lag336"])
    nat = nat.reset_index().rename(columns={"index": "timestamp", "date": "timestamp"})
    tgt = torch.load(os.path.join(HERE, "results_all_convolutions_netload/target.pt"),
                     weights_only=False).numpy()
    train = nat[nat["timestamp"] < "2019-01-01"]
    t0 = int(train.index.max()) + 1
    test = nat.iloc[t0:t0 + len(tgt)]
    y = test["NetLoad"].values.astype(float)
    assert len(test) == len(tgt) and np.max(np.abs(y - tgt)) < 1e-6, "target mismatch"
    print(f"National net-load test 2019: {len(test)} steps, matches target.pt.")

    results = {}
    if which in ("all", "arimax"):
        print("ARIMA-X (per-tod SARIMAX + exog) ...")
        results["ARIMA-X"] = CB.arimax_forecast(train, test, FEATURES, "NetLoad", "tod")
    if which in ("all", "lstm"):
        print("LSTM (seq2seq + covariates) ...")
        results["LSTM"] = CB.lstm_forecast(train, test, FEATURES, "NetLoad", step=STEP)
    if which in ("all", "chronos2"):
        print("Chronos-2 (covariate regime, zero-shot) ...")
        results["Chronos-2"] = CB.chronos2_forecast(nat, t0, len(tgt), FEATURES, "NetLoad", step=STEP)

    print(f'\n{"Baseline":<12} {"sMAPE (%)":>10} {"RMSE (MW)":>12}')
    print("-" * 36)
    rows = []
    for n, p in results.items():
        sm, rm = smape(p, y), rmse(p, y)
        rows.append({"model": n, "smape": sm, "rmse": rm})
        print(f"{n:<12} {sm:>10.2f} {rm:>12.0f}")
    print("\nReference: GAM 1763 | XGBoost 2031 | Chronos-Bolt 3676 | Persistence 5636")
    out = os.path.join(HERE, "results_baselines_extra_netload.csv")
    prev = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    keep = [r for r in prev if r["model"] not in results]
    pd.DataFrame(keep + rows).to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
