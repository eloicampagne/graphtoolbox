"""Covariate-aware national baselines for gross load: ARIMA-X, LSTM, Chronos-2.

National load series (sum over the 12 regions) and features aggregated from the
raw CSVs, trained on 2015-2018 and scored on the 2019 national target (target.pt,
the same series used by the GNN sweep and the regional GAM). Metrics: RMSE and
MAPE, so the rows sit alongside the borrowed INFORMS load baselines.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
from graphtoolbox.models import baseline as CB

STEP = 48
FEATURES = ["temperature", "temp_liss_fort", "temp_liss_faible", "nebu", "wind",
            "heat_deg", "cool_deg", "Instant", "Posan", "JourSemaine", "JourFerie",
            "DayType", "Weekend", "lag48", "lag96", "lag336"]


def smape(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return 100.0 * np.mean(2 * np.abs(p - y) / (np.abs(p) + np.abs(y) + 1e-8))


def mape(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return 100.0 * np.mean(np.abs((p - y) / np.clip(np.abs(y), 1e-6, None)))


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def build_national():
    cols = ["date", "Region", "load", "temp", "nebu", "wind", "temp_liss_fort",
            "temp_liss_faible", "Instant", "Posan", "JourSemaine", "JourFerie",
            "DayType", "Weekend"]
    df = pd.concat([pd.read_csv(os.path.join(HERE, "train.csv"), usecols=cols),
                    pd.read_csv(os.path.join(HERE, "test.csv"), usecols=cols)], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    g = df.groupby("date")
    nat = pd.DataFrame({
        "load": g["load"].sum(),
        "temperature": g["temp"].mean(),
        "temp_liss_fort": g["temp_liss_fort"].mean(),
        "temp_liss_faible": g["temp_liss_faible"].mean(),
        "nebu": g["nebu"].mean(), "wind": g["wind"].mean(),
        "Instant": g["Instant"].first(), "Posan": g["Posan"].first(),
        "JourSemaine": g["JourSemaine"].first(), "JourFerie": g["JourFerie"].first(),
        "DayType": g["DayType"].first(), "Weekend": g["Weekend"].first(),
    }).sort_index()
    nat["heat_deg"] = (15.0 - nat["temperature"]).clip(lower=0)
    nat["cool_deg"] = (nat["temperature"] - 15.0).clip(lower=0)
    for L in (48, 96, 336):
        nat[f"lag{L}"] = nat["load"].shift(L)
    return nat


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    nat = build_national().dropna(subset=["lag336"])
    nat = nat.reset_index().rename(columns={"date": "timestamp"})
    tgt = torch.load(os.path.join(HERE, "results_all_convolutions/target.pt"),
                     weights_only=False).numpy()
    train = nat[(nat["timestamp"] >= "2015-01-01") & (nat["timestamp"] < "2019-01-01")]
    t0 = int(nat[nat["timestamp"] >= "2019-01-01"].index.min())
    test = nat.iloc[t0:t0 + len(tgt)]
    y = test["load"].values.astype(float)
    assert len(test) == len(tgt), f"len {len(test)} vs {len(tgt)}"
    print(f"[check] national test vs target.pt: RMSE {rmse(y, tgt):.1f} MW (~0), {len(test)} steps")

    results = {}
    if which in ("all", "arimax"):
        print("ARIMA-X (per-Instant SARIMAX + exog) ...")
        results["ARIMA-X"] = CB.arimax_forecast(train, test, FEATURES, "load", "Instant")
    if which in ("all", "lstm"):
        print("LSTM (seq2seq + covariates) ...")
        results["LSTM"] = CB.lstm_forecast(train, test, FEATURES, "load", step=STEP)
    if which in ("all", "chronos2"):
        print("Chronos-2 (covariate regime, zero-shot) ...")
        results["Chronos-2"] = CB.chronos2_forecast(nat, t0, len(tgt), FEATURES, "load", step=STEP)

    print(f'\n{"Baseline":<12} {"MAPE (%)":>10} {"RMSE (MW)":>12}')
    print("-" * 36)
    rows = []
    for n, p in results.items():
        mp, rm = mape(p, tgt), rmse(p, tgt)
        rows.append({"model": n, "mape": mp, "rmse": rm})
        print(f"{n:<12} {mp:>10.2f} {rm:>12.0f}")
    print("\nReference (INFORMS/GNN): GAM 1200 | XGBoost 1416 | Chronos-Bolt 2408 | "
          "MLpol 788 | GAM reg 1248")
    out = os.path.join(HERE, "results_baselines_extra_load.csv")
    prev = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    keep = [r for r in prev if r["model"] not in results]
    pd.DataFrame(keep + rows).to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
