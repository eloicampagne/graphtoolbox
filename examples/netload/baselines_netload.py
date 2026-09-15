"""National-direct classical baselines for net-load, to match the load table.

Builds the national net-load series (sum over the 12 regions) and national
features (aggregated weather, shared calendar, day-ahead net-load lags) from the
raw CSVs, fits XGBoost and a GAM on 2014-2018, runs Chronos-Bolt zero-shot, and
reports RMSE and sMAPE on the 2019 test year, which is identical to the GNN
target (target.pt). These rows are produced with GraphToolbox for this paper,
unlike the load baselines borrowed from the companion study.
"""
import os
import numpy as np
import pandas as pd
import torch

from graphtoolbox.models.baseline import (
    chronos_bolt_forecast,
    gam_forecast_by_slot,
    rmse,
    smape,
    xgboost_forecast,
)

HERE = os.path.dirname(__file__)
STEP = 48


def build_national():
    cols = ["date", "NetLoad", "temperature", "temperature_lisse_950", "temperature_lisse_990",
            "wind", "nebulosity", "wind_by_wind_power_weights", "nebulosity_by_solar_power_weights",
            "tod", "toy", "month", "day_type_week", "day_type_jf", "year",
            "period_holiday", "period_hour_changed", "period_summer", "period_christmas"]
    df = pd.concat([pd.read_csv(os.path.join(HERE, "train.csv"), usecols=cols),
                    pd.read_csv(os.path.join(HERE, "test.csv"), usecols=cols)], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    g = df.groupby("date")
    nat = pd.DataFrame({
        "NetLoad": g["NetLoad"].sum(),
        "temperature": g["temperature"].mean(),
        "temp950": g["temperature_lisse_950"].mean(),
        "temp990": g["temperature_lisse_990"].mean(),
        "wind": g["wind"].mean(),
        "nebulosity": g["nebulosity"].mean(),
        "wind_pw": g["wind_by_wind_power_weights"].mean(),
        "nebu_pw": g["nebulosity_by_solar_power_weights"].mean(),
        "tod": g["tod"].first(), "toy": g["toy"].first(), "month": g["month"].first(),
        "day_type_week": g["day_type_week"].first(), "day_type_jf": g["day_type_jf"].first(),
        "year": g["year"].first(),
        "period_holiday": g["period_holiday"].first(), "period_hour_changed": g["period_hour_changed"].first(),
        "period_summer": g["period_summer"].first(), "period_christmas": g["period_christmas"].first(),
    }).sort_index()
    nat.index = nat.index.tz_localize(None)
    nat["heat_deg"] = (15.0 - nat["temperature"]).clip(lower=0)
    nat["cool_deg"] = (nat["temperature"] - 15.0).clip(lower=0)
    for L in (48, 96, 336):                       # J-1, J-2, J-7 same slot (day-ahead)
        nat[f"lag{L}"] = nat["NetLoad"].shift(L)
    return nat


FEATURES = ["temperature", "temp950", "temp990", "wind", "nebulosity", "heat_deg", "cool_deg",
            "tod", "toy", "month", "day_type_week", "day_type_jf",
            "period_holiday", "period_summer", "period_christmas", "lag48", "lag96", "lag336"]


def main():
    nat = build_national().dropna(subset=["lag336"])
    train = nat.loc["2014-01-01":"2018-12-31"]
    tgt = torch.load(os.path.join(HERE, "results_all_convolutions_netload/target.pt"),
                     weights_only=False).numpy()
    # GNN test window is the first 364 days of 2019 (target.pt has 17472 steps).
    test = nat.loc["2019-01-01":].iloc[:len(tgt)]
    y_test = test["NetLoad"].values.astype(float)
    assert len(test) == len(tgt) and np.max(np.abs(y_test - tgt)) < 1e-6, \
        f"test target mismatch: len {len(test)} vs {len(tgt)}"
    print(f"National test 2019: {len(test)} steps, target matches GNN target.pt.")

    results = {}

    # XGBoost parameters from the INFORMS benchmark.
    results["XGBoost"] = xgboost_forecast(train, test, FEATURES, "NetLoad")

    # One GAM per half-hour slot, following the INFORMS methodology.
    # Mirrors models_GAM.fit_gam_by(by_feature='tod'): 48 independent LinearGAMs,
    # with the additive formula of dev.ipynb adapted to the features available here.
    try:
        from pygam import s, f, l
        gorder = ["temperature", "temp950", "temp990", "toy", "wind_pw", "nebu_pw",
                  "day_type_week", "day_type_jf", "period_holiday", "period_hour_changed", "year"]
        gi = {c: i for i, c in enumerate(gorder)}
        formula = (s(gi["temperature"]) + s(gi["temp950"]) + s(gi["temp990"])
                   + s(gi["toy"], by=gi["day_type_week"])
                   + f(gi["day_type_week"]) + f(gi["day_type_jf"])
                   + f(gi["period_holiday"]) + f(gi["period_hour_changed"])
                   + s(gi["toy"], basis="cp")
                   + s(gi["wind_pw"]) + s(gi["toy"], by=gi["wind_pw"], basis="cp")
                   + s(gi["nebu_pw"]) + s(gi["toy"], by=gi["nebu_pw"], basis="cp")
                   + l(gi["year"]))
        results["GAM"] = gam_forecast_by_slot(
            train, test, gorder, "NetLoad", "tod", formula)
    except Exception as e:
        print(f"GAM failed: {str(e).splitlines()[0]}")

    # Chronos-Bolt zero-shot rolling day-ahead forecast.
    try:
        series = nat["NetLoad"].values.astype(np.float32)
        idx = nat.index
        test_start = np.where(idx == test.index[0])[0][0]
        results["Chronos-Bolt"] = chronos_bolt_forecast(
            series, test_start, len(test), step=STEP)
    except Exception as e:
        print(f"Chronos failed: {str(e).splitlines()[0]}")

    # Report results.
    print(f'\n{"Baseline":<16} {"sMAPE (%)":>10} {"RMSE (MW)":>12}')
    print("-" * 40)
    rows = [(n, smape(p, y_test), rmse(p, y_test)) for n, p in results.items()]
    for n, sm, rm in sorted(rows, key=lambda x: x[2], reverse=True):
        print(f"{n:<16} {sm:>10.2f} {rm:>12.0f}")
    out = os.path.join(HERE, "results_baselines_netload.csv")
    pd.DataFrame([{"model": n, "smape": sm, "rmse": rm} for n, sm, rm in rows]).to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
