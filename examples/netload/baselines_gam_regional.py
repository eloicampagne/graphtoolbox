"""Regional (bottom-up) GAM baseline for net-load.

Fits one per-tod GAM per region on that region's net-load and regional
covariates, then sums the twelve regional forecasts to the national series, the
symmetric counterpart to the regional GNN. Complements the national-direct
(top-level) GAM of baselines_netload.py. Evaluated on the same 2019 national
target as the GNN (target.pt).
"""
import os
import numpy as np
import pandas as pd
import torch
from pygam import s, f, l

import netload_pipeline as P
from graphtoolbox.models.baseline import gam_forecast_by_slot, rmse, smape

HERE = P.HERE
T_target = torch.load(os.path.join(HERE, "results_all_convolutions_netload/target.pt"), weights_only=False).numpy()
T = len(T_target)

COLS = ["date", "Region", "NetLoad", "temperature", "temperature_lisse_950", "temperature_lisse_990",
        "wind_by_wind_power_weights", "nebulosity_by_solar_power_weights",
        "tod", "toy", "day_type_week", "day_type_jf", "year", "period_holiday", "period_hour_changed"]
GORDER = ["temperature", "temp950", "temp990", "toy", "wind_pw", "nebu_pw",
          "day_type_week", "day_type_jf", "period_holiday", "period_hour_changed", "year"]
gi = {c: i for i, c in enumerate(GORDER)}
FORMULA = (s(gi["temperature"]) + s(gi["temp950"]) + s(gi["temp990"]) + s(gi["toy"], by=gi["day_type_week"])
           + f(gi["day_type_week"]) + f(gi["day_type_jf"]) + f(gi["period_holiday"]) + f(gi["period_hour_changed"])
           + s(gi["toy"], basis="cp") + s(gi["wind_pw"]) + s(gi["toy"], by=gi["wind_pw"], basis="cp")
           + s(gi["nebu_pw"]) + s(gi["toy"], by=gi["nebu_pw"], basis="cp") + l(gi["year"]))


def main():
    df = pd.concat([pd.read_csv(HERE / "train.csv", usecols=COLS),
                    pd.read_csv(HERE / "test.csv", usecols=COLS)], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.rename(columns={"temperature_lisse_950": "temp950", "temperature_lisse_990": "temp990",
                            "wind_by_wind_power_weights": "wind_pw",
                            "nebulosity_by_solar_power_weights": "nebu_pw"})
    regions = sorted(df["Region"].unique())
    print(f"{len(regions)} regions; national test length {T}")

    nat_pred = np.zeros(T)
    nat_true = np.zeros(T)
    for r in regions:
        sub = df[df["Region"] == r].sort_values("date").reset_index(drop=True)
        tr = sub[(sub["date"] >= "2014-01-01") & (sub["date"] < "2018-12-31")]
        te = sub[sub["date"] >= "2019-01-01"].iloc[:T]
        pred = gam_forecast_by_slot(
            tr, te, GORDER, "NetLoad", "tod", FORMULA)
        nat_pred += pred
        nat_true += te["NetLoad"].values.astype(float)
        print(f"  {r:<24} region RMSE {rmse(pred, te['NetLoad'].values):.0f} MW")

    # Sanity: summed regional truth should equal the national target.
    print(f"\n[check] summed regional truth vs national target: RMSE {rmse(nat_true, T_target):.1f} MW (~0)")
    y = T_target
    print("\n=== National net-load, 2019 test ===")
    print(f"Regional GAM (bottom-up, per region per tod)  sMAPE {smape(nat_pred, y):.2f}%  RMSE {rmse(nat_pred, y):.0f} MW")
    print("National GAM (top-level, reference)            sMAPE 2.92%  RMSE 1763 MW")
    print("Decomposed GNN aggregation (reference)         sMAPE 2.71%  RMSE 1739 MW")
    pd.DataFrame({"national_gam_pred": nat_pred}).to_csv(os.path.join(HERE, "results_gam_regional.csv"), index=False)


if __name__ == "__main__":
    main()
