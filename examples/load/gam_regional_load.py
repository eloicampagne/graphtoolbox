"""Regional (bottom-up) GAM baseline for gross load.

One per-instant GAM per region on that region's load and regional covariates,
summed to the national series, the symmetric counterpart to the regional GNN.
Evaluated on the same 2019 national load target as the sweep (target.pt).
"""
import os
import numpy as np
import pandas as pd
import torch
from pygam import s, f, l

from graphtoolbox.models.baseline import gam_forecast_by_slot, mape, rmse

HERE = os.path.dirname(os.path.abspath(__file__))
tgt = torch.load(os.path.join(HERE, "results_all_convolutions/target.pt"), weights_only=False).numpy()
T = len(tgt)

COLS = ["date", "Region", "load", "temp", "temp_liss_fort", "temp_liss_faible", "nebu", "wind",
        "Instant", "Posan", "JourSemaine", "JourFerie", "DayType", "Weekend", "offset"]
GORDER = ["temp", "temp_liss_fort", "temp_liss_faible", "Posan", "nebu", "wind",
          "JourSemaine", "JourFerie", "DayType", "Weekend", "offset"]
gi = {c: i for i, c in enumerate(GORDER)}
FORMULA = (s(gi["temp"]) + s(gi["temp_liss_fort"]) + s(gi["temp_liss_faible"])
           + s(gi["Posan"], by=gi["JourSemaine"]) + s(gi["Posan"], basis="cp")
           + f(gi["JourSemaine"]) + f(gi["JourFerie"]) + f(gi["DayType"]) + f(gi["Weekend"])
           + s(gi["nebu"]) + s(gi["wind"]) + l(gi["offset"]))


def main():
    df = pd.concat([pd.read_csv(os.path.join(HERE, "train.csv"), usecols=COLS),
                    pd.read_csv(os.path.join(HERE, "test.csv"), usecols=COLS)], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    regions = sorted(df["Region"].unique())
    print(f"{len(regions)} regions; national test length {T}")

    nat_pred = np.zeros(T); nat_true = np.zeros(T)
    for r in regions:
        sub = df[df["Region"] == r].sort_values("date").reset_index(drop=True)
        tr = sub[(sub["date"] >= "2015-01-01") & (sub["date"] < "2019-01-01")]
        te = sub[sub["date"] >= "2019-01-01"].iloc[:T]
        pred = gam_forecast_by_slot(
            tr, te, GORDER, "load", "Instant", FORMULA)
        nat_pred += pred
        nat_true += te["load"].values.astype(float)
        print(f"  {r:<28} region RMSE {rmse(pred, te['load'].values):.0f} MW")

    print(f"\n[check] summed regional truth vs national target: RMSE {rmse(nat_true, tgt):.1f} MW (~0)")
    y = tgt
    print("\n=== National load, 2019 test ===")
    print(f"Regional GAM (bottom-up, per region per instant)  MAPE {mape(nat_pred, y):.2f}%  RMSE {rmse(nat_pred, y):.0f} MW")
    print("National GAM (INFORMS reference)                   MAPE 1.67%  RMSE 1200 MW")
    print("MLpol aggregation (GNN sweep, reference)           MAPE 1.02%  RMSE 788 MW")
    pd.DataFrame({"national_load_pred": nat_pred}).to_csv(os.path.join(HERE, "results_gam_regional_load.csv"), index=False)


if __name__ == "__main__":
    main()
