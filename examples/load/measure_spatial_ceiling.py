"""Estimate spatial forecasting gains with own-region and all-region ridge models."""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = float(os.environ.get("CEIL_ALPHA", 10.0))
SLOTS = 48


def build():
    cols = ["date", "Region", "load", "temp", "temp_liss_fort", "temp_liss_faible",
            "nebu", "wind", "Instant", "Posan", "JourSemaine", "JourFerie", "DayType", "Weekend"]
    df = pd.concat([pd.read_csv(os.path.join(HERE, "train.csv"), usecols=cols),
                    pd.read_csv(os.path.join(HERE, "test.csv"), usecols=cols)], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["day"] = df["date"].dt.normalize()
    regions = sorted(df["Region"].unique())

    # [day, slot] matrices per region, and day-level calendar / weather
    load = {r: df[df["Region"] == r].pivot_table(index="day", columns="Instant", values="load")
            for r in regions}
    temp = {r: df[df["Region"] == r].pivot_table(index="day", columns="Instant", values="temp")
            for r in regions}
    days = load[regions[0]].index
    for r in regions:                       # keep only fully observed days, common to all regions
        days = days.intersection(load[r].dropna().index).intersection(temp[r].dropna().index)
    days = pd.DatetimeIndex(sorted(days))
    cal = (df.drop_duplicates("day").set_index("day")
             .loc[days, ["Posan", "JourSemaine", "JourFerie", "DayType", "Weekend"]])
    return regions, days, load, temp, cal


def main():
    regions, days, load, temp, cal = build()
    # Day D is predicted from day D-1, so keep consecutive pairs only.
    prev = days[:-1]; cur = days[1:]
    ok = (cur - prev) == pd.Timedelta("1D")
    prev, cur = prev[ok], cur[ok]
    is_test = cur >= pd.Timestamp("2019-01-01")
    tr, te = ~is_test, is_test
    print(f"{len(regions)} regions, {len(cur)} usable day pairs "
          f"({tr.sum()} train, {te.sum()} test)")

    cal_m = pd.get_dummies(cal.loc[cur], columns=["JourSemaine", "DayType"]).values.astype(float)

    nat = {}
    per_region = {}
    for name, use_neighbors in (("own", False), ("all", True)):
        pred_nat = np.zeros((te.sum(), SLOTS))
        true_nat = np.zeros((te.sum(), SLOTS))
        rr = []
        for r in regions:
            own_lag = load[r].loc[prev].values                 # [D, 48] previous day
            own_tmp = temp[r].loc[cur].values                  # [D, 48] target-day weather
            X = [own_lag, own_tmp, cal_m]
            if use_neighbors:
                X += [load[o].loc[prev].values for o in regions if o != r]
            X = np.hstack(X)
            y = load[r].loc[cur].values                        # [D, 48] target
            sx = StandardScaler().fit(X[tr])
            m = Ridge(alpha=ALPHA).fit(sx.transform(X[tr]), y[tr])
            p = m.predict(sx.transform(X[te]))
            pred_nat += p; true_nat += y[te]
            rr.append((r, float(np.sqrt(np.mean((p - y[te]) ** 2)))))
        per_region[name] = rr
        nat[name] = float(np.sqrt(np.mean((pred_nat - true_nat) ** 2)))
        print(f"\n[{name:3}] national RMSE {nat[name]:.0f} MW   ({X.shape[1]} features per region)")
        for r, v in rr:
            print(f"      {r:<28} {v:7.1f} MW")

    print("\n===== what the graph is worth at best =====")
    d = nat["own"] - nat["all"]
    print(f"national  own {nat['own']:.0f} -> all {nat['all']:.0f} MW   "
          f"({d:+.0f} MW, {100*d/nat['own']:+.2f}%)")
    for (r, a), (_, b) in zip(per_region["own"], per_region["all"]):
        print(f"  {r:<28} {a:7.1f} -> {b:7.1f}  ({100*(a-b)/a:+.2f}%)")


if __name__ == "__main__":
    main()
