"""TAM (Time series Additive Model, EDF-Lab/tam) references for the load and net-load benchmarks.

Mirrors the pygam references of the paper with the same inputs and the same structure, so that
only the additive-model implementation changes:
  - regional: one model per region and per half-hour slot, summed to the national series
    (load/gam_regional_load.py, netload/baselines_gam_regional.py);
  - national: one model per half-hour slot on the national aggregate, load summed and covariates
    averaged over the regions (netload/baselines_netload.py:build_national).
TAM fits the 48 slots jointly as independent groups (group_col) and selects one smoothing
parameter per effect by GCV (StaticTAM.auto_fit). Each static model is also coupled with an
AdaptiveTAM over a sliding window. The variant (static or adaptive and its window, national or
regional) is selected on 2018 with models fitted before 2018, then refitted for the 2019 test. The paper keeps the
selected static variant, since every model of the benchmark is trained once before the test year; the adaptive
variants, which refit a correction every day of 2019, are written to results_tam.csv for information.
Computation runs on CPU in float64.
Evaluated on the 2019 national targets of the sweeps (target.pt).

Usage: python tam_baselines.py [--static] [load|netload ...]   (pip install tam-ml)
--static fits the static models only and writes tam_static_forecasts.npz (2018 and 2019 forecasts of the
regional model), the top level of the MinT reconciliation.
"""
import os
import sys
import time
import warnings

import torch
torch.backends.mps.is_available = lambda: False  # TAM falls back to float32 on MPS
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tam as ta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from graphtoolbox.models.baseline import mape, rmse, smape  # noqa: E402

TASKS = {
    "load": dict(
        folder="load", target_pt="results_all_convolutions/target.pt", y="load", slot="Instant",
        train=("2015-01-01", "2019-01-01"), score=mape, score_name="MAPE",
        cols=["date", "Region", "load", "temp", "temp_liss_fort", "temp_liss_faible", "nebu", "wind",
              "Instant", "Posan", "JourSemaine", "JourFerie", "DayType", "Weekend", "offset"],
        mean_cols=["temp", "temp_liss_fort", "temp_liss_faible", "nebu", "wind"],
        rename={"offset": "period_code"}, weekend="Weekend",
        meteo=["temp", "nebu", "wind"],
        formula=("load ~ s(temp) + te(s(temp), c(Weekend)) + s(temp_liss_fort) + s(temp_liss_faible)"
                 " + te(s(Posan), c(JourSemaine)) + f(Posan, cyclic=True) + c(JourSemaine) + c(JourFerie)"
                 " + c(DayType) + c(Weekend) + s(nebu) + s(wind) + l(period_code)")),
    "netload": dict(
        folder="netload", target_pt="results_all_convolutions_netload/target.pt", y="NetLoad", slot="tod",
        train=("2014-01-01", "2018-12-31"), score=smape, score_name="sMAPE",
        cols=["date", "Region", "NetLoad", "temperature", "temperature_lisse_950", "temperature_lisse_990",
              "wind_by_wind_power_weights", "nebulosity_by_solar_power_weights", "tod", "toy",
              "day_type_week", "day_type_jf", "year", "period_holiday", "period_hour_changed"],
        mean_cols=["temperature", "temperature_lisse_950", "temperature_lisse_990",
                   "wind_by_wind_power_weights", "nebulosity_by_solar_power_weights"],
        rename={}, weekend=None, meteo=["temperature", "wind", "nebulosity"],
        formula=("NetLoad ~ s(temperature) + te(s(temperature), c(weekend)) + s(temperature_lisse_950)"
                 " + s(temperature_lisse_990)"
                 " + te(s(toy), c(day_type_week)) + c(day_type_week) + c(day_type_jf) + c(period_holiday)"
                 " + c(period_hour_changed) + f(toy, cyclic=True)"
                 " + s(wind_by_wind_power_weights) + te(f(toy, cyclic=True), l(wind_by_wind_power_weights))"
                 " + s(nebulosity_by_solar_power_weights)"
                 " + te(f(toy, cyclic=True), l(nebulosity_by_solar_power_weights)) + l(year)")),
}


WINDOWS = (182, 365, 730)  # days of the sliding window of AdaptiveTAM, selected on 2018


def with_adaptive_inputs(cfg, base, full):
    """Adds the inputs of the adaptive correction to ``full``: the static effects summed into a calendar and a
    weather group, and the residual of the static model one and seven days before, within each half-hour slot
    (the lags of AutoTAM). The residual of day D-1 is known at the midnight origin of day D."""
    dec = base.decompose_prediction(full.copy())
    effects = [c for c in dec.columns if c.startswith("effect_") and c != "effect_offset"]  # effect_offset: intercept
    is_meteo = lambda c: any(v in c for v in cfg["meteo"])
    full = full.copy()
    full["grp_meteo"] = dec[[c for c in effects if is_meteo(c)]].sum(axis=1).to_numpy()
    full["grp_calendar"] = dec[[c for c in effects if not is_meteo(c)]].sum(axis=1).to_numpy()
    residual = full[cfg["y"]].to_numpy() - dec[[c for c in dec.columns if c.startswith("effect_")]].sum(axis=1).to_numpy()
    full["_res"] = residual
    by_slot = full.sort_values("date").groupby(cfg["slot"])["_res"]
    for lag in (1, 7):
        full[f"Residual{cfg['y']}_lag_{lag}"] = by_slot.shift(lag)
    return full.drop(columns="_res")


def variants(cfg, frame, split, n_eval, windows=WINDOWS):
    """Static TAM and AdaptiveTAM (one per window) fitted before ``split``, forecasting ``n_eval`` steps from it.

    AdaptiveTAM corrects the residual of the static model by a linear model of its effects, refitted every day on
    the last W days of each half-hour slot. With one row per slot and day (steps_per_period=1, horizon_steps=1),
    the forecast of day D uses the days up to D-1 only, so it respects the midnight origin of the benchmark."""
    frame = frame.sort_values("date").reset_index(drop=True)
    train = frame[(frame["date"] >= cfg["train"][0]) & (frame["date"] < split)]
    evalp = frame[frame["date"] >= split].iloc[:n_eval]
    base = ta.StaticTAM(formula=cfg["formula"], group_col=cfg["slot"], date_col="date").auto_fit(train)
    out = base.predict(evalp)
    col = next(c for c in out.columns if c.startswith("Estimated"))
    preds = {"static": evalp[["date"]].merge(out[["date", col]], on="date", how="left")[col].to_numpy(float)}
    full = pd.concat([train, evalp]).reset_index(drop=True)
    full = with_adaptive_inputs(cfg, base, full)
    formula = (f"Residual{cfg['y']} ~ l(grp_calendar) + l(grp_meteo)"
               f" + l(Residual{cfg['y']}_lag_1) + l(Residual{cfg['y']}_lag_7)")
    full = full.dropna(subset=[f"Residual{cfg['y']}_lag_7"])
    for W in windows:
        adaptive = ta.AdaptiveTAM(adaptive_formula=formula, update_interval_periods=1, training_window_periods=W,
                                  steps_per_period=1, base_model=base, horizon_steps=1,
                                  group_col=cfg["slot"], date_col="date")
        o = adaptive.predict_online(full)
        preds[f"adaptive{W}"] = evalp[["date"]].merge(
            o[["date", f"AdaptedEstimated{cfg['y']}"]], on="date", how="left")[f"AdaptedEstimated{cfg['y']}"].to_numpy(float)
    for k, v in preds.items():
        assert not np.isnan(v).any(), f"TAM returned no forecast for some steps ({k})"
    return preds, evalp[cfg["y"]].to_numpy(float)


def national_frame(cfg, df):
    g = df.groupby("date")
    nat = g[cfg["mean_cols"]].mean()
    nat[cfg["y"]] = g[cfg["y"]].sum()
    for c in df.columns:
        if c not in nat.columns and c not in ("date", "Region"):
            nat[c] = g[c].first()
    return nat.reset_index()


def run_split(cfg, df, split, n_eval, windows=WINDOWS):
    """Forecasts of every variant on ``n_eval`` steps from ``split``, national and regional (summed)."""
    regional, truth = None, 0.0
    for r in sorted(df["Region"].unique()):
        p, y = variants(cfg, df[df["Region"] == r], split, n_eval, windows)
        regional = p if regional is None else {k: regional[k] + p[k] for k in p}
        truth = truth + y
        print(f"  {r:<28} static RMSE {rmse(p['static'], y):6.0f} MW", flush=True)
    national, y_nat = variants(cfg, national_frame(cfg, df), split, n_eval, windows)
    assert np.max(np.abs(y_nat - truth)) < 1e-6
    preds = {f"{k} national": v for k, v in national.items()}
    preds.update({f"{k} regional": v for k, v in regional.items()})
    return preds, truth


def run(task, static_only=False):
    cfg = TASKS[task]
    d = os.path.join(HERE, cfg["folder"])
    target = torch.load(os.path.join(d, cfg["target_pt"]), weights_only=False).numpy().astype(float)
    T = len(target)
    df = pd.concat([pd.read_csv(os.path.join(d, "train.csv"), usecols=cfg["cols"]),
                    pd.read_csv(os.path.join(d, "test.csv"), usecols=cfg["cols"])], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.rename(columns=cfg["rename"])
    if cfg["weekend"] is None:
        df["weekend"] = (df["day_type_week"] >= 5).astype(int)  # 0 = Monday ... 6 = Sunday

    t0 = time.time()
    print(f"[{task}] validation: fit before 2018, forecast 2018")
    windows = () if static_only else WINDOWS
    val, y_val = run_split(cfg, df, "2018-01-01", 365 * 48, windows)
    print(f"[{task}] test: fit before {cfg['train'][1]}, forecast 2019")
    df_test = df[df["date"] < cfg["train"][1]]
    df_test = pd.concat([df_test, df[df["date"] >= "2019-01-01"]])
    test, y_test = run_split(cfg, df_test, "2019-01-01", T, windows)
    assert np.max(np.abs(y_test - target)) < 1e-6, "national truth differs from target.pt"

    # The benchmark trains every model once, before the test year: only the static variants qualify, and the adaptive
    # ones, which refit a correction every day of the test year, are reported for information.
    if static_only:
        # Top-level forecasts of the MinT reconciliation (load/hpo_v030/mint_tam.py): 2018 for the error variance.
        np.savez_compressed(os.path.join(d, "tam_static_forecasts.npz"), val=val["static regional"],
                            y_val=y_val, test=test["static regional"])
        return
    best = min((k for k in val if k.startswith("static")), key=lambda k: rmse(val[k], y_val))
    print(f"\n=== {task} ({time.time() - t0:.0f} s); static variant selected on 2018: {best} ===")
    print(f"{'variant':<24} {'RMSE 2018':>10} {cfg['score_name'] + ' 2019':>11} {'RMSE 2019':>10}")
    for k in val:
        print(f"{k:<24} {rmse(val[k], y_val):10.0f} {cfg['score'](test[k], target):10.2f}% {rmse(test[k], target):10.0f}"
              + ("   <- selected" if k == best else ""))
    out = pd.DataFrame(test)
    out.attrs["selected"] = best
    out.to_csv(os.path.join(d, "results_tam.csv"), index=False)
    with open(os.path.join(d, "results_tam_selected.txt"), "w") as fh:
        fh.write(best + "\n")


if __name__ == "__main__":
    static_only = "--static" in sys.argv
    for task in [a for a in sys.argv[1:] if a != "--static"] or ["load", "netload"]:
        run(task, static_only)
