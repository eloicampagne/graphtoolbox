"""Shared net-load data pipeline for the rigorous test scripts.

Mirrors the feature engineering and dataset construction of
``example.ipynb`` (cells 6, 8 and 12) so that the temporal,
SOTA-benchmark, Optuna, additive and attention scripts all train on exactly the
same data, graph and reconciliation as the notebook.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from graphtoolbox.data.dataset import DataClass, GraphDataset

HERE = Path(__file__).resolve().parent
GRAPH_FOLDER = str(HERE.parent / "graph_representations")
OUT_CHANNELS = 48
ADJ = "dtw"

# Feature engineering (notebook cell 8)
FRANCE_LAT_DEG = 46.6


def _day_of_year(df):
    return pd.to_datetime(df["date"]).dt.dayofyear.astype(float)


def solar_declination(df):
    return 23.45 * np.sin(np.deg2rad(360.0 * (284.0 + _day_of_year(df)) / 365.0))


def day_length_hours(df, lat_deg=FRANCE_LAT_DEG):
    decl, lat = np.deg2rad(solar_declination(df)), np.deg2rad(lat_deg)
    cos_ws = np.clip(-np.tan(lat) * np.tan(decl), -1.0, 1.0)
    return (24.0 / np.pi) * np.arccos(cos_ws)


def extraterrestrial_daily(df, lat_deg=FRANCE_LAT_DEG):
    decl, lat = np.deg2rad(solar_declination(df)), np.deg2rad(lat_deg)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1.0, 1.0))
    dr = 1.0 + 0.033 * np.cos(np.deg2rad(360.0 * _day_of_year(df) / 365.0))
    return dr * (ws * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(ws))


COMPUTED_FEATURES = {
    "heat_deg": lambda df: (15.0 - df["temperature"]).clip(lower=0),
    "cool_deg": lambda df: (df["temperature"] - 15.0).clip(lower=0),
    "temp_sq":  lambda df: df["temperature"] ** 2,
    "wind_cubed":     lambda df: df["wind"].clip(lower=0) ** 3,
    "wind_cap_cubed": lambda df: df["wind_by_wind_power_weights"].clip(lower=0) ** 3,
    "clearness":    lambda df: (1.0 - df["nebulosity"]).clip(lower=0, upper=1),
    "clearness_sq": lambda df: ((1.0 - df["nebulosity"]).clip(lower=0, upper=1)) ** 2,
    "day_length": lambda df: day_length_hours(df),
    "solar_pot":  lambda df: extraterrestrial_daily(df),
}
ENG_LOAD  = ["heat_deg", "cool_deg", "temp_sq"]
ENG_WIND  = ["wind_cubed", "wind_cap_cubed"]
ENG_SOLAR = ["clearness", "clearness_sq", "day_length", "solar_pot"]

_features_load  = ["temperature", "temperature_lisse_990", "temperature_lisse_950", "wind", "nebulosity",
                   "toy", "year", "month", "day_type_jf", "day_type_week", "period_holiday", "period_hour_changed",
                   "period_holiday_zone_a", "period_holiday_zone_b", "period_holiday_zone_c",
                   "period_christmas", "period_summer"]
_features_wind  = ["wind", "wind_by_wind_power_weights", "toy", "year", "month"]
_features_solar = ["nebulosity", "nebulosity_by_solar_power_weights", "toy", "year", "month"]

FEATURES = sorted(set(_features_load) | set(_features_wind) | set(_features_solar)
                  | set(ENG_LOAD) | set(ENG_WIND) | set(ENG_SOLAR))


# Dataset construction (notebook cell 12)
def build_datasets(out_channels=OUT_CHANNELS, adj=ADJ):
    data_kwargs = {
        "node_var": "Region",
        "features_to_lag": {"NetLoad": (1, 48), "temperature": (-47, -1)},
        "dummies": ["tod", "day_type_week"],
        "day_inf_train": "2014-01-01", "day_sup_train": "2018-01-01",
        "day_inf_val": "2018-01-01", "day_sup_val": "2018-12-31",
        "day_inf_test": "2019-01-01", "day_sup_test": "2019-12-31",
        "computed_features": COMPUTED_FEATURES,
    }
    dataset_kwargs = {
        "batch_size": 32,
        "adj_matrix": adj,
        "features_base": (FEATURES
                          + [f"NetLoad_l{t}" for t in range(1, 49)]
                          + [f"temperature_l{t}" for t in range(-47, 0)]),
        "target_base": "NetLoad",
    }
    data = DataClass(path_train=str(HERE / "train.csv"), path_test=str(HERE / "test.csv"),
                     data_kwargs=data_kwargs, folder_config=str(HERE))
    common = dict(graph_folder=GRAPH_FOLDER, dataset_kwargs=dataset_kwargs, out_channels=out_channels)
    train = GraphDataset(data=data, period="train", **common)
    val = GraphDataset(data=data, period="val", scalers_feat=train.scalers_feat,
                       scalers_target=train.scalers_target, **common)
    test = GraphDataset(data=data, period="test", scalers_feat=train.scalers_feat,
                        scalers_target=train.scalers_target, **common)
    nodes = [str(n) for n in data.nodes]
    return train, val, test, nodes


# Metrics (notebook cell 6) — net-load can cross zero, so plain MAPE is masked.
def metrics_dict(p, t):
    p, t = np.asarray(p, float).ravel(), np.asarray(t, float).ravel()
    mask = np.abs(t) > 1e-6
    return {"rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "mae": float(np.mean(np.abs(p - t))),
            "mape": float(np.mean(np.abs((p[mask] - t[mask]) / t[mask])) * 100)}
