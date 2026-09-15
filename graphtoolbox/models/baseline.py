"""Reusable non-graph forecasting baselines used by the examples."""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


def rmse(prediction, target) -> float:
    """Root mean squared error."""
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def mape(prediction, target, eps: float = 1e-6) -> float:
    """Mean absolute percentage error, in percent."""
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    denominator = np.clip(np.abs(target), eps, None)
    return float(100.0 * np.mean(np.abs(prediction - target) / denominator))


def smape(prediction, target, eps: float = 1e-8) -> float:
    """Symmetric mean absolute percentage error, in percent."""
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    denominator = np.abs(prediction) + np.abs(target) + eps
    return float(100.0 * np.mean(2.0 * np.abs(prediction - target) / denominator))


def arimax_forecast(train: pd.DataFrame, test: pd.DataFrame,
                    features: Sequence[str], target: str, slot_col: str,
                    order=(2, 0, 2)) -> np.ndarray:
    """Fit one regression-with-ARMA-errors model per forecast slot."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    prediction = np.empty(len(test), dtype=float)
    for slot in sorted(train[slot_col].unique()):
        train_slot = train[train[slot_col] == slot]
        test_mask = test[slot_col].to_numpy() == slot
        if not test_mask.any():
            continue
        varying = [name for name in features
                   if train_slot[name].to_numpy(dtype=float).std() > 1e-8]
        x_train = train_slot[varying].to_numpy(dtype=float)
        x_test = test.loc[test_mask, varying].to_numpy(dtype=float)
        y_train = train_slot[target].to_numpy(dtype=float)
        try:
            model = SARIMAX(y_train, exog=x_train, order=order, trend="c",
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            result = model.fit(disp=False, maxiter=50, method="lbfgs")
        except Exception:
            model = SARIMAX(y_train, exog=x_train, order=(1, 0, 0), trend="c",
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            result = model.fit(disp=False, maxiter=50, method="lbfgs")
        forecast = result.get_forecast(steps=int(test_mask.sum()),
                                       exog=x_test).predicted_mean
        prediction[test_mask] = np.asarray(forecast, dtype=float)
    return prediction


class CovariateSeq2Seq(nn.Module):
    """Small LSTM encoder-decoder used as a covariate-aware baseline."""

    def __init__(self, n_features: int, hidden_channels: int = 64,
                 horizon: int = 48):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.LSTM(1 + n_features, hidden_channels,
                               batch_first=True)
        self.decoder = nn.LSTM(n_features, hidden_channels, batch_first=True)
        self.readout = nn.Linear(hidden_channels, 1)

    def forward(self, history, future_covariates):
        _, state = self.encoder(history)
        decoded, _ = self.decoder(future_covariates, state)
        return self.readout(decoded).squeeze(-1)


def lstm_forecast(train: pd.DataFrame, test: pd.DataFrame,
                  features: Sequence[str], target: str, step: int = 48,
                  hist_len: int = 336, hidden: int = 64, epochs: int = 40,
                  batch: int = 64, lr: float = 1e-3, seed: int = 42,
                  device: str = "cpu") -> np.ndarray:
    """Train a day-ahead seq2seq LSTM with known future covariates."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    full = pd.concat([train, test], ignore_index=False)
    features_array = full[list(features)].to_numpy(dtype=np.float32)
    target_array = full[target].to_numpy(dtype=np.float32)
    n_train = len(train)
    feature_mean = features_array[:n_train].mean(0)
    feature_std = features_array[:n_train].std(0) + 1e-6
    target_mean = target_array[:n_train].mean()
    target_std = target_array[:n_train].std() + 1e-6
    x = (features_array - feature_mean) / feature_std
    y = (target_array - target_mean) / target_std

    def make(starts: Iterable[int]):
        histories, futures, targets = [], [], []
        for start in starts:
            if start - hist_len < 0:
                continue
            histories.append(np.concatenate(
                [y[start - hist_len:start, None], x[start - hist_len:start]],
                axis=1))
            futures.append(x[start:start + step])
            targets.append(y[start:start + step])
        return (torch.tensor(np.asarray(histories)),
                torch.tensor(np.asarray(futures)),
                torch.tensor(np.asarray(targets)))

    train_starts = range(hist_len, n_train - step + 1, step)
    test_starts = range(n_train, len(full) - step + 1, step)
    history_train, future_train, target_train = make(train_starts)
    model = CovariateSeq2Seq(len(features), hidden, step).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(history_train))
        for start in range(0, len(history_train), batch):
            indices = permutation[start:start + batch]
            optimizer.zero_grad()
            output = model(history_train[indices].to(device),
                           future_train[indices].to(device))
            loss = loss_function(output, target_train[indices].to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    prediction = np.full(len(test), np.nan, dtype=float)
    with torch.no_grad():
        for start in test_starts:
            history = np.concatenate(
                [y[start - hist_len:start, None], x[start - hist_len:start]],
                axis=1)
            output = model(torch.tensor(history[None]).to(device),
                           torch.tensor(x[start:start + step][None]).to(device))
            values = output.squeeze(0).cpu().numpy() * target_std + target_mean
            offset = start - n_train
            prediction[offset:offset + step] = values
    if np.isnan(prediction).any():
        valid = np.flatnonzero(~np.isnan(prediction))
        if not len(valid):
            raise ValueError("The test set contains no complete forecast window.")
        prediction[valid[-1] + 1:] = prediction[valid[-1]]
    return prediction


def chronos2_forecast(full: pd.DataFrame, t0: int, n_test: int,
                      features: Sequence[str], target: str, step: int = 48,
                      ctx: int = 1024, model_id: str = "amazon/chronos-2",
                      batch_size: int = 64) -> np.ndarray:
    """Run Chronos-2 zero-shot with past and known-future covariates."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from chronos.chronos2 import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map="cpu")
    full = full.reset_index(drop=True)
    t0 = int(t0)
    timestamps = pd.Series(pd.date_range(
        start=pd.to_datetime(full["timestamp"]).iloc[0],
        periods=len(full), freq="30min"))
    context_rows, future_rows = [], []
    for number, offset in enumerate(range(0, n_test, step)):
        end = t0 + offset
        begin = max(0, end - ctx)
        item_id = f"d{number}"
        context_rows.append(pd.DataFrame({
            "item_id": item_id,
            "timestamp": timestamps.iloc[begin:end].to_numpy(),
            "target": full[target].iloc[begin:end].to_numpy(dtype=float),
            **{name: full[name].iloc[begin:end].to_numpy(dtype=float)
               for name in features},
        }))
        horizon = min(step, n_test - offset)
        future_rows.append(pd.DataFrame({
            "item_id": item_id,
            "timestamp": timestamps.iloc[end:end + horizon].to_numpy(),
            **{name: full[name].iloc[end:end + horizon].to_numpy(dtype=float)
               for name in features},
        }))
    output = pipeline.predict_df(
        pd.concat(context_rows, ignore_index=True),
        future_df=pd.concat(future_rows, ignore_index=True),
        prediction_length=step, quantile_levels=[0.5], batch_size=batch_size)
    prediction = np.empty(n_test, dtype=float)
    for number, offset in enumerate(range(0, n_test, step)):
        horizon = min(step, n_test - offset)
        values = output.loc[output["item_id"] == f"d{number}",
                            "predictions"].to_numpy()[:horizon]
        prediction[offset:offset + horizon] = values
    return prediction


def xgboost_forecast(train: pd.DataFrame, test: pd.DataFrame,
                     features: Sequence[str], target: str,
                     **model_kwargs) -> np.ndarray:
    """Fit an XGBoost regressor and return its test forecast."""
    import xgboost as xgb

    defaults = dict(n_estimators=2000, max_depth=5, eta=0.025,
                    subsample=0.9, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1)
    defaults.update(model_kwargs)
    model = xgb.XGBRegressor(**defaults)
    model.fit(train[list(features)].to_numpy(), train[target].to_numpy(dtype=float))
    return model.predict(test[list(features)].to_numpy())


def gam_forecast_by_slot(train: pd.DataFrame, test: pd.DataFrame,
                         features: Sequence[str], target: str, slot_col: str,
                         formula, max_iter: int = 1000) -> np.ndarray:
    """Fit one supplied ``pygam`` formula per forecast slot."""
    from pygam import LinearGAM

    prediction = np.empty(len(test), dtype=float)
    for slot in sorted(train[slot_col].unique()):
        train_slot = train[train[slot_col] == slot]
        test_mask = test[slot_col].to_numpy() == slot
        if not test_mask.any():
            continue
        model = LinearGAM(formula, max_iter=max_iter).fit(
            train_slot[list(features)].to_numpy(),
            train_slot[target].to_numpy(dtype=float))
        prediction[test_mask] = model.predict(
            test.loc[test_mask, list(features)].to_numpy())
    return prediction


def chronos_bolt_forecast(series, test_start: int, n_test: int,
                          step: int = 48, context: int = 1024,
                          model_id: str = "amazon/chronos-bolt-small",
                          batch_size: int = 32) -> np.ndarray:
    """Run a rolling zero-shot Chronos-Bolt forecast."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from chronos.chronos_bolt import ChronosBoltPipeline

    pipeline = ChronosBoltPipeline.from_pretrained(model_id, device_map="cpu")
    values = np.asarray(series, dtype=np.float32)
    contexts = []
    for offset in range(0, n_test, step):
        end = test_start + offset
        contexts.append(torch.tensor(values[max(0, end - context):end]))
    chunks = []
    for start in range(0, len(contexts), batch_size):
        quantiles, _ = pipeline.predict_quantiles(
            contexts[start:start + batch_size], prediction_length=step,
            quantile_levels=[0.5])
        chunks.append(quantiles[:, :, 0].cpu().numpy())
    return np.concatenate(chunks, axis=0).reshape(-1)[:n_test]


def fit_tabular_baseline(model_type: str, x_train, y_train, x_test,
                         x_validation=None, y_validation=None):
    """Fit a tabular baseline used for top-level reconciliation.

    Parameters are already scaled by the caller. The return value is
    ``(model, train_prediction, validation_prediction, test_prediction)``.
    """
    if model_type == "ridge":
        from sklearn.linear_model import RidgeCV

        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        fit_kwargs = {}
    elif model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=20,
            n_jobs=-1, random_state=42)
        fit_kwargs = {}
    elif model_type == "xgb":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for model_type='xgb'. "
                "Install it with: pip install xgboost") from exc
        evaluation = ([(x_validation, y_validation)]
                      if x_validation is not None else None)
        model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbosity=0,
            early_stopping_rounds=20 if evaluation else None)
        fit_kwargs = {"eval_set": evaluation, "verbose": False}
    elif model_type == "gam":
        try:
            from pygam import LinearGAM, l, s
        except ImportError as exc:
            raise ImportError(
                "pygam is required for model_type='gam'. "
                "Install it with: pip install pygam") from exc
        terms = (l(0) + l(1) + l(2) + l(3) + s(4) + s(5)
                 + s(6) + s(7) + s(8) + s(9))
        model = LinearGAM(terms)
        model.gridsearch(x_train, y_train)
        return (model, model.predict(x_train),
                model.predict(x_validation) if x_validation is not None else None,
                model.predict(x_test))
    else:
        raise ValueError("model_type must be 'ridge', 'rf', 'xgb', or 'gam'")

    model.fit(x_train, y_train, **fit_kwargs)
    return (model, model.predict(x_train),
            model.predict(x_validation) if x_validation is not None else None,
            model.predict(x_test))
