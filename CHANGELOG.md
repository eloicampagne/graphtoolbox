# Changelog

## 0.2.0 — 2026-09-15

### Added

- Reusable ARIMA-X, GAM, LSTM, XGBoost, Chronos-Bolt, and Chronos-2 forecasting
  baselines.
- Edge-type support for RGCN, FastRGCN, and RGAT convolutions.
- Reusable attention, GNNExplainer, and grouped-ALE diagnostic helpers.
- Load and net-load benchmark, ablation, attribution, and significance workflows.
- Cached forecasts and scripts for reproducing the paper's reported statistics.

### Changed

- Prevented within-horizon leakage by freezing exponentially weighted features
  at each forecast origin.
- Centralized top-level tabular baseline fitting in
  `graphtoolbox.models.baseline`.

## 0.1.1 — 2026-07-10

- Restricted supported Python versions to 3.10–3.12.
- Improved installation and GPU-support documentation.

## 0.1.0

- Initial release.
