# Changelog

## 0.3.0 — 2026-11-02

### Added

- `graphtoolbox.utils.set_seed(seed, deterministic=False)` seeds Python, NumPy and
  PyTorch and can request deterministic kernels.
- `seed` and `deterministic` arguments for `Trainer` (training loader, dropout)
  and `Optimizer` (TPE sampler, and trial `n` under `seed + n`).
- `RollingTrainer` honours `trainer_kwargs={"seed": ...}`, including the initial
  weights of the model it builds.
- `fused_adam` argument for `Trainer` and `Optimizer` (default False): the fused
  Adam kernel shortens a CPU epoch by up to a sixth, but its rounding differs
  from the default kernel in the last digits.
- `Optimizer` stops a trial after `patience` epochs (default 20) without a
  validation loss lower by `min_delta`; `patience=None` restores fixed-length
  trials. Each trial records `epochs` and `best_epoch` as user attributes.
- `Optimizer.optimize(pruner=...)` sets the Optuna pruner (default unchanged,
  `MedianPruner`).

### Changed

- Faster epochs, with results identical to 0.2.0 to the last digit: evaluation
  runs without autograd, batch losses are read from the device once per epoch,
  batches are tested for NaNs or empty targets only when the dataset holds some,
  and edge weights are loaded as float32. On the French load example, an MPS
  epoch is a quarter shorter, and validation 40 % shorter.
- `Optimizer` keeps the best weights of a trial in memory and writes them once,
  when the trial completes, instead of at every improvement; pruned trials
  write nothing.
- `Optimizer.optimize` warns when `timeout` stops the study before `n_trials`,
  and when no trial completed, where it used to fail on `study.best_trial`.

### Fixed

- `Optimizer` trained and validated every epoch on a single batch: the return
  of its epoch loop sat inside the loop. Hyperparameter searches run with 0.1.0
  to 0.2.0 selected their configurations on these truncated epochs. Trials now
  see the whole training set, and take correspondingly longer.
- `Optimizer` offers the candidate graphs in sorted order and ignores stray files
  in `graph_folder` (such as `.DS_Store`): the `os.listdir` order made a seeded
  search depend on the filesystem.
- `Trainer.train` falls back to `lr=1e-3` when `model_kwargs` has no `lr`, so a
  `RollingTrainer` without an explicit `lr` no longer fails with `KeyError`.

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
