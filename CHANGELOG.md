# Changelog

## 0.3.1 — 2026-11-06

### Added

- `graphtoolbox.training.reconciliation`: MinT reconciliation of node forecasts with a forecast of
  their total, from held-out errors, usable on cached forecasts (`reconcile`). The weight matrix
  holds either the error variances (`"diag"`) or the full error covariance shrunk toward its
  diagonal with the intensity of Schafer and Strimmer (`"shrink"`, `shrinkage_intensity`).
  `Trainer(reconcile_covariance="shrink")` uses the second estimator; the default stays `"diag"`.
- `graphtoolbox.interpretability.additive_ale` and `additive_ale_from_dataset`: first-order
  accumulated local effects (Apley and Zhu, 2020) of an input feature of an
  `AdditiveGraphModel`, evaluated through the model. The feature of one node is moved across
  each quantile bin, the other inputs keep their observed values, and the change of the
  prediction is summed over the nodes. `AdditiveGraphModel.group_output` returns the output of
  one feature group.
- `Trainer(lam_national=...)`: adds the squared error of the national sum, with each node
  weighted by the slope of its target scaler, to the per-node loss, which ignores the
  covariance of the node errors that the national score depends on.
- `RollingTrainer(update="replay")`: after each window, `replay_steps` Adam steps on
  mini-batches that join the `replay_recent` most recent samples to samples drawn from the
  whole history, for a daily online update without refitting. `initial_model` starts from a
  trained model, and `warmup_on_val=True` first rolls through the validation samples.
- `myGNN(edge_weight_as_attr=True)` passes the graph weights as a one-dimensional
  edge attribute to the convolutions that take one (GATConv, GATv2Conv,
  TransformerConv, ...), which otherwise receive a constant attribute; `myGNN`
  now also forwards a user-supplied `edge_attr` to its layers. Off by default,
  so forecasts are unchanged.
- `myGNN(..., return_attention=True)` also returns `per_graph` (`[graphs, edges,
  heads]` per layer) and `edge_index`; the number of graphs is read from
  `num_graphs` (or `batch_size`) when no batch vector is passed.
- `graphtoolbox.interpretability`: `attention_by_snapshot` (attention of every
  layer on every snapshot, matched to edges), `occlusion_influence` (change of
  each node's forecast when the inputs of another node are replaced by a
  baseline), `target_unit_scale` (influence in target units),
  `neighbourhood_deviation` (deviation of an edge from the uniform share of its
  receiver) and `sender_variance_share` (share of the variance of per-edge
  values explained by the sender).

### Fixed

- `RollingTrainer` trains window k on the training set and every past test window; it kept
  only the previous window, so the test data of windows 0 to k-2 left the history. Every
  window also keeps the scalers and the graph of `dataset_train_0`: the warm-started weights
  were updated under refitted scalers, and the window datasets fell back to the default graph
  folder, hence to the identity graph when the graph was stored elsewhere.
- `GraphDataset(period="train")` keeps the scalers it is given instead of refitting them.
- The top-level model that `Trainer` fits for MinT reads the national mean of the covariates
  of the dataset at each step (`top_level_covariates=True` by default), weather included; it
  read only the lags and calendar of the aggregate series, so its forecasts ignored the
  weather that every other model of the pipeline reads.
- `Trainer` estimates the validation error variances of the MinT weights per node: the rows
  of a batch stack its graphs one after the other, and the former reshape mixed nodes and
  graphs, which degraded the reconciled forecasts.
- `Trainer` returns the group contributions of `AdditiveGraphModel` in target units as
  `slope * Y_g / G`, with the intercept in `group_intercept_test`, so that they sum to the
  prediction. They were unscaled as full forecasts (`slope * Y_g + offset`), which made the
  effect curves `G` times too large and mixed the per-node offsets into the binned means.
- The convolution coverage table counts the operators of
  `torch_geometric.nn.conv.__all__`: 53 of 65 run in `myGNN` (APPNP, PointTransformerConv
  and WLConvContinuous were missing, and the alias ECConv of NNConv was counted twice).
  `examples/convolution_coverage.py` reproduces the count.
- Attention of a batch of several graphs is matched to its edges. GATConv and
  GATv2Conv remove the self-loops of their input and append new ones after every
  other edge of the batch, so the former reshape of the weights into
  `[graphs, edges]` attached them to the wrong edges and mixed graphs. `myGNN`
  now matches each weight through the edge index the convolution returns
  (`graphtoolbox.models.gnn.attention_per_graph`).
- `Trainer(return_attention=True)` with `save=True` writes the attention files it
  announced; it previously wrote none.
- `dump_attention_batches` and both `load_attention_batches` share one cache
  format (`format == "per_graph"`: one `[graphs, edges, heads]` tensor per layer
  and the edge index it refers to). Legacy caches holding several graphs per
  file are refused, since their order cannot be recovered.

### Removed

- `Trainer(lam_reg=...)`: its pairwise penalty compared rows of the reshaped batch that were
  not nodes, and it pulled the forecasts of all nodes toward each other whatever the graph.
  A nonzero value now raises an error.

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
