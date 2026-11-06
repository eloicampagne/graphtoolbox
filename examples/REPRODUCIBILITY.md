# Reproducing the GraphToolbox paper

Install the exact paper environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
```

The forecasts of the tables were computed with version 0.3.0, and the interpretability diagnostics, the reconciliations and the daily updates with version 0.3.1, whose training loop is unchanged for the configurations of the paper.

## From cached forecasts (no data, no training)

The committed forecast archives are sufficient to recompute the reported metrics and statistical tests. Run `python examples/reproduce_paper_statistics.py` from the repository root. It writes `results_cached_statistics/metrics.csv` and `results_cached_statistics/significance.json`. Its default 2,000 block-bootstrap resamples match the paper; pass `--n-boot 100` for a quick smoke test.

| Archive | Content |
|---|---|
| `load/forecasts_paper/load_sweep.npz` | regional forecasts of the 46 static operators, targets, uniform and MLpol aggregates |
| `load/forecasts_paper/load_temporal_oneshot.npz` | one-shot recurrent cells on load |
| `netload/forecasts_paper/netload_direct.npz` | direct net-load sweep and its aggregates |
| `netload/forecasts_paper/netload_decomposed.npz` | per-component forecasts, component targets, aggregates of the recombined forecasts |
| `load/forecasts_paper/load_seeds.npz`, `netload/forecasts_paper/netload_seeds.npz` | five-seed retraining of the leading models (seed table) |

`package_paper_forecasts.py` rebuilds these archives from the outputs of the runners below.

## Retraining

The load and net-load `train.csv`/`test.csv` inputs are not committed because of their size. They are needed only for retraining and must be placed beside the corresponding example (`load/`, `netload/`). Their temporal coverage and provenance are documented in the paper. Trained weights are attached to the matching GitHub release.

The runners of the paper live in `load/hpo_v030/` and cover both case studies. Run them from that folder with the repository root on `PYTHONPATH`. Every runner is resumable: it skips forecasts that already exist.

### Convolution coverage

The coverage table of the paper (53 of the 65 operators of PyTorch Geometric 2.6.1) is reproduced by `python examples/convolution_coverage.py` from the repository root, without data.

### Load

| Paper result | Runner |
|---|---|
| Hyperparameter search of LEConv, GATConv and APPNP (100 trials) | `run_hpo.py`, then `cap_at_100.py` |
| Table of the load sweep: one row | `sweep_row.py CONV CONFIG.json OUT.pt [--seed S] [--dir CKPT]` |
| Table of the load sweep: metrics, bootstrap, Diebold-Mariano tests, Model Confidence Set | `recompute_load_stats.py` (writes `load_sweep_v030.npz`, `stats_v030.json`) |
| Seed table, load row | `run_seeds.sh` |
| MinT reconciliation paragraph: top level fitted by the `Trainer` | `mint_all.py`, `mint_mixture.py`, `graph_xgb.py` |
| MinT onto TAM, pooled XGBoost and the regional MLP (last blocks of both tables) | `mint_tam.py load netload`, `mint_tam.py load --top xgb`, `mint_tam.py netload --top mlp`, then `mint_table_rows.py` |
| Daily update of GatedGraphConv (`RollingTrainer(update="replay")`) | `online_gnn.py select`, then `online_gnn.py test` |
| Regional TAM adapted by a Kalman filter | `../../tam_baselines.py --static load`, then `../kalman_graph.py effects`, `grid`, `stats` |
| Regional errors against pooled XGBoost (figure), error decomposition, five-seed average | `calendar_diagnosis.py`, then `figure_regional_vs_xgb_v030.py` |
| One-shot recurrent cells | `load_temporal_v030.py` |
| Training time per epoch (figure) | `../../netload/timing_bench.py`, then `figures_v030.py` |
| Load sweep (figure) | `figures_v030.py` |
| Aggregation over time (figure) | `figure_aggregation_v030.py` |
| Regional mixture map (figure) | `figure_regional_mixture_v030.py` |
| Operator colors shared by the two figures above | `operator_colors.py` |
| Identity-graph ablation | `../identity_graph_ablation.py` (prints the committed results, retrains with `--force`) |
| Baselines | `../benchmark_sota.py`, `../baselines_extra_load.py`, `../gam_regional_load.py`, `../../lstm_seeds.py`, `../../tam_baselines.py load` |
| XGBoost (pooled and regional) and MLP baselines | `nongraph_baselines.py` (models in `nongraph_models.py`) |

### Net-load

All net-load models receive the trajectories of temperature, wind and cloud cover over the forecast day (`--leads`, see `netload_data_v030.py`).

| Paper result | Runner |
|---|---|
| Direct and decomposed sweeps | `netload_sweep_v030.py --leads [--only direct\|decomposed]` |
| Net-load table and the statistics of its section | `netload_stats_v030.py leads` (writes `netload_stats_v030_leads.json`) |
| Seed table, net-load rows | `run_netload_seeds.sh` |
| Net-load figure (direct and decomposed per operator, error by time of day and by month) | `netload_figure_v030.py` |
| Recurrent cell (GConvGRU) | `netload_temporal_v030.py --leads --only GConvGRU` |
| Additive graph model and accumulated local effects | `additive_ale_v030.py --leads`, then `ale_figure_v030.py` |
| Baselines | `../../netload/baselines_netload.py`, `../../netload/baselines_extra_netload.py`, `../../netload/baselines_gam_regional.py`, `../../lstm_seeds.py`, `../../tam_baselines.py netload` |
| XGBoost (pooled and regional) and MLP baselines | `netload_nongraph.py` (models in `nongraph_models.py`) |

### Scale

| Paper result | Runner |
|---|---|
| Parameter count and inference latency on the 12 French regions and the 500 WEAVE-UK feeders | `footprint_v030.py` (writes `footprint_v030.csv`; needs the WEAVE-UK files beside the repository, see its docstring) |

### Interpretability

| Paper result | Runner |
|---|---|
| Attention of every layer and test day, matched to edges | `attention_v030.py CONV GRAPH [prior] [--seed S]` |
| Occlusion influence | `influence_v030.py CONV GRAPH [prior] [--seed S]` |
| Attention figure and its statistics | `attention_figure_v030.py` |
| Agreement across training seeds | `attention_seed_consistency.py` |
| Sender share and DTW prior | `attention_variants_stats.py` |

The figure uses `GATv2Conv` on the `space` graph with seeds 42, 0 and 1, and `GATv2Conv` on the `dtw` graph. The text also uses `GATConv` on `dtw` and both operators with `prior` on `dtw`.

The ARIMA-X, LSTM, GAM and Chronos baselines use `graphtoolbox/models/baseline.py`; XGBoost and the MLPs are defined in `load/hpo_v030/nongraph_models.py`, and TAM comes from the `tam-ml` package. MinT reconciliation, with either covariance estimator, is in `graphtoolbox.training.reconciliation` and `Trainer(reconcile_covariance=...)`. The attention and occlusion diagnostics are in `graphtoolbox.interpretability`.
