# Net-load test suite (mirror of the load example)

The same battery of tests run for `examples/load` is reproduced here for net-load,
on the same DTW graph, the same reconciliation, and the same 2019 test year.

## Data pipeline
`netload_pipeline.py` — shared feature engineering and dataset construction
(cells 6, 8, 12 of `example.ipynb`), reused by every script.

## Tests

| # | Test | Runner | Output | In notebook |
|---|------|--------|--------|-------------|
| 1 | Significance of the convolution sweep (block-bootstrap SE, Diebold-Mariano vs best, Model Confidence Set) | notebook cell "Significance testing" | printed table | ✅ runs on cached sweep |
| 2 | Significance of the aggregation rows (bootstrap CI, MCS, paired DM) | notebook cell "Significance of the aggregation rows" | LaTeX-ready | ✅ |
| 3 | Temporal convolutions (GConvGRU, DCRNN, TGCN, A3TGCN) | `train_temporal_netload.py` | `results_temporal_netload/*.pt` | ✅ appendix |
| 4 | SOTA benchmark (fair + recurrent regime, block-bootstrap CIs) | `benchmark_sota_netload.py --mode both` | `benchmark_sota_netload*.csv/.npz` | ✅ appendix |
| 5 | Optuna HPO (APPNP, GATConv, LEConv) | `optimize_netload.py` | `results_optim_{Conv}/myGNN_best.json` | feeds sweep |
| 6 | Additive graph model + ALE feature importance | `train_additive_ale_netload.py` | `results_additive_netload/` | ✅ appendix |
| 7 | Attention matrices (GATConv, GATv2Conv) | `extract_attention_netload.py` | `attention_matrix/{model}_dtw/` | ✅ appendix |
| 8 | Multi-config convolution sweep | notebook cell 34 | picks up `results_optim_*` automatically | ✅ |

## Run everything (resumable, cache-aware)
```bash
./run_all_tests_netload.sh                                   # full rigour (Optuna 200, SOTA 500 epochs)
OPTUNA_TRIALS=50 SOTA_EPOCHS=300 ./run_all_tests_netload.sh  # cheaper, documented settings
```
Each step skips work already cached, so the chain can be stopped and resumed.

## Notes
- Net-load crosses zero, so plain MAPE is masked away from zero; RMSE / nRMSE are the headline.
- CPU beats MPS by ~5x for the recurrent temporal cells on this 12-node graph; the temporal
  and attention scripts force CPU via `set_device('cpu')`.
- Significance tests read cached predictions and run in seconds — no retraining.
