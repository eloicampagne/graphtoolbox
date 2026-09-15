#!/usr/bin/env bash
# Resumable orchestrator for the full net-load test suite, mirroring the load example.
# Every step is cache-aware: re-running skips what is already done. Runs cheap -> expensive.
#
#   ./run_all_tests_netload.sh              # full rigour (like load): Optuna 200 trials, SOTA 500 epochs
#   OPTUNA_TRIALS=50 SOTA_EPOCHS=300 ./run_all_tests_netload.sh   # cheaper, documented settings
#
set -u
cd "$(dirname "$0")"
PY="${PY:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

OPTUNA_TRIALS="${OPTUNA_TRIALS:-200}"
OPTUNA_EPOCHS="${OPTUNA_EPOCHS:-100}"
SOTA_EPOCHS="${SOTA_EPOCHS:-500}"

echo "==== [1/5] Additive graph model + ALE ===="
$PY train_additive_ale_netload.py

echo "==== [2/5] Attention matrices (GAT / GATv2) ===="
$PY extract_attention_netload.py

echo "==== [3/5] Temporal convolutions (GConvGRU/DCRNN/TGCN/A3TGCN) ===="
$PY train_temporal_netload.py

echo "==== [4/5] SOTA benchmark (fair + sota, ${SOTA_EPOCHS} epochs) ===="
$PY benchmark_sota_netload.py --mode both --epochs "$SOTA_EPOCHS"

echo "==== [5/5] Optuna HPO (APPNP, GATConv, LEConv: ${OPTUNA_TRIALS} trials x ${OPTUNA_EPOCHS} epochs) ===="
$PY optimize_netload.py --trials "$OPTUNA_TRIALS" --epochs "$OPTUNA_EPOCHS"

echo "==== DONE. Re-run the notebook convolution sweep to pick up the tuned params (multi-config). ===="
