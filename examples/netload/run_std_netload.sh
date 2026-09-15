#!/usr/bin/env bash
# Standardized net-load rerun: direct and decomposed, same 10 convolutions,
# same config (hidden 64, 1 layer, 300 epochs, batch 16, DTW, patience 30),
# for an honest same-budget comparison. Resumable (per-conv cache).
set -u
cd "$(dirname "$0")"
PY="${PY:-/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "==== DIRECT GNN (h64/l1/300ep) ===="
$PY fair_sweep_netload.py

echo "==== DECOMPOSED GNN, one per component (h64/l1/300ep) ===="
$PY decomp_sweep_netload.py

echo "==== STANDARDIZED NET-LOAD RERUN DONE ===="
