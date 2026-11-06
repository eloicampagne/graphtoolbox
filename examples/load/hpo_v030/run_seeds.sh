#!/bin/sh
# Five-seed retraining of the leading load operator at its table configuration (sweep defaults).
cd "$(dirname "$0")/.."
for s in 42 0 1 2 3; do
  "${PYTHON:-python3}" hpo_v030/sweep_row.py GatedGraphConv hpo_v030/old_configs/default.json \
    hpo_v030/preds/GatedGraphConv_seed$s.pt --dir hpo_v030/checkpoints/GatedGraphConv_seed$s --seed $s
done
