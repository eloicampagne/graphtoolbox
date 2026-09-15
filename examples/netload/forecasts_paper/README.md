# Cached net-load forecasts

`netload_direct.npz` contains the unreconciled regional forecasts of the ten
direct GNNs, their node-level and national targets, and the uniform and MLpol
aggregates. `netload_decomposed.npz` contains each operator/component forecast,
the component targets, the shared net-load targets, and the aggregates. Arrays
use `float32`; model arrays are shaped `[nodes, half-hours]`.

These caches reproduce the paper's metrics and block-bootstrap intervals without
retraining. From the repository root, run
`python examples/reproduce_paper_statistics.py`.
