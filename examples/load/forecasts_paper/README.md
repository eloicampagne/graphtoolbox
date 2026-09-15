# Cached load forecasts

`load_sweep.npz` contains the unreconciled regional forecasts of the 47 static
operators, their node-level and national targets, and the uniform, top-level
MLpol, and bottom-up MLpol aggregates used by the paper.
`load_temporal_unrolled.npz` contains the four recurrent-cell forecasts and their
target. Arrays use `float32`; model arrays are shaped `[nodes, half-hours]`.

These caches are the inputs to the block bootstrap, Diebold--Mariano tests, and
Model Confidence Set. From the repository root, run
`python examples/reproduce_paper_statistics.py` to reproduce the analysis
without retraining.
