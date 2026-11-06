# Cached load forecasts

`load_sweep.npz` contains the unreconciled regional forecasts of the 46 static
operators, their node-level and national targets, and the uniform, top-level
MLpol, and bottom-up MLpol aggregates used by the paper.
`load_temporal_oneshot.npz` contains the four recurrent cells run one-shot (a
single update on the full feature vector, as a static convolution does) and their
targets. Arrays use `float32`; model arrays are shaped `[nodes, half-hours]`.

These caches are the inputs to the block bootstrap, Diebold--Mariano tests, and
Model Confidence Set. From the repository root, run
`python examples/reproduce_paper_statistics.py` to reproduce the analysis
without retraining.
