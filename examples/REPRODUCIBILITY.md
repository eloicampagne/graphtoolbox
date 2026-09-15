# Reproducing the GraphToolbox paper

Install the exact paper environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
```

The committed forecast archives are sufficient to recompute the reported metrics and statistical tests without the processed input tables or trained weights. Run `python examples/reproduce_paper_statistics.py` from the repository root. The load and net-load `train.csv`/`test.csv` inputs are intentionally not committed because of their size; they are needed only for retraining, and must be placed beside the corresponding example. Their temporal coverage and source provenance are documented in the paper.

The runner writes `results_cached_statistics/metrics.csv` and `results_cached_statistics/significance.json`. Its default 2,000 block-bootstrap resamples match the paper; pass `--n-boot 100` for a quick smoke test.

## Load

- `load/example.ipynb`: complete convolution sweep, aggregation and statistical
  analysis.
- `load/identity_graph_ablation.py`: the ten-operator DTW versus identity-graph
  ablation. It prints the committed results without data and retrains only with
  `--force`. Outputs are under `load/results_identity_graph_ablation/`.
- `load/benchmark_sota.py`, `load/baselines_extra_load.py`, and
  `load/gam_regional_load.py`: recurrent and classical baselines.
- `load/regen_explanation_figure.py`: regenerate the paper's attention figure
  from cached attention. Reusable attribution code is in
  `graphtoolbox.interpretability`.

## Net-load

- `netload/example.ipynb`: direct convolution sweep and aggregation.
- `netload/run_std_netload.sh`: standardized direct and decomposed sweeps.
- `netload/seed_variance.py`: five-seed retraining summarized in
  `netload/results_seed_variance/results.json`.
- `netload/train_additive_ale_netload.py`: intrinsically additive graph model,
  feature-level ALE values, grouped ALE summary, and the paper figure.
- `netload/run_all_tests_netload.sh`: cache-aware end-to-end experiment runner.

Model implementations for the classical baselines are centralized in `graphtoolbox/models/baseline.py`; the files above only construct the paper's datasets, formulas, and output tables.  Cached predictions needed by the bootstrap, Diebold--Mariano tests, and Model Confidence Set are kept beside the examples that consume them.  Large trained weights can instead be downloaded from the matching GitHub release.
