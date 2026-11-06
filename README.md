<p><img src="docs/source/_static/banner_toolbox.png" alt="logo" width="1000" /></p>

[![PyPI version](https://img.shields.io/pypi/v/graphtoolbox.svg)](https://pypi.org/project/graphtoolbox/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![CI](https://github.com/eloicampagne/GraphToolbox/actions/workflows/ci.yml/badge.svg)
![Maintainer](https://img.shields.io/badge/maintainer-E.Campagne-blue) 
[![Documentation](https://img.shields.io/badge/Documentation-eloicampagne.fr%2Fgraphtoolbox-green)](https://eloicampagne.fr/graphtoolbox)

# GraphToolbox

GraphToolbox is a Python framework on PyTorch Geometric for forecasting time series observed on the nodes of a graph, such as regional electricity load. One configuration-driven pipeline covers graph construction, the choice among the convolution operators of PyTorch Geometric, training, reconciliation, expert aggregation, significance testing and interpretability, so that a benchmark changes one choice at a time.

## Features

- **Data and graphs**: `DataClass` and `GraphDataset` build lagged, scaled node samples with train, validation and test splits; `GraphBuilder` derives the adjacency from coordinates or from the series themselves (spatial distance, correlation, precision, GL3SR, dynamic time warping), any precomputed adjacency stored as a matrix is loaded by name, and a graph can be selected on validation error.
- **Models**: `myGNN` runs 53 of the 65 convolutions of PyTorch Geometric through one adapter (see the table below), `TemporalGNN` the recurrent cells of PyTorch Geometric Temporal, and `AdditiveGraphModel` an additive graph network whose forecast splits into feature-group contributions.
- **Training**: `Trainer` (early stopping, forecasting metrics, MinT reconciliation), `Optimizer` for Optuna searches over the model and the graph, and `RollingTrainer` for rolling refits or daily online updates of a trained model.
- **Reconciliation**: MinT combines the node forecasts with a forecast of their total, fitted by the framework or supplied by the user, with a diagonal or a shrunk full error covariance (`graphtoolbox.training.reconciliation`).
- **Aggregation**: online expert aggregation (MLpol, EWA, BOA) across architectures or seeds, on the total or node by node (`graphtoolbox.aggregation`).
- **Evaluation**: Diebold-Mariano tests with Holm correction, moving-block bootstrap and Model Confidence Set, on cached forecasts (`graphtoolbox.evaluation`).
- **Interpretability**: attention matched to edges, occlusion influence, accumulated local effects, and maps of edge attributions (`graphtoolbox.interpretability`).
- **Baselines**: ARIMA-X, GAM, LSTM, XGBoost, Chronos-Bolt and Chronos-2 (`graphtoolbox.models.baseline`).

## Convolutions 
We benchmarked the entire collection of `torch_geometric.nn.conv` layers against `myGNN`, evaluating whether each operator can be instantiated and run end-to-end with standard node-feature inputs and a homogeneous graph structure.

Legend:
- 🟢 = Working (fully compatible with `myGNN`)
- 🔴 = Skipped (requires the `dgNN` package, not available on all platforms)
- ⚪️ = Skipped (requires CUDA-specific dependencies or device-restricted libraries)

| Convolution Type        | Status | Convolutions |
|-------------------------|--------|--------------|
| Propagation             | 🟢 | SimpleConv, GCNConv, ChebConv, SGConv, SSGConv, APPNP, GCN2Conv, ARMAConv, TAGConv, LGConv, ClusterGCNConv, MixHopConv, PANConv |
| Attention               | 🟢 | GATConv, GATv2Conv, SuperGATConv, TransformerConv, AGNNConv, RGATConv, FAConv, DNAConv, GPSConv, PointTransformerConv |
| Gating and modulation   | 🟢 | GatedGraphConv, ResGatedGraphConv, FiLMConv |
| Relational and edge-conditioned | 🟢 | RGCNConv, FastRGCNConv, NNConv, CGConv, GMMConv, SplineConv, FeaStConv, PDNConv, GINEConv |
| Point-cloud and dynamic graphs | 🟢 | EdgeConv, DynamicEdgeConv, GravNetConv, XConv |
| Other aggregators       | 🟢 | SAGEConv, GraphConv, MFConv, LEConv, PNAConv, GENConv, EGConv, GINConv, GeneralConv, DirGNNConv, AntiSymmetricConv, SignedConv, WLConv, WLConvContinuous |
| Missing optional deps   | 🔴 | FusedGATConv (`dgNN`) |
| Heterogeneous graphs    | ⚪️ | HANConv, HGTConv, HEATConv, HeteroConv |
| Point-cloud             | ⚪️ | PointNetConv, PointGNNConv, PPFConv |
| CuGraph (CUDA only)     | ⚪️ | CuGraphGATConv, CuGraphRGCNConv, CuGraphSAGEConv |
| Hypergraph              | ⚪️ | HypergraphConv |

Detailed statistics per status:

| Status | Count | Percentage |
|--------|-------|------------|
| 🟢 | **53** | **81.5 %** |
| 🔴 | **1**  | **1.5 %**  |
| ⚪️ | **11** | **16.9 %** |
| **Total Tested** | **65** | **100 %** |

Counts follow `torch_geometric.nn.conv.__all__` in PyTorch Geometric 2.6.1; the aliases ECConv (NNConv) and PointConv (PointNetConv) are not counted twice.

FusedGATConv is not broken; it requires the `dgNN` package which is not available on all platforms. Installing it will make it pass.
Installing `torch-cluster`, `torch-sparse`, and `torch-spline-conv` unlocked DynamicEdgeConv, GravNetConv, XConv, PANConv, and SplineConv.

If you spot a missing convolution, find an incompatibility, or want to help extend support, contributions are warmly welcomed! Feel free to open an issue or submit a PR so we can improve these results.


## Installation

The package is available on [PyPI](https://pypi.org/project/graphtoolbox/):

```sh
pip install graphtoolbox
```

Alternatively, install the latest development version from source:

```sh
git clone git@github.com:eloicampagne/GraphToolbox.git
cd GraphToolbox
pip install .
```

The geographic-map figures rely on the deprecated `basemap` package, which is kept out of the core dependencies; install it on demand with `pip install graphtoolbox[maps]`.

### GPU support

GraphToolbox runs on CPU, CUDA, or Apple MPS and selects the device automatically at run time (CUDA first, then MPS, then CPU). Override it with the `GRAPHTOOLBOX_DEVICE` environment variable or `graphtoolbox.training.set_device(...)`. Note that `pip install graphtoolbox` does **not** pin a CUDA build: it installs the default PyTorch wheel for your platform, which is the CUDA build on Linux (used automatically if a compatible GPU and driver are present) and the CPU/MPS build on macOS. To force a specific variant, install `torch` yourself from the [appropriate PyTorch index](https://pytorch.org/get-started/locally/) before installing GraphToolbox.

To unlock the full set of supported convolutions, install the optional PyTorch Geometric extensions that match your PyTorch and platform versions. Replace `${TORCH}` and `${CUDA}` with the appropriate values (e.g. `2.5.1` and `cpu`):

```sh
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

Without these packages, DynamicEdgeConv, GravNetConv, XConv, PANConv, and SplineConv are unavailable. FusedGATConv additionally requires `dgNN`, which is not available on all platforms.

## Usage

A GATv2 forecasting the next 48 half-hours of regional load on the DTW graph, reconciled with a national XGBoost forecast:

```python
from torch_geometric.nn import GATv2Conv

from graphtoolbox.data import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer

data_kwargs = {
    "node_var": "Region",                    # one node per region
    "features_to_lag": {"load": (1, 48)},    # load of the previous day
    "dummies": ["Instant", "JourSemaine"],   # one-hot calendar
    "day_inf_train": "2015-01-01", "day_sup_train": "2018-01-01",
    "day_inf_val": "2018-01-01", "day_sup_val": "2018-12-31",
    "day_inf_test": "2019-01-01", "day_sup_test": "2019-12-31",
}
dataset_kwargs = {
    "adj_matrix": "dtw",                     # graph read from graph_folder/dtw/
    "features_base": ["temp", "nebu", "wind", "Instant", "JourSemaine"]
                     + [f"load_l{t}" for t in range(1, 49)],
    "target_base": "load",
}
data = DataClass(path_train="train.csv", path_test="test.csv",
                 data_kwargs=data_kwargs, folder_config=".")
common = dict(data=data, dataset_kwargs=dataset_kwargs, out_channels=48,
              graph_folder="../graph_representations")
train = GraphDataset(period="train", **common)
scalers = dict(scalers_feat=train.scalers_feat, scalers_target=train.scalers_target)
val, test = (GraphDataset(period=p, **scalers, **common) for p in ("val", "test"))

model = myGNN(in_channels=train.num_node_features, num_layers=2, hidden_channels=64,
              out_channels=48, conv_class=GATv2Conv)   # any PyG convolution
trainer = Trainer(model=model, dataset_train=train, dataset_val=val, dataset_test=test,
                  batch_size=32, model_kwargs={"lr": 1e-3, "num_epochs": 300},
                  reconcile=True, top_level_model="xgb")
pred, target, edge_index, edge_weight = trainer.train(patience=30, plot_loss=False)
trainer.evaluate()
```

## Reproducing the paper

The exact dependency lock, the runners of every table and figure, the cached forecasts, the identity-graph ablation, the multi-seed runs and the interpretability outputs of the paper are documented in [`examples/REPRODUCIBILITY.md`](examples/REPRODUCIBILITY.md). The forecasts of the tables were computed with version 0.3.0, and the interpretability diagnostics, the reconciliations and the daily updates with version 0.3.1. Trained weights are attached to the GitHub release.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

Special **thanks** to all contributors of the GraphToolbox project:

- Eloi Campagne 
- Itai Zehavi

## Citation
If you use the GraphToolbox in your work, please cite the corresponding [preprint](https://arxiv.org/abs/2609.24609):

```bibtex
@misc{campagne2026graphtoolbox,
    author = {Campagne, Eloi and Amara-Ouali, Yvenn and Goude, Yannig and Kalogeratos, Argyris},
    title = {GraphToolbox: A Configurable Framework for Controlled Benchmarking of Graph Neural Network Forecasting},
    year = {2026},
    eprint = {2609.24609},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
}
```


## License

This project is licensed under the GPL License - see the LICENSE file for details.
