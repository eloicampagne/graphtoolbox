GraphToolbox
============

GraphToolbox is a Python framework on PyTorch Geometric for forecasting time series observed on the nodes of a graph, such as regional electricity load. One configuration-driven pipeline covers graph construction, the choice among the convolution operators of PyTorch Geometric, training, reconciliation, expert aggregation, significance testing and interpretability, so that a benchmark changes one choice at a time.

Features
--------

- **Data and graphs**: ``DataClass`` and ``GraphDataset`` build lagged, scaled node samples with train, validation and test splits; ``GraphBuilder`` derives the adjacency from coordinates or from the series themselves (spatial distance, correlation, precision, GL3SR, dynamic time warping), any precomputed adjacency stored as a matrix is loaded by name, and a graph can be selected on validation error.
- **Models**: ``myGNN`` runs 53 of the 65 convolutions of PyTorch Geometric through one adapter (see `Convolutions`_), ``TemporalGNN`` the recurrent cells of PyTorch Geometric Temporal, and ``AdditiveGraphModel`` an additive graph network whose forecast splits into feature-group contributions.
- **Training**: ``Trainer`` (early stopping, forecasting metrics, MinT reconciliation), ``Optimizer`` for Optuna searches over the model and the graph, and ``RollingTrainer`` for rolling refits or daily online updates of a trained model.
- **Reconciliation**: MinT combines the node forecasts with a forecast of their total, fitted by the framework or supplied by the user, with a diagonal or a shrunk full error covariance (see `Reconciliation`_).
- **Aggregation**: online expert aggregation (MLpol, EWA, BOA) across architectures or seeds, on the total or node by node (see `Expert Aggregation`_).
- **Evaluation**: Diebold-Mariano tests with Holm correction, moving-block bootstrap and Model Confidence Set, on cached forecasts (``graphtoolbox.evaluation``).
- **Interpretability**: attention matched to edges, occlusion influence, accumulated local effects, and maps of edge attributions (``graphtoolbox.interpretability``).
- **Baselines**: ARIMA-X, GAM, LSTM, XGBoost, Chronos-Bolt and Chronos-2 (``graphtoolbox.models.baseline``).

Installation
------------

Clone the repository and install the package and dependencies:

.. code-block:: bash

   git clone git@github.com:eloicampagne/GraphToolbox.git
   cd GraphToolbox
   pip install .

To unlock the full set of supported convolutions, install the optional PyTorch Geometric extensions that match your PyTorch and platform versions. Replace ``${TORCH}`` and ``${CUDA}`` with the appropriate values (e.g. ``2.5.1`` and ``cpu``):

.. code-block:: bash

   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
       -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

Without these packages, DynamicEdgeConv, GravNetConv, XConv, PANConv, and SplineConv are unavailable. FusedGATConv additionally requires ``dgNN``, which is not available on all platforms.

Convolutions
------------

GraphToolbox benchmarks the entire ``torch_geometric.nn.conv`` collection against ``myGNN``, evaluating whether each operator can be instantiated and run end-to-end with standard node-feature inputs and a homogeneous graph structure.

**Legend**

- |green| Working (fully compatible with ``myGNN``)
- |red| Skipped (requires the ``dgNN`` package, not available on all platforms)
- |white| Skipped (requires CUDA-specific dependencies or device-restricted libraries)

.. |green| unicode:: U+1F7E2
.. |red|   unicode:: U+1F534
.. |white| unicode:: U+26AA

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Convolution Type
     - Status
     - Convolutions
   * - Propagation
     - |green|
     - SimpleConv, GCNConv, ChebConv, SGConv, SSGConv, APPNP, GCN2Conv, ARMAConv, TAGConv, LGConv, ClusterGCNConv, MixHopConv, PANConv
   * - Attention
     - |green|
     - GATConv, GATv2Conv, SuperGATConv, TransformerConv, AGNNConv, RGATConv, FAConv, DNAConv, GPSConv, PointTransformerConv
   * - Gating and modulation
     - |green|
     - GatedGraphConv, ResGatedGraphConv, FiLMConv
   * - Relational and edge-conditioned
     - |green|
     - RGCNConv, FastRGCNConv, NNConv, CGConv, GMMConv, SplineConv, FeaStConv, PDNConv, GINEConv
   * - Point-cloud and dynamic graphs
     - |green|
     - EdgeConv, DynamicEdgeConv, GravNetConv, XConv
   * - Other aggregators
     - |green|
     - SAGEConv, GraphConv, MFConv, LEConv, PNAConv, GENConv, EGConv, GINConv, GeneralConv, DirGNNConv, AntiSymmetricConv, SignedConv, WLConv, WLConvContinuous
   * - Missing optional deps
     - |red|
     - FusedGATConv (``dgNN``)
   * - Heterogeneous graphs
     - |white|
     - HANConv, HGTConv, HEATConv, HeteroConv
   * - Point-cloud
     - |white|
     - PointNetConv, PointGNNConv, PPFConv
   * - CuGraph (CUDA only)
     - |white|
     - CuGraphGATConv, CuGraphRGCNConv, CuGraphSAGEConv
   * - Hypergraph
     - |white|
     - HypergraphConv

.. list-table::
   :header-rows: 1
   :widths: 20 15 15

   * - Status
     - Count
     - Percentage
   * - |green|
     - **53**
     - **81.5 %**
   * - |red|
     - **1**
     - **1.5 %**
   * - |white|
     - **11**
     - **16.9 %**
   * - **Total Tested**
     - **65**
     - **100 %**

Counts follow ``torch_geometric.nn.conv.__all__`` in PyTorch Geometric 2.6.1; the aliases ECConv (NNConv) and PointConv (PointNetConv) are not counted twice.

FusedGATConv is not broken; it requires the ``dgNN`` package which is not available on all platforms.
Installing ``torch-cluster``, ``torch-sparse``, and ``torch-spline-conv`` unlocked DynamicEdgeConv, GravNetConv, XConv, PANConv, and SplineConv.

If you spot a missing convolution, find an incompatibility, or want to help extend support, contributions are warmly welcomed. Feel free to open an issue or submit a PR.

Usage
-----

Basic example of how to use GraphToolbox:

.. code-block:: python

   from torch_geometric.nn import GATConv

   from graphtoolbox.data import DataClass, GraphDataset
   from graphtoolbox.models import myGNN
   from graphtoolbox.training import Trainer

   # Load datasets
   out_channels = 48
   data = DataClass(path_train='./train.csv',
                    path_test='./test.csv',
                    data_kwargs=data_kwargs,
                    folder_config='.')

   graph_dataset_train = GraphDataset(data=data, period='train',
                                      graph_folder='../graph_representations',
                                      dataset_kwargs=dataset_kwargs,
                                      out_channels=out_channels)
   graph_dataset_val = GraphDataset(data=data, period='val',
                                    scalers_feat=graph_dataset_train.scalers_feat,
                                    scalers_target=graph_dataset_train.scalers_target,
                                    graph_folder='../graph_representations',
                                    dataset_kwargs=dataset_kwargs,
                                    out_channels=out_channels)
   graph_dataset_test = GraphDataset(data=data, period='test',
                                     scalers_feat=graph_dataset_train.scalers_feat,
                                     scalers_target=graph_dataset_train.scalers_target,
                                     graph_folder='../graph_representations',
                                     dataset_kwargs=dataset_kwargs,
                                     out_channels=out_channels)

   # Initialize model
   conv_class = GATConv
   conv_kwargs = {}
   params = {'num_layers': 3,
             'hidden_channels': 364,
             'heads': 2,
             'lr': 1e-3,
             'batch_size': 16,
             'adj_matrix': 'gl3sr'}

   model = myGNN(
       in_channels=graph_dataset_train.num_node_features,
       num_layers=params["num_layers"],
       hidden_channels=params["hidden_channels"],
       out_channels=out_channels,
       conv_class=conv_class,
       conv_kwargs=conv_kwargs,
       heads=params["heads"]
   )

   # Initialize trainer
   trainer = Trainer(
       model=model,
       dataset_train=graph_dataset_train,
       dataset_val=graph_dataset_val,
       dataset_test=graph_dataset_test,
       batch_size=params["batch_size"],
       return_attention=False,
       model_kwargs={'lr': params["lr"], 'num_epochs': 200}
   )

   # Train model
   pred_model_test, target_test, edge_index, edge_weight = trainer.train(
       plot_loss=True,
       force_training=True,
       save=False,
       patience=75
   )

   # Evaluate model
   trainer.evaluate()

Reconciliation
--------------

MinT reconciliation makes the node forecasts sum to a forecast of their total,
weighting each series by its errors on held-out data. With ``reconcile=True``,
the ``Trainer`` fits a top-level model (``top_level_model="xgb"``, ``"ridge"``,
``"rf"`` or ``"gam"``) on the lags, the calendar and the national mean of the
covariates, or takes an external forecast through ``top_forecasts_val`` and
``top_forecasts_test``. ``reconcile_covariance="diag"`` (default) weights the
series by their validation error variances; ``"shrink"`` uses their full error
covariance, shrunk toward its diagonal with the Schafer-Strimmer intensity.

.. code-block:: python

   trainer = Trainer(model=model, dataset_train=graph_dataset_train,
                     dataset_val=graph_dataset_val, dataset_test=graph_dataset_test,
                     batch_size=16, reconcile=True, top_level_model="xgb",
                     reconcile_covariance="shrink")

Cached forecasts can be reconciled without a model:

.. code-block:: python

   from graphtoolbox.training.reconciliation import reconcile

   # base [nodes, T] and top [T] to reconcile; their errors on a validation period.
   coherent = reconcile(base, top, base_errors_val, top_errors_val, covariance="shrink")
   total = coherent[-1]          # coherent forecast of the total

Expert Aggregation
------------------

The ``graphtoolbox.aggregation`` module combines several competing forecasts
into a single one whose weights adapt over time, in the spirit of the ``opera``
package (Gaillard & Goude). It is useful for blending the GNN forecast with
external experts such as a top-level XGBoost or GAM model, either sequentially
(online) or as a static combiner fitted on a history.

The estimator consumes a matrix of expert forecasts of shape ``[T, K]`` (``T``
time steps, ``K`` experts) and the observations ``[T]``, and exposes the full
weight trajectory in ``weights_`` (handy for interpretability) as well as the
final convex weights in ``coefficients_`` for out-of-sample reuse.

.. code-block:: python

   import numpy as np
   from graphtoolbox.aggregation import Aggregation

   # Toy example: 4 experts of increasing noise around a signal.
   rng = np.random.default_rng(0)
   T, K = 1000, 4
   y = np.sin(np.linspace(0, 30, T)) + 0.5 * np.linspace(0, 1, T)
   experts = y[:, None] + rng.normal(0, [0.1, 0.3, 0.6, 1.0], size=(T, K))

   # MLpol is parameter-free and a robust default.
   agg = Aggregation(model="MLpol", loss="square").run(experts, y)

   print(agg.summary())          # aggregated vs best/mean expert loss
   forecast = agg.prediction_    # aggregated forecast, shape [T]
   weight_path = agg.weights_    # weight trajectory, shape [T, K]

   # EWA and BOA take a learning rate, auto-calibrated when y is given.
   ewa = Aggregation(model="EWA").run(experts, y)
   print("calibrated eta:", ewa.learning_rate_)

Fit on a history and reuse the learned static weights on unseen experts:

.. code-block:: python

   agg = Aggregation(model="MLpol").run(experts[:800], y[:800])
   future = agg.predict(experts[800:])   # applies agg.coefficients_

For a genuine streaming setting, alternate ``partial_fit`` (to obtain the
forecast for the current step) and ``update`` (to feed back the observation):

.. code-block:: python

   agg = Aggregation(model="BOA", learning_rate=1.0).reset(n_experts=K)
   preds = []
   for t in range(T):
       preds.append(agg.partial_fit(experts[t]))  # forecast at time t
       agg.update(y[t])                            # reveal observation

When forecasts are served in batches rather than one step at a time, use
``block_size``, which follows ``opera``'s ``predict`` / ``update`` split. A
day-ahead forecast delivered as daily blocks of 48 half-hours commits each
block with the weights known at its start; once the block is observed the
weights then advance step by step within it before the next block:

.. code-block:: python

   agg = Aggregation(model="MLpol").run(experts, y, block_size=48)

Available rules are ``"MLpol"`` (default, parameter-free), ``"EWA"`` and
``"BOA"`` (adaptive, learning-rate based), plus the ``"uniform"`` mean and the
``"best"`` single-expert oracle as baselines.

Baselines and interpretability
------------------------------

Classical forecasting baselines are reusable from ``graphtoolbox.models.baseline``:
per-slot ARIMA-X and GAM fitting, the covariate-conditioned LSTM, XGBoost,
Chronos-Bolt and Chronos-2. Dataset assembly, the paper-specific formulas, and the
pooled and per-node XGBoost and MLP baselines of the paper remain in ``examples``.

``graphtoolbox.interpretability`` exposes the attribution tools:
``attention_by_snapshot`` returns the attention of every layer and snapshot,
matched to edges; ``occlusion_influence`` measures the change of a node's
forecast when the inputs of another node are replaced by a baseline;
``neighbourhood_deviation`` and ``sender_variance_share`` summarize them;
``additive_ale_from_dataset`` computes accumulated local effects through an
``AdditiveGraphModel``; ``compute_edge_masks`` runs GNNExplainer; and
``plot_ale_group_importance`` renders grouped ALE diagnostics. The complete reproduction commands and cached
output locations are listed in the `reproducibility guide`_.

.. _reproducibility guide: https://github.com/eloicampagne/GraphToolbox/blob/main/examples/REPRODUCIBILITY.md

Changelog
---------

Release notes are maintained in the project `changelog`_.

.. _changelog: https://github.com/eloicampagne/GraphToolbox/blob/main/CHANGELOG.md

Contributing
------------

Contributions are welcome! Please fork the repository and submit a pull request.

Special thanks to all contributors of the GraphToolbox project:

- Eloi Campagne
- Itai Zehavi

License
-------

This project is licensed under the GPL License - see the ``LICENSE`` file for details.


Documentation
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   graphtoolbox
