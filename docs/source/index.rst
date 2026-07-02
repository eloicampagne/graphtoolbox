GraphToolbox
============

GraphToolbox is a Python package designed for graph machine learning focused on electricity load forecasting. It provides tools for data handling, model building, training, evaluation, and visualization.

Features
--------

- Data handling and preprocessing for graph datasets.
- Various graph neural network models including Graph Convolutional Networks (GCNs), GraphSAGE, and Graph Attention Networks (GATs).
- Training and evaluation utilities for graph-based models.
- Visualization tools for graph data and model results.

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
   * - GCN / Spectral
     - |green|
     - GCNConv, ChebConv, SGConv, SSGConv, LGConv, GCN2Conv, ClusterGCNConv, FAConv
   * - Attention-based
     - |green|
     - GATConv, GATv2Conv, SuperGATConv, TransformerConv, AGNNConv, DNAConv
   * - MPNN / Aggregation
     - |green|
     - SAGEConv, GENConv, GraphConv, MFConv, LEConv, SimpleConv, EGConv, GravNetConv
   * - MLP-based (GIN-style)
     - |green|
     - GINConv, GINEConv
   * - Edge-conditioned
     - |green|
     - NNConv, ECConv, CGConv, GMMConv, GeneralConv, XConv
   * - Recurrent / Gated
     - |green|
     - GatedGraphConv, ARMAConv, TAGConv
   * - Residual / Deep
     - |green|
     - DirGNNConv, AntiSymmetricConv, FiLMConv, ResGatedGraphConv, PDNConv
   * - Spectral / Poly
     - |green|
     - MixHopConv, GPSConv, FeaStConv, SplineConv, PANConv
   * - Dynamic aggregators
     - |green|
     - PNAConv, EdgeConv, DynamicEdgeConv
   * - Relational
     - |green|
     - RGCNConv, RGATConv, FastRGCNConv
   * - Graph-level
     - |green|
     - WLConv, SignedConv
   * - Missing optional deps
     - |red|
     - FusedGATConv (``dgNN``)
   * - Heterogeneous graphs
     - |white|
     - HANConv, HGTConv, HEATConv, HeteroConv
   * - Point-cloud
     - |white|
     - PointNetConv, PointConv, PointGNNConv, PointTransformerConv, PPFConv
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
     - **51**
     - **78.5 %**
   * - |red|
     - **1**
     - **1.5 %**
   * - |white|
     - **13**
     - **20.0 %**
   * - **Total Tested**
     - **65**
     - **100 %**

FusedGATConv is not broken; it requires the ``dgNN`` package which is not available on all platforms.
Installing ``torch-cluster``, ``torch-sparse``, and ``torch-spline-conv`` unlocked DynamicEdgeConv, GravNetConv, XConv, PANConv, and SplineConv.

If you spot a missing convolution, find an incompatibility, or want to help extend support, contributions are warmly welcomed. Feel free to open an issue or submit a PR.

Usage
-----

Basic example of how to use GraphToolbox:

.. code-block:: python

   from graphtoolbox.data.dataset import *
   from graphtoolbox.training.trainer import Trainer
   from graphtoolbox.utils.helper_functions import *
   from torch_geometric.nn.models import *

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
   conv_kwargs = {'heads': 2}
   params = {'num_layers': 3,
             'hidden_channels': 364,
             'lr': 1e-3,
             'batch_size': 16,
             'adj_matrix': 'gl3sr',
             'lam_reg': 0}

   model = myGNN(
       in_channels=graph_dataset_train.num_node_features,
       num_layers=params["num_layers"],
       hidden_channels=params["hidden_channels"],
       out_channels=out_channels,
       conv_class=conv_class,
       conv_kwargs=conv_kwargs
   )

   # Initialize trainer
   trainer = Trainer(
       model=model,
       dataset_train=graph_dataset_train,
       dataset_val=graph_dataset_val,
       dataset_test=graph_dataset_test,
       batch_size=params["batch_size"],
       return_attention=False,
       model_kwargs={'lr': params["lr"], 'num_epochs': 200},
       lam_reg=params["lam_reg"]
   )

   # Train model
   pred_model_test, target_test, edge_index, attention_weights = trainer.train(
       plot_loss=True,
       force_training=True,
       save=False,
       patience=75
   )

   # Evaluate model
   trainer.evaluate()

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
