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
     - GCNConv, ChebConv, SGConv, SSGConv, LGConv, GCN2Conv, ClusterGCNConv
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
     - NNConv, CGConv, GMMConv, GeneralConv, XConv
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
     - **49**
     - **77.8 %**
   * - |red|
     - **1**
     - **1.6 %**
   * - |white|
     - **13**
     - **20.6 %**
   * - **Total Tested**
     - **63**
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
