import colorama
from datetime import timedelta
from graphtoolbox.data.preprocessing import *
from graphtoolbox.training.metrics import *
from graphtoolbox.utils.helper_functions import *
import numpy as np
import os
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import MinMaxScaler
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from typing import List
import warnings

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class DataClass:
    """
    DataClass handles the loading, preprocessing, and temporal segmentation of graph-based datasets
    used for machine learning and graph neural networks.

    This class automates several common data preparation steps for graph-based time series:
    - Reading training and test datasets (CSV or Parquet format).
    - Creating lagged versions of numerical features.
    - Splitting data into train, validation, and test sets based on time boundaries.
    - Encoding categorical variables into dummy features.
    - Ensuring consistent node indexing across splits.

    Attributes
    ----------
    df_train_original : pandas.DataFrame
        Original training DataFrame as loaded from disk.
    df_test_original : pandas.DataFrame
        Original test DataFrame as loaded from disk.
    df_train, df_val, df_test : pandas.DataFrame
        Preprocessed train/validation/test sets ready for model input.
    node_var : str
        Column name identifying graph nodes.
    nodes : numpy.ndarray
        Sorted array of unique node identifiers.
    features_to_lag : dict or None
        Dictionary describing temporal lags to apply on selected features.
        Format: ``{'feature': (min_lag, max_lag)}``.
    dummies : dict or None
        Mapping of categorical features to be one-hot encoded.
    day_inf_train, day_sup_train, day_inf_val, day_sup_val, day_inf_test, day_sup_test : str
        Date boundaries for temporal splits.
    folder_config : str
        Path to the configuration folder containing data preprocessing parameters.
    data_kwargs : dict
        Loaded data-related configuration options.

    Parameters
    ----------
    path_train : str
        Path to the training dataset (CSV or Parquet file).
    path_test : str
        Path to the test dataset (CSV or Parquet file).
    folder_config : str
        Path to the folder containing configuration files (used by ``load_kwargs``).
    data_kwargs : dict, optional
        Custom dictionary of preprocessing arguments. If not provided, it is loaded
        from the configuration folder.
    col0 : bool, optional
        Whether to treat the first column as the index column. Default is False.
    csv : bool, optional
        Whether the files are CSVs (if False, Parquet is assumed). Default is True.
    node_var : str, optional
        Name of the column identifying nodes. If not provided, retrieved from ``data_kwargs``.
    computed_features : dict, optional
        Mapping of ``{column_name: callable}`` for derived features.  Each callable
        receives the full sorted dataframe (all splits concatenated, including any
        lag columns already added) and must return a same-length Series or array.
        Example::

            data_kwargs['computed_features'] = {
                'heat_deg': lambda df: (15 - df['temp']).clip(lower=0),
                'cool_deg': lambda df: (df['temp'] - 18).clip(lower=0),
            }

        The resulting columns are appended before train/val/test splitting and
        can be referenced in ``dataset_kwargs['features_base']`` like any raw column.
    features_to_lag : dict, optional
        Temporal lags to compute, e.g. ``{'temperature': (1, 3)}`` to add columns
        ``temperature_l1``, ``temperature_l2``, ``temperature_l3``.
    get_dummies : bool, optional
        Whether to apply one-hot encoding on categorical variables. Default is True.
    **kwargs :
        Additional keyword arguments passed to internal preprocessing utilities.

    Raises
    ------
    AssertionError
        If mandatory columns ('date', node variable, lagged features) are missing.
    ValueError
        If invalid lag intervals are specified.

    Examples
    --------
    >>> data = DataClass(
    ...     path_train="data/train.csv",
    ...     path_test="data/test.csv",
    ...     folder_config="config/",
    ... )
    >>> data.df_train.shape
    (12000, 45)
    >>> list(data.df_train.columns[:5])
    ['node_id', 'date', 'feature1', 'feature1_l1', 'feature1_l2']
    """
    def __init__(self, path_train: str, 
                 path_test: str,
                 folder_config: str,
                 data_kwargs: Optional[Dict] = None,
                 **kwargs):
        
        col0 = kwargs.get('col0', False)
        csv = kwargs.get('csv', True)
        if col0:
            if csv:
                self.df_train_original = pd.read_csv(path_train, index_col=0)
                self.df_test_original = pd.read_csv(path_test, index_col=0)
            else:
                self.df_train_original = pd.read_parquet(path_train, index_col=0)
                self.df_test_original = pd.read_parquet(path_test, index_col=0)
        else:
            if csv:
                self.df_train_original = pd.read_csv(path_train)
                self.df_test_original = pd.read_csv(path_test)
            else:
                self.df_train_original = pd.read_parquet(path_train)
                self.df_test_original = pd.read_parquet(path_test)
        assert 'date' in self.df_train_original.columns, "Column 'date' is not in the DataFrame!"
            
        self.folder_config = folder_config
        if data_kwargs is None:
            data_kwargs = load_kwargs(folder_config=folder_config, kwargs='data_kwargs')
        self.data_kwargs = data_kwargs
        
        node_var = None
        if 'node_var' in self.data_kwargs:
            node_var = self.data_kwargs['node_var']     
        self.node_var = kwargs.get('node_var', node_var)
        assert (self.node_var is not None) and (self.node_var in self.df_train_original.columns), f"'{self.node_var}' is either None or not in the DataFrame!"
        self.nodes = np.sort(self.df_train_original[self.node_var].unique())

        features_to_lag = None
        if 'features_to_lag' in self.data_kwargs:
            features_to_lag = self.data_kwargs['features_to_lag']
        self.features_to_lag = kwargs.get('features_to_lag', features_to_lag)
        
        if self.features_to_lag is not None:
            df_concat = pd.concat([self.df_train_original, self.df_test_original], axis=0)
            df_concat = df_concat.sort_values([self.node_var, 'date'])
            lagged_cols = []
            for feature, (min_shift, max_shift) in self.features_to_lag.items():
                assert feature in df_concat.columns, f"Column '{feature}' is not in the DataFrame!"
                min_shift, max_shift = int(min_shift), int(max_shift)
                assert min_shift <= max_shift, f"min_shift ({min_shift}) > max_shift ({max_shift})"

                for s in range(min_shift, max_shift + 1):
                    lagged = df_concat.groupby(self.node_var)[feature].shift(s)
                    lagged.name = f'{feature}_l{s}'
                    lagged_cols.append(lagged)

            lagged_df = pd.concat(lagged_cols, axis=1)
            df_concat = pd.concat([df_concat.reset_index(drop=True), lagged_df.reset_index(drop=True)], axis=1)
            self.df_train_original = df_concat[df_concat['date'].isin(self.df_train_original['date'])]
            self.df_test_original = df_concat[df_concat['date'].isin(self.df_test_original['date'])]

        # Computed features: arbitrary callables that receive the full sorted
        # dataframe and return a same-length Series / array.  Applied after lags
        # so they can reference lag columns if needed.
        # Example entry: {'heat_deg': lambda df: (18 - df['temp']).clip(lower=0)}
        computed_features = self.data_kwargs.get('computed_features', None)
        if computed_features:
            _df = pd.concat(
                [self.df_train_original, self.df_test_original], axis=0
            ).sort_values([self.node_var, 'date']).reset_index(drop=True)
            for col_name, fn in computed_features.items():
                _df[col_name] = fn(_df).values
            _train_dates = set(self.df_train_original['date'])
            _test_dates  = set(self.df_test_original['date'])
            self.df_train_original = _df[_df['date'].isin(_train_dates)].reset_index(drop=True)
            self.df_test_original  = _df[_df['date'].isin(_test_dates)].reset_index(drop=True)

        day_cut = str(pd.to_datetime(self.df_train_original.date).max() - timedelta(days=365))
        self.day_sup_train = day_cut
        if 'day_inf_train' in self.data_kwargs.keys():
            self.day_inf_train = self.data_kwargs['day_inf_train']
        if 'day_sup_train' in self.data_kwargs.keys():
            self.day_sup_train = self.data_kwargs['day_sup_train']
        self.day_inf_val = day_cut
        if 'day_inf_val' in self.data_kwargs.keys():
            self.day_inf_val = self.data_kwargs['day_inf_val']
        if 'day_sup_val' in self.data_kwargs.keys():
            self.day_sup_val = self.data_kwargs['day_sup_val']
        if 'day_inf_test' in self.data_kwargs.keys():
            self.day_inf_test = self.data_kwargs['day_inf_test']
        if 'day_sup_test' in self.data_kwargs.keys():
            self.day_sup_test = self.data_kwargs['day_sup_test']
        
        df_train_ = extract_dataframe(self.df_train_original, day_inf=self.day_inf_train, day_sup=self.day_sup_train)
        df_val_ = extract_dataframe(self.df_train_original, day_inf=self.day_inf_val, day_sup=self.day_sup_val)
        df_test_ = extract_dataframe(self.df_test_original, day_inf=self.day_inf_test, day_sup=self.day_sup_test)

        for i, node_name in enumerate(self.nodes):
            df_train_.loc[df_train_[self.node_var] == node_name, f'{self.node_var}Int'] = i
            df_val_.loc[df_val_[self.node_var] == node_name, f'{self.node_var}Int'] = i
            df_test_.loc[df_test_[self.node_var] == node_name, f'{self.node_var}Int'] = i

        get_dummies = kwargs.get('get_dummies', True)
        dummies = None
        if ('dummies' in self.data_kwargs) and get_dummies:
            dummies = self.data_kwargs['dummies']
        self.dummies = dummies
        if self.dummies is not None:
            self.df_train = extract_dummies(df_train_, self.dummies)
            self.df_val = extract_dummies(df_val_, self.dummies)
            self.df_test = extract_dummies(df_test_, self.dummies)
        else:
            self.df_train = df_train_
            self.df_val = df_val_
            self.df_test = df_test_
            
        for df_attr in ["df_train", "df_val", "df_test"]:
            df = getattr(self, df_attr)
            num_cols = df.select_dtypes(include=["number", "bool"]).columns
            df[num_cols] = df[num_cols].astype("float32")
            setattr(self, df_attr, df)
                        
class GraphDataset:
    """
    GraphDataset organizes time-dependent node features and targets into
    graph-structured tensors compatible with PyTorch Geometric.

    This class acts as the bridge between tabular time series data and graph neural
    network inputs. It handles:
    - feature and target extraction from the preprocessed `DataClass` object,
    - normalization per node using train-based MinMax scaling,
    - construction of temporal tensors (node × time × features),
    - association with graph topology (`edge_index`, `edge_weight`),
    - packaging of graph snapshots as `torch_geometric.data.Data` objects.

    Parameters
    ----------
    data : DataClass
        Preprocessed data container including train/val/test splits.
    period : str
        Dataset split to use, one of ``{'train', 'val', 'test'}``.
    scalers_feat : dict, optional
        Dictionary of fitted feature scalers per node (from training phase).
        Required for validation and test datasets.
    scalers_target : dict, optional
        Dictionary of fitted target scalers per node (from training phase).
    dataset_kwargs : dict, optional
        Dataset-level configuration (loaded via ``load_kwargs`` if not provided).
        Must include keys like ``'features_base'`` and ``'target_base'``.
    out_channels : int, default 1
        Number of temporal steps grouped per graph sample (sliding window width).
    **kwargs :
        Additional options such as:
        - ``graph_folder`` (str): path to saved adjacency matrices.
        - ``adj_matrix`` (str): graph construction algorithm (default: `'space'`).
        - ``get_dummies`` (bool): whether to expand categorical dummy variables.

    Attributes
    ----------
    dataframe : pandas.DataFrame
        Subset of data corresponding to the specified period.
    features_base : list of str
        List of input feature column names.
    feature_groups : dict or None
        Optional mapping of feature groups for grouped GNN inputs.
    target_base : str
        Name of the prediction target column.
    X_scaled : torch.Tensor
        Normalized feature tensor of shape ``[num_nodes, T, num_features]``.
    Y_scaled : torch.Tensor
        Normalized target tensor of shape ``[num_nodes, T, 1]``.
    mask_X, mask_Y : torch.BoolTensor
        Boolean masks indicating valid (non-NaN) temporal positions.
    edge_index : torch.LongTensor
        Graph connectivity in COO format for PyTorch Geometric.
    edge_weight : torch.FloatTensor
        Edge weights (typically similarities).
    pyg_data : list[torch_geometric.data.Data]
        List of graph snapshots ready for batching or iteration.
    num_nodes : int
        Number of graph nodes.
    num_node_features : int
        Number of input features per node.

    Raises
    ------
    AssertionError
        If expected columns or scalers are missing.
    FileNotFoundError
        If the adjacency matrix file is missing.

    Examples
    --------
    >>> dataset = GraphDataset(data=data, period='train', out_channels=3)
    >>> len(dataset)
    120  # number of temporal graph snapshots
    >>> sample = dataset[0]
    >>> sample.x.shape, sample.y.shape
    (torch.Size([N, F]), torch.Size([N, 3]))
    >>> sample.edge_index.shape
    torch.Size([2, E])
    """
    def __init__(self, data, period: str, scalers_feat=None, scalers_target=None,
                 dataset_kwargs: Optional[Dict] = None,
                 out_channels: int = 1, **kwargs):

        self.data = data
        assert period in ['train', 'val', 'test'], f"{period} should be in ['train', 'val', 'test']"
        self.period = period

        self.graph_folder = kwargs.get('graph_folder', './graph_representations')
        self.folder_config = self.data.folder_config

        self.data_kwargs = self.data.data_kwargs
        if dataset_kwargs is None:
            dataset_kwargs = load_kwargs(folder_config=self.folder_config, kwargs='dataset_kwargs')
        self.dataset_kwargs = dataset_kwargs

        self.adj_matrix = kwargs.get('adj_matrix', self.dataset_kwargs.get('adj_matrix', 'space'))
        self.dataframe = getattr(self.data, f'df_{period}')
        self.nodes = np.sort(kwargs.get('nodes', self.dataframe[self.data.node_var].unique()))
        self.dataframe = self.dataframe[self.dataframe[self.data.node_var].isin(self.nodes)].reset_index(drop=True)

        self.features_base = [f for f in self.dataset_kwargs['features_base'] if f not in self.data_kwargs['dummies']]
        if kwargs.get('get_dummies', True):
            for dummy in self.data_kwargs['dummies']:
                self.features_base += self.dataframe.filter(regex=f'{dummy}_').columns.tolist()

        self.feature_groups = dataset_kwargs.get('feature_groups', None)
        self.target_base = self.dataset_kwargs['target_base']

        self.scalers_feat = scalers_feat
        self.scalers_target = scalers_target

        self.out_channels = out_channels

        self._init_nodes()
        self._init_data()

    def _init_nodes(self):
        """Initialize node count (`num_nodes`) based on the unique node identifiers."""
        self.num_nodes = len(self.nodes)

    def _init_data(self):
        """
        Initialize dataset internals:
        - set features and targets,
        - build graph connectivity,
        - prepare PyTorch Geometric data objects.
        """
        self._set_features(self.features_base)
        self._set_target(self.target_base)
        self.edge_index, self.edge_weight = self._get_edge_info()
        self._prepare_graph_data()

    def _set_features(self, features: List[str]):
        """
        Define feature variables and prepare feature tensors.

        Checks feature existence, warns against data leakage if a target is included,
        and builds the 3D tensor `[num_nodes, T, num_features]`.

        Parameters
        ----------
        features : list of str
            Names of the input feature columns.
        """
        self.features = features
        for feat in features:
            assert feat in self.dataframe.columns, f"{feat} not in dataframe."
            if feat == self.target_base:
                warnings.warn(colorama.Fore.RED + f"Data-leakage: {feat} is also the target!" + colorama.Style.RESET_ALL)

        self.X = self.dataframe[self.features].to_numpy()
        self.num_node_features = len(self.features)
        self.X_node = self._get_nodewise(self.features)
        self._init_feat_scalers()
        self._normalize_features()

    def _set_target(self, target: str):
        """
        Define target variable and prepare target tensors.

        Parameters
        ----------
        target : str
            Name of the column to predict.
        """
        assert target in self.dataframe.columns, f"{target} not in dataframe."
        self.target = target
        self.Y = self.dataframe[self.target].to_numpy().reshape(-1, 1)
        self.Y_node = self._get_nodewise([self.target]).squeeze()
        self._init_target_scalers()
        self._normalize_targets()

    def _init_feat_scalers(self):
        """
        Fit or load feature scalers (MinMax per node).

        - During training: fits a new scaler for each node.
        - During validation/test: checks that scalers are provided.
        """
        if self.period == 'train':
            self.scalers_feat = {}
            for node in self.nodes:
                values = self.dataframe[self.dataframe[self.data.node_var] == node][self.features].values
                values = values[~np.isnan(values).any(axis=1)] 
                self.scalers_feat[node] = MinMaxScaler().fit(values)
        else:
            assert self.scalers_feat is not None, 'Feature scaler not set!'

    def _init_target_scalers(self):
        """
        Fit or load target scalers (MinMax per node).

        - During training: fits a new scaler for each node.
        - During validation/test: requires existing scalers.
        """
        if self.period == 'train':
            self.scalers_target = {}
            for node in self.nodes:
                values = self.dataframe[self.dataframe[self.data.node_var] == node][self.target].values.reshape(-1, 1)
                values = values[~np.isnan(values).any(axis=1)]
                self.scalers_target[node] = MinMaxScaler().fit(values)
        else:
            assert self.scalers_target is not None, 'Target scaler not set!'
    
    def _get_nodewise(self, cols):
        """
        Convert feature columns into a node-wise padded tensor.

        Groups samples by node and pads sequences with NaNs to equal length.

        Parameters
        ----------
        cols : list of str
            Feature or target column names.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``[num_nodes, T_max, len(cols)]``.
        """
        grouped = []
        max_len = 0
        for node in self.nodes:
            values = self.dataframe[self.dataframe[self.data.node_var] == node][cols].values
            grouped.append(values)
            max_len = max(max_len, len(values))

        padded = []
        for g in grouped:
            if len(g) < max_len:
                pad_width = ((0, max_len - len(g)), (0, 0))
                g = np.pad(g, pad_width, constant_values=np.nan)
            padded.append(g)

        tensor = torch.tensor(np.stack(padded), dtype=torch.float32)  # [num_nodes, Tmax, features]
        return tensor

    def _normalize_features(self):
        """
        Normalize feature tensors per node using fitted scalers.

        Produces:
        - `X_scaled`: normalized tensor of shape `[num_nodes, T, features]`,
        - `mask_X`: boolean mask for valid (non-NaN) entries.
        """
        shape = self.X_node.shape  # (num_nodes, time, features)
        X_scaled = torch.full(shape, float('nan'), dtype=torch.float32)
        mask_X = torch.zeros((shape[0], shape[1]), dtype=torch.bool)

        for i, node in enumerate(self.nodes):
            x_node = self.X_node[i, :, :]  # [T, F]
            mask = (~torch.isnan(x_node).any(dim=1))  # [T]
            mask_X[i] = mask
            if mask.sum() > 0:
                scaled = torch.tensor(
                    self.scalers_feat[node].transform(x_node[mask, :].numpy()),
                    dtype=torch.float32
                )
                X_scaled[i, mask, :] = scaled

        self.X_scaled = X_scaled
        self.mask_X = mask_X

    def _normalize_targets(self):
        """
        Normalize target tensors per node using fitted scalers.

        Produces:
        - `Y_scaled`: normalized target tensor `[num_nodes, T, 1]`,
        - `mask_Y`: boolean mask for valid entries.
        """
        ...
        shape = self.Y_node.shape
        Y_scaled = torch.full(shape, float('nan'), dtype=torch.float32)
        mask_Y = torch.zeros_like(self.Y_node, dtype=torch.bool)

        for i, node in enumerate(self.nodes):
            y_node = self.Y_node[i, :]
            mask = (~torch.isnan(y_node))
            mask_Y[i] = mask
            if mask.sum() > 0:
                scaled = torch.tensor(
                    self.scalers_target[node].transform(y_node[mask].numpy().reshape(-1, 1)).flatten(),
                    dtype=torch.float32
                )
                Y_scaled[i, mask] = scaled

        self.Y_scaled = Y_scaled
        self.Y_node = self.Y_node.float()
        self.mask_Y = mask_Y

    def _get_edge_info(self):
        """
        Load the adjacency matrix and return its sparse representation.

        Returns
        -------
        tuple of (torch.LongTensor, torch.FloatTensor)
            Edge index and edge weights for the graph.

        Notes
        -----
        If the specified adjacency file is missing, uses an identity matrix instead.
        """
        adj_path = os.path.join(self.graph_folder, self.adj_matrix, 'W.txt')
        if not os.path.exists(adj_path):
            print(f"{self.adj_matrix} is invalid. Using identity matrix.")
            return dense_to_sparse(torch.eye(self.num_nodes))
        A = torch.tensor(np.loadtxt(adj_path, delimiter=' ', usecols=range(self.num_nodes)))
        return dense_to_sparse(A)
    
    def _infer_group_dims(self):
        """
        Compute feature_group_dims = {group_name: num_features_in_group}.
        Called after _set_features() and before model creation.
        """

        if not self.feature_groups:
            self.feature_group_dims = None
            return

        feature_group_dims = {}

        for group_name, feature_names in self.feature_groups.items():
            count = 0

            for name in feature_names:
                # Exact feature match
                if name in self.features_base:
                    count += 1
                    continue

                # Dummy encoding expansion
                matched = [f for f in self.features_base if f.startswith(name + "_")]
                count += len(matched)

            if count == 0:
                raise KeyError(
                    f"[GraphDataset] Group '{group_name}' matched 0 features "
                    f"(check feature names or dummy expansion)."
                )

            feature_group_dims[group_name] = count

        self.feature_group_dims = feature_group_dims

    def _prepare_graph_data(self):
        """
        Prepare graph snapshots (PyG Data) for each time step.

        If feature_groups is provided:
            - builds Data.<group_name> for each group
            - still includes Data.x (all features)

        Otherwise:
            - same behavior as original.
        """

        T = self.X_scaled.shape[1]
        X_all = self.X_scaled
        Y_scaled = self.Y_scaled
        Y_raw = self.Y_node
        edge_index, edge_weight = self.edge_index, self.edge_weight

        group_index_map = {}
        if self.feature_groups:
            for group_name, feature_names in self.feature_groups.items():
                idxs = []

                for name in feature_names:

                    # Exact match
                    if name in self.features_base:
                        idxs.append(self.features_base.index(name))
                        continue

                    # Prefix match for dummies
                    matched = [i for i, f in enumerate(self.features_base)
                            if f.startswith(name + "_")]
                    if matched:
                        idxs.extend(matched)
                        continue

                    raise KeyError(
                        f"[GraphDataset] Feature '{name}' in group '{group_name}' "
                        f"not found among features_base."
                    )

                group_index_map[group_name] = idxs

        self.pyg_data = []

        max_start = T - self.out_channels
        if max_start < 0:
            return

        for t in range(0, max_start + 1, self.out_channels):
            start, end = t, t + self.out_channels  # always full horizon

            x_t = X_all[:, start]            # [N, F]
            y_t = Y_scaled[:, start:end]     # [N, out_channels]
            y_raw_t = Y_raw[:, start:end]

            mask_x = (~torch.isnan(x_t)).float()
            mask_y = (~torch.isnan(y_t)).float()

            x_t = torch.nan_to_num(x_t, nan=0)
            y_t = torch.nan_to_num(y_t, nan=0)
            y_raw_t = torch.nan_to_num(y_raw_t, nan=0)

            data_kwargs = {
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "mask_X": mask_x,
                "mask_y": mask_y,
                "y_scaled": y_t,
                "y": y_raw_t,
            }

            if not self.feature_groups:
                data_kwargs["x"] = x_t
                self.pyg_data.append(Data(**data_kwargs))
                continue

            data_kwargs["x"] = x_t
            data_kwargs["all"] = x_t

            for group_name, idxs in group_index_map.items():
                data_kwargs[group_name] = x_t[:, idxs]

            data = Data(**data_kwargs)
            self.pyg_data.append(data)

        if self.feature_groups:
            self._infer_group_dims()


    def _set_adj_matrix(self, adj_matrix):
        """
        Update the adjacency matrix and rebuild the corresponding graph data.

        Parameters
        ----------
        adj_matrix : str
            Name or path of the new adjacency matrix to load.
        """
        self.adj_matrix = adj_matrix
        self.edge_index, self.edge_weight = self._get_edge_info()
        self._prepare_graph_data()

    def __getitem__(self, idx):
        """
        Retrieve a temporal graph snapshot.

        Parameters
        ----------
        idx : int
            Index of the graph snapshot (time step group).

        Returns
        -------
        torch_geometric.data.Data
            Graph object containing node features, targets, and connectivity.
        """
        return self.pyg_data[idx]

    def __len__(self):
        """
        Compute the number of graph samples in the dataset.

        Returns
        -------
        int
            Number of temporal graph windows (floor(T / out_channels)).
        """
        return len(self.pyg_data)