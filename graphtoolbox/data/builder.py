from fastdtw import fastdtw
from graphtoolbox.data.preprocessing import *
from graphtoolbox.models.gnn import myGNN
from graphtoolbox.training.metrics import *
from graphtoolbox.training.trainer import Trainer
from graphtoolbox.utils import GL_3SR
from graphtoolbox.utils.helper_functions import *
from scipy.linalg import pinv
from scipy.spatial.distance import pdist, squareform
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

class GraphBuilder:
    """
    Constructs graph representations from tabular or temporal datasets,
    combining feature reduction and graph construction algorithms.

    This class provides a unified interface for transforming time series or feature
    datasets into adjacency matrices suitable for graph neural networks. It can:

    - Reduce temporal or feature signals (e.g., via SVD or RESITER)
    - Build graphs based on spatial distance, correlation, precision matrices, GL-3SR,
      or dynamic time warping (DTW)
    - Optionally reuse previously computed signals or graphs from disk

    Parameters
    ----------
    graph_dataset_train : Dataset
        Dataset containing training graph data (with features, nodes, etc.).
    graph_dataset_val : Dataset
        Dataset for validation.
    graph_dataset_test : Dataset
        Dataset for testing.
    model_vgae : object, optional
        Pre-trained VGAE (Variational Graph AutoEncoder) model to initialize
        the graph builder.
    load_graph : bool, default=False
        If True, load a previously saved adjacency matrix instead of recomputing it.
    load_signal : bool, default=False
        If True, load a pre-computed reduced signal representation from disk.
    reduce_method : str, default='svd'
        Method to reduce the signal before graph construction.
        Options are ``'svd'`` or ``'resiter'``.
    folder_config : str, optional
        Path to a configuration folder (used to load positional data and parameters
        via ``load_kwargs``).
    **kwargs
        Additional keyword arguments (e.g., algorithm hyperparameters or model options).

    Attributes
    ----------
    model_vgae : object or None
        VGAE model instance, if provided.
    load_graph : bool
        Whether an existing graph should be loaded instead of generated.
    load_signal : bool
        Whether to reuse a pre-computed reduced signal.
    reduce_method : str
        Signal reduction strategy used by :meth:`reduce_signal`.
    folder_config : str or None
        Folder path containing saved positional or configuration data.
    df_pos : pandas.DataFrame or None
        Positional data for nodes (longitude, latitude) loaded from configuration.
    graph_dataset_train : Dataset
        Dataset used for training.
    graph_dataset_val : Dataset
        Dataset used for validation.
    graph_dataset_test : Dataset
        Dataset used for testing.
    dataframe : pandas.DataFrame
        The raw DataFrame from the training dataset.
    data : DataFrame-like
        The training dataset’s data container.

    Notes
    -----
    The :meth:`build_graph` method always calls :meth:`reduce_signal` before constructing
    an adjacency matrix, unless ``load_graph=True``. The resulting graph can be
    fed into GNNs (e.g., GCN, GraphSAGE).

    Examples
    --------
    >>> gb = GraphBuilder(train_set, val_set, test_set, reduce_method='svd')
    >>> W = gb.build_graph(algo='space', threshold=0.1)
    >>> W.shape
    torch.Size([N, N])
    """

    def __init__(self, graph_dataset_train, graph_dataset_val, graph_dataset_test, **kwargs):
        self.model_vgae = kwargs.get('model_vgae', None)
        self.load_graph = kwargs.get('load_graph', False)
        self.load_signal = kwargs.get('load_signal', False)
        self.reduce_method = kwargs.get('reduce_method', 'svd')
        self.folder_config = kwargs.get('folder_config', '.')
        if self.folder_config is not None:
            self.df_pos = load_kwargs(folder_config=self.folder_config, kwargs='df_pos')
        else:
            self.df_pos = None
        self.graph_dataset_train = graph_dataset_train
        self.graph_dataset_val = graph_dataset_val
        self.graph_dataset_test = graph_dataset_test
        self.dataframe = self.graph_dataset_train.dataframe
        self.data = self.graph_dataset_train.data

    def build_graph(self, algo, **kwargs):
        """
        Build or load an adjacency matrix using a specified graph construction algorithm.

        Parameters
        ----------
        algo : str
            Graph construction method. Options: ``'space'``, ``'correlation'``,
            ``'precision'``, ``'gl3sr'``, or ``'dtw'``.
        **kwargs : dict
            Algorithm-specific hyperparameters (e.g., threshold, alpha, beta).

        Returns
        -------
        torch.Tensor
            Adjacency matrix of shape (N, N).

        Raises
        ------
        NotImplementedError
            If the specified algorithm is not supported.
        """
        print(f"Algorithm to build graph: {algo}")
        if self.load_graph:
            file = os.path.join("graph_representations", algo, "W.txt")
            W = np.loadtxt(file)
            print(f"Loaded graph file {file}.")
        else:
            Y = np.asarray(self.reduce_signal(**kwargs), float)
            bad_mask = ~np.isfinite(Y)
            if bad_mask.any():
                print(f"[GraphBuilder] Warning: found {bad_mask.sum()} non-finite entries in Y, setting to 0.")
                Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
            N = self.graph_dataset_train.num_nodes
            if algo == "space":
                threshold = kwargs.get("threshold", 0.0)
                stations_np = self.df_pos[["LONGITUDE", "LATITUDE"]].to_numpy()
                dist_mat_condensed = pdist(stations_np, metric=get_geodesic_distance)
                sigma = np.median(dist_mat_condensed)
                W_space = squareform(
                    get_exponential_similarity(dist_mat_condensed, sigma, threshold)
                )
                W = W_space + np.eye(N)
            elif algo == "correlation":
                corr = 1 - np.corrcoef(Y)
                bandwidth = np.median(corr)
                W = get_exponential_similarity(corr, bandwidth=bandwidth, threshold=0.0)
            elif algo == "precision":
                cov = np.cov(Y)
                # guard again in case cov picks up numerical issues
                if not np.isfinite(cov).all():
                    print("[GraphBuilder] Warning: non-finite entries in cov, nan_to_num.")
                    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
                prec = pinv(cov)
                prec = 1 - (prec - np.min(prec)) / (np.max(prec) - np.min(prec))
                bandwidth = np.median(prec)
                W = get_exponential_similarity(prec, bandwidth=bandwidth, threshold=0.0)
            elif algo == "gl3sr":
                a = kwargs.get("a", 0.98)
                alpha = kwargs.get("alpha", 1e-4)
                beta = kwargs.get("beta", 1500)
                gl3sr = GL_3SR.FGL_3SR(trace=N, beta=beta, alpha=alpha,
                                       maxit=100, verbose=True, cv_crit=1e-11)
                gl3sr.fit(Y.astype(np.double))
                X, H, lbd, err = gl3sr.get_coeffs()
                Lpred = X.dot(np.diag(lbd)).dot(X.T)
                Wpred = np.diag(np.diag(Lpred)) - Lpred
                Wpred = (Wpred + Wpred.T) / 2
                W = a * np.eye(N) + (1 - a) * Wpred
            elif algo == "dtw":
                l2_norm = lambda x, y: (x - y) ** 2
                mat = np.zeros((N, N), dtype=float)
                for i in tqdm(range(N)):
                    for k in range(N - i):
                        j = k + i
                        a, b = Y[i], Y[j]
                        distance, _ = fastdtw(a, b, dist=l2_norm)
                        mat[i, j] += distance
                mat += mat.T
                bandwidth = np.median(mat)
                W = get_exponential_similarity(mat, bandwidth=bandwidth, threshold=0.0)
            else:
                raise NotImplementedError(f"Algorithm {algo} not implemented.")
        return torch.tensor(W)

    def reduce_signal(self, **kwargs):
        """
        Compute or load a reduced signal representation from the dataset.
        
        Parameters
        ----------
        **kwargs : dict
            Method-specific parameters (e.g., ``k_max``, ``model_base``, ``num_epochs``).

        Returns
        -------
        np.ndarray
            Reduced feature matrix of shape (num_nodes, num_features).
        """
        if self.load_signal:
            file = os.path.join("signal_representations", self.reduce_method, "Y.txt")
            reduced_signal = np.loadtxt(file)
            print(f"Loaded signal file {file}.")
        else:
            if self.reduce_method.lower() == "svd":
                signals = []
                node_var_int = f"{self.data.node_var}Int"
                for r in range(self.graph_dataset_train.num_nodes):
                    signal = self.dataframe[
                        self.dataframe[node_var_int] == r
                    ][self.graph_dataset_train.features].to_numpy()
                    # signal: (T_r, F).
                    if signal.size == 0:
                        # no data -> use zeros
                        F = len(self.graph_dataset_train.features)
                        signals.append(np.zeros((1, F), dtype=float))
                        continue
                    node_vec = signal.mean(axis=0, keepdims=True)  # (1, F)
                    signals.append(node_vec)
                # shape: (num_nodes, F)
                reduced_signal = np.concatenate(signals, axis=0)
            elif self.reduce_method.lower() == "identity":
                # e.g., mean over time per node-feature
                node_var_int = f"{self.data.node_var}Int"
                signals = []
                for r in range(self.graph_dataset_train.num_nodes):
                    signal = self.dataframe[
                        self.dataframe[node_var_int] == r
                    ][self.graph_dataset_train.features].to_numpy()
                    if signal.size == 0:
                        F = len(self.graph_dataset_train.features)
                        signals.append(np.zeros((1, F), dtype=float))
                    else:
                        signals.append(signal.mean(axis=0, keepdims=True))
                reduced_signal = np.concatenate(signals, axis=0)
            elif self.reduce_method.lower() == "resiter":
                k_max = kwargs.get("k_max", 10)
                threshold = kwargs.get("threshold", 0.71)
                model_base = kwargs.get("model_base", "sage").lower()
                hidden_channels = kwargs.get("hidden_channels", 128)
                num_layers = kwargs.get("num_layers", 3)
                num_epochs = kwargs.get("num_epochs", 10)
                if model_base == "sage":
                    gnn_base = myGNN(
                        in_channels=self.graph_dataset_train.num_node_features,
                        num_layers=num_layers,
                        hidden_channels=hidden_channels,
                        out_channels=1,
                        conv_class=SAGEConv,
                        conv_kwargs={'project': True},
                    )
                elif model_base == "gcn":
                    gnn_base = myGNN(
                        in_channels=self.graph_dataset_train.num_node_features,
                        num_layers=num_layers,
                        hidden_channels=hidden_channels,
                        out_channels=1,
                        conv_class=GCNConv,
                    )
                else:
                    raise NotImplementedError("Only GCN and SAGE are implemented!")

                for k in range(k_max):
                    if k == 0:
                        W_iter = torch.tensor(
                            self.build_graph(algo="space", threshold=threshold),
                            dtype=torch.float32,
                        )
                    else:
                        W_iter = torch.tensor(np.corrcoef(reduced_signal), dtype=torch.float32)

                    edge_index, edge_weight = dense_to_sparse(W_iter)
                    trainer = Trainer(
                        model=gnn_base,
                        dataset_train=self.graph_dataset_train,
                        dataset_val=self.graph_dataset_val,
                        dataset_test=self.graph_dataset_val,
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                    )
                    pred_model_val, target_val, _, _ = trainer.train(
                        num_epochs=num_epochs, force_training=True
                    )
                    reduced_signal = target_val - pred_model_val
                    print(
                        f"distance between matrices: {np.linalg.norm(np.corrcoef(reduced_signal) - W_iter.numpy())}"
                    )
            else:
                raise NotImplementedError(f"Reduction method {self.reduce_method} not implemented.")
        return reduced_signal
