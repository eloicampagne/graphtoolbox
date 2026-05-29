from graphtoolbox.data.dataset import *
from graphtoolbox.utils.helper_functions import *
import inspect
import optuna
from optuna_dashboard import run_server
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn.conv import GATConv
from tqdm import tqdm

class Optimizer():
    """
    Hyperparameter optimizer for graph neural networks using Optuna.

    The `Optimizer` class automates hyperparameter tuning of GNN models
    on training and validation datasets. It supports structured logging,
    pruning of poor trials, and optional dashboard visualization via
    `optuna-dashboard`.

    Parameters
    ----------
    model : torch.nn.Module
        GNN model class to be optimized (not an instance).
    dataset_train : GraphDataset
        Training dataset.
    dataset_val : GraphDataset
        Validation dataset.
    optim_kwargs : dict, optional
        Search space definition for hyperparameters.
        Example: ``{"hidden_channels": (32, 128), "num_layers": (2, 5), "lr": (1e-4, 1e-2)}``.
        Loaded from the configuration folder if not provided.
    num_epochs : int, optional
        Number of epochs to train each trial. Default is 200.
    conv_class : torch_geometric.nn.MessagePassing, optional
        Convolution class to use in the model (default: ``GATv2Conv``).

    Attributes
    ----------
    study : optuna.Study
        Optuna study object containing all trials and results.
    storage : optuna.storages.InMemoryStorage
        In-memory storage backend for optimization results.
    is_optimized : bool
        Whether the optimization process has been executed.
    logger : logging.Logger
        Logger instance for progress and diagnostic output.

    Examples
    --------
    >>> opt = Optimizer(model=myGNN, dataset_train=train_set, dataset_val=val_set)
    >>> opt.optimize(n_trials=30)
    >>> opt.run_on_server()  # visualize results
    """
    def __init__(self, model, dataset_train: GraphDataset, dataset_val: GraphDataset, out_channels: int = 48, optim_kwargs: Dict = None, **kwargs):
        self.model_class = model
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.out_channels = out_channels
        if optim_kwargs is None:
            self.optim_kwargs = load_kwargs(folder_config=dataset_train.folder_config, kwargs='optim_kwargs')
        else:
            self.optim_kwargs = optim_kwargs
        self.num_epochs = kwargs.get('num_epochs', 200)
        self.conv_class = kwargs.get('conv_class', GATConv)
        self.loss_fn    = kwargs.get('loss_fn', 'mse')  # 'mse' or 'nmae'
        self.is_optimized = False

    def _run_epoch(self, optimizer, mode: str, loader: PyGDataLoader) -> float:
        """
        Run a single epoch of training or evaluation.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer instance (used only when `mode='train'`).
        mode : {'train', 'eval'}
            Operating mode for model update or evaluation.
        loader : torch_geometric.loader.DataLoader
            Data loader providing graph batches.

        Returns
        -------
        float
            Average RMSE loss for the epoch.

        Notes
        -----
        - Skips batches containing NaN values in features or targets.
        - Applies gradient clipping to stabilize training.
        """
 
        assert mode in ['train', 'eval']
        num_nodes = self.dataset_train.num_nodes
        self.model.train() if mode == 'train' else self.model.eval()
        total_loss, count = 0.0, 0
        for i, batch in enumerate(loader):
            batch = batch.to(DEVICE)
            batch.x = batch.x.float()
            if hasattr(batch, 'edge_weight') and batch.edge_weight is not None:
                batch.edge_weight = batch.edge_weight.float()
            if torch.isnan(batch.x).any() or torch.isnan(batch.y_scaled).any():
                print(f"[WARN] NaN detected in batch {i}. Skipping batch.")
                continue
            
            out = self.model(batch.x, batch.edge_index, edge_weight=getattr(batch, 'edge_weight', None)).squeeze().view(-1, num_nodes).T           
 
            y_s = batch.y_scaled.view(-1, num_nodes).T
            mask = batch.mask_y.view(-1, num_nodes).T  
            if mask.sum() > 0:
                if self.loss_fn == 'nmae':
                    loss = (torch.sum(torch.abs(out - y_s) * mask)
                            / (torch.sum(torch.abs(y_s) * mask) + 1e-6))
                else:
                    loss = torch.sum(((out - y_s) ** 2) * mask) / mask.sum()
            else:
                print(f"[WARN] Batch {i} ignoré car aucune cible valide")
                continue
            del y_s, out
            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
 
            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs

            if count > 0:
                return total_loss / count, optimizer
            else:
                return 0, optimizer
            
    def _build_model_tag(self, hp, batch_size):
        """
        Build checkpoint tag from sampled hyperparameters (except lr, batch_size, adj_matrix).
        """
        if not isinstance(hp, dict):
            raise TypeError(f"_build_model_tag expects a dict, got {type(hp)}")

        skip = {"lr", "batch_size", "adj_matrix"}
        parts = [f"batch{batch_size}"]

        for key, val in hp.items():
            if key not in skip:
                parts.append(f"{key}{val}")

        return "_".join(parts)


    def _define_model(self, trial):
        """
        Build a model instance for a given Optuna trial.

        Supports both:
        - Classic GNNs: model(in_channels, hidden_channels, num_layers, ...)
        - AdditiveGraphModel: model(feature_group_dims, hidden_channels, gnn_layers, ...)
        """
        sampled = {}
        for param, (vmin, vmax) in self.optim_kwargs.items():
            if param == "batch_size":
                continue

            if isinstance(vmin, int):
                sampled[param] = trial.suggest_int(param, vmin, vmax)
            elif isinstance(vmin, float):
                sampled[param] = (
                    trial.suggest_float(param, vmin, vmax, log=(param == "lr"))
                )
                if param == "lr":
                    self.lr = sampled[param]
            else:
                raise NotImplementedError(
                    f"[Optimizer] Unsupported type for param {param}: {type(vmin)}"
                )

        sig = inspect.signature(self.model_class.__init__)
        # Parameters explicitly belonging to the model
        model_param_names = {
            p.name
            for p in sig.parameters.values()
            if p.name not in ("self", "args", "kwargs")
        }
        model_kwargs = {k: v for k, v in sampled.items() if k in model_param_names}
        conv_kwargs = {k: v for k, v in sampled.items() if k not in model_param_names}

        is_additive = self.model_class.__name__ == "AdditiveGraphModel"
        if is_additive:
            feature_group_dims = self.dataset_train.feature_group_dims
            if feature_group_dims is None:
                raise RuntimeError(
                    "[Optimizer] feature_group_dims is required for AdditiveGraphModel."
                )

            model = self.model_class(
                feature_group_dims=feature_group_dims,
                out_channels=self.out_channels,
                conv_class=self.conv_class,
                conv_kwargs=conv_kwargs,
                **model_kwargs,
            )

        else:
            model = self.model_class(
                in_channels=self.dataset_val.num_node_features,
                out_channels=self.out_channels,
                conv_class=self.conv_class,
                conv_kwargs=conv_kwargs,
                **model_kwargs,
            )

        return model


    def _objective(self, trial):
        """
        Objective function evaluated by Optuna for each trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial used to sample hyperparameters.

        Returns
        -------
        float
            Validation RMSE (lower is better).

        Notes
        -----
        - Defines loaders for train/val sets.
        - Saves best-performing model state per trial.
        - Supports pruning based on intermediate results.
        """
        self.model = self._define_model(trial).to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        adj_matrix = trial.suggest_categorical('adj_matrix', os.listdir(self.dataset_train.graph_folder))
        self.dataset_train._set_adj_matrix(adj_matrix=adj_matrix)
        self.dataset_val._set_adj_matrix(adj_matrix=adj_matrix)
        tag = self._build_model_tag(trial.params, batch_size)
        mname = self.model.__class__.__name__
        saving_directory = f"./checkpoints_optim/{mname}/{adj_matrix}/{tag}"
        os.makedirs(saving_directory, exist_ok=True)

        train_loader = PyGDataLoader(self.dataset_train, batch_size=batch_size, shuffle=True)
        val_loader = PyGDataLoader(self.dataset_val, batch_size=batch_size, shuffle=False)
        train_losses = []
        val_losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(self.num_epochs)):
            params_filename = 'epoch{}.params'.format(epoch)
            train_loss, optimizer = self._run_epoch(optimizer=optimizer, mode='train', loader=train_loader)
            train_losses.append(train_loss)
            val_loss, _ = self._run_epoch(optimizer=optimizer, mode='eval', loader=val_loader)
            val_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                clean_dir(saving_directory)
                torch.save(self.model.state_dict(), os.path.join(saving_directory, params_filename))
            trial.report(best_loss, epoch)
            if trial.should_prune():
                shutil.rmtree(saving_directory)
                raise optuna.exceptions.TrialPruned()
        return best_loss
    
    def optimize(self, **kwargs):
        """
        Run the Optuna optimization loop.

        Parameters
        ----------
        study_name : str, optional
            Name for the Optuna study. Default is derived from the model class.
        n_trials : int, default=100
            Number of hyperparameter trials to perform.
        direction : {'minimize', 'maximize'}, default='minimize'
            Optimization direction for the objective function.
        timeout : int, optional
            Maximum runtime in seconds.

        Side Effects
        ------------
        - Saves best parameters and statistics to `./results_optim_<ConvClass>/`.
        - Logs progress to `./logs/optimization_<date>.log`.

        Notes
        -----
        Uses in-memory storage by default but can be extended for database-backed
        studies if persistence is needed.
        """
        self.storage = optuna.storages.InMemoryStorage()
        self.is_optimized = True
        self.study = optuna.create_study(storage=self.storage,
                                         study_name=kwargs.get('study_name', f'{self.model_class.__name__}_hpo'),
                                         direction=kwargs.get('direction', 'minimize'))
        self.study.optimize(self._objective, n_trials=kwargs.get('n_trials', 100), timeout=kwargs.get('timeout', 10000))
        self.pruned_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        self.complete_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of pruned trials: ", len(self.pruned_trials))
        print("  Number of complete trials: ", len(self.complete_trials))

        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        result_dir = f"./results_optim_{self.conv_class.__name__}"
        os.makedirs(result_dir, exist_ok=True)

        result_file = os.path.join(
            result_dir,
            f"{self.model_class.__name__}.txt"
        )

        with open(result_file, 'a') as f:
            f.write("Study statistics:\n")
            f.write(f"  Number of finished trials: {len(self.study.trials)}\n")
            f.write(f"  Number of pruned trials: {len(self.pruned_trials)}\n")
            f.write(f"  Number of complete trials: {len(self.complete_trials)}\n\n")

            f.write("Best trial:\n")
            f.write(f"  Value: {trial.value}\n\n")
            f.write("  Params:\n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")

        import json as _json
        json_file = os.path.join(result_dir, f"{self.model_class.__name__}_best.json")
        with open(json_file, 'w') as f:
            _json.dump({'value': trial.value, 'params': trial.params}, f, indent=2)

    def run_on_server(self):
        """
        Launch an interactive Optuna dashboard to visualize study results.

        Requires that `optimize()` has already been executed.

        Raises
        ------
        RuntimeError
            If called before the optimization is completed.
        """
        if self.is_optimized:
            run_server(self.storage)
        else:
            print('You need to optimize your model first!')