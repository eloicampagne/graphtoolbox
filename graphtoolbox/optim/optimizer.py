from graphtoolbox.data.dataset import *
from graphtoolbox.utils.helper_functions import *
from graphtoolbox.utils.reproducibility import set_seed, loader_generator
import inspect
import warnings
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
    seed : int, optional
        Seeds the TPE sampler, and seeds trial ``n`` with ``seed + n`` before its
        model is built and trained. Without it the study is not reproducible.
    deterministic : bool, optional
        With ``seed``, request deterministic kernels (CPU, CUDA). Default False.
    fused_adam : bool, optional
        Fused Adam kernel: faster, not bitwise equal to the default. Default False.
    patience : int or None, optional
        Stop a trial after this many epochs without a better validation loss;
        the trial returns its best loss. Default 20; None trains every trial for
        ``num_epochs``.
    min_delta : float, optional
        Smallest decrease of the validation loss that counts as better. Default 0.

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
        # With a seed, the TPE sampler and every trial are seeded: trial n is
        # built and trained under seed + n, so a study replays identically.
        self.seed = kwargs.get('seed', None)
        self.deterministic = kwargs.get('deterministic', False)
        self.fused_adam = kwargs.get('fused_adam', False)
        self.patience = kwargs.get('patience', 20)
        self.min_delta = kwargs.get('min_delta', 0.0)
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
        # Batch losses read from the device once per epoch; batches tested
        # for NaNs or empty targets only if the dataset holds some (see
        # batch_guards): both avoid a device synchronization per step.
        batch_losses, batch_sizes = [], []
        guards = self.__dict__.setdefault('_batch_guard_cache', {})
        if id(loader.dataset) not in guards:
            guards[id(loader.dataset)] = batch_guards(loader.dataset)
        has_nan, has_empty = guards[id(loader.dataset)]
        # No autograd graph in evaluation.
        with torch.set_grad_enabled(mode == 'train'):
            for i, batch in enumerate(loader):
                batch = batch.to(DEVICE)
                batch.x = batch.x.float()
                if hasattr(batch, 'edge_weight') and batch.edge_weight is not None:
                    batch.edge_weight = batch.edge_weight.float()
                if has_nan and (torch.isnan(batch.x).any() or torch.isnan(batch.y_scaled).any()):
                    print(f"[WARN] NaN detected in batch {i}. Skipping batch.")
                    continue

                out = self.model(batch.x, batch.edge_index, edge_weight=getattr(batch, 'edge_weight', None)).squeeze().view(-1, num_nodes).T

                y_s = batch.y_scaled.view(-1, num_nodes).T
                mask = batch.mask_y.view(-1, num_nodes).T
                if has_empty and mask.sum() == 0:
                    print(f"[WARN] Skipping batch {i}: no valid target")
                    continue
                if self.loss_fn == 'nmae':
                    loss = (torch.sum(torch.abs(out - y_s) * mask)
                            / (torch.sum(torch.abs(y_s) * mask) + 1e-6))
                else:
                    loss = torch.sum(((out - y_s) ** 2) * mask) / mask.sum()
                del y_s, out
                if mode == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()

                batch_losses.append(loss.detach())
                batch_sizes.append(batch.num_graphs)

        # After the loop: this return used to sit inside it, so every epoch
        # trained and validated on a single batch.
        return epoch_mean(batch_losses, batch_sizes), optimizer
            
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
        - Stops after ``patience`` epochs without improvement.
        - Keeps the best weights in memory and saves them once, when the trial
          completes; pruned trials save nothing.
        - Supports pruning based on intermediate results.
        """
        trial_seed = None if self.seed is None else self.seed + trial.number
        if trial_seed is not None:
            set_seed(trial_seed, deterministic=self.deterministic)
        self.model = self._define_model(trial).to(DEVICE)
        optimizer = make_adam(self.model.parameters(), lr=self.lr, fused=self.fused_adam)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        # Sorted, and folders only: os.listdir order depends on the filesystem, so
        # a seeded TPE would draw other graphs on another machine, and stray files
        # (.DS_Store) would be offered as graphs.
        graph_folder = self.dataset_train.graph_folder
        adj_choices = sorted(d for d in os.listdir(graph_folder)
                             if os.path.isdir(os.path.join(graph_folder, d)))
        adj_matrix = trial.suggest_categorical('adj_matrix', adj_choices)
        self.dataset_train._set_adj_matrix(adj_matrix=adj_matrix)
        self.dataset_val._set_adj_matrix(adj_matrix=adj_matrix)
        tag = self._build_model_tag(trial.params, batch_size)
        mname = self.model.__class__.__name__
        saving_directory = f"./checkpoints_optim/{mname}/{adj_matrix}/{tag}"

        train_loader = PyGDataLoader(self.dataset_train, batch_size=batch_size, shuffle=True,
                                     generator=loader_generator(trial_seed))
        val_loader = PyGDataLoader(self.dataset_val, batch_size=batch_size, shuffle=False)
        train_losses = []
        val_losses = []
        best_loss = float('inf')
        best_epoch, best_state, epoch = -1, None, -1
        for epoch in tqdm(range(self.num_epochs)):
            train_loss, optimizer = self._run_epoch(optimizer=optimizer, mode='train', loader=train_loader)
            train_losses.append(train_loss)
            val_loss, _ = self._run_epoch(optimizer=optimizer, mode='eval', loader=val_loader)
            val_losses.append(val_loss)
            if val_loss < best_loss - self.min_delta:
                best_loss, best_epoch = val_loss, epoch
                # In memory: a checkpoint written at every improvement costs disk
                # I/O, and a pruned trial would throw it away.
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            trial.report(best_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            # A trial whose validation loss has stopped improving would otherwise
            # train until num_epochs.
            if self.patience is not None and epoch - best_epoch >= self.patience:
                break
        trial.set_user_attr('epochs', epoch + 1)
        trial.set_user_attr('best_epoch', best_epoch + 1)
        if best_state is not None:
            os.makedirs(saving_directory, exist_ok=True)
            clean_dir(saving_directory)
            torch.save(best_state, os.path.join(saving_directory, f'epoch{best_epoch}.params'))
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
            Maximum runtime in seconds (default 10000). The study stops at the
            first trial that ends after it, possibly before ``n_trials``; a
            warning reports how many trials ran.
        pruner : optuna.pruners.BasePruner, optional
            Pruner of the study. Default: Optuna's ``MedianPruner``. A
            ``optuna.pruners.HyperbandPruner()`` stops poor trials earlier.

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
        sampler = None if self.seed is None else optuna.samplers.TPESampler(seed=self.seed)
        self.study = optuna.create_study(storage=self.storage,
                                         study_name=kwargs.get('study_name', f'{self.model_class.__name__}_hpo'),
                                         direction=kwargs.get('direction', 'minimize'),
                                         sampler=sampler,
                                         pruner=kwargs.get('pruner', None))
        n_trials, timeout = kwargs.get('n_trials', 100), kwargs.get('timeout', 10000)
        self.study.optimize(self._objective, n_trials=n_trials, timeout=timeout)
        if n_trials is not None and len(self.study.trials) < n_trials:
            warnings.warn(f"The study stopped after {len(self.study.trials)} of {n_trials} "
                          f"trials: the timeout of {timeout} s was reached. Raise timeout, "
                          "or pass timeout=None, to run every trial.", stacklevel=2)
        self.pruned_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        self.complete_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of pruned trials: ", len(self.pruned_trials))
        print("  Number of complete trials: ", len(self.complete_trials))

        if not self.complete_trials:
            # study.best_trial would raise: every trial was pruned, failed, or
            # never started.
            warnings.warn("No trial completed: there is no best configuration to report.",
                          stacklevel=2)
            return

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