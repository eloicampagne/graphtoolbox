import copy
import re
from graphtoolbox.data.dataset import GraphDataset
import graphtoolbox.training.metrics
from graphtoolbox.utils.helper_functions import *
from graphtoolbox.utils.visualizations import *
import numpy as np
import os
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
from typing import List, Tuple, Type, Union


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.

    Parameters
    ----------
    patience : int, default=10
        Number of epochs with no improvement after which training will be stopped.
    min_delta : float, default=0.0
        Minimum change in validation loss to qualify as an improvement.

    Attributes
    ----------
    counter : int
        Number of consecutive epochs without improvement.
    best_loss : float
        Lowest recorded validation loss.
    early_stop : bool
        Whether the early stopping condition has been met.

    Examples
    --------
    >>> stopper = EarlyStopping(patience=5, min_delta=0.01)
    >>> for epoch in range(100):
    ...     val_loss = compute_validation_loss()
    ...     stopper(val_loss)
    ...     if stopper.early_stop:
    ...         print("Stopped early at epoch", epoch)
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Update early stopping state with the latest validation loss.

        Parameters
        ----------
        val_loss : float
            Current validation loss for this epoch.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Trainer:
    """
    Train, validate, and evaluate a graph neural network model on temporal graph datasets.

    This class handles the full training loop, including:
    - batched training with PyTorch Geometric loaders,
    - validation and early stopping,
    - checkpointing and loss tracking,
    - optional attention collection and saving,
    - optional per-group outputs for additive models (sum to final prediction),
    - inference and hierarchical reconciliation (MinT).

    Parameters
    ----------
    model : torch.nn.Module
        Graph neural network model to train.
    dataset_train : GraphDataset
        Training dataset.
    dataset_val : GraphDataset
        Validation dataset.
    dataset_test : GraphDataset
        Test dataset.
    batch_size : int
        Number of graph samples per batch.
    model_kwargs : dict, optional
        Dictionary of model hyperparameters (loaded from config if None).
    reconcile : bool, default=True
        Whether to apply MinT reconciliation to predictions.
    **kwargs :
        Optional keyword-only arguments:
            - edge_index (torch.Tensor[2, E])
            - edge_weight (torch.Tensor[E]) or None
            - return_attention (bool): collect attention during validation/test.
            - return_group_outputs (bool): ask additive models to return group-wise contributions.
            - lam_reg (float): graph smoothness regularizer weight (pairwise node prediction L2).

    Attributes
    ----------
    is_trained : bool
        Whether the model has been trained.
    train_loader, val_loader, test_loader : PyGDataLoader
        Dataloaders for training, validation, and test.
    saving_directory : str
        Path to saved model checkpoints.
    S, G, P : torch.Tensor
        Matrices for hierarchical MinT reconciliation.
    nodes : list[str]
        Node names (order used by the model).
    num_nodes : int
        Number of base nodes.

    Notes
    -----
    When the model class name is 'AdditiveGraphModel' and return_group_outputs is not set,
    Trainer will auto-enable group output collection.
    """
    def __init__(self, model, dataset_train, dataset_val, dataset_test, batch_size,
                 model_kwargs: Optional[Dict] = None, reconcile: bool = True,
                 top_forecasts_train: Optional[torch.Tensor] = None,
                 top_forecasts_val: Optional[torch.Tensor] = None,
                 top_forecasts_test: Optional[torch.Tensor] = None,
                 top_level_model: Optional[str] = 'ridge',
                 **kwargs):
        self.edge_index = kwargs.get('edge_index', None)
        self.edge_weight = kwargs.get('edge_weight', None)
        self.return_attention = kwargs.get('return_attention', False)
        self.return_group_outputs = kwargs.get('return_group_outputs', False)
        self.lam_reg = kwargs.get('lam_reg', 0)
        self.loss_fn = kwargs.get('loss_fn', 'mse')  # 'mse' or 'nmae'

        self.model = model.to(DEVICE)
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.reconcile = reconcile
        self.nodes = self.dataset_train.nodes
        self.num_nodes = self.dataset_train.num_nodes
        self.folder_config = dataset_train.data.folder_config

        self.top_forecasts_train = top_forecasts_train.to(DEVICE) if top_forecasts_train is not None else None
        self.top_forecasts_val = top_forecasts_val.to(DEVICE) if top_forecasts_val is not None else None
        self.top_forecasts_test = top_forecasts_test.to(DEVICE) if top_forecasts_test is not None else None

        dataset_kwargs = dataset_train.dataset_kwargs
        if dataset_kwargs is None:
            dataset_kwargs = load_kwargs(folder_config=self.folder_config, kwargs='dataset_kwargs')
        self.dataset_kwargs = dataset_kwargs

        if model_kwargs is None:
            model_kwargs = load_kwargs(folder_config=self.folder_config, kwargs='model_kwargs')
        self.model_kwargs = model_kwargs
        self.is_trained = False

        self.batch_size = batch_size
        self.train_loader = PyGDataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        self.val_loader = PyGDataLoader(dataset_val, batch_size=self.batch_size, shuffle=False)
        self.test_loader = PyGDataLoader(dataset_test, shuffle=False, drop_last=False)

        self._build_summing_matrix()
        # Use identity W at init (model not yet trained); recomputed with error covariance after training.
        self._compute_min_trace_projection(W=torch.eye(self.S.shape[0], device=DEVICE))

        if self.reconcile:
            if any(f is not None for f in [self.top_forecasts_train, self.top_forecasts_val, self.top_forecasts_test]):
                print("[Trainer] Using external top-level forecasts for MinT reconciliation.")
            elif top_level_model is not None:
                self._fit_top_level_model(top_level_model)

    def train(self, **kwargs) -> Tuple:
        """
        Train the model and optionally evaluate during training.

        Supports early stopping, checkpoint saving, attention visualization and
        (for additive models) returning per-group outputs.

        Parameters
        ----------
        num_epochs : int, optional
            Number of epochs to train (default: from model_kwargs).
        optimizer : torch.optim.Optimizer, optional
            Optimizer (default: Adam with model_kwargs['lr']).
        patience : int, default=20
            Early stopping patience.
        min_delta : float, default=0.0
            Minimum delta to count as improvement.
        force_training : bool, default=False
            If True, retrains even if checkpoint exists.
        saving_directory : str, optional
            Folder to store model weights.
        plot_loss : bool, optional
            If True, plots training/validation curves.
        dynamic_graph : bool, optional
            Enable dynamic adjacency matrix updates per epoch.
        save : bool, optional
            If True, saves attention maps.

        Returns
        -------
        Tuple
            If return_attention is False and return_group_outputs is False:
                (preds, targets, edge_index, edge_weight)
            If return_attention is True:
                (preds, targets, edge_index, attention_mats)
            If return_group_outputs is True and return_attention is False:
                (preds, targets, edge_index, edge_weight, group_outputs)
            If both are True:
                (preds, targets, edge_index, attention_mats, group_outputs)

            Where:
            - preds : torch.Tensor[num_nodes, T] in original units (reconciled if enabled)
            - targets : torch.Tensor[num_nodes, T] in original units
            - attention_mats : dict[str, list[torch.Tensor]] if collected
            - group_outputs : dict[str, torch.Tensor[num_nodes, T]] if collected
        
        Notes
        -----
        - Applies graph smoothness regularization weighted by lam_reg.
        - Uses early stopping with best-checkpoint saving.
        """
        self.num_epochs = kwargs.get('num_epochs', self.model_kwargs['num_epochs'])
        optimizer = kwargs.get('optimizer', torch.optim.Adam(self.model.parameters(), lr=self.model_kwargs['lr']))
        force_training = kwargs.get('force_training', False)
        patience = kwargs.get('patience', 20)
        min_delta = kwargs.get('min_delta', 0.0)
        self.dynamic_graph = kwargs.get('dynamic_graph', False)
        save = kwargs.get('save', False)

        # Auto-enable group outputs for additive model if not explicitly set
        if (self.model.__class__.__name__ == 'AdditiveGraphModel') and not getattr(self, 'return_group_outputs', False):
            self.return_group_outputs = True
        
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        if hasattr(self.model, 'conv_class'):
            self.model_name = self.model.conv_class.__name__
        else:
            self.model_name = self.model.__class__.__name__
        self.hidden_channels = self.model.hidden_channels
        self.num_layers = self.model.num_layers
        self.adj_matrix = self.dataset_train.adj_matrix
        Lattention_mat = []
        try:
            self.heads = self.model.heads
        except:
            self.heads = 0
        saving_directory = kwargs.get('saving_directory',
                                    f'./checkpoints/{self.model_name}_{self.adj_matrix}/batch{self.batch_size}_hidden{self.hidden_channels}_layers{self.num_layers}_epochs{self.num_epochs}')
        if hasattr(self.model, 'conv_kwargs'):
                for k, v in self.model.conv_kwargs.items():
                    saving_directory += f'_{k}{v}'
        self.saving_directory = saving_directory

        # Resume from a partial checkpoint (fewer epochs, same other hyperparams)
        start_epoch = 0
        if not force_training and (not os.path.exists(saving_directory) or len(os.listdir(saving_directory)) == 0):
            parent_dir = os.path.dirname(saving_directory)
            dir_name   = os.path.basename(saving_directory)
            m = re.match(r'^(.*_epochs)(\d+)(.*?)$', dir_name)
            if m and os.path.isdir(parent_dir):
                prefix, target_epochs, suffix = m.group(1), int(m.group(2)), m.group(3)
                candidates = []
                for d in os.listdir(parent_dir):
                    dm = re.match(rf'^{re.escape(prefix)}(\d+){re.escape(suffix)}$', d)
                    if dm:
                        n = int(dm.group(1))
                        full = os.path.join(parent_dir, d)
                        if n < target_epochs and os.path.isdir(full) and len(os.listdir(full)) > 0:
                            candidates.append((n, full))
                if candidates:
                    best_n, best_path = max(candidates, key=lambda x: x[0])
                    ckpt = os.path.join(best_path, os.listdir(best_path)[0])
                    print(f"Resuming from {best_n}-epoch checkpoint: {ckpt}")
                    self.model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                    start_epoch = best_n

        if not os.path.exists(saving_directory) or len(os.listdir(saving_directory)) == 0 or force_training:
            print("Training model...")
            os.makedirs(saving_directory, exist_ok=True)
            clean_dir(saving_directory)
            train_losses = []
            val_losses = []
            best_loss = float('inf')
            num_epochs_final = self.num_epochs
            for epoch in tqdm(range(start_epoch, self.num_epochs)):
                params_filename = 'epoch{}.params'.format(epoch)
                train_loss = self._run_epoch(optimizer, 'train', self.train_loader, return_attention=False)
                if self.return_attention:
                    val_loss, attention_mat = self._run_epoch(optimizer, 'eval', self.val_loader, return_attention=self.return_attention)
                    Lattention_mat.append(attention_mat)
                else:
                    val_loss = self._run_epoch(optimizer, 'eval', self.val_loader)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    clean_dir(saving_directory)
                    torch.save(self.model.state_dict(), os.path.join(saving_directory, params_filename))

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    num_epochs_final = epoch + 1
                    break

            if kwargs.get('plot_loss', False):
                plot_losses(num_epochs_final, train_losses, val_losses, start_epoch=start_epoch)
        else:
            print("Loading pretrained model.")
            self.model.load_state_dict(torch.load(os.path.join(saving_directory, os.listdir(saving_directory)[0]), map_location=DEVICE))

        self.batch_size_save = kwargs.get('batch_size_save', self.batch_size)
        self.test_loader = PyGDataLoader(self.dataset_test, shuffle=False, drop_last=False)
        if self.return_attention and save:
            self.val_loader = PyGDataLoader(self.dataset_val, batch_size=self.batch_size_save, shuffle=False)
            self.train_loader = PyGDataLoader(self.dataset_train, batch_size = self.batch_size_save, shuffle=False)
            _ = self._run_epoch(optimizer, 'eval', self.train_loader, return_attention=self.return_attention, save=True, dataset_name='train')
            _ = self._run_epoch(optimizer, 'eval', self.val_loader, return_attention=self.return_attention, save=True, dataset_name='val')
            _ = self._run_epoch(optimizer, 'eval', self.test_loader, return_attention=self.return_attention, save=True, dataset_name='test')

        _, preds, targets = self._predict(self.test_loader, split='test')
        output_groups = getattr(self, 'group_outputs_test', None) if self.return_group_outputs else None

        self.is_trained = True
        if self.return_attention:
            self.return_attention = False
            if self.return_group_outputs:
                return (preds, targets, self.edge_index, Lattention_mat, output_groups)
            else:
                return (preds, targets, self.edge_index, Lattention_mat)
        else:
            if self.return_group_outputs:
                return (preds, targets, self.edge_index, self.edge_weight, output_groups)
            else:
                return (preds, targets, self.edge_index, self.edge_weight)

    def _run_epoch(self, optimizer, mode: str, loader: PyGDataLoader, return_attention: bool = False, save: bool = False, dataset_name: str = 'test') -> float:
        """
        Run a single training or evaluation epoch.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer used for parameter updates (only when mode='train').
        mode : {'train', 'eval'}
            Whether to update weights or evaluate.
        loader : PyGDataLoader
            DataLoader providing batched graph data.
        return_attention : bool, default=False
            Whether to collect and return attention weights.
        save : bool, default=False
            Whether to save attention maps on disk.
        dataset_name : str, default='test'
            Dataset label for saved outputs.

        Returns
        -------
        float or (float, dict)
            - If return_attention is False: average epoch loss (float).
            - If return_attention is True: (average loss, dict of aggregated attention).

        Notes
        -----
        - Converts batch attributes ['x','y_scaled','y','edge_weight','mask_y'] to float32 on DEVICE.
        - Loss = MSE over masked targets + lam_reg * graph-smoothness penalty.
        - For additive models with attention, averages attention across batches in eval.
        """
        assert mode in ['train', 'eval']
        num_nodes = self.dataset_train.num_nodes
        self.model.train() if mode == 'train' else self.model.eval()
        total_loss, count = 0.0, 0

        if save: 
            save_path = f"./attention_matrix/{self.model_name}_{self.adj_matrix}/{dataset_name}_batch{self.batch_size}_hidden{self.hidden_channels}_layers{self.num_layers}_epochs{self.num_epochs}_heads{self.heads}"
            self.save_path_MA = save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            clean_dir(save_path)

        tot_dict_attention = None
        attention_batches = 0

        # TODO: add dynamic graph condition
        for i, batch in enumerate(loader):
            for attr in ['x', 'y_scaled', 'y', 'edge_weight', 'mask_y']:
                if getattr(batch, attr, None) is not None:
                    setattr(batch, attr, getattr(batch, attr).to(torch.float32))
            batch = batch.to(DEVICE)
            if torch.isnan(batch.x).any() or torch.isnan(batch.y_scaled).any():
                print(f"[WARN] NaN detected in batch {i}. Skipping batch.")
                continue

            if (mode == 'eval') and return_attention:
                if save:
                    save_path = f"./attention_matrix/{self.model_name}_{self.adj_matrix}/{dataset_name}_batch{self.batch_size}_hidden{self.hidden_channels}_layers{self.num_layers}_epochs{self.num_epochs}_heads{self.heads}/num_batch{i}.pt"
                    model_out = self.model(
                        batch.x,
                        batch.edge_index,
                        edge_weight=getattr(batch, 'edge_weight', None),
                        mask=getattr(batch, 'mask_y', None),
                        return_attention=True,
                        return_group_outputs=self.return_group_outputs,
                        batch_size=self.batch_size,
                        save=True,
                        save_path=save_path
                    )
                    # Forward contract:
                    # (y_hat) or (y_hat, attention) or (y_hat, group_outputs, attention)
                    if isinstance(model_out, tuple):
                        out = model_out[0]
                        attention_per_group = model_out[-1]
                    else:
                        out = model_out
                    out = out.squeeze().view(-1, num_nodes).T
                else:
                    same_size = (getattr(batch, 'num_graphs', None) == self.batch_size)
                    if same_size:
                        model_out = self.model(
                            batch.x,
                            batch.edge_index,
                            edge_weight=getattr(batch, 'edge_weight', None),
                            mask=getattr(batch, 'mask_y', None),
                            return_attention=True,
                            return_group_outputs=self.return_group_outputs,
                            batch_size=self.batch_size
                        )
                        if isinstance(model_out, tuple):
                            out = model_out[0]
                            dict_attention = model_out[-1]  # attention_per_group
                        else:
                            out = model_out
                            dict_attention = {}
                        out = out.squeeze().view(-1, num_nodes).T

                        if tot_dict_attention is None:
                            tot_dict_attention = {k: [v_i.clone() if torch.is_tensor(v_i) else v_i
                                                      for v_i in v] for k, v in dict_attention.items()}
                            attention_batches = 1
                        else:
                            for k in tot_dict_attention.keys():
                                for j in range(len(tot_dict_attention[k])):
                                    a = tot_dict_attention[k][j]
                                    b = dict_attention.get(k, [None]*len(tot_dict_attention[k]))[j]
                                    if (a is None) or (b is None):
                                        continue
                                    if torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape:
                                        tot_dict_attention[k][j] = a + b
                            attention_batches += 1
                    else:
                        # Fallback: no attention aggregation for irregular batch size
                        model_out = self.model(
                            batch.x,
                            batch.edge_index,
                            edge_weight=getattr(batch, 'edge_weight', None),
                            mask=getattr(batch, 'mask_y', None),
                            return_attention=False,
                            return_group_outputs=False
                        )
                        out = model_out.squeeze().view(-1, num_nodes).T
            else:
                # Standard forward (training or eval without attention)
                model_out = self.model(
                    batch.x,
                    batch.edge_index,
                    edge_weight=getattr(batch, 'edge_weight', None),
                    mask=getattr(batch, 'mask_y', None),
                    return_attention=False,
                    return_group_outputs=False
                )
                out = model_out.squeeze().view(-1, num_nodes).T
 
            y_s = batch.y_scaled.view(-1, num_nodes).T
            mask = batch.mask_y.view(-1, num_nodes).T  
            if mask.sum() > 0:
                if self.loss_fn == 'nmae':
                    base_loss = (torch.sum(torch.abs(out - y_s) * mask)
                                 / (torch.sum(torch.abs(y_s) * mask) + 1e-6))
                else:
                    base_loss = torch.sum(((out - y_s) ** 2) * mask) / mask.sum()
            else:
                print(f"[WARN] Batch {i} ignoré car aucune cible valide")
                continue
            pred_diff = out[:, None, :] - out[None, :, :]
            norms = torch.norm(pred_diff, p=2, dim=2)
            reg_loss = norms.mean()
            loss = base_loss + self.lam_reg * reg_loss
            del y_s
            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
 
            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs

        # Finalize attention averaging using only the number of aggregated batches
        if (mode == 'eval') and (return_attention) and (not save) and (tot_dict_attention is not None) and (attention_batches > 0):
            for k in tot_dict_attention.keys():
                for j in range(len(tot_dict_attention[k])):
                    v = tot_dict_attention[k][j]
                    if torch.is_tensor(v):
                        tot_dict_attention[k][j] = v / attention_batches

        if (return_attention) and (not save):
            if count > 0:
                return (total_loss / count, tot_dict_attention if tot_dict_attention is not None else {})
            else:
                return (0, tot_dict_attention if tot_dict_attention is not None else {})
        else:
            if count > 0:
                return total_loss / count
            else:
                return 0

    def _predict(self, loader: PyGDataLoader, split: str = 'test') -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Inference over a loader and compute loss against ground truth (sum over nodes).

        Parameters
        ----------
        loader : PyGDataLoader
            DataLoader for the split to predict.
        split : str, default='test'
            Which split is being predicted ('train', 'val', or 'test').
            Used to select the appropriate top-level forecasts for MinT reconciliation.

        Returns
        -------
        loss : float
            RMSE between summed predictions and summed targets (in original units).
        preds : torch.Tensor
            Predictions in original units with shape [num_nodes, T]; reconciled if enabled.
        targets : torch.Tensor
            Ground-truth targets with shape [num_nodes, T].

        Side Effects
        ------------
        - When return_group_outputs is True, populates self.group_outputs_test as:
          dict[group_name] -> torch.Tensor[num_nodes, T] of unscaled group contributions.

        Notes
        -----
        - Predictions are inverse-transformed per node using dataset_train.scalers_target.
        - If reconcile=True, performs MinT reconciliation using external top forecasts if provided,
          otherwise uses the sum of base predictions.
        """
        self.model.eval()
        num_nodes = self.dataset_train.num_nodes
        y_preds, y_targets = [], []
        collect_groups = self.return_group_outputs
        group_outputs_list = []
        with torch.no_grad():
            for batch in loader:
                for attr in ['x', 'y_scaled', 'y', 'edge_weight', 'mask_y']:
                    if getattr(batch, attr, None) is not None:
                        setattr(batch, attr, getattr(batch, attr).to(torch.float32))
                batch = batch.to(DEVICE)
                if hasattr(batch, 'edge_weight') and batch.edge_weight is not None:
                    batch.edge_weight = batch.edge_weight.float()

                if collect_groups:
                    y_hat, group_outputs = self.model(
                        batch.x,
                        batch.edge_index,
                        edge_weight=getattr(batch, 'edge_weight', None),
                        mask=getattr(batch, 'mask_y', None),
                        return_group_outputs=True,
                        return_attention=False
                    )
                    group_outputs_list.append(group_outputs)
                else:
                    y_hat = self.model(
                        batch.x,
                        batch.edge_index,
                        edge_weight=getattr(batch, 'edge_weight', None),
                        mask=getattr(batch, 'mask_y', None),
                        return_group_outputs=False,
                        return_attention=False
                    )

                y_preds.append(y_hat)
                y_targets.append(batch.y.cpu().detach())
        y_targets = torch.hstack(y_targets)
        y_targets = y_targets.reshape(num_nodes, -1)
        y_preds = torch.hstack(y_preds)
        y_preds = y_preds.reshape(num_nodes, -1)[:, :y_targets.shape[1]]
        pred_rescaled = self._rescale_predictions(y_preds).cpu().detach()
        del y_preds

        top_forecast = {'train': self.top_forecasts_train, 'val': self.top_forecasts_val, 'test': self.top_forecasts_test}.get(split)

        if self.reconcile:
            pred_rescaled = self._min_trace_reconciliation(
                preds=pred_rescaled, top_forecast=top_forecast
            ).cpu().detach()
            # Strip appended national node if reconciliation added it.
            if pred_rescaled.shape[0] == self.num_nodes + 1:
                pred_rescaled = pred_rescaled[:-1]

        dataset_split = {'train': self.dataset_train, 'val': self.dataset_val, 'test': self.dataset_test}.get(split, self.dataset_test)
        if hasattr(dataset_split, 'mask_Y') and dataset_split.mask_Y is not None:
            mask = dataset_split.mask_Y  # [num_nodes, T] or [num_nodes, T, 1]
            T_pred = pred_rescaled.shape[1]
            m = mask[:, :T_pred]
            if m.dim() == 3:
                m = m.squeeze(-1)
            pred_rescaled = pred_rescaled * m.float().to(pred_rescaled.device)

        loss = getattr(graphtoolbox.training.metrics, 'RMSE')(
            preds=pred_rescaled.cpu().detach().sum(dim=0),
            targets=y_targets.sum(dim=0)
        ).item()

        if collect_groups:
            group_concat = {}  # final dict[group] = [num_nodes, T_full]
            group_names = list(group_outputs_list[0].keys())
            for g in group_names:
                group_concat[g] = []

            for group_dict in group_outputs_list:
                for gname, g_tensor in group_dict.items():
                    group_concat[gname].append(g_tensor.cpu())

            for gname in group_names:
                group_concat[gname] = torch.cat(group_concat[gname], dim=1)  # [num_nodes, T]

            unscaled_groups = {}
            for gname, g_mat in group_concat.items():   # g_mat: [num_nodes, T]

                g_unscaled = []
                for node_idx, node in enumerate(self.nodes):
                    scaler = self.dataset_train.scalers_target[node]
                    g_np = g_mat[node_idx].reshape(-1, 1).numpy()        # [T, 1]
                    g_unscaled_np = scaler.inverse_transform(g_np).reshape(-1)  # [T]
                    g_unscaled.append(torch.tensor(g_unscaled_np, dtype=torch.float32))

                unscaled_groups[gname] = torch.stack(g_unscaled, dim=0).to(DEVICE)  # [num_nodes, T]
            self.group_outputs_test = unscaled_groups
        return loss, pred_rescaled, y_targets

    def evaluate(self, losses: Union[List[str], str] = ['mape', 'rmse']):
        """
        Evaluate trained model on test set using given metrics.

        Parameters
        ----------
        losses : str or list of str, default=['mape', 'rmse']
            Metrics to compute. Supported: 'mape', 'rmse'.

        Returns
        -------
        None
            Prints evaluation metrics.
        """
        if not self.is_trained:
            print("You need to train the model first!")
            return

        _, preds, targets = self._predict(self.test_loader, split='test')

        if isinstance(losses, str):
            losses = [losses]

        for loss in losses:
            try:
                eps = 1e-6
                loss_fn = getattr(graphtoolbox.training.metrics, loss.upper())
                result = loss_fn(preds=preds.cpu().detach().sum(dim=0) + eps, targets=targets.sum(dim=0).cpu().detach() + eps)
                unit = "%" if loss.lower() == 'mape' else "MW"
                val = result.item() * 100 if loss.lower() == 'mape' else result.item()
                print(f"{loss.upper()} on test set: {val:.4f} {unit}")
            except AttributeError:
                print(f"Loss function {loss} not found.")
        
    def _rescale_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transform model predictions using stored target scalers.

        Parameters
        ----------
        preds : torch.Tensor
            Normalized predictions (num_nodes × T).

        Returns
        -------
        torch.Tensor
            Rescaled predictions (num_nodes × T) in original units (float32).
        """
        preds_rescaled = []
        for node_idx, node in enumerate(self.nodes):
            scaler = self.dataset_train.scalers_target[node]
            pred_np = preds[node_idx].detach().cpu().numpy().reshape(-1, 1)
            pred_rescaled_np = scaler.inverse_transform(pred_np).reshape(-1)
            preds_rescaled.append(torch.as_tensor(pred_rescaled_np, dtype=torch.float32))
        return torch.stack(preds_rescaled, dim=0).to(dtype=torch.float32, device=preds.device)
            
    def _build_summing_matrix(self) -> torch.Tensor:
        """
        Build the summing matrix S for hierarchical aggregation.

        Returns
        -------
        torch.Tensor
            Structure matrix combining base and total nodes.
        """
        I = torch.eye(self.num_nodes)  
        total = torch.ones((1, self.num_nodes))
        self.S = torch.cat([I, total], dim=0).to(DEVICE)
    
    def _compute_min_trace_projection(self, W: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the MinT projection matrix for hierarchical reconciliation.

        Parameters
        ----------
        W : torch.Tensor, optional
            Weight matrix. Defaults to the identity (OLS / ordinary reconciliation).
            Pass a custom W to use WLS or full MinT — but beware that the full error
            covariance can be near-singular for highly correlated series.

        Returns
        -------
        torch.Tensor
            Projection matrix P such that reconciled forecasts = S @ G @ forecasts.
        """
        if W is None:
            W = torch.eye(self.S.shape[0], device=DEVICE)
        self.W = W.to(DEVICE)

        if W.shape[0] == W.shape[1] and torch.allclose(W, torch.diag(torch.diagonal(W))):
            W_inv = torch.diag(1.0 / torch.diagonal(W))
        else:
            W_inv = torch.inverse(W)

        S_t = self.S.T
        middle = torch.inverse(S_t @ W_inv @ self.S)
        self.G = middle @ S_t @ W_inv
        self.P = self.S @ self.G

    def _compute_validation_error_covariance(self) -> torch.Tensor:
        """
        Estimate the forecast error covariance matrix on the validation set.

        Not called automatically — only use this after training if you want to
        pass a data-driven W to ``_compute_min_trace_projection``. For highly
        correlated series the full covariance is near-singular; prefer the
        diagonal (variance-only) version or stick with the identity default.

        Returns
        -------
        torch.Tensor
            Error covariance W of shape [num_nodes+1, num_nodes+1].
        """
        self.model.eval()
        num_nodes = self.dataset_train.num_nodes
        errors = []

        with torch.no_grad():
            for batch in self.val_loader:
                for attr in ['x', 'y_scaled', 'y', 'edge_weight', 'mask_y']:
                    if getattr(batch, attr, None) is not None:
                        setattr(batch, attr, getattr(batch, attr).to(torch.float32))
                batch = batch.to(DEVICE)

                y_hat = self.model(
                    batch.x,
                    batch.edge_index,
                    edge_weight=getattr(batch, 'edge_weight', None),
                    mask=getattr(batch, 'mask_y', None),
                    return_group_outputs=False,
                    return_attention=False
                )

                y_hat = y_hat.reshape(num_nodes, -1)
                y_true = batch.y.reshape(num_nodes, -1)

                y_hat_rescaled = self._rescale_predictions(y_hat)
                base_errors = y_hat_rescaled.cpu() - y_true.cpu()

                top_pred = y_hat_rescaled.sum(dim=0, keepdim=True)
                top_true = y_true.sum(dim=0, keepdim=True).cpu()
                top_error = top_pred.cpu() - top_true

                errors.append(torch.cat([base_errors, top_error], dim=0))

        all_errors = torch.cat(errors, dim=1)
        centered = all_errors - all_errors.mean(dim=1, keepdim=True)
        W = (centered @ centered.T) / (all_errors.shape[1] - 1)
        W = W + 1e-6 * torch.eye(W.shape[0], device=W.device)
        return W.to(DEVICE)

    def _min_trace_reconciliation(self, preds: torch.Tensor,
                                   top_forecast: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply MinT reconciliation to hierarchical forecasts.

        Parameters
        ----------
        preds : torch.Tensor
            Base-level forecasts [num_nodes, T].
        top_forecast : torch.Tensor, optional
            External top-level forecast [1, T] or [T]. If None, uses the sum of
            base predictions as the top-level constraint.

        Returns
        -------
        torch.Tensor
            Reconciled forecasts [num_nodes+1, T].
        """
        if top_forecast is not None:
            if top_forecast.dim() == 1:
                top_forecast = top_forecast.unsqueeze(0)
            T = min(preds.shape[1], top_forecast.shape[1])
            preds = preds[:, :T]
            top_forecast = top_forecast[:, :T]
            y_combined = torch.cat([preds, top_forecast.to(preds.device)], dim=0)
        else:
            national_pred = preds.sum(dim=0, keepdim=True)
            y_combined = torch.cat([preds, national_pred], dim=0)

        return self.S @ self.G @ y_combined.to(DEVICE)

    def _fit_top_level_model(self, model_type: str = 'ridge') -> None:
        """
        Fit a univariate model on the national-level (aggregated) target to produce
        top-level forecasts for MinT reconciliation when none are provided externally.

        Parameters
        ----------
        model_type : {'ridge', 'rf', 'xgb', 'gam', 'gcn'}
            Model family to use.
            - 'ridge' : RidgeCV (fast, smooth, good default).
            - 'rf'    : Random Forest (non-linear, no external dependency).
            - 'xgb'   : XGBoost (fast, accurate; requires ``pip install xgboost``).
            - 'gam'   : Generalised Additive Model (interpretable; requires ``pip install pygam``).
            - 'gcn'   : 2-layer GCN trained on graph batches; node forecasts are summed
                        to form the national aggregate. Uses the same edge structure as
                        the main model.

        Notes
        -----
        Features: lag-48 (1-day), lag-336 (7-day), EWM-48, EWM-336, and cyclical
        encodings of hour-of-day, day-of-week, and month.  All features are
        computed on the chronologically concatenated train+val+test series so that
        lag values at split boundaries are correct.  Predictions are stored in
        ``self.top_forecasts_train``, ``self.top_forecasts_val``, and
        ``self.top_forecasts_test``.
        """
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        print(f"[Trainer] Fitting top-level model ({model_type}) for MinT reconciliation...")

        target_col = self.dataset_kwargs.get('target_base', 'load')

        def aggregate(dataset) -> pd.Series:
            # Use the dataframe that GraphDataset itself was built from so that
            # the number of dates matches len(dataset) exactly.
            df = dataset.dataframe
            return df.groupby('date')[target_col].sum().sort_index()

        s_train = aggregate(self.dataset_train)
        s_val   = aggregate(self.dataset_val)   if self.dataset_val   is not None else None
        s_test  = aggregate(self.dataset_test)

        # Infer temporal resolution from the training index.
        train_idx = pd.to_datetime(s_train.index)
        delta_hours = (train_idx[1] - train_idx[0]).total_seconds() / 3600
        steps_per_day  = max(1, round(24 / delta_hours))
        steps_per_week = steps_per_day * 7

        # Concatenate chronologically so lag features span across split boundaries.
        # Deduplicate by INDEX (not by value) so dates that happen to share the same
        # national sum are not silently dropped, which would misalign X and y.
        s_full = pd.concat([s for s in [s_train, s_val, s_test] if s is not None])
        s_full = s_full.sort_index()
        s_full = s_full[~s_full.index.duplicated(keep='first')]

        # EWM must be computed on past values only (shift by 1) and span must be
        # ≥ 2 to avoid alpha=1 (which would set ewm[t] = y[t], i.e. leakage).
        ewm_day_span  = max(2, steps_per_day)
        ewm_week_span = max(2, steps_per_week)

        def make_features(s: pd.Series) -> pd.DataFrame:
            df = pd.DataFrame({'y': s.values}, index=s.index)
            df['lag_day']  = df['y'].shift(steps_per_day)
            df['lag_week'] = df['y'].shift(steps_per_week)
            df['ewm_day']  = df['y'].shift(1).ewm(span=ewm_day_span,  adjust=False).mean()
            df['ewm_week'] = df['y'].shift(1).ewm(span=ewm_week_span, adjust=False).mean()
            idx = pd.to_datetime(s.index)
            hour = idx.hour + idx.minute / 60
            df['hour_sin']  = np.sin(2 * np.pi * hour / 24)
            df['hour_cos']  = np.cos(2 * np.pi * hour / 24)
            df['dow_sin']   = np.sin(2 * np.pi * idx.dayofweek / 7)
            df['dow_cos']   = np.cos(2 * np.pi * idx.dayofweek / 7)
            df['month_sin'] = np.sin(2 * np.pi * idx.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * idx.month / 12)
            return df

        FEAT = ['lag_day', 'lag_week', 'ewm_day', 'ewm_week',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']

        feat_full = make_features(s_full)

        def get_X_y(s: pd.Series):
            # Both X and y are sliced from feat_full so they are always the same length.
            mask = feat_full.index.isin(s.index)
            X = feat_full.loc[mask, FEAT].bfill().fillna(0).values
            y = feat_full.loc[mask, 'y'].values
            return X, y

        X_train, y_train = get_X_y(s_train)
        X_val,   y_val   = get_X_y(s_val)  if s_val  is not None else (None, None)
        X_test,  y_test  = get_X_y(s_test)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)  if X_val  is not None else None
        X_test_s  = scaler.transform(X_test)

        if model_type == 'ridge':
            from sklearn.linear_model import RidgeCV
            mdl = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            mdl.fit(X_train_s, y_train)
            pred_train = mdl.predict(X_train_s)
            pred_val   = mdl.predict(X_val_s)  if X_val_s  is not None else None
            pred_test  = mdl.predict(X_test_s)

        elif model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            mdl = RandomForestRegressor(n_estimators=200, max_depth=15,
                                        min_samples_leaf=20, n_jobs=-1, random_state=42)
            mdl.fit(X_train_s, y_train)
            pred_train = mdl.predict(X_train_s)
            pred_val   = mdl.predict(X_val_s)  if X_val_s  is not None else None
            pred_test  = mdl.predict(X_test_s)

        elif model_type == 'xgb':
            try:
                import xgboost as xgb
            except ImportError:
                raise ImportError("xgboost is required for top_level_model='xgb'. "
                                  "Install it with: pip install xgboost")
            eval_set = [(X_val_s, y_val)] if X_val_s is not None else None
            mdl = xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0,
                early_stopping_rounds=20 if eval_set else None,
            )
            mdl.fit(X_train_s, y_train, eval_set=eval_set, verbose=False)
            pred_train = mdl.predict(X_train_s)
            pred_val   = mdl.predict(X_val_s)  if X_val_s  is not None else None
            pred_test  = mdl.predict(X_test_s)

        elif model_type == 'gam':
            try:
                from pygam import LinearGAM, s as gam_s, l as gam_l
            except ImportError:
                raise ImportError("pygam is required for top_level_model='gam'. "
                                  "Install it with: pip install pygam")
            # Lag features modelled linearly; calendar features with splines.
            terms = gam_l(0) + gam_l(1) + gam_l(2) + gam_l(3) \
                  + gam_s(4) + gam_s(5) + gam_s(6) + gam_s(7) + gam_s(8) + gam_s(9)
            mdl = LinearGAM(terms)
            mdl.gridsearch(X_train_s, y_train)
            pred_train = mdl.predict(X_train_s)
            pred_val   = mdl.predict(X_val_s)  if X_val_s  is not None else None
            pred_test  = mdl.predict(X_test_s)

        elif model_type == 'gcn':
            # Small 2-layer GCN trained end-to-end: each node produces a forecast,
            # node outputs are summed to the national aggregate.
            from torch_geometric.nn import GCNConv
            import torch.nn as nn

            _in_ch  = self.dataset_train[0].x.shape[-1]
            _out_ch = self.dataset_train[0].y.shape[-1]
            _N      = self.num_nodes

            class _GCNNational(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = GCNConv(_in_ch, 64)
                    self.conv2 = GCNConv(64, _out_ch)

                def forward(self, x, edge_index, edge_weight=None):
                    x = torch.relu(self.conv1(x, edge_index, edge_weight))
                    return self.conv2(x, edge_index, edge_weight)  # [N_total, out_ch]

            _gcn = _GCNNational().to(DEVICE)
            _opt = torch.optim.Adam(_gcn.parameters(), lr=1e-3, weight_decay=1e-4)

            _bs = min(32, len(self.dataset_train))
            _tr_loader  = PyGDataLoader(self.dataset_train, batch_size=_bs, shuffle=True)
            _val_loader = PyGDataLoader(self.dataset_val,   batch_size=_bs, shuffle=False) \
                          if self.dataset_val is not None else None

            _best_val, _best_state, _no_imp = float('inf'), None, 0
            _patience_gcn = 10

            def _prep(b):
                # Cast all floating-point attributes to float32 on CPU before
                # moving to device — MPS rejects float64 tensors outright.
                for _attr in ['x', 'y', 'edge_weight']:
                    _t = getattr(b, _attr, None)
                    if _t is not None and _t.is_floating_point():
                        setattr(b, _attr, _t.float())
                return b.to(DEVICE)

            def _ew(b):
                ew = getattr(b, 'edge_weight', None)
                return ew if ew is not None else None  # already float32 after _prep

            for _ep in range(100):
                _gcn.train()
                for _b in _tr_loader:
                    _b = _prep(_b)
                    _B = _b.num_graphs
                    # batch.y: [N*B, out_ch] → sum over nodes → [B, out_ch]
                    _y_nat  = _b.y.view(_B, _N, _out_ch).sum(dim=1)
                    _out    = _gcn(_b.x, _b.edge_index, _ew(_b))
                    _o_nat  = _out.view(_B, _N, _out_ch).sum(dim=1)
                    _l = torch.nn.functional.mse_loss(_o_nat, _y_nat)
                    _opt.zero_grad(); _l.backward(); _opt.step()

                if _val_loader is not None:
                    _gcn.eval()
                    _vl = 0.0
                    with torch.no_grad():
                        for _b in _val_loader:
                            _b = _prep(_b)
                            _B = _b.num_graphs
                            _y_nat = _b.y.view(_B, _N, _out_ch).sum(dim=1)
                            _o     = _gcn(_b.x, _b.edge_index, _ew(_b))
                            _o_nat = _o.view(_B, _N, _out_ch).sum(dim=1)
                            _vl   += torch.nn.functional.mse_loss(_o_nat, _y_nat).item()
                    if _vl < _best_val - 1e-6:
                        _best_val   = _vl
                        _best_state = {k: v.clone() for k, v in _gcn.state_dict().items()}
                        _no_imp     = 0
                    else:
                        _no_imp += 1
                        if _no_imp >= _patience_gcn:
                            print(f"[Trainer] GCN early stop at epoch {_ep + 1}.")
                            break

            if _best_state is not None:
                _gcn.load_state_dict(_best_state)

            def _gcn_predict(dataset):
                _loader = PyGDataLoader(dataset, batch_size=_bs, shuffle=False)
                _gcn.eval()
                _preds = []
                with torch.no_grad():
                    for _b in _loader:
                        _b   = _prep(_b)
                        _B   = _b.num_graphs
                        _out = _gcn(_b.x, _b.edge_index, _ew(_b))
                        _o_nat = _out.view(_B, _N, _out_ch).sum(dim=1).cpu()  # [B, out_ch]
                        _preds.append(_o_nat)
                return torch.cat(_preds, dim=0).flatten().numpy()  # [T * out_ch]

            pred_train = _gcn_predict(self.dataset_train)
            pred_val   = _gcn_predict(self.dataset_val) if self.dataset_val is not None else None
            pred_test  = _gcn_predict(self.dataset_test)

        else:
            raise ValueError(
                f"Unknown top_level_model='{model_type}'. "
                "Choose from 'ridge', 'rf', 'xgb', 'gam', 'gcn', or None."
            )

        def _rmse(p, t): return np.sqrt(np.mean((p - t) ** 2))
        def _mape(p, t): return np.mean(np.abs((p - t) / np.where(np.abs(t) < 1e-8, 1e-8, t))) * 100
        def _nmae(p, t): return np.mean(np.abs(p - t)) / (np.mean(np.abs(t)) + 1e-8) * 100

        header = f"{'split':<6} {'RMSE':>12} {'MAPE%':>10} {'nMAE%':>10}"
        rows = [header, "-" * len(header)]
        for tag, p, t in [("train", pred_train, y_train),
                           ("val",   pred_val,   y_val),
                           ("test",  pred_test,  y_test)]:
            if p is None or t is None:
                continue
            rows.append(f"{tag:<6} {_rmse(p,t):>12.2f} {_mape(p,t):>10.2f} {_nmae(p,t):>10.2f}")
        print(f"[Trainer] Top-level {model_type} — national aggregate errors:\n" + "\n".join(rows))

        self.top_forecasts_train = torch.tensor(pred_train, dtype=torch.float32).to(DEVICE)
        self.top_forecasts_val   = torch.tensor(pred_val,   dtype=torch.float32).to(DEVICE) \
                                   if pred_val is not None else None
        self.top_forecasts_test  = torch.tensor(pred_test,  dtype=torch.float32).to(DEVICE)

class RollingTrainer:
    """
    Warm-start rolling retraining without future leakage.

    Behavior:
    ---------
    - Window 0:
        TRAIN = df_train_base
        VAL   = df_val_base
        TEST  = first test window

    - Window k >= 1:
        TRAIN = df_train_base + test_window_{k-1}    (historical + past month)
        VAL   = df_val_base
        TEST  = test_window_k

    - Last window:
        If remaining timestamps < window_size, a final window is created with
        size = remaining timestamps.

    All windows warm-start from the previous model.
    """

    def __init__(
        self,
        dataset_train_0: "GraphDataset",
        dataset_val_0: "GraphDataset",
        dataset_test_full: "GraphDataset",
        model_class: Type[torch.nn.Module],
        model_kwargs: Dict[str, Any],
        window_size: int,
        step_size: int,
        batch_size: int = 32,
        reconcile: bool = True,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        num_epochs_initial: int = 50,
        num_epochs_update: int = 5,
    ):
        self.ds_train_0 = dataset_train_0
        self.ds_val_0 = dataset_val_0
        self.ds_test_full = dataset_test_full

        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.batch_size = batch_size
        self.reconcile = reconcile
        self.trainer_kwargs = trainer_kwargs or {}
        self.num_epochs_initial = int(num_epochs_initial)
        self.num_epochs_update = int(num_epochs_update)

        self.window_size = int(window_size)
        self.step_size = int(step_size)

        # Base DataClass and splits
        self.base_data = self.ds_train_0.data
        self.df_train_base = self.base_data.df_train.copy()
        self.df_val_base   = self.base_data.df_val.copy()
        df_test_full       = self.ds_test_full.data.df_test.copy()

        # Normalize all date columns to naive datetime64[ns]
        def normalize(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], utc=False)
            return df

        self.df_train_base = normalize(self.df_train_base)
        self.df_val_base   = normalize(self.df_val_base)
        self.df_test_full  = normalize(df_test_full)

        # Unique test timestamps
        self.dates = np.sort(self.df_test_full["date"].unique())
        self.num_time = len(self.dates)

        if self.window_size <= 0 or self.window_size > self.num_time:
            raise ValueError(f"Invalid window_size={self.window_size} for num_time={self.num_time}.")
        if self.step_size <= 0:
            raise ValueError("step_size must be > 0.")

        # Dataset config / nodes
        self.dataset_kwargs = getattr(self.ds_train_0, "dataset_kwargs", None)
        self.out_channels   = getattr(self.ds_train_0, "out_channels", None)
        if self.dataset_kwargs is None or self.out_channels is None:
            raise AttributeError("dataset_train_0 must expose dataset_kwargs and out_channels.")

        self.nodes = self.ds_train_0.nodes
        self.num_nodes = self.ds_train_0.num_nodes

    def _iter_windows(self) -> List[Tuple[int, int]]:
        """
        Return list of (start_idx, end_idx) over self.dates.

        Windows:
          - [t, t + window_size)
          - step_size between successive windows
          - if trailing timestamps remain (< window_size), add a final tail window:
                (t, num_time)
        """
        windows: List[Tuple[int, int]] = []
        t = 0
        while t + self.window_size <= self.num_time:
            windows.append((t, t + self.window_size))
            t += self.step_size

        # Tail window: remaining timestamps if any
        if t < self.num_time:
            windows.append((t, self.num_time))

        return windows

    def _make_datasets(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple["GraphDataset", "GraphDataset", "GraphDataset"]:
        """
        Build GraphDataset(train / val / test) for this window.

        - train_df : training rows for this window
        - test_df  : test rows for this window
        - val_df   : always df_val_base
        """
        DatasetCls = self.ds_train_0.__class__

        data_w = copy.copy(self.base_data)
        data_w.df_train = train_df
        data_w.df_val   = self.df_val_base
        data_w.df_test  = test_df

        ds_train = DatasetCls(
            data_w,
            period="train",
            out_channels=self.out_channels,
            nodes=self.nodes,
            dataset_kwargs=self.dataset_kwargs,
        )

        ds_val = DatasetCls(
            data_w,
            period="val",
            scalers_feat=ds_train.scalers_feat,
            scalers_target=ds_train.scalers_target,
            out_channels=self.out_channels,
            nodes=self.nodes,
            dataset_kwargs=self.dataset_kwargs,
        )

        ds_test = DatasetCls(
            data_w,
            period="test",
            scalers_feat=ds_train.scalers_feat,
            scalers_target=ds_train.scalers_target,
            out_channels=self.out_channels,
            nodes=self.nodes,
            dataset_kwargs=self.dataset_kwargs,
        )

        return ds_train, ds_val, ds_test

    # -------------------------------------------------------------
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the rolling training + evaluation.

        Returns
        -------
        List[Dict[str, Any]] with one entry per window:
            {
                "window_index": int,
                "window_start": Timestamp,
                "window_end":   Timestamp,
                "preds":  Tensor[num_nodes, T_eff],
                "targets":Tensor[num_nodes, T_eff],
            }
        """
        results: List[Dict[str, Any]] = []
        windows = self._iter_windows()

        print(f"[RollingTrainer] total windows={len(windows)} "
              f"(window_size={self.window_size}, step_size={self.step_size})")

        current_model: Optional[torch.nn.Module] = None
        previous_test_df: Optional[pd.DataFrame] = None

        # Split Trainer kwargs into:
        #  - init_kwargs: passed to Trainer(...)
        #  - train_kwargs: passed to trainer.train(...)
        exclude_train_keys = {
            "num_epochs", "patience", "min_delta", "force_training",
            "saving_directory", "plot_loss", "dynamic_graph", "save", "batch_size_save",
        }
        init_kwargs  = {k: v for k, v in self.trainer_kwargs.items() if k not in exclude_train_keys}
        train_kwargs = {k: v for k, v in self.trainer_kwargs.items()}

        for w_id, (t0, t1) in enumerate(windows):
            win_dates = self.dates[t0:t1]
            win_start, win_end = win_dates[0], win_dates[-1]

            # Slice test df for this window
            test_mask = (
                (self.df_test_full["date"] >= win_start) &
                (self.df_test_full["date"] <= win_end)
            )
            test_df_w = self.df_test_full.loc[test_mask].copy()
            if test_df_w.empty:
                print(f"[RollingTrainer] Window {w_id}: empty test slice, skipping.")
                continue

            # Build TRAIN df:
            if w_id == 0:
                # Window 0: pure historical training
                train_df = self.df_train_base.copy()
                desc = "historical train"
            else:
                # Windows k >= 1: base train + previous test month
                if previous_test_df is None:
                    # Should not happen, but guard anyway
                    train_df = self.df_train_base.copy()
                    desc = "historical train (fallback, no prev test)"
                else:
                    train_df = pd.concat(
                        [self.df_train_base, previous_test_df],
                        axis=0
                    ).sort_values("date")
                    desc = "historical + previous test window"

            ds_train, ds_val, ds_test = self._make_datasets(train_df, test_df_w)

            num_epochs = self.num_epochs_initial if w_id == 0 else self.num_epochs_update

            if w_id == 0:
                print(
                    f"[RollingTrainer] Window 0 | "
                    f"train='{desc}' (rows={len(train_df)}), "
                    f"test={win_start} → {win_end} (rows={len(test_df_w)}), "
                    f"epochs={num_epochs}"
                )
                current_model = self.model_class(**self.model_kwargs)
            else:
                print(
                    f"[RollingTrainer] Window {w_id} | "
                    f"train='{desc}' (rows={len(train_df)}), "
                    f"test={win_start} → {win_end} (rows={len(test_df_w)}), "
                    f"epochs={num_epochs}"
                )
                if current_model is None:
                    raise RuntimeError("current_model is None for w_id >= 1.")

            trainer_model_kwargs: Dict[str, Any] = {"num_epochs": num_epochs}
            if "lr" in self.trainer_kwargs:
                trainer_model_kwargs["lr"] = self.trainer_kwargs["lr"]

            trainer = Trainer(
                model=current_model,
                dataset_train=ds_train,
                dataset_val=ds_val,
                dataset_test=ds_test,
                batch_size=self.batch_size,
                reconcile=self.reconcile,
                model_kwargs=trainer_model_kwargs,
                **init_kwargs,
            )

            out = trainer.train(**train_kwargs)
            preds, targets = out[0], out[1]

            print(
                f"[RollingTrainer] Window {w_id} | "
                f"preds.shape={preds.shape}, targets.shape={targets.shape}"
            )

            current_model = trainer.model
            previous_test_df = test_df_w

            results.append(
                {
                    "window_index": w_id,
                    "window_start": win_start,
                    "window_end": win_end,
                    "preds": preds,
                    "targets": targets,
                }
            )

        return results