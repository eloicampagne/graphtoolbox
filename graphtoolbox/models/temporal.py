"""Temporal graph models built on PyG-Temporal recurrent cells.

The static architecture :class:`~graphtoolbox.models.gnn.myGNN` relies on
:class:`~graphtoolbox.models.gnn.ConvAdapter` to expose every ``torch_geometric``
convolution behind a single ``(x, edge_index)`` interface. This module mirrors
that design for the recurrent graph cells of ``torch_geometric_temporal``:
:class:`ConvAdapterTemporal` absorbs the calling conventions of those cells and
:class:`TemporalGNN` plays the role of ``myGNN``, turning a lag-augmented node
feature vector into a horizon forecast through a genuine sequence unrolling.
"""
import re
import warnings
from inspect import signature

import torch
import torch.nn as nn


def _cheb_basis(x, prop, K):
    """Chebyshev basis [T_0(L̂)x, ..., T_{K-1}(L̂)x] given a sparse propagation matrix."""
    Ts = [x]
    if K > 1:
        Ts.append(torch.sparse.mm(prop, x))
    for _ in range(2, K):
        Ts.append(2. * torch.sparse.mm(prop, Ts[-1]) - Ts[-2])
    return Ts


def _apply_cheb_lins(conv, Ts):
    """Recombine a Chebyshev basis with the per-order linear maps of a ChebConv."""
    out = conv.lins[0](Ts[0])
    for lin, Tk in zip(conv.lins[1:], Ts[1:]):
        out = out + lin(Tk)
    if conv.bias is not None:
        out = out + conv.bias
    return out


def _gconvgru_supported(cell):
    """Structural check that `cell` matches the GConvGRU layout the fast path relies on."""
    if type(cell).__name__ != "GConvGRU":
        return False
    gates = ("conv_x_z", "conv_h_z", "conv_x_r", "conv_h_r", "conv_x_h", "conv_h_h")
    if not all(hasattr(cell, g) for g in gates):
        return False
    conv = cell.conv_x_z
    return hasattr(conv, "lins") and hasattr(conv, "__norm__") and hasattr(conv, "normalization")


def _gconvgru_fast_unroll(cell, seq, edge_index, edge_weight=None):
    """Numerically-equivalent fast unroll of a ``torch_geometric_temporal.GConvGRU``.

    The stock cell issues six ChebConv calls per time step, each recomputing the
    graph normalisation. Following the classical cuDNN GRU optimisation, the
    input-side transforms of the three gates are computed for the whole sequence
    in a single Chebyshev expansion (time folded into channels, valid because
    the convolution is linear in its input), the normalisation is computed once,
    and the hidden-side gates z and r share one Chebyshev basis per step. Only
    what genuinely depends on the recurrent state remains sequential.

    Parameters
    ----------
    cell : torch_geometric_temporal.nn.recurrent.GConvGRU
        The recurrent cell whose weights are used (state_dict untouched).
    seq : torch.Tensor
        Sequence of shape ``[N, T, C]``.
    edge_index : torch.LongTensor
        Graph connectivity in COO format.
    edge_weight : torch.Tensor, optional
        Optional edge weights.

    Returns
    -------
    torch.Tensor
        Final hidden state ``[N, hidden_channels]``.
    """
    N, T, C = seq.shape
    conv = cell.conv_x_z
    K = len(conv.lins)
    ei_n, norm = conv.__norm__(edge_index, N, edge_weight, conv.normalization,
                               None, dtype=seq.dtype, batch=None)
    # PyG propagation aggregates source -> target: out[dst] += norm * x[src].
    prop = torch.sparse_coo_tensor(torch.stack([ei_n[1], ei_n[0]]), norm,
                                   (N, N)).coalesce()

    # Input side: one Chebyshev expansion for all T steps and all three gates.
    Ts_x = _cheb_basis(seq.reshape(N, T * C), prop, K)

    def gate_x(g):
        out = sum(lin(Tk.view(N, T, C)) for lin, Tk in zip(g.lins, Ts_x))
        if g.bias is not None:
            out = out + g.bias
        return out

    Zx, Rx, Hx = gate_x(cell.conv_x_z), gate_x(cell.conv_x_r), gate_x(cell.conv_x_h)

    h = torch.zeros(N, cell.out_channels, dtype=seq.dtype, device=seq.device)
    for t in range(T):
        Ts_h = _cheb_basis(h, prop, K)                     # shared by gates z and r
        z = torch.sigmoid(Zx[:, t] + _apply_cheb_lins(cell.conv_h_z, Ts_h))
        r = torch.sigmoid(Rx[:, t] + _apply_cheb_lins(cell.conv_h_r, Ts_h))
        Ts_rh = _cheb_basis(h * r, prop, K)
        h_tilde = torch.tanh(Hx[:, t] + _apply_cheb_lins(cell.conv_h_h, Ts_rh))
        h = z * h + (1 - z) * h_tilde
    return h


def lag_sequence_indices(features, target="load"):
    """Split a feature list into a lag sequence and the remaining static columns.

    The lag columns of ``target`` (named ``{target}_l{k}``) form the temporal
    sequence, ordered from the oldest lag to the most recent one so that a
    recurrent cell consumes them in chronological order. Every other column is
    treated as a static side feature.

    Parameters
    ----------
    features : list[str]
        Ordered feature names, as exposed by ``GraphDataset.features``.
    target : str, optional
        Base name of the lagged target (default ``"load"``).

    Returns
    -------
    tuple[list[int], list[int]]
        ``(seq_idx, static_idx)``: column indices of the lag sequence (oldest
        first) and of the static features.
    """
    pattern = re.compile(rf"{re.escape(target)}_l(\d+)$")
    lags = []
    for i, name in enumerate(features):
        match = pattern.search(name)
        if match is not None:
            lags.append((int(match.group(1)), i))
    lags.sort(key=lambda item: -item[0])          # l48, l47, ... l1 (oldest -> newest)
    seq_idx = [i for _, i in lags]
    seq_set = set(seq_idx)
    static_idx = [i for i in range(len(features)) if i not in seq_set]
    return seq_idx, static_idx


class ConvAdapterTemporal(nn.Module):
    """Wrap a PyG-Temporal recurrent graph cell for use inside :class:`TemporalGNN`.

    Analogous to :class:`~graphtoolbox.models.gnn.ConvAdapter`, this adapter
    hides the two calling conventions found in ``torch_geometric_temporal``:

    * *step-recurrent* cells (``GConvGRU``, ``DCRNN``, ``TGCN``) accept one step
      at a time as ``cell(x_t, edge_index, edge_weight, H)`` and carry the hidden
      state ``H`` across the sequence;
    * *windowed* cells (``A3TGCN``) receive the whole window at once as
      ``cell(X, edge_index, edge_weight)`` with ``X`` shaped ``[N, C, periods]``
      and therefore need the number of periods at construction time.

    Both regimes are detected from the constructor signature, so callers only
    provide the cell class and its hyper-parameters.

    Parameters
    ----------
    cell_class : type
        A ``torch_geometric_temporal.nn.recurrent`` cell class (not an instance).
    in_channels : int
        Number of features carried at each sequence step (``1`` for a plain load
        lag sequence).
    hidden_channels : int
        Hidden state dimension produced by the cell.
    seq_len : int, optional
        Sequence length, required for windowed cells to set ``periods`` when it
        is not already given in ``cell_kwargs``.
    cell_kwargs : dict, optional
        Extra keyword arguments forwarded to the cell constructor (e.g. ``K``).
    """

    def __init__(self, cell_class, in_channels, hidden_channels, seq_len=None,
                 cell_kwargs=None):
        super().__init__()
        cell_kwargs = dict(cell_kwargs or {})
        self.cell_class = cell_class
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        try:
            ctor_params = set(signature(cell_class).parameters.keys())
        except (ValueError, TypeError):
            ctor_params = set()

        # Windowed cells consume the whole window and need 'periods' up front.
        self.windowed = "periods" in ctor_params
        if self.windowed and "periods" not in cell_kwargs:
            if seq_len is None:
                raise ValueError(
                    f"{cell_class.__name__} is a windowed cell and needs the number "
                    f"of periods; pass seq_len or cell_kwargs['periods']."
                )
            cell_kwargs["periods"] = seq_len

        kwargs = {"in_channels": in_channels, "out_channels": hidden_channels}
        kwargs.update(cell_kwargs)
        if ctor_params:
            kwargs = {k: v for k, v in kwargs.items() if k in ctor_params}
        self.cell = cell_class(**kwargs)

        # Fast-path state: None = undecided, True/False after first-forward check.
        self._use_fast = None

    def forward(self, seq, edge_index, edge_weight=None):
        """Reduce a per-node sequence to a hidden representation.

        Parameters
        ----------
        seq : torch.Tensor
            Sequence tensor of shape ``[N, T]`` (single channel) or
            ``[N, T, in_channels]``.
        edge_index : torch.LongTensor
            Graph connectivity in COO format.
        edge_weight : torch.Tensor, optional
            Optional edge weights.

        Returns
        -------
        torch.Tensor
            Hidden state of shape ``[N, hidden_channels]``.
        """
        if seq.dim() == 2:
            seq = seq.unsqueeze(-1)                    # [N, T, 1]

        if self.windowed:
            window = seq.permute(0, 2, 1).contiguous()  # [N, C, T]
            return self.cell(window, edge_index, edge_weight)

        # Decide once whether the optimised unroll is safe for this cell: it must
        # match the generic unroll on a real batch before it is trusted.
        if self._use_fast is None:
            self._use_fast = self._check_fast_path(seq, edge_index, edge_weight)

        if self._use_fast:
            return _gconvgru_fast_unroll(self.cell, seq, edge_index, edge_weight)
        return self._generic_unroll(seq, edge_index, edge_weight)

    def _generic_unroll(self, seq, edge_index, edge_weight=None):
        """Reference step-by-step unroll, valid for any step-recurrent cell."""
        h = None
        for t in range(seq.size(1)):
            h = self.cell(seq[:, t, :], edge_index, edge_weight, h)
        return h

    def _check_fast_path(self, seq, edge_index, edge_weight):
        """Enable the fast unroll only after verifying numerical equivalence."""
        if not _gconvgru_supported(self.cell):
            return False
        try:
            with torch.no_grad():
                ref = self._generic_unroll(seq, edge_index, edge_weight)
                fast = _gconvgru_fast_unroll(self.cell, seq, edge_index, edge_weight)
            scale = ref.abs().max().clamp(min=1.0)
            if ((ref - fast).abs().max() / scale).item() < 1e-4:
                return True
            warnings.warn(f"{type(self.cell).__name__}: fast unroll disagrees with the "
                          f"reference implementation; falling back to the generic loop.")
        except Exception as exc:
            warnings.warn(f"{type(self.cell).__name__}: fast unroll unavailable "
                          f"({exc}); falling back to the generic loop.")
        return False


class TemporalGNN(nn.Module):
    """Recurrent spatio-temporal forecaster, the temporal counterpart of ``myGNN``.

    The lag columns of the target are unrolled through a :class:`ConvAdapterTemporal`
    that carries a graph-aware hidden state across the sequence. The final hidden
    state is concatenated with the static features and projected to the forecast
    horizon by a linear head. The class exposes the attributes the GraphToolbox
    :class:`~graphtoolbox.training.trainer.Trainer` relies on (``conv_class``,
    ``conv_kwargs``, ``hidden_channels``, ``num_layers``, ``heads``) so it trains,
    reconciles and caches exactly like the static models.

    Parameters
    ----------
    in_channels : int
        Total number of node features (kept for API symmetry with ``myGNN``; the
        sequence/static split is driven by ``seq_idx`` / ``static_idx``).
    hidden_channels : int
        Hidden state dimension of the recurrent cell.
    out_channels : int
        Forecast horizon (output dimension per node).
    cell_class : type
        A ``torch_geometric_temporal`` recurrent cell class.
    seq_idx : list[int]
        Column indices forming the temporal sequence, oldest lag first.
    static_idx : list[int]
        Column indices of the static side features.
    cell_kwargs : dict, optional
        Extra keyword arguments forwarded to the cell (e.g. ``K``).
    seq_channels : int, optional
        Number of features per sequence step (default ``1``).
    """

    def __init__(self, in_channels, hidden_channels, out_channels, cell_class,
                 seq_idx, static_idx, cell_kwargs=None, seq_channels=1, **kwargs):
        super().__init__()
        self.seq_idx = list(seq_idx)
        self.static_idx = list(static_idx)
        self.seq_channels = seq_channels
        # Index tensors follow the model device; excluded from the state_dict so
        # checkpoints stay identical to earlier versions.
        self.register_buffer("_seq_index", torch.as_tensor(self.seq_idx, dtype=torch.long),
                             persistent=False)
        self.register_buffer("_static_index", torch.as_tensor(self.static_idx, dtype=torch.long),
                             persistent=False)

        # Attributes read by the Trainer for naming, checkpointing and logging.
        self.conv_class = cell_class
        self.conv_kwargs = dict(cell_kwargs or {})
        self.hidden_channels = hidden_channels
        self.num_layers = 1
        self.heads = 0

        seq_len = len(self.seq_idx) // seq_channels
        self.adapter = ConvAdapterTemporal(
            cell_class, in_channels=seq_channels, hidden_channels=hidden_channels,
            seq_len=seq_len, cell_kwargs=cell_kwargs,
        )
        self.head = nn.Linear(hidden_channels + len(self.static_idx), out_channels)

    @classmethod
    def from_features(cls, features, cell_class, hidden_channels, out_channels,
                      target="load", cell_kwargs=None, **kwargs):
        """Build a :class:`TemporalGNN` by deriving the lag split from feature names.

        Parameters
        ----------
        features : list[str]
            Ordered feature names (``GraphDataset.features``).
        cell_class : type
            Recurrent cell class.
        hidden_channels, out_channels : int
            Hidden and output dimensions.
        target : str, optional
            Base name of the lagged target (default ``"load"``).
        cell_kwargs : dict, optional
            Extra keyword arguments forwarded to the cell.

        Returns
        -------
        TemporalGNN
        """
        seq_idx, static_idx = lag_sequence_indices(features, target=target)
        return cls(
            in_channels=len(features), hidden_channels=hidden_channels,
            out_channels=out_channels, cell_class=cell_class,
            seq_idx=seq_idx, static_idx=static_idx, cell_kwargs=cell_kwargs,
            **kwargs,
        )

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        seq = x.index_select(1, self._seq_index)       # [N, T] or [N, T*C]
        static = x.index_select(1, self._static_index) # [N, S]
        if self.seq_channels > 1:
            seq = seq.reshape(x.size(0), -1, self.seq_channels)
        h = self.adapter(seq, edge_index, edge_weight)  # [N, hidden]
        return self.head(torch.cat([h, static], dim=1))
