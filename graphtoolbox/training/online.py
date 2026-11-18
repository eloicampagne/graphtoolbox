"""
Forecasting on a data stream.

``OnlineForecaster`` receives the raw rows of a stream (the columns of the files read by
``DataClass``), forecasts each block of ``out_channels`` timestamps before its target is
observed, and updates the model by replay once the observations arrive. The scalers and the
graph stay those of the training dataset, as in ``RollingTrainer(update="replay")``, whose
update rule ``ReplayEngine`` implements for both classes.

The replay engine keeps every sample of the pool in preallocated tensors. A mini-batch is
an index into these tensors joined to a block-diagonal ``edge_index`` built once, which
gives the same inputs as ``torch_geometric.data.Batch.from_data_list`` on a static graph
without collating Python objects at every step.
"""
from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data


__all__ = ["OnlineForecaster", "ReplayEngine"]


# --------------------------------------------------------------------------- #
# Scalers
# --------------------------------------------------------------------------- #
class NodeScaler:
    """
    Per-node scalers applied to a whole ``[nodes, rows, features]`` array at once.

    When every scaler is an unclipped ``MinMaxScaler`` fitted on the same number of
    features, ``transform`` and ``inverse_transform`` repeat the float32 operations of
    scikit-learn on the stacked arrays, so the result equals the per-node calls bit for
    bit. Other scalers fall back to one call per node.
    """

    def __init__(self, scalers: Dict, nodes: Sequence):
        self.scalers = [scalers[n] for n in nodes]
        self.fast = all(type(s) is MinMaxScaler and not getattr(s, "clip", False)
                        for s in self.scalers)
        if self.fast:
            try:
                self.scale = np.stack([s.scale_ for s in self.scalers])[:, None, :]  # [N, 1, F]
                self.min = np.stack([s.min_ for s in self.scalers])[:, None, :]
            except ValueError:                                                  # unequal F
                self.fast = False

    def transform(self, X: np.ndarray) -> np.ndarray:
        """``X`` of shape ``[N, T, F]`` without NaN, in float32."""
        X = X.astype(np.float32, copy=False)
        if self.fast:
            return ((X * self.scale).astype(np.float32) + self.min).astype(np.float32)
        return np.stack([s.transform(x) for s, x in zip(self.scalers, X)]).astype(np.float32)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """``Z`` of shape ``[N, T]`` (one target per node), in float32."""
        Z = Z.astype(np.float32, copy=False)
        if self.fast:
            Z = Z[:, :, None]
            return ((Z - self.min).astype(np.float32) / self.scale).astype(np.float32)[:, :, 0]
        return np.stack([s.inverse_transform(z.reshape(-1, 1)).reshape(-1)
                         for s, z in zip(self.scalers, Z)]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Replay engine
# --------------------------------------------------------------------------- #
def _block_edges(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                 num_nodes: int, repeats: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``repeats`` disjoint copies of the graph, in the order of ``Batch.from_data_list``."""
    offsets = torch.arange(repeats, device=edge_index.device).repeat_interleave(edge_index.shape[1])
    ei = edge_index.repeat(1, repeats) + offsets * num_nodes
    ew = None if edge_weight is None else edge_weight.repeat(repeats)
    return ei, ew


class ReplayEngine:
    """
    Forecast blocks of samples and update a model by replay.

    After each block, :meth:`observe` adds its samples to the pool and takes
    ``replay_steps`` Adam steps (learning rate ``replay_lr``, gradient norm clipped at 5)
    on mini-batches that join the ``replay_recent`` most recent samples to
    ``batch_size - replay_recent`` samples drawn uniformly from all earlier ones. The
    loss is the mean squared error on the observed targets in scaled units.

    All samples must share the ``edge_index`` and ``edge_weight`` of the first one, which
    holds for the static graphs of ``GraphDataset``.

    Parameters
    ----------
    model : torch.nn.Module
        Model called as ``model(x, edge_index, edge_weight=...)``; updated in place.
    pool : sequence of torch_geometric.data.Data
        Initial samples (typically the training samples).
    num_nodes : int
        Number of nodes of every graph.
    batch_size, replay_steps, replay_recent : int
    replay_lr : float
    seed : int, optional
        Seeds the draw of the replayed samples and ``torch``'s global generator.
    device : torch.device or str, optional
        Device of the model and of the pool. Defaults to the device of the model's
        parameters, so that the model of a ``Trainer`` stays where the ``Trainer`` put it.
    """

    def __init__(self, model: torch.nn.Module, pool: Sequence[Data], num_nodes: int,
                 batch_size: int = 32, replay_steps: int = 10, replay_lr: float = 1e-4,
                 replay_recent: int = 7, seed: Optional[int] = None, device=None):
        if not 0 < replay_recent < batch_size:
            raise ValueError("replay_recent must lie in (0, batch_size).")
        if len(pool) == 0:
            raise ValueError("The initial pool is empty.")
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.N = int(num_nodes)
        self.batch_size = int(batch_size)
        self.replay_steps = int(replay_steps)
        self.replay_recent = int(replay_recent)
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(replay_lr))

        ref = pool[0]
        ew = getattr(ref, "edge_weight", None)
        self._ref_ei, self._ref_ew = ref.edge_index, ew
        self._ref_ei_cpu = ref.edge_index.cpu()
        self._ref_ew_cpu = None if ew is None else ew.detach().cpu().float()
        self.edge_index = ref.edge_index.to(self.device)
        self.edge_weight = None if ew is None else ew.to(self.device, torch.float32)
        self._edges: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}

        x0, y0 = ref.x, ref.y_scaled
        self.F, self.H = x0.shape[1], y0.shape[1]
        self._cap = 0
        self.size = 0
        self._X = self._Y = self._M = None
        self.add(pool)

    # -- pool --------------------------------------------------------------- #
    def _check_graph(self, d: Data) -> None:
        ew = getattr(d, "edge_weight", None)
        if d.edge_index is self._ref_ei and ew is self._ref_ew:
            return
        same = torch.equal(d.edge_index.cpu(), self._ref_ei_cpu) and (
            (ew is None) == (self._ref_ew_cpu is None)
            and (ew is None or torch.equal(ew.detach().cpu().float(), self._ref_ew_cpu)))
        if not same:
            raise ValueError("ReplayEngine needs every sample on the graph of the first one.")

    def _stack(self, samples: Sequence[Data]) -> Tuple[torch.Tensor, ...]:
        for d in samples:
            self._check_graph(d)
        x = torch.stack([d.x for d in samples]).to(self.device, torch.float32)
        y = torch.stack([d.y_scaled for d in samples]).to(self.device, torch.float32)
        m = torch.stack([d.mask_y for d in samples]).to(self.device, torch.float32)
        return x, y, m

    def add(self, samples: Sequence[Data]) -> None:
        """Append samples to the pool (amortised constant cost per sample)."""
        if len(samples) == 0:
            return
        x, y, m = self._stack(samples)
        need = self.size + len(samples)
        if need > self._cap:
            cap = max(need, 2 * self._cap, 64)
            X = torch.empty((cap, self.N, self.F), device=self.device)
            Y = torch.empty((cap, self.N, self.H), device=self.device)
            M = torch.empty((cap, self.N, self.H), device=self.device)
            if self.size:
                X[:self.size], Y[:self.size], M[:self.size] = (
                    self._X[:self.size], self._Y[:self.size], self._M[:self.size])
            self._X, self._Y, self._M, self._cap = X, Y, M, cap
        self._X[self.size:need], self._Y[self.size:need], self._M[self.size:need] = x, y, m
        self.size = need

    # -- model -------------------------------------------------------------- #
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` of shape ``[B, N, F]`` -> output ``[B, N, H]``."""
        B = x.shape[0]
        if B not in self._edges:
            self._edges[B] = _block_edges(self.edge_index, self.edge_weight, self.N, B)
        ei, ew = self._edges[B]
        out = self.model(x.reshape(B * self.N, self.F), ei, edge_weight=ew)
        return out.view(B, self.N, -1)

    def predict(self, samples: Sequence[Data]) -> np.ndarray:
        """Scaled forecasts ``[samples, nodes, H]`` of the current model."""
        x, _, _ = self._stack(samples)
        self.model.eval()
        with torch.no_grad():
            return self._forward(x).cpu().numpy()

    def update(self) -> None:
        """``replay_steps`` Adam steps on the most recent and on replayed samples."""
        if self.size <= self.replay_recent:
            raise ValueError("The pool must hold more samples than replay_recent.")
        self.model.train()
        R = self.replay_recent
        recent = torch.arange(self.size - R, self.size, device=self.device)
        for _ in range(self.replay_steps):
            past = self.rng.integers(0, self.size - R, self.batch_size - R)
            idx = torch.cat([recent, torch.as_tensor(past, device=self.device)])
            out = self._forward(self._X[idx])
            y, m = self._Y[idx], self._M[idx]
            loss = (((out - y) ** 2) * m).sum() / m.sum()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.opt.step()

    def observe(self, samples: Sequence[Data]) -> None:
        """Add observed samples to the pool, then update the model."""
        self.add(samples)
        if self.replay_steps:
            self.update()

    def to(self, device) -> "ReplayEngine":
        """
        Move the model, its optimiser state and the pool to ``device``.

        Needed after ``torch.load(..., map_location=...)``, which moves the tensors but not
        the recorded device.
        """
        self.device = torch.device(device)
        self.model.to(self.device)
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and k != "step":       # Adam keeps its step on CPU
                    state[k] = v.to(self.device)
        self.edge_index = self.edge_index.to(self.device)
        if self.edge_weight is not None:
            self.edge_weight = self.edge_weight.to(self.device)
        self._edges = {}
        if self._X is not None:
            self._X, self._Y, self._M = (t.to(self.device) for t in (self._X, self._Y, self._M))
        return self


# --------------------------------------------------------------------------- #
# Stream forecaster
# --------------------------------------------------------------------------- #
class OnlineForecaster:
    """
    Forecast a stream block by block and update the model by replay.

    The stream alternates two calls. :meth:`forecast` receives the rows of the next
    timestamps (covariates known in advance, target missing or NaN) and returns the
    forecast of every node. :meth:`update` receives the observed targets of timestamps
    already sent, and possibly revised covariates; it adds the completed samples to the
    replay pool and takes ``replay_steps`` Adam steps on them and on replayed history.

    Rows are prepared as ``DataClass`` prepares its files: lags of ``features_to_lag``
    computed node by node on a buffer of past rows, ``computed_features`` evaluated on that
    buffer and the new rows, then one-hot columns of ``dummies`` matched to the columns of
    the training dataset (a category absent from training gets zeros). Features and
    targets are scaled with the scalers of ``dataset_train`` and the graph is that of
    ``dataset_train``.

    A sample predicts ``out_channels`` consecutive timestamps from the features of the
    first one, as in ``GraphDataset``. Each call must therefore cover whole blocks of
    ``out_channels`` timestamps, aligned like the blocks of the training data.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model; updated in place.
    dataset_train : GraphDataset
        Dataset the model was trained on (scalers, features, graph, configuration).
    history : pandas.DataFrame, optional
        Past raw rows (date, node, raw covariates and target) used to compute the lags of
        the first blocks. Defaults to the rows read by the ``DataClass`` of
        ``dataset_train``, without the columns it derived.
    history_steps : int, optional
        Number of past timestamps kept in the buffer, the largest lag plus two blocks by
        default. The buffer also keeps every date forecast and not yet updated, with the
        rows its lags read, so that late observations are prepared like timely ones. Raise
        it when a ``computed_features`` callable looks further back.
    pool : sequence of GraphDataset, optional
        Datasets whose samples initialise the replay pool. Defaults to ``[dataset_train]``.
    batch_size, replay_steps, replay_lr, replay_recent, seed :
        Replay update, as in ``RollingTrainer(update="replay")``. ``replay_steps=0``
        forecasts without updating.
    device : torch.device or str, optional
        Device of the replay; defaults to the device of the model's parameters.

    Notes
    -----
    Lags are row shifts within a node, as in ``DataClass``; the buffer holds every node at
    every date, so that a missing row becomes an empty row instead of shifting the lags.
    Rows with a missing feature are zeroed, with a warning when they start a block.

    The forecaster can be saved with ``torch.save`` and restored with
    ``torch.load(path, weights_only=False)``, followed by :meth:`to` when the device
    changes; ``computed_features`` must then be module-level functions, not lambdas.

    Examples
    --------
    >>> fc = OnlineForecaster(model, ds_train, pool=[ds_train, ds_val], seed=0)
    >>> pred = fc.forecast(rows_next_day)      # date, node, forecast
    >>> fc.update(observed_next_day)           # date, node, target
    """

    def __init__(self, model: torch.nn.Module, dataset_train, history: Optional[pd.DataFrame] = None,
                 history_steps: Optional[int] = None, pool: Optional[Iterable] = None,
                 batch_size: int = 32, replay_steps: int = 10, replay_lr: float = 1e-4,
                 replay_recent: int = 7, seed: Optional[int] = None, device=None):
        ds = dataset_train
        data = ds.data
        self.node_var = data.node_var
        self.target = ds.target
        self.features = list(ds.features)
        self.nodes = np.asarray(ds.nodes)
        self._data_nodes = np.asarray(data.nodes)
        self._node_int = f"{self.node_var}Int"
        self.N = len(self.nodes)
        self.H = int(ds.out_channels)
        self.features_to_lag = dict(getattr(data, "features_to_lag", None) or {})
        self.computed = dict(data.data_kwargs.get("computed_features", None) or {})
        self.dummies = list(getattr(data, "dummies", None) or [])

        self._max_lag = max([int(b) for _, b in self.features_to_lag.values()], default=0)
        self.history_steps = int(history_steps if history_steps is not None
                                 else self._max_lag + 2 * self.H)

        self.scale_x = NodeScaler(ds.scalers_feat, self.nodes)
        self.scale_y = NodeScaler(ds.scalers_target, self.nodes)
        self.edge_index = ds.edge_index
        self.edge_weight = None if ds.edge_weight is None else ds.edge_weight.float()

        derived = {f"{f}_l{s}" for f, (a, b) in self.features_to_lag.items()
                   for s in range(int(a), int(b) + 1)} | set(self.computed) | {self._node_int}
        if history is None:
            history = pd.concat([data.df_train_original, data.df_test_original])
            history = history.drop(columns=[c for c in history.columns if c in derived])
        self.raw_columns = [c for c in history.columns if c not in derived]
        # One-hot columns of the training features, with the category each one encodes; a
        # raw or derived column that only shares the prefix (e.g. Instant_sin) is not one.
        self._dummy_cols = {v: [c for c in self.features if c.startswith(f"{v}_")
                                and c not in self.raw_columns and c not in derived]
                            for v in self.dummies}

        # Phase of the training blocks: they start every H steps from the first date.
        train_dates = np.sort(pd.to_datetime(ds.dataframe["date"]).unique())
        self._t0 = train_dates[0] if len(train_dates) else None
        diffs = np.diff(train_dates)
        self._step = pd.Series(diffs).mode().iloc[0] if len(diffs) else None

        self._pending: set = set()        # dates forecast and not yet updated
        self._updated: set = set()        # block starts already added to the pool
        self.history = self._complete(self._normalise(history[self.raw_columns]))
        self._trim()

        samples = [d for p in (pool or [ds]) for d in p.pyg_data]
        self.engine = ReplayEngine(model, samples, self.N, batch_size, replay_steps,
                                   replay_lr, replay_recent, seed, device=device)

    # -- rows ---------------------------------------------------------------- #
    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        unknown = set(df[self.node_var].unique()) - set(self.nodes)
        if unknown:
            raise ValueError(f"Nodes absent from training: {sorted(unknown)}")
        return df.drop_duplicates([self.node_var, "date"], keep="last")

    def _complete(self, df: pd.DataFrame) -> pd.DataFrame:
        """Every node at every date of ``df``, so that a lag of k rows is a lag of k steps."""
        keys = [self.node_var, "date"]
        grid = pd.MultiIndex.from_product([self.nodes, np.sort(df["date"].unique())], names=keys)
        if len(grid) == len(df):
            return df.sort_values(keys).reset_index(drop=True)
        return df.set_index(keys).reindex(grid).reset_index()

    def _ingest(self, df: pd.DataFrame, require_target: bool = False) -> np.ndarray:
        """
        Write ``df`` into the buffer and return its sorted dates. Its non-NaN values
        replace those of the buffer; its NaN values leave them unchanged. A node absent
        from ``df`` gets an empty row at each of its dates. Nothing is written when ``df``
        is rejected.
        """
        df = self._normalise(df)
        extra = [c for c in df.columns if c not in self.raw_columns]
        if extra:
            raise ValueError(f"Columns absent from the history: {extra}")
        if require_target and self.target not in df:
            raise ValueError(f"update() needs the target column '{self.target}'.")
        dates = np.sort(df["date"].unique())
        if len(dates) % self.H:
            raise ValueError(f"A call must cover whole blocks of {self.H} timestamps, "
                             f"got {len(dates)}.")
        if self._t0 is not None and self._step is not None:
            phase = ((dates[0] - self._t0) // self._step) % self.H
            if phase:
                warnings.warn(f"The block starting at {pd.Timestamp(dates[0])} is shifted by "
                              f"{phase} step(s) from the blocks of the training data.")
        in_buffer = self.history["date"].isin(dates)
        missing = [c for c in self.raw_columns if c not in df.columns and c != self.target]
        if missing:
            new = np.setdiff1d(dates, self.history.loc[in_buffer, "date"].unique())
            if len(new):
                raise ValueError(f"{len(new)} timestamp(s) (first {pd.Timestamp(new[0])}) are not "
                                 f"in the buffer and their covariates {missing} were not passed: "
                                 "pass them, or raise history_steps if they were sent earlier.")
        df = self._complete(df)
        if not in_buffer.any():
            hist = pd.concat([self.history, df.reindex(columns=self.raw_columns)], ignore_index=True)
        else:
            keys = [self.node_var, "date"]
            merged = df.set_index(keys).combine_first(self.history[in_buffer].set_index(keys))
            hist = pd.concat([self.history[~in_buffer],
                              merged.reset_index().reindex(columns=self.raw_columns)],
                             ignore_index=True)
        self.history = hist
        return dates

    def _trim(self) -> None:
        """Keep ``history_steps`` timestamps, and the lags of every date still pending."""
        dates = np.sort(self.history["date"].unique())
        cut = len(dates) - self.history_steps
        if self._pending:
            cut = min(cut, int(np.searchsorted(dates, np.datetime64(min(self._pending)))) - self._max_lag)
        if cut > 0:
            self.history = self.history[self.history["date"] >= dates[cut]].reset_index(drop=True)
            self._updated = {d for d in self._updated if d >= dates[cut]}

    def _dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot columns of the training features, matched by value (1 == 1.0 == "1")."""
        cols = {}
        for var, names in self._dummy_cols.items():
            vals = df[var]
            num = pd.to_numeric(vals, errors="coerce")
            text = vals.astype(str)
            for c in names:
                cat = c[len(var) + 1:]
                try:
                    hit = num.to_numpy() == float(cat)
                except ValueError:
                    hit = (text == cat).to_numpy()
                cols[c] = hit.astype(np.float32)
        if not cols:
            return df
        return pd.concat([df.drop(columns=[c for c in cols if c in df]),
                          pd.DataFrame(cols, index=df.index)], axis=1)

    def prepare(self, dates: Sequence) -> pd.DataFrame:
        """Model-ready rows of the buffered ``dates``: lags, computed features, dummies."""
        df = self.history.sort_values([self.node_var, "date"]).reset_index(drop=True)
        buffered = np.sort(df["date"].unique())
        first = np.searchsorted(buffered, np.min(np.asarray(dates, dtype="datetime64[ns]")))
        if first < self._max_lag:
            warnings.warn(f"The buffer holds {first} timestamp(s) before {pd.Timestamp(buffered[first])}, "
                          f"fewer than the largest lag ({self._max_lag}): those lags are NaN and "
                          "their rows are zeroed. Raise history_steps or pass a longer history.")
        if self.features_to_lag:
            g = df.groupby(self.node_var, sort=False)
            lags = {f"{f}_l{s}": g[f].shift(s) for f, (a, b) in self.features_to_lag.items()
                    for s in range(int(a), int(b) + 1)}
            df = pd.concat([df, pd.DataFrame(lags)], axis=1)
        for name, fn in self.computed.items():
            df[name] = np.asarray(fn(df))
        df = df[df["date"].isin(pd.DatetimeIndex(dates))].reset_index(drop=True)
        if self._node_int in self.features or self._node_int in self.dummies:
            # DataClass numbers the nodes in float (RegionInt_1.0, ...).
            df[self._node_int] = np.searchsorted(self._data_nodes, df[self.node_var]).astype(float)
        return self._dummies(df)

    def _samples(self, df: pd.DataFrame, dates: np.ndarray) -> List[Data]:
        T = len(dates)
        n = np.searchsorted(self.nodes, df[self.node_var].to_numpy())
        t = np.searchsorted(dates, df["date"].to_numpy())
        X = np.full((self.N, T, len(self.features)), np.nan, dtype=np.float32)
        Y = np.full((self.N, T), np.nan, dtype=np.float32)
        X[n, t] = df[self.features].to_numpy(dtype=np.float32)
        Y[n, t] = df[self.target].to_numpy(dtype=np.float32)

        # As GraphDataset: a row with a missing feature is zeroed, and so are missing targets.
        ok = ~np.isnan(X).any(axis=2)
        starts = ok[:, ::self.H]
        if not starts.all():
            bad = [self.features[j] for j in np.flatnonzero(np.isnan(X[:, ::self.H]).any(axis=(0, 1)))]
            warnings.warn(f"{int((~starts).sum())} block-start row(s) have missing features "
                          f"({bad[:5]}{'...' if len(bad) > 5 else ''}) and are zeroed.")
        Xs = np.zeros_like(X)
        Xs[ok] = self.scale_x.transform(np.where(ok[..., None], X, 0))[ok]
        Ys = self.scale_y.transform(np.nan_to_num(Y)[..., None])[..., 0]
        mask_y = ~np.isnan(Y)
        Ys[~mask_y] = 0

        out = []
        for s in range(0, T, self.H):
            x = torch.from_numpy(Xs[:, s])
            out.append(Data(x=x, edge_index=self.edge_index, edge_weight=self.edge_weight,
                            mask_X=torch.from_numpy(np.repeat(ok[:, s, None], X.shape[2], 1)).float(),
                            mask_y=torch.from_numpy(mask_y[:, s:s + self.H]).float(),
                            y_scaled=torch.from_numpy(np.ascontiguousarray(Ys[:, s:s + self.H])),
                            y=torch.from_numpy(np.nan_to_num(Y[:, s:s + self.H]))))
        return out

    # -- stream ------------------------------------------------------------- #
    def to(self, device) -> "OnlineForecaster":
        """Move the replay (model, optimiser state, pool) to ``device``."""
        self.engine.to(device)
        return self

    def forecast(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast the timestamps of ``rows``.

        Parameters
        ----------
        rows : pandas.DataFrame
            Raw rows of the next blocks: date, node and the raw covariates. The target
            column may be absent or NaN.

        Returns
        -------
        pandas.DataFrame
            Columns ``date``, node variable and ``forecast`` (target units), one row per
            node and timestamp.
        """
        rows = rows.copy()
        if self.target not in rows:
            rows[self.target] = np.nan
        dates = self._ingest(rows)
        self._pending.update(pd.DatetimeIndex(dates))
        samples = self._samples(self.prepare(dates), dates)
        z = self.engine.predict(samples)                             # [S, N, H]
        z = z.transpose(1, 0, 2).reshape(self.N, -1)                 # [N, T]
        pred = self.scale_y.inverse_transform(z)
        self._trim()
        return pd.DataFrame({
            "date": np.tile(dates, self.N),
            self.node_var: np.repeat(self.nodes, len(dates)),
            "forecast": pred.reshape(-1),
        })

    def update(self, observed: pd.DataFrame) -> None:
        """
        Feed back observations and update the model.

        Parameters
        ----------
        observed : pandas.DataFrame
            Date, node and target of whole blocks, plus any covariate to revise (for
            instance realised weather in place of its forecast). Missing covariates are
            read from the rows sent to :meth:`forecast`. A block already added to the pool
            is not added again.
        """
        dates = self._ingest(observed, require_target=True)
        self._pending.difference_update(pd.DatetimeIndex(dates))
        samples = self._samples(self.prepare(dates), dates)
        starts = [pd.Timestamp(d) for d in dates[::self.H]]
        keep = [i for i, d in enumerate(starts) if d not in self._updated]
        if len(keep) < len(starts):
            warnings.warn(f"{len(starts) - len(keep)} block(s) were already added to the pool "
                          "and are skipped.")
        samples = [samples[i] for i in keep if bool(samples[i].mask_y.any())]
        if not samples:
            warnings.warn("update() received no new observed target; the pool is unchanged.")
        else:
            self.engine.observe(samples)
            self._updated.update(starts[i] for i in keep)
        self._trim()
