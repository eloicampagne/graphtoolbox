"""
Online and batch expert aggregation (Opera-style robust prediction).

This module provides :class:`Aggregation`, a self-contained implementation of
sequential expert aggregation in the spirit of the R package ``opera``
(Gaillard & Goude). Given a matrix of expert forecasts and the corresponding
observations, the class combines the experts into a single forecast whose
weights adapt over time so as to track, and provably compete with, the best
expert (or the best fixed convex combination) in hindsight.

Implemented aggregation rules
-----------------------------
- ``"EWA"``    : Exponentially Weighted Average (Hedge). Fixed learning rate,
                 calibrated automatically when observations are available.
- ``"MLpol"``  : ML-Poly with the polynomial potential of Gaillard, Stoltz &
                 van Erven (2014). Fully parameter-free (per-expert adaptive
                 learning rate); the robust default.
- ``"BOA"``    : Bernstein Online Aggregation (Wintenberger, 2017). Second-order
                 update with an exponential potential.
- ``"uniform"``: Static uniform mean of the experts (baseline).
- ``"best"``   : Oracle constant weight on the single best expert over the whole
                 horizon (baseline; requires observations).

References
----------
- Cesa-Bianchi, N. & Lugosi, G. (2006). *Prediction, Learning, and Games.*
- Gaillard, P., Stoltz, G. & van Erven, T. (2014). A second-order bound with
  excess losses. *COLT.*
- Wintenberger, O. (2017). Optimal learning with Bernstein online aggregation.
  *Machine Learning.*
- Gaillard, P. & Goude, Y. *opera: Online Prediction by Expert Aggregation*
  (R package).
"""

from typing import Optional, Union, Dict

import numpy as np

try:
    import torch
    _TorchTensor = torch.Tensor
except ImportError:  # torch is a hard dependency of the package, but stay safe.
    torch = None
    _TorchTensor = ()

ArrayLike = Union[np.ndarray, "_TorchTensor"]

_MODELS = ("EWA", "MLpol", "BOA", "uniform", "best")
_LOSSES = ("square", "absolute", "percentage")


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Coerce a numpy array or torch tensor to a float64 numpy array."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _loss(pred: ArrayLike, y: ArrayLike, kind: str) -> np.ndarray:
    """Point-wise loss between a prediction and an observation."""
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if kind == "square":
        return (pred - y) ** 2
    if kind == "absolute":
        return np.abs(pred - y)
    if kind == "percentage":
        return np.abs(pred - y) / np.maximum(np.abs(y), 1e-12)
    raise ValueError(f"Unknown loss '{kind}'. Choose from {_LOSSES}.")


def _loss_gradient(pred: float, y: float, kind: str) -> float:
    """Sub-gradient of the loss with respect to the prediction."""
    if kind == "square":
        return 2.0 * (pred - y)
    if kind == "absolute":
        return float(np.sign(pred - y))
    if kind == "percentage":
        return float(np.sign(pred - y)) / max(abs(y), 1e-12)
    raise ValueError(f"Unknown loss '{kind}'. Choose from {_LOSSES}.")


class Aggregation:
    """
    Sequential aggregation of expert forecasts.

    The estimator consumes a matrix of expert predictions of shape ``[T, K]``
    (``T`` time steps, ``K`` experts) and, when observations ``y`` of shape
    ``[T]`` are available, produces an aggregated forecast whose weights are
    updated online. The full trajectory of weights is exposed in
    :attr:`weights_`, and the final convex weight vector in
    :attr:`coefficients_`, so that the fitted mixture can be reused as a static
    combiner on out-of-sample experts through :meth:`predict`.

    Parameters
    ----------
    model : {"MLpol", "EWA", "BOA", "uniform", "best"}, default "MLpol"
        Aggregation rule. ``"MLpol"`` is parameter-free and is a robust default.
    loss : {"square", "absolute", "percentage"}, default "square"
        Loss used both to evaluate experts and to drive the weight updates.
    learning_rate : float or None, default None
        Learning rate ``eta`` for ``"EWA"`` and ``"BOA"``. When ``None`` and
        observations are provided, it is calibrated automatically by replaying
        the sequence over a logarithmic grid and keeping the value with the
        lowest cumulative loss (see :attr:`learning_rate_`). Ignored by
        ``"MLpol"``, ``"uniform"`` and ``"best"``.
    gradient : bool, default True
        If ``True``, updates use the linearised (gradient) pseudo-loss
        ``g_t * x_{t,k}``, as in ``opera``. This guarantees bounded, convex
        regrets. If ``False``, updates use the raw per-expert losses.
    prior : numpy.ndarray or None, default None
        Optional prior weights of shape ``[K]`` (non-negative, summing to one).
        Defaults to the uniform prior.

    Attributes
    ----------
    weights_ : numpy.ndarray
        Weight trajectory of shape ``[T, K]``; row ``t`` holds the weights used
        to form the forecast at time ``t`` (before observing ``y_t``).
    coefficients_ : numpy.ndarray
        Final weight vector of shape ``[K]`` (the weights that would be applied
        to the next, unseen time step).
    prediction_ : numpy.ndarray
        Aggregated forecast of shape ``[T]`` from the last online run.
    experts_loss_ : numpy.ndarray
        Per-expert cumulative loss of shape ``[K]``.
    loss_ : float
        Cumulative loss of the aggregated forecast.
    learning_rate_ : float
        Learning rate actually used (relevant for ``"EWA"`` / ``"BOA"``).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> T, K = 500, 4
    >>> y = np.sin(np.linspace(0, 20, T))
    >>> experts = y[:, None] + rng.normal(0, [0.1, 0.3, 0.6, 1.0], size=(T, K))
    >>> agg = Aggregation(model="MLpol").run(experts, y)
    >>> agg.loss_ < (experts - y[:, None]).__pow__(2).mean(0).min() * T
    True
    """

    def __init__(
        self,
        model: str = "MLpol",
        loss: str = "square",
        learning_rate: Optional[float] = None,
        gradient: bool = True,
        prior: Optional[ArrayLike] = None,
    ) -> None:
        if model not in _MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from {_MODELS}.")
        if loss not in _LOSSES:
            raise ValueError(f"Unknown loss '{loss}'. Choose from {_LOSSES}.")
        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate
        self.gradient = gradient
        self.prior = None if prior is None else _to_numpy(prior).ravel()

        # Fitted state (populated by ``run`` / ``fit``).
        self.weights_: Optional[np.ndarray] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.prediction_: Optional[np.ndarray] = None
        self.experts_loss_: Optional[np.ndarray] = None
        self.loss_: Optional[float] = None
        self.learning_rate_: Optional[float] = None

        # Streaming state (populated by ``reset`` / ``partial_fit`` / ``update``).
        self._K: Optional[int] = None
        self._stream_ready = False

    # ------------------------------------------------------------------ #
    # Batch / online sequential engine
    # ------------------------------------------------------------------ #
    def run(self, experts: ArrayLike, y: ArrayLike, block_size: int = 1) -> "Aggregation":
        """
        Run the online aggregation over a full sequence.

        Parameters
        ----------
        experts : array-like of shape [T, K]
            Expert forecasts (numpy array or torch tensor).
        y : array-like of shape [T]
            Observations.
        block_size : int, default 1
            Forecast-commitment granularity, following ``opera``'s
            ``predict`` / ``update`` split. With ``block_size=1`` the weights are
            updated at every step. With ``block_size=h`` the forecast for a whole
            block of ``h`` steps is committed with the weights known at the
            block's start; once the block is observed the internal weights still
            advance one step at a time within it, as ``opera`` loops over the
            block inside ``update``. For a day-ahead forecast served in daily
            batches of 48 half-hours, set ``block_size=48``.

        Returns
        -------
        Aggregation
            ``self``, with :attr:`weights_`, :attr:`prediction_`,
            :attr:`coefficients_`, :attr:`experts_loss_` and :attr:`loss_` set.
        """
        X = _to_numpy(experts)
        yv = _to_numpy(y).ravel()
        if X.ndim != 2:
            raise ValueError(f"`experts` must be 2-D [T, K], got shape {X.shape}.")
        if X.shape[0] != yv.shape[0]:
            raise ValueError(
                f"Time dimension mismatch: experts has T={X.shape[0]}, y has T={yv.shape[0]}."
            )
        if block_size < 1:
            raise ValueError(f"`block_size` must be >= 1, got {block_size}.")
        T, K = X.shape
        if self.prior is not None and self.prior.shape[0] != K:
            raise ValueError(f"`prior` has length {self.prior.shape[0]} but K={K}.")

        if self.model == "uniform":
            return self._run_static(X, yv, np.full(K, 1.0 / K))
        if self.model == "best":
            per_expert = _loss(X, yv[:, None], self.loss).sum(axis=0)
            w = np.zeros(K)
            w[int(np.argmin(per_expert))] = 1.0
            return self._run_static(X, yv, w)

        eta = self._resolve_learning_rate(X, yv, block_size)
        self.learning_rate_ = eta
        weights, preds = self._run_sequential(X, yv, eta, block_size)

        self.weights_ = weights
        self.prediction_ = preds
        # After the replay the sufficient statistics hold the full history, so the
        # current weights are exactly those that would serve the next block.
        self.coefficients_ = self._current_weights()
        self.experts_loss_ = _loss(X, yv[:, None], self.loss).sum(axis=0)
        self.loss_ = float(_loss(preds, yv, self.loss).sum())
        return self

    def fit(self, experts: ArrayLike, y: ArrayLike, block_size: int = 1) -> "Aggregation":
        """Alias of :meth:`run` that returns ``self`` (scikit-style)."""
        return self.run(experts, y, block_size)

    def predict(self, experts: ArrayLike) -> np.ndarray:
        """
        Apply the fitted static weights (:attr:`coefficients_`) to new experts.

        Use this for out-of-sample combination once the mixture has been fitted
        on a history via :meth:`run`. For time-varying online weights on a fresh
        sequence, call :meth:`run` again with the new observations.

        Parameters
        ----------
        experts : array-like of shape [T, K]
            New expert forecasts.

        Returns
        -------
        numpy.ndarray
            Aggregated forecast of shape ``[T]``.
        """
        if self.coefficients_ is None:
            raise RuntimeError("Aggregation is not fitted; call `run` first.")
        X = _to_numpy(experts)
        if X.shape[1] != self.coefficients_.shape[0]:
            raise ValueError(
                f"Expected {self.coefficients_.shape[0]} experts, got {X.shape[1]}."
            )
        return X @ self.coefficients_

    # ------------------------------------------------------------------ #
    # Streaming interface (genuine online use)
    # ------------------------------------------------------------------ #
    def reset(self, n_experts: int) -> "Aggregation":
        """
        Initialise streaming state for ``n_experts`` experts.

        After ``reset``, alternate :meth:`partial_fit` (to get the forecast for
        the current step) and :meth:`update` (to feed back the observation).

        Parameters
        ----------
        n_experts : int
            Number of experts ``K``.

        Returns
        -------
        Aggregation
            ``self``.
        """
        if self.model in ("uniform", "best"):
            if self.model == "best":
                raise ValueError("model='best' is an offline oracle; use `run`.")
        K = int(n_experts)
        self._K = K
        self._prior = self.prior if self.prior is not None else np.full(K, 1.0 / K)
        # Sufficient statistics shared across rules.
        self._cum_loss = np.zeros(K)   # EWA: cumulative (pseudo-)loss
        self._R = np.zeros(K)          # cumulative regret (MLpol, BOA)
        self._B = np.zeros(K)          # cumulative squared regret (MLpol, BOA)
        self._eta = self.learning_rate  # may be None -> requires a float below
        self._last_p: Optional[np.ndarray] = None
        self._weights_hist = []
        self._stream_ready = True
        return self

    def partial_fit(self, expert_row: ArrayLike) -> float:
        """
        Return the aggregated forecast for the current step.

        Parameters
        ----------
        expert_row : array-like of shape [K]
            The experts' forecasts for the current time step.

        Returns
        -------
        float
            Aggregated forecast.
        """
        if not self._stream_ready:
            raise RuntimeError("Call `reset(n_experts)` before streaming.")
        x = _to_numpy(expert_row).ravel()
        if x.shape[0] != self._K:
            raise ValueError(f"Expected {self._K} experts, got {x.shape[0]}.")
        p = self._current_weights()
        self._last_p = p
        self._last_x = x
        self._weights_hist.append(p.copy())
        return float(x @ p)

    def update(self, y_t: float) -> None:
        """
        Feed back the observation for the step served by :meth:`partial_fit`.

        Parameters
        ----------
        y_t : float
            Observation for the current time step.
        """
        if self._last_p is None:
            raise RuntimeError("Call `partial_fit` before `update`.")
        x = self._last_x
        p = self._last_p
        pred = float(x @ p)
        self._accumulate(x, pred, float(y_t))
        self._last_p = None

    @property
    def streaming_weights_(self) -> np.ndarray:
        """Weight trajectory accumulated through the streaming interface."""
        return np.asarray(self._weights_hist)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _run_static(self, X: np.ndarray, yv: np.ndarray, w: np.ndarray) -> "Aggregation":
        T = X.shape[0]
        preds = X @ w
        self.coefficients_ = w
        self.weights_ = np.tile(w, (T, 1))
        self.prediction_ = preds
        self.experts_loss_ = _loss(X, yv[:, None], self.loss).sum(axis=0)
        self.loss_ = float(_loss(preds, yv, self.loss).sum())
        self.learning_rate_ = None
        return self

    def _run_sequential(self, X: np.ndarray, yv: np.ndarray, eta: float,
                        block_size: int = 1):
        """
        Replay the online rule in blocks; return (weights [T, K], preds [T]).

        Reproduces ``opera``'s ``predict`` / ``update`` split. Every step in a
        block is *forecast* with the weights held at the block's start (the
        forecast for the whole block is committed before any of it is observed),
        but once the block is revealed the internal state advances one step at a
        time within it, exactly as ``opera`` loops over the block inside
        ``update``. So the reported weight trajectory :attr:`weights_` is constant
        within a block, while the sufficient statistics evolve at every step.
        ``block_size=1`` recovers the plain per-step regime. On return the
        statistics reflect the full history (used for :attr:`coefficients_`).
        """
        self.reset(X.shape[1])
        self._eta = eta
        T, K = X.shape
        weights = np.empty((T, K))
        preds = np.empty(T)
        for s in range(0, T, block_size):
            e = min(s + block_size, T)
            p_block = self._current_weights()    # committed forecast for the block
            weights[s:e] = p_block
            preds[s:e] = X[s:e] @ p_block
            for t in range(s, e):                # internal state advances per step
                p_now = self._current_weights()  # evolving weights, as in opera
                self._accumulate(X[t], float(X[t] @ p_now), yv[t])
        self._stream_ready = False
        return weights, preds

    def _current_weights(self) -> np.ndarray:
        """Weights used at the current step given the accumulated statistics."""
        K = self._K
        if self.model == "uniform":
            return np.full(K, 1.0 / K)
        if self.model == "EWA":
            eta = self._require_eta()
            pot = -eta * (self._cum_loss - self._cum_loss.min())
            w = self._prior * np.exp(pot - pot.max())
            s = w.sum()
            return w / s if s > 0 else np.full(K, 1.0 / K)
        if self.model == "BOA":
            eta = self._require_eta()
            pot = eta * self._R - (eta ** 2) * self._B
            w = self._prior * np.exp(pot - pot.max())
            s = w.sum()
            return w / s if s > 0 else np.full(K, 1.0 / K)
        if self.model == "MLpol":
            pos = np.clip(self._R, 0.0, None)
            eta_k = np.where(self._B > 0, 1.0 / np.maximum(self._B, 1e-12), 0.0)
            num = eta_k * pos
            s = num.sum()
            return num / s if s > 0 else np.full(K, 1.0 / K)
        raise ValueError(f"Unsupported streaming model '{self.model}'.")

    def _accumulate(self, x: np.ndarray, pred: float, y_t: float) -> None:
        """Update sufficient statistics after observing ``y_t``."""
        if self.gradient:
            g = _loss_gradient(pred, y_t, self.loss)
            expert_pseudo = g * x          # linearised expert loss
            mixture_pseudo = g * pred      # linearised mixture loss
        else:
            expert_pseudo = _loss(x, y_t, self.loss)
            mixture_pseudo = _loss(np.array(pred), y_t, self.loss)
        r = mixture_pseudo - expert_pseudo  # instantaneous regret vs each expert
        self._cum_loss += expert_pseudo
        self._R += r
        self._B += r ** 2

    def _require_eta(self) -> float:
        if self._eta is None:
            raise RuntimeError(
                f"model='{self.model}' needs a learning_rate. Provide one at "
                "construction, or call `run` with observations for auto-calibration."
            )
        return float(self._eta)

    def _resolve_learning_rate(self, X: np.ndarray, yv: np.ndarray,
                               block_size: int = 1) -> Optional[float]:
        """Return the learning rate to use, calibrating on a grid if needed."""
        if self.model not in ("EWA", "BOA"):
            return None
        if self.learning_rate is not None:
            return float(self.learning_rate)
        # Grid calibration by sequential replay, keeping the lowest cumulative loss.
        scale = max(np.ptp(X), np.ptp(yv), 1e-8)
        base = 1.0 / (scale ** 2 if self.loss == "square" else scale)
        grid = base * np.logspace(-3, 3, 13)
        best_eta, best_loss = grid[0], np.inf
        for eta in grid:
            _, preds = self._run_sequential(X, yv, float(eta), block_size)
            cum = float(_loss(preds, yv, self.loss).sum())
            if cum < best_loss:
                best_loss, best_eta = cum, float(eta)
        return best_eta

    def summary(self) -> Dict[str, float]:
        """
        Return a compact performance summary of the fitted mixture.

        Returns
        -------
        dict
            Keys: ``model``, ``loss`` (mean aggregated loss), ``best_expert_loss``
            (mean loss of the best single expert), ``mean_expert_loss`` (average
            over experts of their mean loss) and ``learning_rate``.
        """
        if self.prediction_ is None:
            raise RuntimeError("Aggregation is not fitted; call `run` first.")
        T = self.prediction_.shape[0]
        return {
            "model": self.model,
            "loss": self.loss_ / T,
            "best_expert_loss": float(self.experts_loss_.min()) / T,
            "mean_expert_loss": float(self.experts_loss_.mean()) / T,
            "learning_rate": self.learning_rate_,
        }

    def __repr__(self) -> str:
        return (
            f"Aggregation(model='{self.model}', loss='{self.loss}', "
            f"gradient={self.gradient})"
        )
