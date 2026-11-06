"""First-order accumulated local effects of an additive graph model.

The accumulated local effect (ALE) of a feature (Apley and Zhu, 2020) partitions its
observed values into bins at quantiles, measures in each bin the mean change of the
prediction when the feature moves from the lower to the upper edge of the bin, the other
inputs of each observation being kept, and accumulates these local changes. Because every
change is evaluated at observed values of the other inputs, correlated inputs are not
moved jointly, which a marginal plot of the prediction against the feature would do.

For an :class:`~graphtoolbox.models.AdditiveGraphModel` the effect of a feature acts only
through the output of its group. An observation is one node of one snapshot: the feature
of that node alone is moved, and the change is summed over all nodes of the graph, so that
it includes the effect transmitted by message passing to the neighbors.
"""
from typing import Optional, Sequence

import numpy as np
import torch


def _bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile edges of the finite values, with duplicates removed (binary features
    give the two edges of their range)."""
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        raise ValueError("The feature is constant; its ALE is undefined.")
    return edges


@torch.no_grad()
def additive_ale(model, x_scaled: torch.Tensor, x_raw: np.ndarray, feature_index: int,
                 edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None,
                 feature_scale: Optional[Sequence[float]] = None,
                 feature_offset: Optional[Sequence[float]] = None,
                 target_slope: Optional[Sequence[float]] = None, n_bins: int = 20,
                 batch_size: int = 512) -> dict:
    """
    First-order ALE of one input feature of an additive graph model.

    :param model: Fitted ``AdditiveGraphModel`` (set to evaluation mode here).
    :param x_scaled: Model inputs of the snapshots, [D, N, F].
    :param x_raw: Raw values of the feature, [D, N]; bins are defined on these values.
    :param feature_index: Column of the feature in ``x_scaled``.
    :param edge_index: Edge index [2, E] of one snapshot.
    :param edge_weight: Optional edge weights [E].
    :param feature_scale: Per-node slope ``a_i`` of the feature scaling, so that the scaled
        value is ``a_i * raw + b_i``. Defaults to 1.
    :param feature_offset: Per-node offset ``b_i``. Defaults to 0.
    :param target_slope: Per-node slope of the inverse target scaling, used to express the
        group output in target units. Defaults to 1 (scaled units).
    :param n_bins: Number of quantile bins (fewer when quantiles coincide).
    :param batch_size: Number of observations evaluated per forward pass.
    :returns: dict with ``edges`` [K+1] (raw units), ``ale`` [out_channels, K+1] (target
        units, centered, effect on the sum over nodes of the prediction), ``counts`` [K],
        ``group`` and ``importance``, the standard deviation of the effect over
        observations, pooled over horizons.
    """
    model.eval()
    device = next(model.parameters()).device
    D, N, F = x_scaled.shape
    group = next((g for g, sl in model.group_index_map.items()
                  if sl.start <= feature_index < sl.stop), None)
    if group is None:
        raise ValueError(f"Feature column {feature_index} belongs to no group of the model.")
    a = torch.as_tensor(np.ones(N) if feature_scale is None else feature_scale, dtype=torch.float32)
    b = torch.as_tensor(np.zeros(N) if feature_offset is None else feature_offset, dtype=torch.float32)
    slope = torch.as_tensor(np.ones(N) if target_slope is None else target_slope, dtype=torch.float32)
    # Contribution of the group to the prediction in target units: slope * Y_g / G.
    weight = (slope / model.num_groups).to(device)

    raw = np.asarray(x_raw, dtype=float)
    d_idx, n_idx = np.nonzero(np.isfinite(raw))
    values = raw[d_idx, n_idx]
    edges = _bin_edges(values, n_bins)
    K = edges.size - 1
    # Bin k (1-based) holds (z_{k-1}, z_k]; the minimum joins the first bin.
    bins = np.clip(np.searchsorted(edges, values, side="left"), 1, K)

    out = model.out_channels
    effect_sum = np.zeros((K, out))
    counts = np.bincount(bins - 1, minlength=K)
    x_scaled = x_scaled.to(torch.float32)
    for start in range(0, values.size, batch_size):
        sl = slice(start, start + batch_size)
        d, n, k = d_idx[sl], n_idx[sl], bins[sl]
        B = d.size
        x = x_scaled[torch.as_tensor(d)].clone()                     # [B, N, F]
        rows = torch.arange(B)
        node = torch.as_tensor(n)
        lo = torch.as_tensor(edges[k - 1], dtype=torch.float32) * a[node] + b[node]
        hi = torch.as_tensor(edges[k], dtype=torch.float32) * a[node] + b[node]
        x_lo, x_hi = x.clone(), x
        x_lo[rows, node, feature_index] = lo
        x_hi[rows, node, feature_index] = hi
        both = torch.cat([x_lo, x_hi], dim=0).reshape(2 * B * N, F).to(device)
        ei = model._repeat_edge_index(edge_index, num_nodes=N, repeats=2 * B, device=device)
        ew = model._repeat_edge_weight(edge_weight, repeats=2 * B, device=device)
        y = model.group_output(group, both, ei, ew).reshape(2, B, N, out)
        delta = ((y[1] - y[0]) * weight[None, :, None]).sum(dim=1).cpu().numpy()   # [B, out]
        np.add.at(effect_sum, k - 1, delta)

    local = effect_sum / np.maximum(counts, 1)[:, None]                  # [K, out]
    acc = np.vstack([np.zeros((1, out)), np.cumsum(local, axis=0)])       # [K+1, out]
    # Effect at each observation, interpolated within its bin, so that a binary feature
    # (a single bin) keeps its effect. Centering makes it average to zero over observations.
    frac = (values - edges[bins - 1]) / (edges[bins] - edges[bins - 1])
    at_obs = acc[bins - 1] + frac[:, None] * local[bins - 1]              # [n_obs, out]
    centre = at_obs.mean(axis=0)
    acc -= centre
    importance = float(np.sqrt(((at_obs - centre) ** 2).mean()))
    return {"edges": edges, "ale": acc.T, "counts": counts, "group": group, "importance": importance}


def additive_ale_from_dataset(model, dataset, feature: str, n_bins: int = 20,
                              batch_size: int = 512) -> dict:
    """
    :func:`additive_ale` on the snapshots of a :class:`~graphtoolbox.data.GraphDataset`.

    The model reads the features of each node at the first time step of each snapshot;
    those raw values define the bins. The per-node MinMax feature scalers and the target
    scalers of the dataset convert the bin edges to model inputs and the group output to
    target units.
    """
    j = list(dataset.features).index(feature)
    starts = range(0, dataset.X_scaled.shape[1] - dataset.out_channels + 1, dataset.out_channels)
    starts = torch.as_tensor(list(starts))
    x_scaled = torch.nan_to_num(dataset.X_scaled[:, starts, :], nan=0.0).permute(1, 0, 2)  # [D, N, F]
    x_raw = dataset.X_node[:, starts, j].numpy().T                                        # [D, N]
    nodes = list(dataset.nodes)
    scale = [float(dataset.scalers_feat[v].scale_[j]) for v in nodes]
    offset = [float(dataset.scalers_feat[v].min_[j]) for v in nodes]
    tgt = []
    for v in nodes:
        sc = dataset.scalers_target[v]
        tgt.append(float(sc.inverse_transform(np.ones((1, 1)))[0, 0] - sc.inverse_transform(np.zeros((1, 1)))[0, 0]))
    ew = None if dataset.edge_weight is None else torch.as_tensor(dataset.edge_weight, dtype=torch.float32)
    return additive_ale(model, x_scaled, x_raw, j, dataset.edge_index, ew,
                        feature_scale=scale, feature_offset=offset, target_slope=tgt,
                        n_bins=n_bins, batch_size=batch_size)
