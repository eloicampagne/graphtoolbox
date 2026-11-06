"""Reusable edge-attribution and ALE diagnostics."""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch import nn


class RegressionExplanationWrapper(nn.Module):
    """Expose a GraphToolbox model as a plain node-regression callable."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, **kwargs):
        output = self.model(x, edge_index)
        return output[0] if isinstance(output, (tuple, list)) else output


def make_edge_explainer(model, epochs: int = 100):
    """Build a PyG ``GNNExplainer`` configured for node regression."""
    from torch_geometric.explain import Explainer, GNNExplainer

    return Explainer(
        model=RegressionExplanationWrapper(model),
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model", edge_mask_type="object",
        model_config=dict(mode="regression", task_level="node",
                          return_type="raw"),
    )


def compute_edge_masks(dataset, model, epochs: int = 100, limit: int | None = None,
                       output_path: str | None = None) -> np.ndarray:
    """Run an edge explainer on graph snapshots and optionally cache its masks."""
    explainer = make_edge_explainer(model, epochs=epochs)
    count = len(dataset) if not limit else min(limit, len(dataset))
    masks = []
    for index in range(count):
        graph = dataset[index]
        explanation = explainer(graph.x, graph.edge_index)
        masks.append(explanation.edge_mask.detach().cpu().numpy())
    stacked = np.stack(masks)
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.save(output_path, stacked)
    return stacked


def topk_indices(values, count: int) -> set[int]:
    """Indices of the ``count`` largest values."""
    return set(np.argsort(-np.asarray(values))[:count])


def explainer_reproducibility(dataset, model, day: int = 0, runs: int = 2,
                              epochs: int = 100) -> dict[str, float]:
    """Measure agreement between independent explanations of one snapshot."""
    if runs < 2:
        raise ValueError("runs must be at least two")
    explainer = make_edge_explainer(model, epochs=epochs)
    graph = dataset[day]
    masks = [explainer(graph.x, graph.edge_index).edge_mask.detach().cpu().numpy()
             for _ in range(runs)]
    top_count = max(1, len(masks[0]) // 10)
    top_overlap = topk_indices(masks[0], top_count).intersection(
        topk_indices(masks[1], top_count)
    )
    return {
        "correlation": float(np.corrcoef(masks[0], masks[1])[0, 1]),
        "top_count": top_count,
        "top_overlap": len(top_overlap),
        "chance_overlap": float(top_count * top_count / len(masks[0])),
    }


def _attention_cache_tensor(data: dict, name: str = "") -> torch.Tensor:
    """Return the attention of one cache file as a [L, B, E_graph, H] tensor.

    Files written since GraphToolbox 0.3.1 (``format == "per_graph"``) store one
    [B, E_graph, H] tensor per layer, already matched to ``edge_idx``. Older files
    store the raw [B * E_graph, H] output of the convolution, which can be tiled by
    graph only when it holds a single graph: GATConv appends the self-loops of the
    whole batch after every other edge, so a batch of several graphs is not ordered
    graph by graph.
    """
    layers = data["attention_weights"]
    if data.get("format") == "per_graph":
        return torch.stack([a.float() if a.dim() == 3 else a.float().unsqueeze(-1) for a in layers])
    E_graph = data["edge_idx"].shape[1]
    E_total = layers[0].shape[0]
    B = int(data.get("num_graphs") or E_total // E_graph)
    if B * E_graph != E_total:
        raise ValueError(f"{name}: attention of {E_total} edges does not tile {B} graphs x {E_graph} edges")
    if B > 1:
        raise ValueError(f"{name}: legacy cache with {B} graphs per file; its weights are not ordered "
                         "graph by graph and cannot be matched to edges. Regenerate it with "
                         "dump_attention_batches or Trainer(return_attention=True).")
    return torch.stack([(a if a.dim() == 2 else a.unsqueeze(-1)).float().view(B, E_graph, -1) for a in layers])


@torch.no_grad()
def dump_attention_batches(model, dataset, output_dir: str, batch_size: int = 32,
                           require_full_batch: bool = True) -> int:
    """Save the attention of successive PyG mini-batches, matched to the edges.

    Each file ``num_batch{i}.pt`` holds ``attention_weights`` (one [B, E, H] tensor
    per layer), ``edge_idx`` (the [2, E] local edge index the weights refer to,
    including the self-loops the convolution adds), ``num_graphs`` and
    ``format == "per_graph"``.
    """
    from torch_geometric.loader import DataLoader

    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    saved = 0
    for index, batch in enumerate(loader):
        for attribute in ("x", "y_scaled", "y", "edge_weight", "mask_y"):
            value = getattr(batch, attribute, None)
            if value is not None:
                setattr(batch, attribute, value.to(torch.float32))
        batch = batch.to(device)
        if require_full_batch and getattr(batch, "num_graphs", None) != batch_size:
            continue
        output = model(
            batch.x, batch.edge_index,
            edge_weight=getattr(batch, "edge_weight", None),
            mask=getattr(batch, "mask_y", None),
            return_attention=True, num_graphs=batch.num_graphs,
        )
        attention = output[-1] if isinstance(output, tuple) else None
        if not attention or not attention.get("per_graph"):
            continue
        torch.save({"attention_weights": attention["per_graph"], "edge_idx": attention["edge_index"][0],
                    "num_graphs": int(batch.num_graphs), "format": "per_graph"},
                   os.path.join(output_dir, f"num_batch{index}.pt"))
        saved += 1
    return saved


def load_attention_batches(attention_dir: str):
    """Load cached attention as a ``[snapshots, edges]`` tensor.

    Heads and message-passing layers are averaged.  The returned edge index is
    checked for consistency across cached batches.
    """
    files = sorted(
        glob.glob(os.path.join(attention_dir, "num_batch*.pt")),
        key=lambda path: int(
            "".join(filter(str.isdigit, os.path.basename(path)))
        ),
    )
    if not files:
        raise FileNotFoundError(f"no cached attention under {attention_dir}")
    masks = []
    edge_index = None
    for path in files:
        cache = torch.load(path, weights_only=False, map_location="cpu")
        current_edges = cache["edge_idx"]
        if edge_index is None:
            edge_index = current_edges
        elif not torch.equal(edge_index, current_edges):
            raise ValueError("edge_index changes between cached attention batches")
        attention = _attention_cache_tensor(cache, os.path.basename(path))   # [L, B, E, H]
        masks.append(attention.mean(dim=(0, 3)))
    return torch.cat(masks, dim=0), edge_index


def _float_graph(graph, device):
    edge_weight = getattr(graph, "edge_weight", None)
    return (graph.x.float().to(device), graph.edge_index.to(device),
            None if edge_weight is None else edge_weight.float().to(device))


@torch.no_grad()
def attention_by_snapshot(model, dataset, limit: int | None = None):
    """Attention of every layer on every snapshot, matched to a fixed edge list.

    Each snapshot is run on its own and every weight is matched to its edge through
    the edge index the convolution returns. Heads are averaged.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``alpha`` [snapshots, layers, edges] and ``edge_index`` [2, edges], the edge
        list of the attention layers (input edges plus the self-loops they add).
    """
    device = next(model.parameters()).device
    count = len(dataset) if not limit else min(limit, len(dataset))
    model.eval()
    alpha, edge_index = [], None
    for index in range(count):
        x, ei, ew = _float_graph(dataset[index], device)
        output = model(x, ei, edge_weight=ew, return_attention=True)
        attention = output[-1] if isinstance(output, tuple) else None
        if not attention or not attention.get("per_graph"):
            raise ValueError("the model returns no attention weights")
        layers = attention["edge_index"]
        if any(not torch.equal(layers[0], e) for e in layers[1:]):
            raise ValueError("attention layers use different edge lists")
        if edge_index is None:
            edge_index = layers[0]
        elif not torch.equal(edge_index, layers[0]):
            raise ValueError("edge list changes between snapshots")
        alpha.append(torch.stack([a[0].mean(dim=-1) for a in attention["per_graph"]]))
    return torch.stack(alpha).numpy(), edge_index.numpy()


def target_unit_scale(scalers, nodes) -> np.ndarray:
    """Size, in target units, of one unit of each node's scaled target.

    Valid for affine scalers (MinMaxScaler, StandardScaler) as stored in
    ``GraphDataset.scalers_target``.
    """
    scale = []
    for node in nodes:
        low, high = scalers[node].inverse_transform(np.array([[0.0], [1.0]])).ravel()
        scale.append(high - low)
    return np.asarray(scale, dtype=float)


@torch.no_grad()
def occlusion_influence(model, dataset, baseline=None, node_scale=None, limit: int | None = None) -> np.ndarray:
    """Influence of each node's inputs on each node's forecast, by occlusion.

    For every snapshot and every node j, the input features of j are replaced by
    a baseline (by default their mean over the snapshots) and the model is run
    again. The influence of j on i is the mean absolute change of the outputs of
    i. It counts every path from j to i, whatever the convolution.

    Parameters
    ----------
    baseline : torch.Tensor, optional
        [nodes, features] replacement values. Defaults to the mean over snapshots.
    node_scale : array-like, optional
        Target units per scaled unit for each node (see ``target_unit_scale``),
        to express the influence in target units.

    Returns
    -------
    np.ndarray
        [snapshots, receiver, sender] influence.
    """
    device = next(model.parameters()).device
    count = len(dataset) if not limit else min(limit, len(dataset))
    X = torch.stack([dataset[t].x.float() for t in range(count)])
    baseline = X.mean(dim=0) if baseline is None else torch.as_tensor(baseline, dtype=torch.float32)
    _, ei, ew = _float_graph(dataset[0], device)
    n = X.shape[1]
    scale = np.ones(n) if node_scale is None else np.asarray(node_scale, dtype=float)
    model.eval()
    influence = np.zeros((count, n, n))
    for t in range(count):
        reference = model(X[t].to(device), ei, edge_weight=ew)
        reference = (reference[0] if isinstance(reference, tuple) else reference).cpu().numpy()
        for j in range(n):
            x = X[t].clone()
            x[j] = baseline[j]
            out = model(x.to(device), ei, edge_weight=ew)
            out = (out[0] if isinstance(out, tuple) else out).cpu().numpy()
            influence[t, :, j] = np.abs(out - reference).reshape(n, -1).mean(axis=1) * scale
    return influence


def neighbourhood_deviation(values, edge_index, exclude_self: bool = True) -> np.ndarray:
    """Relative deviation of each edge from the uniform share of its receiver.

    ``values`` holds one non-negative value per edge (an attention weight, an
    influence) in the order of ``edge_index``, with a leading batch dimension
    allowed. Within the neighbors of receiver i, the share of sender j is
    s_ij = v_ij / sum_k v_ik, and the deviation is m_i * s_ij - 1, where m_i is
    the number of neighbors: 0 for a uniform share, 0.1 for a share 10% above it.
    Self-loops are excluded by default and receive NaN.
    """
    values = np.asarray(values, dtype=float)
    src, dst = np.asarray(edge_index)
    keep = src != dst if exclude_self else np.ones(src.shape, bool)
    out = np.full(values.shape, np.nan)
    for i in np.unique(dst[keep]):
        e = np.where(keep & (dst == i))[0]
        total = values[..., e].sum(axis=-1, keepdims=True)
        out[..., e] = len(e) * values[..., e] / total - 1
    return out


def sender_variance_share(deviation, edge_index) -> float:
    """Share of the variance of per-edge values explained by the sending node.

    A share of 1 means that every receiver weighs a given sender identically, as
    static attention does on graphs whose neighborhoods coincide.
    """
    deviation = np.asarray(deviation, dtype=float)
    src = np.asarray(edge_index)[0]
    ok = np.isfinite(deviation)
    v, s = deviation[ok], src[ok]
    fitted = np.array([v[s == j].mean() for j in s])
    total = ((v - v.mean()) ** 2).sum()
    return float(1 - ((v - fitted) ** 2).sum() / total) if total > 0 else 1.0


def aggregate_ale_importance(table, group_col: str = "group",
                             value_col: str = "importance",
                             reduction: str = "mean"):
    """Aggregate feature-level ALE importance into normalized group shares.

    The default is a mean so groups with many lag columns are not rewarded merely
    for containing more features. Pass ``reduction='sum'`` when total feature
    importance, rather than typical within-group importance, is desired.
    """
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")
    grouped = getattr(table.groupby(group_col, as_index=False)[value_col],
                      reduction)().copy()
    total = float(grouped[value_col].sum())
    grouped["share"] = grouped[value_col] / total if total > 0 else 0.0
    return grouped.sort_values("share", ascending=False).reset_index(drop=True)


def plot_ale_group_importance(table, output_path: str | None = None,
                              group_col: str = "group",
                              value_col: str = "importance",
                              color: str = "#7B2CBF", figsize=(3.3, 1.8)):
    """Render normalized ALE importance by feature group."""
    import matplotlib.pyplot as plt

    grouped = aggregate_ale_importance(table, group_col=group_col,
                                       value_col=value_col)
    plotted = grouped.sort_values("share", ascending=True)
    figure, axis = plt.subplots(figsize=figsize)
    axis.barh(plotted[group_col].str.replace("_", " "),
              100.0 * plotted["share"], color=color, alpha=0.85)
    axis.set_xlabel("ALE importance (%)")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    figure.tight_layout(pad=0.4)
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
    return figure, grouped
