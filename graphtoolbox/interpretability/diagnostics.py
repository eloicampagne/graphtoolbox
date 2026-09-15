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


@torch.no_grad()
def dump_attention_batches(model, dataset, output_dir: str, batch_size: int = 32,
                           require_full_batch: bool = True) -> int:
    """Save attention dictionaries for successive PyG mini-batches."""
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
            return_attention=True, batch_size=batch_size,
        )
        attention = output[-1] if isinstance(output, tuple) else None
        if attention is None:
            continue
        torch.save(attention, os.path.join(output_dir, f"num_batch{index}.pt"))
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
        graphs = int(cache["num_graphs"])
        current_edges = cache["edge_idx"]
        if edge_index is None:
            edge_index = current_edges
        elif not torch.equal(edge_index, current_edges):
            raise ValueError("edge_index changes between cached attention batches")
        edge_count = current_edges.shape[1]
        layers = []
        for attention in cache["attention_weights"]:
            attention = attention.detach().float()
            if attention.dim() > 1:
                attention = attention.mean(dim=-1)
            if attention.numel() != graphs * edge_count:
                raise ValueError(
                    f"{os.path.basename(path)}: attention {tuple(attention.shape)} "
                    f"does not tile {graphs} graphs x {edge_count} edges")
            layers.append(attention.view(graphs, edge_count))
        masks.append(torch.stack(layers).mean(dim=0))
    return torch.cat(masks, dim=0), edge_index


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
