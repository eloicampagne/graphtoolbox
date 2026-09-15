"""Post-hoc explanation and accumulated local effects (ALE) analysis."""

from graphtoolbox._lazy import install_lazy_exports

_SYMBOL_MODULES = {
    "VisualizationConfig": ".explain",
    "plot_explanation_graph": ".explain",
    "get_group_feature_mats": ".explain",
    "compute_ALE": ".explain",
    "compute_ALE_avg_over_instants": ".explain",
    "compute_ALE_per_node": ".explain",
    "compute_ALE_per_node_avg_over_instants": ".explain",
    "plot_ALE": ".explain",
    "plot_ALE_avg": ".explain",
    "plot_ALE_nodes": ".explain",
    "ale_scalar_importance": ".explain",
    "compute_feature_importances_from_ALE": ".explain",
    "plot_feature_importance_bar": ".explain",
    "RegressionExplanationWrapper": ".diagnostics",
    "make_edge_explainer": ".diagnostics",
    "compute_edge_masks": ".diagnostics",
    "topk_indices": ".diagnostics",
    "explainer_reproducibility": ".diagnostics",
    "dump_attention_batches": ".diagnostics",
    "load_attention_batches": ".diagnostics",
    "aggregate_ale_importance": ".diagnostics",
    "plot_ale_group_importance": ".diagnostics",
}

__getattr__, __dir__, __all__ = install_lazy_exports(__name__, globals(), _SYMBOL_MODULES)
