"""Graph learning, attention analysis and plotting helpers."""

from graphtoolbox._lazy import install_lazy_exports

_SYMBOL_MODULES = {
    # Graph structure learning (GL-3SR).
    "FGL_3SR": ".GL_3SR",
    "f_obj": ".GL_3SR",
    "get_X_complete_graph": ".GL_3SR",
    # Configuration and general helpers.
    "clean_dir": ".helper_functions",
    "batch_y_to_tensor": ".helper_functions",
    "batch_x_to_tensor": ".helper_functions",
    "load_config": ".helper_functions",
    "load_kwargs": ".helper_functions",
    "save_config": ".helper_functions",
    "update_config": ".helper_functions",
    "parser_config": ".helper_functions",
    "extract_parameters": ".helper_functions",
    "change_cwd": ".helper_functions",
    "get_geodesic_distance": ".helper_functions",
    "get_exponential_similarity": ".helper_functions",
    "build_adjacency_matrix": ".helper_functions",
    # Attention weight analysis.
    "load_attention_batches": ".attention",
    "compute_attention_statistics": ".attention",
    "plot_attention_statistics": ".attention",
    "animate_grouped_attention": ".attention",
    "pca_analysis_attention": ".attention",
    "umap_analysis_attention": ".attention",
    "normalized_laplacian": ".attention",
    "plot_spectral_gap": ".attention",
    "spectral_embedding": ".attention",
    "cosine_similarity_matrix": ".attention",
    "spectral_fusion": ".attention",
    "hierarchical_attention_fusion": ".attention",
    "attention_to_dense": ".attention",
    "pca_per_head": ".attention",
    "pca_global_mean": ".attention",
    "plot_explained_variance": ".attention",
    "plot_components": ".attention",
    # Plotting.
    "plot_losses": ".visualizations",
    "plot_nodes": ".visualizations",
    "plot_graph_map": ".visualizations",
    "plot_node_errors_map": ".visualizations",
    "plot_all_graph_maps": ".visualizations",
}

__getattr__, __dir__, __all__ = install_lazy_exports(__name__, globals(), _SYMBOL_MODULES)
