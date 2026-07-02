from IPython.display import HTML
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import umap

@torch.no_grad()
def load_attention_batches(directory_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load and assemble attention weights dumped in batch files produced by a model.

    Parameters
    ----------
    directory_path : str
        Path to a directory containing attention dump files. The function expects files
        named like "num_batch{n}.pt" (for example "num_batch0.pt"). Files are processed
        in ascending numeric order.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple (all_attentions, edge_index)
        - all_attentions: tensor of shape [L, H, G_total, E_graph]
        - edge_index: tensor of shape [2, E_graph]

    Raises
    ------
    RuntimeError
        If no valid attention dump files are found in the directory.
    ValueError
        If sizes are inconsistent within a file.
    """
    all_graphs = []
    edge_index = None

    batch_files = sorted(
        [f for f in os.listdir(directory_path) if f.startswith("num_batch") and f.endswith(".pt")],
        key=lambda x: int(x.replace("num_batch", "").replace(".pt", ""))
    )

    for file in batch_files:
        data = torch.load(os.path.join(directory_path, file), map_location="cpu")
        if 'attention_weights' not in data or 'edge_idx' not in data:
            print(f"[WARN] Malformed file {file}, skipping.")
            continue

        attn_per_layer = data['attention_weights']  # list len L
        edge_index = data['edge_idx'] if edge_index is None else edge_index

        L = len(attn_per_layer)
        E_total, H = attn_per_layer[0].shape
        E_graph = edge_index.shape[1]
        B = data.get('num_graphs', E_total // E_graph)
        if E_total % E_graph != 0 or B != (E_total // E_graph):
            raise ValueError(f"Inconsistent sizes in {file}: E_total={E_total}, E_graph={E_graph}, B={B}")

        # [L, E_total, H] -> [L, B, E_graph, H] -> [L, H, B, E_graph]
        attn_tensor = torch.stack(attn_per_layer, dim=0)                  # [L, E_total, H]
        attn_tensor = attn_tensor.view(L, B, E_graph, H).permute(0, 3, 1, 2)
        all_graphs.append(attn_tensor)

    if not all_graphs:
        raise RuntimeError(f"Aucun dump d'attention trouvé dans {directory_path}")

    all_attentions = torch.cat(all_graphs, dim=2)  # [L, H, G_total, E_graph]
    return all_attentions, edge_index  # edge_index of the base graph [2, E_graph]


def compute_attention_statistics(
    all_attentions: torch.Tensor,
    edge_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-head, per-layer attention mean and standard deviation mapped to adjacency matrices.

    Parameters
    ----------
    all_attentions : torch.Tensor
        Attention values with shape (L, H, G, E) where
        L = number of layers, H = number of heads, G = number of graphs, E = number of edges.
    edge_index : torch.Tensor or array-like
        Edge indices in shape (2, E) or (E, 2). Node ids need not be contiguous; they
        will be compacted to a contiguous range.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (mean_adj, std_adj) both tensors of shape (L, H, n_used, n_used), where n_used is the
        number of unique nodes present in edge_index. Entries for absent node pairs are zero.

    Raises
    ------
    ValueError
        If edge_index does not have shape (2, E) or (E, 2).
    """
    L, H, G, E = all_attentions.shape

    # Standardize edge_index to [2, E] on same device and long dtype
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index)
    ei = edge_index.detach().long()
    if ei.dim() != 2 or 2 not in ei.shape:
        raise ValueError("edge_index must have shape [2, E] or [E, 2].")
    if ei.shape[0] == 2:
        src, tgt = ei[0], ei[1]
    else:  # [E, 2]
        src, tgt = ei[:, 0], ei[:, 1]
    src = src.to(all_attentions.device).long()
    tgt = tgt.to(all_attentions.device).long()

    # Compact node ids to remove gaps and avoid trailing all-zero rows/cols
    unique_nodes = torch.unique(torch.cat([src, tgt])).sort()[0]
    n_used = int(unique_nodes.numel())
    id_map = torch.full(
        (int(unique_nodes.max().item()) + 1,),
        -1,
        dtype=torch.long,
        device=all_attentions.device
    )
    id_map[unique_nodes] = torch.arange(n_used, device=all_attentions.device)
    src_c = id_map[src]
    tgt_c = id_map[tgt]

    # Compute per-edge statistics across graphs: [L, H, E]
    mean_per_edge = all_attentions.mean(dim=2)
    std_per_edge = all_attentions.std(dim=2)

    # Project to adjacency tensors [L, H, n_used, n_used]
    mean_adj = torch.zeros(
        (L, H, n_used, n_used),
        device=all_attentions.device,
        dtype=all_attentions.dtype
    )
    std_adj = torch.zeros(
        (L, H, n_used, n_used),
        device=all_attentions.device,
        dtype=all_attentions.dtype
    )

    # Fill adjacency using compacted indices
    for l in range(L):
        for h in range(H):
            mean_adj[l, h, src_c, tgt_c] = mean_per_edge[l, h]
            std_adj[l, h, src_c, tgt_c] = std_per_edge[l, h]

    return mean_adj, std_adj

def plot_attention_statistics(
    avg_attn: torch.Tensor,
    std_attn: torch.Tensor,
    **kwargs
) -> None:
    """
    Plot heatmaps of the average and standard deviation attention matrices
    for each layer and attention head.

    :param avg_attn: Mean attention matrices to be visualized, shape [L, H, N, N].
    :type avg_attn: torch.Tensor
    :param std_attn: Standard deviation matrices to be visualized, shape [L, H, N, N].
    :type std_attn: torch.Tensor
    :param kwargs: Optional plotting parameters:
        - figsize: base size for a single head (width, height)
        - fontsize: title font size
    :type kwargs: dict

    :returns: None
    :rtype: None
    """
    L, H, N, _ = avg_attn.shape
    figsize    = kwargs.get('figsize', (3.5, 3))
    fontsize   = kwargs.get('fontsize', 14)
    figsize    = (H * figsize[0], L * figsize[1])

    def plot_matrix_grid(data: torch.Tensor, title_prefix: str):
        fig, axes = plt.subplots(L, H, figsize=figsize,
                                 squeeze=False) 
        for l in range(L):
            for h in range(H):
                ax = axes[l][h]
                sns.heatmap(data[l, h].cpu().numpy().T,
                            ax=ax,
                            cmap='rocket_r',
                            cbar=True)
                ax.set_title(f"{title_prefix} - Layer {l}, Head {h}", fontsize=fontsize/1.6)
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(f"Attention matrices - {title_prefix}", fontsize=fontsize)
        fig.tight_layout(rect=[0, 0, 1, 0.96]) 
        plt.show()

    plot_matrix_grid(avg_attn, "Average")
    plot_matrix_grid(std_attn, "Std")

def animate_grouped_attention(
    all_attentions: torch.Tensor,
    edge_index: torch.Tensor,
    group_variable: list | np.ndarray,
    group_name: str = "Group",
    interval: int = 1000,
    mode: str = "mean",
    save: bool = True,
    **kwargs
):
    """
    Animate attention matrices (mean or standard deviation) across groups of graphs.

    Parameters
    ----------
    all_attentions : torch.Tensor of shape [L, H, G, E]
        Attention scores for all layers (L), heads (H), graphs (G), and edges (E).
    edge_index : torch.Tensor of shape [2, E]
        Edge indices indicating source and target nodes.
    group_variable : array-like of length G
        Group identifier for each graph (e.g., time step, class, or cluster ID).
    group_name : str, optional (default="Group")
        Name of the group variable to display in the animation title.
    interval : int, optional (default=1000)
        Time interval between frames in milliseconds.
    mode : {"mean", "std"}, optional (default="mean")
        Statistic to visualize: either the mean or standard deviation of attention scores.
    save : bool, optional (default=True)
        Save the figure to .gif.
    Returns
    -------
    IPython.display.HTML
        HTML animation of attention matrices for each group, rendered in Jupyter notebooks.
    """
    assert mode in ("mean", "std"), "mode must be 'mean' or 'std'"
    L, H, G, E = all_attentions.shape
    base_figsize = kwargs.get('figsize', (3.5, 3))
    vmin = kwargs.get('vmin')
    vmax = kwargs.get('vmax')
    figsize = (H * base_figsize[0], L * base_figsize[1])
    gif_path = kwargs.get("gif_path", os.path.join('./attention_matrix', f'attention_animation_{mode}.gif'))

    group_variable = np.array(group_variable[:G])
    groups = np.unique(group_variable)

    senders, receivers = edge_index[0].long(), edge_index[1].long()
    N = int(max(senders.max(), receivers.max()).item()) + 1

    matrices_per_group = []
    for group_val in groups:
        mask = (group_variable == group_val)
        indices = torch.tensor(np.where(mask)[0], dtype=torch.long, device=all_attentions.device)
        attn_group = all_attentions[:, :, indices, :]  # [L, H, Gg, E]
        vec = attn_group.mean(dim=2) if mode == "mean" else attn_group.std(dim=2)

        mat = torch.zeros((L, H, N, N), dtype=vec.dtype)
        for l in range(L):
            for h in range(H):
                mat[l, h].index_put_((senders, receivers), vec[l, h], accumulate=True)
        matrices_per_group.append(mat.cpu().numpy())

    fig, axes = plt.subplots(L, H, figsize=figsize, squeeze=False)
    ims = []
    for l in range(L):
        row = []
        for h in range(H):
            ax = axes[l][h]
            im = ax.imshow(matrices_per_group[0][l][h], cmap='rocket_r', vmin=vmin, vmax=vmax)
            ax.set_title(f"Layer {l} - Head {h}", fontsize=10)
            ax.set_xlabel("Target")
            ax.set_ylabel("Source")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            row.append(im)
        ims.append(row)

    def update(i):
        fig.suptitle(f"{mode.capitalize()} Attention - {group_name} = {groups[i]}", fontsize=14)
        for l in range(L):
            for h in range(H):
                ims[l][h].set_data(matrices_per_group[i][l][h])
        return sum(ims, [])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    anim = FuncAnimation(fig, update, frames=len(groups), interval=interval, blit=False)
    
    if save:
        anim.save(gif_path, writer='pillow', dpi=200)
        print(f"GIF saved at: {os.path.abspath(gif_path)}")

    plt.close(fig)
    return HTML(anim.to_jshtml())

def pca_analysis_attention(
    all_attentions: torch.Tensor,
    edge_index: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
    n_components: int = 10
) -> None:
    """
    Perform Principal Component Analysis (PCA) on a selected attention head across graphs.

    This function visualizes:
    - The explained variance of the principal components (PCs),
    - A 2D projection of the data on the first two PCs,
    - Heatmaps of the top principal components reshaped into attention matrices.

    Parameters
    ----------
    all_attentions : torch.Tensor of shape [L, H, G, E] or [G, E]
        Attention values. Can be the full tensor from a model or already flattened for a given head.
    edge_index : torch.Tensor of shape [2, E] or [E, 2]
        Edge indices (source and target nodes).
    layer_idx : int, optional (default=0)
        Index of the attention layer to analyze.
    head_idx : int, optional (default=0)
        Index of the attention head to analyze.
    n_components : int, optional (default=10)
        Number of principal components to extract and visualize.

    Returns
    -------
    None
        Displays plots directly.
    """
    # Normalize edge_index and compact node ids to avoid trailing zero rows/cols
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index)
    ei = edge_index.detach().long()
    if ei.dim() != 2 or 2 not in ei.shape:
        raise ValueError("edge_index must have shape [2, E] or [E, 2].")
    if ei.shape[0] == 2:
        src_t, tgt_t = ei[0], ei[1]
    else:  # [E, 2]
        src_t, tgt_t = ei[:, 0], ei[:, 1]
    src_t = src_t.long()
    tgt_t = tgt_t.long()

    unique_nodes = torch.unique(torch.cat([src_t, tgt_t])).sort()[0]
    n_used = int(unique_nodes.numel())
    id_map = torch.full((int(unique_nodes.max().item()) + 1,), -1, dtype=torch.long)
    id_map[unique_nodes] = torch.arange(n_used, dtype=torch.long)
    src_c = id_map[src_t].cpu().numpy()
    tgt_c = id_map[tgt_t].cpu().numpy()

    # Prepare attention matrix [G, E]
    if isinstance(all_attentions, torch.Tensor):
        if all_attentions.dim() == 4:
            # [L, H, G, E] -> pick layer/head -> [G, E]
            att_flat = all_attentions[layer_idx, head_idx].detach().cpu().numpy()
        elif all_attentions.dim() == 2:
            att_flat = all_attentions.detach().cpu().numpy()
        else:
            raise ValueError("all_attentions must have shape [L, H, G, E] or [G, E].")
    else:
        att_flat = np.asarray(all_attentions)

    # Robust n_components
    n_components_eff = max(1, min(n_components, att_flat.shape[0], att_flat.shape[1]))
    pca = PCA(n_components=n_components_eff)
    X_pca = pca.fit_transform(att_flat)
    explained_variance = pca.explained_variance_ratio_
    components = pca.components_  # [n_components_eff, E]

    # Explained variance plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, n_components_eff + 1), explained_variance, marker='o')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance")
    plt.title("Energy of each principal component")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2D PCA projection
    time_labels = np.linspace(0, 1, X_pca.shape[0])
    hsv_colors = time_labels
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hsv_colors, cmap='hsv', s=1)
    plt.title(f"PCA of Attention vectors\n(Layer {layer_idx}, Head {head_idx})")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    cbar = plt.colorbar(scatter, label="DoY")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['1 Jan', '1 Apr', '1 Jul', '1 Oct', '31 Dec'])
    plt.grid(True)
    plt.show()

    # Heatmaps of components projected back to adjacency (using compacted ids)
    fig, axes = plt.subplots(1, n_components_eff, figsize=(4 * n_components_eff, 4))
    if n_components_eff == 1:
        axes = [axes]

    E_from_edges = src_c.shape[0]
    for idx, comp in enumerate(components):
        if comp.size == n_used * n_used:
            mat = comp.reshape(n_used, n_used)
        elif comp.size == E_from_edges:
            mat = np.zeros((n_used, n_used), dtype=float)
            mat[src_c, tgt_c] = comp
        else:
            mat = np.zeros((n_used, n_used), dtype=float)

        sns.heatmap(
            mat,
            cmap="rocket_r",
            ax=axes[idx],
            cbar=True
        )
        axes[idx].set_title(f"PC{idx+1}")

    plt.suptitle(f"PCA of Attention matrices (Layer {layer_idx}, Head {head_idx})", fontsize=16)
    plt.tight_layout()
    plt.show()

def umap_analysis_attention(
    all_attentions: torch.Tensor,
    edge_index: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2
) -> None:
    """
    Perform UMAP-based dimensionality reduction on a selected attention head across graphs.

    This function visualizes:
    - A low-dimensional embedding of attention vectors using UMAP,
    - A 2D scatter plot colored cyclically to reflect graph ordering (e.g., temporal).

    Parameters
    ----------
    all_attentions : torch.Tensor of shape [L, H, G, E] or [G, E]
        Attention weights either for the entire model or already extracted for a given head.
    edge_index : torch.Tensor of shape [2, E]
        Edge indices (source and target nodes), required to infer node count if needed.
    layer_idx : int, optional (default=0)
        Index of the attention layer to analyze.
    head_idx : int, optional (default=0)
        Index of the attention head to analyze.
    n_neighbors : int, optional (default=15)
        Number of neighbors for the UMAP algorithm (controls local/global structure).
    min_dist : float, optional (default=0.1)
        Minimum distance between embedded points (controls tightness of clusters).
    n_components : int, optional (default=2)
        Number of output dimensions (typically 2 for visualization).

    Returns
    -------
    None
        Displays a UMAP projection plot.
    """
    if len(all_attentions.shape) == 4:
        L, H, G, E = all_attentions.shape
        src, tgt = edge_index
        att_flat = all_attentions[layer_idx, head_idx, :, :]  # [G, E]
    elif len(all_attentions.shape) == 2:
        att_flat = all_attentions

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='euclidean'
    )
    X_umap = reducer.fit_transform(att_flat)  # [G, n_components]

    time_labels = np.linspace(0, 1, X_umap.shape[0])
    hsv_colors = time_labels  
    plt.figure(figsize=(6, 6))
    if n_components == 2:
        scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=hsv_colors, cmap='hsv', s=1)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"UMAP Projection (Layer {layer_idx}, Head {head_idx})")
        cbar = plt.colorbar(scatter, label="DoY")
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.set_ticklabels(['1 Jan', '1 Apr', '1 Jul', '1 Oct', '31 Dec'])
    else:
        plt.plot(X_umap)
        plt.title(f"UMAP Projection on {n_components} dimensions")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def normalized_laplacian(A: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Compute the symmetric normalized Laplacian of an adjacency matrix.

    Parameters
    ----------
    A : torch.Tensor of shape [N, N]
        Adjacency matrix of the graph (must be square and 2D).
    eps : float, optional
        Small value added to avoid division by zero.

    Returns
    -------
    L : torch.Tensor of shape [N, N]
        Symmetric normalized Laplacian matrix: L = I - D^{-1/2} A D^{-1/2}.
    """
    if A.ndim != 2:
        raise ValueError(f"Expected 2D square matrix, got shape {A.shape}")
    deg = torch.sum(A, dim=1)
    deg_sqrt_inv = torch.zeros_like(deg)
    nonzero = deg > 0
    deg_sqrt_inv[nonzero] = 1.0 / torch.sqrt(deg[nonzero] + eps)
    D_inv_sqrt = torch.diag(deg_sqrt_inv)
    L = torch.eye(A.shape[0], device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
    return L

def plot_spectral_gap(L: torch.Tensor, max_k: int) -> tuple[np.ndarray, int]:
    """
    Plot the first `max_k` eigenvalues of a Laplacian and estimate the optimal number of clusters via the spectral gap.

    Parameters
    ----------
    L : torch.Tensor of shape [N, N]
        Laplacian matrix.
    max_k : int
        Number of smallest eigenvalues to consider.

    Returns
    -------
    eigvals : np.ndarray
        First `max_k` eigenvalues of the Laplacian.
    optimal_k : int
        Estimated number of clusters based on the largest spectral gap (elbow method).
    """
    eigvals,_ = torch.linalg.eigh(L)
    eigvals = eigvals[:max_k].numpy()
    gaps = eigvals[1:] - eigvals[:-1]
    optimal_k = int(gaps.argsort()[-2]) + 1
    plt.figure(figsize=(6, 4))
    plt.plot(range(0, max_k), eigvals, marker='o', label="Valeurs propres")
    plt.axvline(optimal_k, color='red', linestyle='--', label=f"Coude (k={optimal_k})")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title("Elbow Method")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return eigvals, optimal_k

def spectral_embedding(L: torch.Tensor, k: int, plot: bool = False) -> torch.Tensor:
    """
    Compute the spectral embedding (first `k` eigenvectors of the Laplacian).

    Parameters
    ----------
    L : torch.Tensor of shape [N, N]
        Laplacian matrix.
    k : int
        Number of leading eigenvectors to return.
    plot : bool, optional
        If True, plots the spectrum of eigenvalues.

    Returns
    -------
    embedding : torch.Tensor of shape [N, k]
        Matrix of the first `k` eigenvectors.
    """
    eigvals, eigvecs = torch.linalg.eigh(L)
    if plot:
        plt.plot(eigvals)
        plt.show()
    return eigvecs[:, :k]  # (N, k)

def cosine_similarity_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarity matrix between rows of a matrix.

    Parameters
    ----------
    X : torch.Tensor of shape [N, d]
        Input feature matrix.

    Returns
    -------
    similarity : torch.Tensor of shape [N, N]
        Cosine similarity between all pairs of rows in X.
    """
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-6)
    return X_norm @ X_norm.T

def spectral_fusion(lA: list[torch.Tensor], k: int, **kwargs) -> torch.Tensor:
    """
    Perform spectral fusion by computing the average cosine similarity 
    of the spectral embeddings of several adjacency matrices.

    Parameters
    ----------
    lA : list of torch.Tensor [N, N]
        List of adjacency matrices to fuse.
    k : int
        Number of eigenvectors to use for each spectral embedding.
    **kwargs : dict
        Optional arguments passed to `spectral_embedding`.

    Returns
    -------
    A_fused : torch.Tensor of shape [N, N]
        Fused similarity matrix.
    """
    lE = []
    for A in lA:
        L = normalized_laplacian(A)
        E = spectral_embedding(L, k, **kwargs)
        lE.append(E)
    max_dim = max(E.shape[1] for E in lE)
    lE_padded = [torch.nn.functional.pad(E, (0, max_dim - E.shape[1])) for E in lE]
    E_fused = torch.mean(torch.stack(lE_padded), axis=0)
    A_fused = cosine_similarity_matrix(E_fused)
    return A_fused

def hierarchical_attention_fusion(attn_tensor: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
    """
    Fuse attention maps hierarchically across heads and layers using spectral fusion.

    Parameters
    ----------
    attn_tensor : torch.Tensor of shape [L, H, N, N]
        Attention matrices for L layers and H heads.
    k : int
        Number of eigenvectors used in spectral embeddings.
    **kwargs : dict
        Optional arguments passed to `spectral_fusion`.

    Returns
    -------
    A_final : torch.Tensor of shape [N, N]
        Final fused similarity matrix after hierarchical fusion.
    """
    L, H, N, _ = attn_tensor.shape
    fused_per_layer = []
    for l in range(L):
        heads = [attn_tensor[l, h] for h in range(H)]
        fused_layer = spectral_fusion(heads, k, **kwargs)
        fused_per_layer.append(fused_layer)
    A_final = spectral_fusion(fused_per_layer, k, **kwargs)
    return A_final

@torch.no_grad()
def attention_to_dense(all_attentions: torch.Tensor, edge_index: torch.Tensor, num_nodes: int = 12) -> torch.Tensor:
    """
    Project attention values from edge-list format to dense adjacency tensor.

    The function maps an attention tensor of shape [L, H, G, E] (layers, heads,
    graphs, edges) into a dense tensor of shape [L, H, G, N, N] by scattering
    edge values on (source, target) indices given by `edge_index`.

    :param all_attentions: Attention tensor with shape [L, H, G, E].
    :type all_attentions: torch.Tensor
    :param edge_index: Edge indices; accepted shapes are [2, E] or [E, 2].
    :type edge_index: torch.Tensor
    :param num_nodes: Number of nodes N for the output dense adjacency matrices.
    :type num_nodes: int

    :returns: Dense attention tensor of shape [L, H, G, N, N].
    :rtype: torch.Tensor
    """
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    src, tgt = edge_index[0].long(), edge_index[1].long()

    L, H, G, E = all_attentions.shape
    N = int(num_nodes)
    dense = all_attentions.new_zeros((L, H, G, N, N))
    # broadcasting: [L,H,G,1] to [L,H,G,E]
    # scatter by 2D index (src,tgt) on the last dim
    for l in range(L):
        for h in range(H):
            for g in range(G):
                dense[l, h, g, src, tgt] = all_attentions[l, h, g]
    return dense  # [L,H,G,N,N]

def pca_per_head(all_attentions: torch.Tensor,
                 edge_index: torch.Tensor,
                 num_nodes: int,
                 n_components: int = 10):
    """
    Perform PCA independently for each (layer, head) pair on dense attention matrices.

    The function converts the edge-form attention tensor [L, H, G, E] into dense
    [L, H, G, N, N], reshapes each (layer, head) block into a [G, N*N] matrix and
    runs PCA, returning per-(layer,head) results.

    :param all_attentions: Attention tensor of shape [L, H, G, E].
    :type all_attentions: torch.Tensor
    :param edge_index: Edge indices, used to project edges into NxN dense format.
    :type edge_index: torch.Tensor
    :param num_nodes: Number of nodes N used for dense matrices.
    :type num_nodes: int
    :param n_components: Maximum number of PCA components to compute per head.
    :type n_components: int

    :returns: A list of dicts, one per (layer, head), each containing:
        - "layer": int
        - "head": int
        - "explained_variance": array of explained variance ratios
        - "components": numpy array shaped [k, N, N] of principal components
        - "scores": numpy array shaped [G, k] of transformed samples
        - "mean_matrix": numpy array [N, N] the mean attention matrix across G
    :rtype: list[dict]
    """
    dense = attention_to_dense(all_attentions, edge_index, num_nodes=num_nodes)  # [L,H,G,N,N]
    L, H, G, N, _ = dense.shape
    results = []
    for l in range(L):
        for h in range(H):
            X = dense[l, h].reshape(G, N * N).cpu().numpy()  # [G, N*N]
            k = max(1, min(n_components, X.shape[0], X.shape[1]))
            pca = PCA(n_components=k)
            X_pca = pca.fit_transform(X)
            comps = pca.components_.reshape(k, N, N)  # [k,N,N]
            results.append({
                "layer": l,
                "head": h,
                "explained_variance": pca.explained_variance_ratio_,
                "components": comps,     # [k,N,N]
                "scores": X_pca,         # [G,k]
                "mean_matrix": X.mean(axis=0).reshape(N, N)
            })
    return results  # len = L*H

def pca_global_mean(all_attentions: torch.Tensor,
                    edge_index: torch.Tensor,
                    num_nodes: int,
                    n_components: int = 10):
    """
    PCA on the global mean attention matrix (averaged over layers and heads).

    :param all_attentions: Attention tensor [L, H, G, E].
    :type all_attentions: torch.Tensor
    :param edge_index: Edge indices used to form dense matrices.
    :type edge_index: torch.Tensor
    :param num_nodes: Number of nodes N.
    :type num_nodes: int
    :param n_components: Number of PCA components.
    :type n_components: int

    :returns: Dictionary containing PCA results similar to `pca_per_head`.
    :rtype: dict
    """
    dense = attention_to_dense(all_attentions, edge_index, num_nodes=num_nodes)  # [L,H,G,N,N]
    G, N = dense.shape[2], num_nodes
    X = dense.mean(dim=(0,1)).reshape(G, N * N).cpu().numpy()  # [G, N*N]
    k = max(1, min(n_components, X.shape[0], X.shape[1]))
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X)
    comps = pca.components_.reshape(k, N, N)
    return {
        "explained_variance": pca.explained_variance_ratio_,
        "components": comps,     # [k,N,N]
        "scores": X_pca,         # [G,k]
        "mean_matrix": X.mean(axis=0).reshape(N, N)
    }

def plot_explained_variance(explained, title="Explained variance", **kwargs):
    """
    Plot explained variance ratio of PCA components.

    :param explained: Iterable of explained variance ratios.
    :type explained: array-like
    :param title: Plot title.
    :type title: str
    :param kwargs: Optional plotting parameters (figsize).
    :type kwargs: dict

    :returns: None
    :rtype: None
    """
    plt.figure(figsize=kwargs.get('figsize', (6,4)))
    plt.plot(range(1, len(explained)+1), explained, marker='o')
    plt.xlabel("PC")
    plt.ylabel("Explained variance")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_components(components: np.ndarray, **kwargs):
    """
    Plot a grid of component heatmaps.

    :param components: Array of principal components with shape [k, N, N].
    :type components: numpy.ndarray
    :param kwargs: Optional arguments:
        - max_cols: maximum columns in the grid (default 5)
        - cmap: colormap (default 'rocket_r')
        - suptitle: overall figure title
        - figsize: figure size override
    :type kwargs: dict

    :returns: None
    :rtype: None
    """
    max_cols = kwargs.get('max_cols', 5)
    cmap = kwargs.get('cmap', 'rocket_r')
    suptitle = kwargs.get('suptitle', "Principal Components")
    k = components.shape[0]
    cols = min(max_cols, k)
    rows = int(np.ceil(k / cols))
    figsize = kwargs.get('figsize', (4*cols, 4*rows))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows*cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    for i in range(rows*cols):
        ax = axes[i]
        ax.axis('off')
        if i < k:
            sns.heatmap(components[i], cmap=cmap, ax=ax, cbar=True)
            ax.set_title(f"PC{i+1}")
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()
