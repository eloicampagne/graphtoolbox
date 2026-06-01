import colorcet as cc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch

def plot_losses(num_epochs, train_losses, val_losses, start_epoch=0):
    """
    Plot training and validation losses across epochs.

    Parameters
    ----------
    num_epochs : int
        Total number of training epochs (used as the upper bound of the x-axis).
    train_losses : list[float]
        List of training loss values per epoch.
    val_losses : list[float]
        List of validation loss values per epoch.
    start_epoch : int, optional
        First epoch index (non-zero when resuming from a checkpoint).

    Notes
    -----
    The function displays the loss evolution and is typically used
    to diagnose convergence and potential overfitting.

    Examples
    --------
    >>> plot_losses(100, train_losses, val_losses)
    """
    epochs = range(start_epoch, start_epoch + len(train_losses))
    plt.plot(epochs, train_losses, label='train')
    plt.plot(epochs, val_losses, label='valid')
    plt.xlabel('Number of epochs')
    plt.ylabel('RMSE (MW)')
    plt.legend()
    plt.show()

def plot_nodes(true, pred, graph_dataset, **kwargs):
    """
    Plot true vs. predicted time series for all nodes in a graph.

    Parameters
    ----------
    true : torch.Tensor
        Ground-truth target values with shape ``[num_nodes, T]``.
    pred : torch.Tensor
        Model predictions with shape ``[num_nodes, T]``.
    graph_dataset : GraphDataset
        Dataset providing node metadata (names, coordinates, etc.).
    nrows : int, optional
        Number of rows in subplot grid (default: 3).
    ncols : int, optional
        Number of columns in subplot grid (default: 4).
    figsize : tuple, optional
        Figure size in inches (default: (7*nrows, 3*ncols)).

    Notes
    -----
    - Each subplot corresponds to a node’s time series.
    - Predictions and true values are plotted over time.

    Examples
    --------
    >>> plot_nodes(true, pred, dataset_val, nrows=2, ncols=3)
    """
    nrows = kwargs.get('nrows', 3)
    ncols = kwargs.get('ncols', 4)
    figsize = kwargs.get('figsize', (7*nrows, 3*ncols))
    true = true.detach().cpu()
    pred = pred.detach().cpu()
    T = pred.shape[1]
    dates_all = pd.to_datetime(graph_dataset.dataframe.date.unique())
    dates = dates_all[-T:]
    palette = sns.color_palette(cc.glasbey, n_colors=graph_dataset.num_nodes)
    _, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = np.array(axs).flatten()
    for n in range(graph_dataset.num_nodes):
        ax = axs[n]
        try:
            ax.set_title(graph_dataset.nodes[n].replace('_', ' '))
        except:
            ax.set_title(graph_dataset.nodes[n])
        ax.plot(dates, pred[n].numpy(), label='predicted', color=palette[n])
        ax.plot(dates, true[n].numpy(), label='true', color='black', alpha=.4)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper right')
    for k in range(graph_dataset.num_nodes, len(axs)):
        axs[k].set_visible(False)
    plt.tight_layout()
    plt.show()    

def plot_graph_map(edge_index: torch.Tensor, edge_weight: torch.Tensor, df_pos: pd.DataFrame, ax):
    """
    Plot a geographic graph with nodes and weighted edges using Basemap.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index tensor of shape ``[2, E]``.
    edge_weight : torch.Tensor
        Edge weight tensor of shape ``[E]``.
    df_pos : pandas.DataFrame
        DataFrame with node coordinates (`LATITUDE`, `LONGITUDE`).
    ax : matplotlib.axes.Axes
        Axis on which to draw the map.

    Notes
    -----
    - Nodes are placed using geographic coordinates.
    - Edge color intensity corresponds to connection weight.
    - Uses `Basemap` for cartographic rendering.

    Examples
    --------
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> plot_graph_map(edge_index, edge_weight, df_pos, ax)
    """
    G = nx.Graph()
    for region, (lat, lon) in enumerate(zip(df_pos.LATITUDE, df_pos.LONGITUDE)):
        G.add_node(region, pos=(lon, lat))
    for u, v, weight in zip(edge_index[0], edge_index[1], edge_weight):
        u, v, weight = u.item(), v.item(), weight.item()
        if weight > 0:
            G.add_edge(u, v, weight=weight)
    pos = nx.get_node_attributes(G, 'pos')
    weights = nx.get_edge_attributes(G, 'weight')
    m = Basemap(projection='merc', llcrnrlat=40, urcrnrlat=52, llcrnrlon=-5, urcrnrlon=10, resolution='i', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='gray')
    pos_basemap = {node: m(lat, lon) for node, (lat, lon) in pos.items()}
    nx.draw_networkx_nodes(G, pos_basemap, node_size=500, node_color='blue', alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos_basemap, font_size=12, font_color='white', ax=ax)
    edges = G.edges()
    colors = [weights[edge] for edge in edges]
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = sns.cm.rocket_r
    nx.draw_networkx_edges(G, 
                        pos_basemap, 
                        edge_color=colors, 
                        arrows=True, 
                        connectionstyle='arc3,rad=0.2',
                        edge_cmap=cmap, 
                        edge_vmin=0, 
                        edge_vmax=1, 
                        width=2, 
                        ax=ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Edge Weight')
    
def plot_node_errors_map(node_errors: dict, nodes: list, df_pos: pd.DataFrame, **kwargs):
    """
    Display a French map with one panel per method, each node coloured by its error.

    Parameters
    ----------
    node_errors : dict[str, array-like]
        Mapping from method label to a 1-D array of length ``num_nodes`` containing
        the per-node error metric (e.g. MAPE in %).
    nodes : list[str]
        Ordered list of node names matching the rows of ``node_errors`` arrays.
        Names are matched against the ``REGION`` column of ``df_pos``.
    df_pos : pandas.DataFrame
        Must contain columns ``REGION``, ``LATITUDE``, and ``LONGITUDE``.
    metric_label : str, optional
        Colorbar label (default: ``'MAPE (%)'``).
    cmap : str or Colormap, optional
        Matplotlib colormap (default: ``'RdYlGn_r'``).
    figsize : tuple, optional
        Figure size per panel (default: ``(7, 6)``).
    vmin, vmax : float, optional
        Shared color-scale limits.  Defaults to the 5th/95th percentile
        across all methods.
    node_size : int, optional
        Scatter marker size (default: 350).

    Examples
    --------
    >>> errors = {'GATConv': mape_per_node, 'Opera BU': opera_mape}
    >>> plot_node_errors_map(errors, graph_dataset_train.nodes, data.df_pos)
    """
    metric_label = kwargs.get('metric_label', 'MAPE (%)')
    cmap         = plt.get_cmap(kwargs.get('cmap', 'RdYlGn_r'))
    node_size    = kwargs.get('node_size', 350)
    pw, ph       = kwargs.get('figsize', (7, 6))

    n_panels = len(node_errors)
    all_vals = np.concatenate([np.asarray(v) for v in node_errors.values()])
    vmin = kwargs.get('vmin', float(np.percentile(all_vals, 5)))
    vmax = kwargs.get('vmax', float(np.percentile(all_vals, 95)))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Build a region → (lat, lon) lookup
    pos_lookup = {row['REGION']: (row['LATITUDE'], row['LONGITUDE'])
                  for _, row in df_pos.iterrows()}

    fig, axes = plt.subplots(1, n_panels, figsize=(pw * n_panels, ph))
    if n_panels == 1:
        axes = [axes]

    for ax, (title, errors) in zip(axes, node_errors.items()):
        errors = np.asarray(errors)
        m = Basemap(
            projection='merc',
            llcrnrlat=41.0, urcrnrlat=51.5,
            llcrnrlon=-5.5, urcrnrlon=9.5,
            resolution='i', ax=ax
        )
        m.drawcoastlines(linewidth=0.8, color='#444')
        m.drawcountries(linewidth=0.8, color='#444')
        m.drawmapboundary(fill_color='#daeef7')
        m.fillcontinents(color='#f2f2ee', lake_color='#daeef7')

        for node_idx, node_name in enumerate(nodes):
            key = node_name  # try exact match first
            if key not in pos_lookup:
                # Normalise underscores / spaces
                key = next((k for k in pos_lookup if k.replace(' ', '_') == node_name.replace(' ', '_')), None)
            if key is None:
                continue
            lat, lon = pos_lookup[key]
            x, y = m(lon, lat)
            err = float(errors[node_idx])
            color = cmap(norm(err))
            m.scatter(x, y, s=node_size, c=[color], zorder=5,
                      alpha=0.88, edgecolors='#222', linewidths=0.6)
            label = node_name.replace('_', ' ').replace('Cote d Azur', 'Côte d\'Azur')
            ax.annotate(
                f'{label}\n{err:.2f}%',
                xy=(x, y), ha='center', va='bottom', fontsize=6,
                zorder=6, xytext=(0, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, lw=0),
            )
        ax.set_title(title, fontsize=12, fontweight='bold', pad=6)

    # Reserve a fixed right margin for the colorbar so it never overlaps a panel.
    fig.subplots_adjust(right=0.88, wspace=0.06)
    cbar_ax = fig.add_axes([0.905, 0.18, 0.018, 0.64])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(metric_label, fontsize=10)
    plt.show()

def plot_all_graph_maps(graph_list, edge_index, df_pos, **kwargs):
    """
    Plot a grid of graph visualizations with varying edge weights.

    Parameters
    ----------
    graph_list : list[torch.Tensor]
        List of edge weight tensors to visualize.
    edge_index : torch.Tensor
        Edge index shared across graphs.
    df_pos : pandas.DataFrame
        Node coordinates for plotting.
    nrows : int, optional
        Number of rows in subplot grid (default: 2).
    ncols : int, optional
        Number of columns in subplot grid (computed automatically).
    figsize : tuple, optional
        Overall figure size.

    Examples
    --------
    >>> plot_all_graph_maps([W1, W2, W3], edge_index, df_pos, nrows=2)
    """
    nrows = kwargs.get('nrows', 2)
    num_graphs = len(graph_list)
    ncols = int(np.ceil(num_graphs / nrows))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 8 * nrows))
    if nrows == 1:
        axs = axs.reshape(-1)
    axs = axs.flatten()
    for i, edge_weight in enumerate(graph_list):
        plot_graph_map(edge_index, edge_weight, df_pos, axs[i])
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()