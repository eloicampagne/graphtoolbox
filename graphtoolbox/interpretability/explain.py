from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import os
import pandas as pd
from scipy.interpolate import UnivariateSpline
import seaborn as sns
import torch
from typing import Optional, Any, Dict, List, Tuple
import warnings

from graphtoolbox.data.dataset import GraphDataset


def _require_basemap():
    """Import Basemap lazily so the package installs and imports without it.

    Basemap only backs the optional explanation-graph maps and is a deprecated
    dependency without wheels for every Python version. Install it on demand
    with ``pip install GraphToolbox[maps]``.
    """
    try:
        from mpl_toolkits.basemap import Basemap
    except ImportError as exc:
        raise ImportError(
            "This figure requires Basemap, an optional dependency. Install it "
            "with `pip install GraphToolbox[maps]` (or `pip install basemap`)."
        ) from exc
    return Basemap

@dataclass
class VisualizationConfig:
    """
    Configuration for graph explanation visualization.

    Fields
    -------
    name : str
        Run/dataset name used in figure titles and output paths.
    output_root : str
        Root directory where figures are saved.

    basemap : dict | None
        Explicit Basemap kwargs (projection, bounds). If None, inferred.
    positions : dict[int, tuple[float, float]] | None
        Explicit node positions {node_id: (lon, lat)}.
    pos_df : pandas.DataFrame | None
        DataFrame holding lon/lat columns (and optionally node id).
    lon_col : str
        Longitude column name in pos_df.
    lat_col : str
        Latitude column name in pos_df.
    node_id_col : str | None
        Column in pos_df specifying node IDs; defaults to row order.

    grouping : dict
        Time grouping specification (mode, ndays, labels, indices).
    start_date_key : str
        Key in data_kwargs for the test start date.
    date_freq : str
        Pandas frequency string to expand dates.

    map_projection : str
        Basemap projection (default 'merc').
    map_resolution : str
        Basemap resolution code.
    draw_coastlines : bool
        Draw coastlines if True.
    draw_countries : bool
        Draw country borders if True.
    fillcontinents_color : str
        Color used to fill continents.
    mapboundary_color : str
        Map boundary fill color.

    show_nodes : bool
        Display nodes.
    show_labels : bool
        Display node labels.
    node_size : int
        Node marker size.
    node_color : str
        Node color.
    node_alpha : float
        Node transparency.
    label_fontsize : int
        Node label font size.
    label_color : str
        Node label font color.

    edge_cmap : str
        Colormap name for importance (non-std modes).
    edge_cmap_std : str
        Colormap name when vis_mode == 'std'.
    edge_width_min : float
        Minimum edge line width.
    edge_width_max : float
        Maximum edge line width.
    edge_alpha : float
        Edge transparency.
    connectionstyle : str
        Matplotlib connection style for edges.

    edge_arrows : bool
        Draw directed edge arrows if True.
    arrowstyle : str
        Arrow style passed to NetworkX.
    arrowsize : int
        Arrow size.
    arrow_max_edges : int
        Max edges allowed for arrow rendering (performance guard).

    normalize_with_edge_weight : bool
        Modulate context importance by normalized edge weights.

    save_dpi : int
        DPI for saved figures.
    file_ext : str
        Output file extension.
    subdir : str
        Subdirectory under output_root.

    fontsize : int
        Global title font size.
    labelsize : int
        Axis tick label size.
    """
    name: str = "default"                       
    output_root: str = "interpretability"       

    # Map and positions
    basemap: Optional[Dict[str, Any]] = None    
    positions: Optional[Dict[int, Tuple[float, float]]] = None  # {node_id: (lon, lat)}
    pos_df: Optional[pd.DataFrame] = None
    lon_col: str = "LONGITUDE"
    lat_col: str = "LATITUDE"
    node_id_col: Optional[str] = None

    # Time grouping config
    grouping: Dict[str, Any] = field(default_factory=lambda: dict(mode="all", ndays=3, labels=None, indices=None))
    start_date_key: str = "day_inf_test"        # key in data_kwargs to reconstruct dates
    date_freq: str = "D"                        # pandas frequency string (e.g. 'D', 'H')

    # Basemap defaults (used if not provided in basemap)
    map_projection: str = "merc"
    map_resolution: str = "i"
    draw_coastlines: bool = True
    draw_countries: bool = True
    fillcontinents_color: str = "gray"
    mapboundary_color: str = "white"

    # Drawing options
    show_nodes: bool = True
    show_labels: bool = True
    node_size: int = 500
    node_color: str = "blue"
    node_alpha: float = 0.6
    label_fontsize: int = 10
    label_color: str = "white"

    edge_cmap: str = "rocket_r"                 # colormap when vis_mode != "std"
    edge_cmap_std: str = "mako"                 # colormap when vis_mode == "std"
    edge_width_min: float = 1.0
    edge_width_max: float = 5.0
    edge_alpha: float = 1.0
    connectionstyle: str = "arc3,rad=0.1"
    
    # Edge arrows (use FancyArrowPatches). Warning: can be slow on large graphs.
    edge_arrows: bool = False
    arrowstyle: str = "-|>"
    arrowsize: int = 10
    arrow_max_edges: int = 500

    # Edge importance weighting when vis_mode == "context"
    normalize_with_edge_weight: bool = True

    # Output options
    save_dpi: int = 150
    file_ext: str = "pdf"
    subdir: str = "explanation_graph"

    # Font sizes
    fontsize: int = 16
    labelsize: int = 12

def _month_indices(month_name, dates):
    """
    Return temporal indices for a given month name.

    :param month_name: Calendar month (e.g., 'January') or 'ALL'.
    :type month_name: str
    :param dates: Datetime index or sequence with month_name() available.
    :type dates: pandas.DatetimeIndex | list | None
    :returns: Indices array for the month, or slice(None) if 'ALL' or dates is None.
    :rtype: numpy.ndarray | slice
    """
    if month_name == "ALL" or dates is None:
        return slice(None)
    return np.where(dates.month_name() == month_name)[0]


def _pick_temporal_mask(exp_dict, vis_mode, idxs):
    """
    Aggregate temporal edge masks according to visualization mode.

    If per-time edge masks exist, aggregate over provided indices:
    - vis_mode in {'model','context'}: mean over time
    - vis_mode == 'std': standard deviation over time

    Falls back to global masks stored in exp_dict.

    :param exp_dict: Explanation dictionary containing keys 'mean', optionally 'std', 'edge_masks'.
    :type exp_dict: dict
    :param vis_mode: Visualization mode ('model', 'context', 'std').
    :type vis_mode: str
    :param idxs: Temporal indices (array) or slice.
    :type idxs: numpy.ndarray | slice
    :returns: Aggregated edge importance array.
    :rtype: numpy.ndarray
    :raises KeyError: If required keys are missing.
    """
    em_time = exp_dict.get("edge_masks", None)
    if em_time is None:
        em_time = globals().get("edge_masks", None)
    if em_time is not None and not isinstance(idxs, slice):
        if vis_mode in ["model", "context"]:
            return em_time[idxs].mean(dim=0).detach().cpu().numpy()
        else:
            return em_time[idxs].std(dim=0).detach().cpu().numpy()
    # fallback: global or per-explanation
    if vis_mode in ["model", "context"]:
        return exp_dict["mean"].edge_mask.detach().cpu().numpy()
    else:
        return exp_dict["std"].detach().cpu().numpy()
    
def _fill_missing_positions(pos_map: Dict[int, Tuple[float, float]],
                            n_nodes: int,
                            map_kwargs: Dict[str, Any],
                            seed: int = 42) -> Dict[int, Tuple[float, float]]:
    """
    Fill missing node positions by uniform sampling within map extent.

    Sampling is deterministic via RandomState(seed).

    :param pos_map: Existing positions {node_id: (lon, lat)}.
    :type pos_map: dict[int, tuple[float, float]]
    :param n_nodes: Total number of nodes expected.
    :type n_nodes: int
    :param map_kwargs: Basemap kwargs containing llcrnrlon/urcrnrlon/llcrnrlat/urcrnrlat.
    :type map_kwargs: dict
    :param seed: RNG seed for reproducibility.
    :type seed: int
    :returns: Completed position mapping with all nodes.
    :rtype: dict[int, tuple[float, float]]
    """
    have = set(int(k) for k in pos_map.keys())
    missing = [i for i in range(n_nodes) if i not in have]
    if not missing:
        return pos_map
    lon_lo = float(map_kwargs.get('llcrnrlon', -5))
    lon_hi = float(map_kwargs.get('urcrnrlon', 10))
    lat_lo = float(map_kwargs.get('llcrnrlat', 40))
    lat_hi = float(map_kwargs.get('urcrnrlat', 52))
    rng = np.random.RandomState(seed)
    updated = dict(pos_map)
    for i in missing:
        lon_i = float(rng.uniform(lon_lo, lon_hi))
        lat_i = float(rng.uniform(lat_lo, lat_hi))
        updated[int(i)] = (lon_i, lat_i)
    return updated

def plot_explanation_graph(
    all_explanations: dict,
    graph_dataset_test: GraphDataset,
    data_kwargs: dict,
    dataset: str = "default",                      # kept for backward compatibility: only used as default cfg.name
    vis_mode: str = "std",
    months_to_plot: list[str] = ["ALL"],           # backward compat; overridden by cfg.grouping if provided
    edge_keep_ratio: float = 0.10,
    df_pos: pd.DataFrame | None = None,            # backward compat; prefer cfg.positions or cfg.pos_df
    viz_cfg: VisualizationConfig | dict | None = None
):
    """
    Visualize explanation graphs on a map for selected period(s).

    :param all_explanations: Mapping from model name to a dict with explanation
        artifacts. Expected keys include:
        - "mean": node or edge importance array/tensor
        - "std": optional uncertainty array/tensor (same shape as "mean")
        - "edge_masks": optional edge-level importance array/tensor
    :type all_explanations: dict[str, dict]

    :param graph_dataset_test: Test graphs (PyG Data objects). Must contain at
        least one element and be aligned with the explanation arrays.
    :type graph_dataset_test: GraphDataset | list[torch_geometric.data.Data]

    :param data_kwargs: Auxiliary information for plotting. Must contain the key
        "day_inf_test" (start date of the test set); additional keys are accepted.
    :type data_kwargs: dict

    :param dataset: Backward compatibility only. If ``viz_cfg`` is provided,
        its ``name`` is used instead.
    :type dataset: str

    :param vis_mode: Visualization mode: "model", "context", or "std".
        - "model": display model-driven importances
        - "context": display context-driven importances if available
        - "std": display uncertainty (std) when provided
    :type vis_mode: str

    :param months_to_plot: Backward compatibility. When ``viz_cfg.grouping`` is
        provided, it takes precedence over this argument.
    :type months_to_plot: list[str]

    :param edge_keep_ratio: Fraction of highest-importance edges to keep in the
        rendering (e.g., 0.10 keeps top 10% edges).
    :type edge_keep_ratio: float

    :param df_pos: Node positions as a DataFrame (backward compatibility).
        Prefer passing positions via ``viz_cfg.positions`` or ``viz_cfg.pos_df``.
    :type df_pos: pandas.DataFrame | None

    :param viz_cfg: Visualization configuration. May be a dataclass or a dict.
        Typical fields include:
        - name: run name to use in titles/outputs
        - positions / pos_df: node positions
        - grouping: list of period keys to render (overrides months_to_plot)
        - map kwargs: styling, CRS, background, etc.
    :type viz_cfg: VisualizationConfig | dict | None

    :returns: A Matplotlib Figure, or a dictionary of Figures keyed by period/group,
        depending on the requested grouping. May return None when running
        non-interactive workflows that save plots to disk.
    :rtype: matplotlib.figure.Figure | dict[str, matplotlib.figure.Figure] | None

    :raises KeyError: If required keys (e.g., "day_inf_test") are missing in ``data_kwargs``.
    :raises ValueError: If inputs are inconsistent (e.g., empty dataset, invalid ``edge_keep_ratio``).
    :raises RuntimeError: If a map background cannot be created or positions are unresolved.

    Notes
    -----
    - ``viz_cfg.grouping`` supersedes ``months_to_plot`` when provided.
    - Edge filtering is applied independently per plot based on importance.
    - This function does not compute explanations; it only renders precomputed
      artifacts contained in ``all_explanations``.
    """
    # --- Build configuration ---
    if viz_cfg is None:
        cfg = VisualizationConfig(name=str(dataset))
        # Legacy: build positions from df_villes if provided
        if df_pos is not None and all(c in df_pos.columns for c in ["LATITUDE", "LONGITUDE"]):
            cfg.pos_df = df_pos
            cfg.lon_col, cfg.lat_col = "LONGITUDE", "LATITUDE"
    elif isinstance(viz_cfg, dict):
        cfg = VisualizationConfig(**viz_cfg)
    else:
        cfg = viz_cfg

    # --- Reconstruct dates from temporal edge_masks ---
    dates = None
    try:
        t_len = None
        for _m, _exp in all_explanations.items():
            if _exp.get("edge_masks", None) is not None:
                t_len = int(_exp["edge_masks"].shape[0])
                break
        if t_len is None:
            em_glob = globals().get("edge_masks", None)
            if em_glob is not None:
                try:
                    t_len = int(em_glob.shape[0])
                except Exception:
                    t_len = None
        if t_len is not None:
            start = pd.to_datetime(data_kwargs["day_inf_test"])
            dates = pd.date_range(start=start, periods=t_len, freq="D")
    except Exception:
        dates = None

    # --- Loop over models ---
    for model_name, exp_dict in all_explanations.items():
        explanation = exp_dict["mean"]
        edge_index = explanation.edge_index.cpu().numpy()

        G_base = nx.Graph()
        pos_map = _build_positions(cfg, graph_dataset_test)
        for nid, (lon, lat) in pos_map.items():
            G_base.add_node(int(nid), pos=(float(lon), float(lat)))

        pos_all = nx.get_node_attributes(G_base, 'pos')
        map_kwargs = _infer_map_extent(pos_all, cfg)

        data_graph = graph_dataset_test[0]
        n_nodes = _infer_num_nodes(data_graph, explanation, edge_index)

        pos_map = _fill_missing_positions(pos_map, n_nodes, map_kwargs)
        G_base.clear()
        for nid, (lon, lat) in pos_map.items():
            G_base.add_node(int(nid), pos=(float(lon), float(lat)))

        # -----------------------------
        # 4. Normalize edge_weight if present
        # -----------------------------
        base_ei = data_graph.edge_index
        base_ew = getattr(data_graph, 'edge_weight', None)
        ew_map, ew_min, ew_max, ew_den = _prepare_edge_weight_map(base_ei, base_ew)

        # -----------------------------
        # 5. Determine temporal panels (groups)
        # -----------------------------
        panel_labels, panel_indices, nrows, ncols, tag = _define_panels(cfg, dates, months_to_plot)

        # -----------------------------
        # 6. Prepare data to plot
        # -----------------------------
        vmin_global, vmax_global = float('inf'), float('-inf')
        panel_data = []
        for lbl, idxs in zip(panel_labels, panel_indices):
            edge_mask = _pick_temporal_mask(exp_dict, vis_mode, idxs)
            edges_draw, weights_draw = _select_top_edges(
                edge_index, edge_mask, vis_mode,
                base_ew if cfg.normalize_with_edge_weight else None,
                ew_map, ew_min, ew_den, edge_keep_ratio
            )
            if weights_draw:
                vmin_global = min(vmin_global, min(weights_draw))
                vmax_global = max(vmax_global, max(weights_draw))
            panel_data.append((lbl, edges_draw, weights_draw))

        if not np.isfinite(vmin_global):
            vmin_global, vmax_global = 0.0, 1.0

        _draw_graph_panels(
            model_name, cfg.name, cfg, vis_mode,
            panel_data, G_base, map_kwargs,
            vmin_global, vmax_global, nrows, ncols, tag,
            output_root=cfg.output_root
        )

def _build_positions(cfg: VisualizationConfig, graph_dataset_test: list) -> Dict[int, Tuple[float, float]]:
    """
    Visualize explanation graphs on a map for selected period(s).

    :param all_explanations: Mapping from model name to a dict with explanation
        artifacts. Expected keys include:
        - "mean": node or edge importance array/tensor
        - "std": optional uncertainty array/tensor (same shape as "mean")
        - "edge_masks": optional edge-level importance array/tensor
    :type all_explanations: dict[str, dict]

    :param graph_dataset_test: Test graphs (PyG Data objects). Must contain at
        least one element and be aligned with the explanation arrays.
    :type graph_dataset_test: GraphDataset | list[torch_geometric.data.Data]

    :param data_kwargs: Auxiliary information for plotting. Must contain the key
        "day_inf_test" (start date of the test set); additional keys are accepted.
    :type data_kwargs: dict

    :param dataset: Backward compatibility only. If ``viz_cfg`` is provided,
        its ``name`` is used instead.
    :type dataset: str

    :param vis_mode: Visualization mode: "model", "context", or "std".
        - "model": display model-driven importances
        - "context": display context-driven importances if available
        - "std": display uncertainty (std) when provided
    :type vis_mode: str

    :param months_to_plot: Backward compatibility. When ``viz_cfg.grouping`` is
        provided, it takes precedence over this argument.
    :type months_to_plot: list[str]

    :param edge_keep_ratio: Fraction of highest-importance edges to keep in the
        rendering (e.g., 0.10 keeps top 10% edges).
    :type edge_keep_ratio: float

    :param df_pos: Node positions as a DataFrame (backward compatibility).
        Prefer passing positions via ``viz_cfg.positions`` or ``viz_cfg.pos_df``.
    :type df_pos: pandas.DataFrame | None

    :param viz_cfg: Visualization configuration. May be a dataclass or a dict.
        Typical fields include:
        - name: run name to use in titles/outputs
        - positions / pos_df: node positions
        - grouping: list of period keys to render (overrides months_to_plot)
        - map kwargs: styling, CRS, background, etc.
    :type viz_cfg: VisualizationConfig | dict | None

    :returns: A Matplotlib Figure, or a dictionary of Figures keyed by period/group,
        depending on the requested grouping. May return None when running
        non-interactive workflows that save plots to disk.
    :rtype: matplotlib.figure.Figure | dict[str, matplotlib.figure.Figure] | None

    :raises KeyError: If required keys (e.g., "day_inf_test") are missing in ``data_kwargs``.
    :raises ValueError: If inputs are inconsistent (e.g., empty dataset, invalid ``edge_keep_ratio``).
    :raises RuntimeError: If a map background cannot be created or positions are unresolved.

    Notes
    -----
    - ``viz_cfg.grouping`` supersedes ``months_to_plot`` when provided.
    - Edge filtering is applied independently per plot based on importance.
    - This function does not compute explanations; it only renders precomputed
      artifacts contained in ``all_explanations``.
    """
    # 1) Explicit mapping
    if cfg.positions is not None:
        return {int(k): (float(v[0]), float(v[1])) for k, v in cfg.positions.items()}

    # 2) DataFrame with lon/lat columns
    if cfg.pos_df is not None and cfg.lon_col in cfg.pos_df.columns and cfg.lat_col in cfg.pos_df.columns:
        if cfg.node_id_col and cfg.node_id_col in cfg.pos_df.columns:
            ids = cfg.pos_df[cfg.node_id_col].astype(int).tolist()
        else:
            ids = list(range(len(cfg.pos_df)))
        lons = cfg.pos_df[cfg.lon_col].astype(float).tolist()
        lats = cfg.pos_df[cfg.lat_col].astype(float).tolist()
        return {int(i): (float(lo), float(la)) for i, lo, la in zip(ids, lons, lats)}

    # 3) Read from graph attributes
    try:
        d = graph_dataset_test[0]
        coords = None
        for attr in ['pos', 'coords']:
            if hasattr(d, attr) and getattr(d, attr) is not None:
                coords = getattr(d, attr).detach().cpu().numpy()
                break
        if coords is None and hasattr(d, 'lon') and hasattr(d, 'lat'):
            coords = np.stack([d.lon.detach().cpu().numpy(), d.lat.detach().cpu().numpy()], axis=1)
        if coords is not None:
            return {int(i): (float(lon), float(lat)) for i, (lon, lat) in enumerate(coords)}
    except Exception:
        pass
    return {}


def _infer_map_extent(pos_all: Dict[int, Tuple[float, float]], cfg: VisualizationConfig):
    """
    Infer Basemap keyword arguments (projection and bounds) from positions or cfg.

    If cfg.basemap is provided, it is returned as-is. Otherwise the geographic
    extent is computed from available positions with a small padding. Falls back
    to a default Europe-like box if no positions are available.

    :param pos_all: Node positions mapping {node_id: (lon, lat)}.
    :type pos_all: dict[int, tuple[float, float]]
    :param cfg: Visualization configuration containing map defaults.
    :type cfg: VisualizationConfig
    :returns: Basemap keyword arguments (projection and bounding box).
    :rtype: dict
    """
    # If user provided explicit Basemap kwargs, use them
    if cfg.basemap is not None:
        return cfg.basemap
    # auto extent from positions
    if len(pos_all) > 0:
        lons = np.array([p[0] for p in pos_all.values()], dtype=float)
        lats = np.array([p[1] for p in pos_all.values()], dtype=float)
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())
        pad_lon = max(0.2, 0.1 * (lon_max - lon_min + 1e-12))
        pad_lat = max(0.2, 0.1 * (lat_max - lat_min + 1e-12))
        return dict(
            projection=cfg.map_projection,
            llcrnrlon=lon_min - pad_lon, urcrnrlon=lon_max + pad_lon,
            llcrnrlat=lat_min - pad_lat, urcrnrlat=lat_max + pad_lat
        )
    # fallback
    return dict(projection=cfg.map_projection, llcrnrlon=-5, urcrnrlon=10, llcrnrlat=40, urcrnrlat=52)

def _infer_num_nodes(data_graph, explanation, edge_index):
    """
    Infer the number of nodes in a graph from multiple possible sources.

    Order of checks:
    - data_graph.num_nodes if present
    - explanation.x (node features) if present
    - otherwise from edge_index.max() + 1

    :param data_graph: A PyG Data-like object.
    :type data_graph: Any
    :param explanation: An object that may contain an 'x' tensor with shape [N, ...].
    :type explanation: Any
    :param edge_index: Edge indices tensor of shape [2, E] (PyTorch).
    :type edge_index: torch.Tensor
    :returns: Number of nodes inferred.
    :rtype: int
    """
    try:
        if hasattr(data_graph, 'num_nodes') and data_graph.num_nodes is not None:
            return int(data_graph.num_nodes)
        if hasattr(explanation, 'x') and explanation.x is not None:
            return int(explanation.x.shape[0])
    except Exception:
        pass
    return int(edge_index.max() + 1)


def _prepare_edge_weight_map(base_ei, base_ew):
    """
    Build a symmetric edge weight lookup and summary stats.

    :param base_ei: Edge index tensor of shape [2, E] (torch.LongTensor).
    :type base_ei: torch.Tensor
    :param base_ew: Edge weight tensor of shape [E] or None.
    :type base_ew: torch.Tensor | None
    :returns: Tuple (ew_map, ew_min, ew_max, ew_den), where:
              - ew_map: dict {(u, v): weight, (v, u): weight}
              - ew_min/ew_max: min/max weight
              - ew_den: denominator (max - min, or 1.0 if constant)
    :rtype: tuple[dict[tuple[int, int], float], float, float, float]
    """
    ew_map = {}
    if base_ew is None:
        return ew_map, 0.0, 0.0, 1.0
    ei_np = base_ei.cpu().numpy()
    ew_np = base_ew.detach().cpu().numpy()
    for k in range(ei_np.shape[1]):
        a, b = int(ei_np[0, k]), int(ei_np[1, k])
        w = float(ew_np[k])
        ew_map[(a, b)] = w
        ew_map[(b, a)] = w
    ew_vals = np.array(list(ew_map.values()))
    ew_min, ew_max = ew_vals.min(), ew_vals.max()
    ew_den = (ew_max - ew_min) if ew_max > ew_min else 1.0
    return ew_map, ew_min, ew_max, ew_den


def _define_panels(cfg: VisualizationConfig, dates, months_to_plot_legacy: List[str]):
    """
    Define subplots (labels and indices) based on configuration.

    Modes supported via cfg.grouping:
      - "month": 12 monthly panels using calendar month names
      - "days": first N days (ndays key, default 3)
      - "custom": user-provided 'labels' and 'indices'
      - fallback: legacy months_to_plot_legacy or ["ALL"]

    :param cfg: Visualization configuration with grouping settings.
    :type cfg: VisualizationConfig
    :param dates: Sequence of datetime-like objects aligned with test horizon, or None.
    :type dates: list[datetime.datetime] | pandas.DatetimeIndex | None
    :param months_to_plot_legacy: Legacy month labels to display if no grouping is provided.
    :type months_to_plot_legacy: list[str]
    :returns: (labels, indices, nrows, ncols, tag)
    :rtype: tuple[list[str], list[numpy.ndarray], int, int, str]
    :raises ValueError: When grouping.mode='custom' and labels/indices are missing.
    """
    mode = cfg.grouping.get("mode", "all")
    if mode == "month":
        labels = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
        indices = [_month_indices(m, dates) for m in labels]
        return labels, indices, 3, 4, 'year'
    if mode == "days":
        ndays = int(cfg.grouping.get("ndays", 3))
        idx_list = list(range(min(ndays, (len(dates) if dates is not None else ndays))))
        labels = [dates[i].strftime('%Y-%m-%d') if dates is not None else f'Day {i+1}' for i in idx_list]
        indices = [np.array([i]) for i in idx_list]
        return labels, indices, 1, len(labels), 'days'
    if mode == "custom":
        labels = cfg.grouping.get("labels", None)
        indices = cfg.grouping.get("indices", None)
        if labels is None or indices is None:
            raise ValueError("For grouping.mode='custom', provide both 'labels' and 'indices' in cfg.grouping.")
        ncols = len(labels)
        return labels, indices, 1, ncols, 'custom'
    # fallback to legacy months_to_plot if provided
    labels = months_to_plot_legacy if months_to_plot_legacy else ["ALL"]
    indices = [_month_indices(m, dates) for m in labels]
    return labels, indices, 1, len(labels), 'custom'


def _select_top_edges(edge_index, edge_mask, vis_mode, base_ew, ew_map, ew_min, ew_den, edge_keep_ratio):
    """
    Select and weight the top-K edges to render.

    Edges with non-positive or non-finite mask values are discarded.
    When ``vis_mode == 'context'`` and base edge weights are provided, the mask
    is modulated by normalized edge weights.

    :param edge_index: Edge list as array of shape [E, 2] or [2, E].
    :type edge_index: numpy.ndarray
    :param edge_mask: Edge importance values of shape [E].
    :type edge_mask: numpy.ndarray
    :param vis_mode: Visualization mode ('model'|'context'|'std').
    :type vis_mode: str
    :param base_ew: Base edge weights tensor or None.
    :type base_ew: torch.Tensor | None
    :param ew_map: Lookup dict {(u, v): weight}.
    :type ew_map: dict[tuple[int, int], float]
    :param ew_min: Minimum edge weight for normalization.
    :type ew_min: float
    :param ew_den: Denominator (max-min) for normalization (>= 1e-12).
    :type ew_den: float
    :param edge_keep_ratio: Fraction of highest-weighted edges to keep (0,1].
    :type edge_keep_ratio: float
    :returns: (edges_draw, weights_draw) filtered by top ratio.
    :rtype: tuple[list[tuple[int, int]], list[float]]
    """
    edges_all, weights_all = [], []
    for (u, v), m_val in zip(edge_index.T, edge_mask):
        m_val = float(m_val)
        if not np.isfinite(m_val) or m_val <= 0:
            continue
        if vis_mode == "context" and base_ew is not None:
            ew_val = ew_map.get((u, v), ew_map.get((v, u), 0.0))
            ew_norm = (ew_val - ew_min) / (ew_den + 1e-12)
            w = m_val * ew_norm
        else:
            w = m_val
        if np.isfinite(w):
            edges_all.append((int(u), int(v)))
            weights_all.append(float(w))
    if not weights_all:
        return [], []
    K = max(1, int(np.ceil(edge_keep_ratio * len(weights_all))))
    order = np.argsort(-np.asarray(weights_all))[:K]
    edges_draw = [edges_all[i] for i in order]
    weights_draw = [weights_all[i] for i in order]
    return edges_draw, weights_draw


def _draw_graph_panels(model_name, dataset_name, cfg, vis_mode, panel_data, G_base,
                       map_kwargs, vmin, vmax, nrows, ncols, tag, output_root="interpretability"):
    """
    Render a grid of graph panels on a Basemap background and save the figure.

    :param model_name: Identifier used in title and output filename.
    :type model_name: str
    :param dataset_name: Dataset/run name used for output directory structure.
    :type dataset_name: str
    :param cfg: Visualization configuration (styling, sizes, arrows, etc.).
    :type cfg: VisualizationConfig
    :param vis_mode: Visualization mode ('model'|'context'|'std').
    :type vis_mode: str
    :param panel_data: Iterable of (label, edges_draw, weights_draw) per subplot.
    :type panel_data: list[tuple[str, list[tuple[int,int]], list[float]]]
    :param G_base: Base NetworkX graph containing nodes and 'pos' (lon,lat).
    :type G_base: networkx.Graph
    :param map_kwargs: Basemap keyword arguments (projection, bounds).
    :type map_kwargs: dict
    :param vmin: Lower bound for colormap normalization.
    :type vmin: float
    :param vmax: Upper bound for colormap normalization.
    :type vmax: float
    :param nrows: Number of subplot rows.
    :type nrows: int
    :param ncols: Number of subplot columns.
    :type ncols: int
    :param tag: Tag appended to output filename (e.g., period/mode).
    :type tag: str
    :param output_root: Root directory for saved figures.
    :type output_root: str
    :returns: None. Saves a figure to disk and closes it.
    :rtype: None
    :raises Exception: Propagates errors from Basemap/Matplotlib or file I/O.
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    cmap = sns.cm.rocket_r if vis_mode != "std" else sns.cm.mako
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    for ax, (lbl, edges_draw, weights_draw) in zip(axes, panel_data):
        G = G_base.copy()
        for (u_i, v_i), w in zip(edges_draw, weights_draw):
            G.add_edge(u_i, v_i, weight=w)

        pos = nx.get_node_attributes(G, 'pos')
        bm_kwargs = dict(map_kwargs) if map_kwargs is not None else {}
        Basemap = _require_basemap()
        m = Basemap(resolution=cfg.map_resolution, ax=ax, **bm_kwargs)
        if cfg.draw_coastlines: m.drawcoastlines()
        if cfg.draw_countries: m.drawcountries()
        m.drawmapboundary(fill_color=cfg.mapboundary_color)
        m.fillcontinents(color=cfg.fillcontinents_color)

        pos_bm = {node: m(lon, lat) for node, (lon, lat) in pos.items()}
        node_list = list(pos_bm.keys())

        if cfg.show_nodes and node_list:
            nx.draw_networkx_nodes(G, pos_bm, nodelist=node_list,
                                   node_size=cfg.node_size, node_color=cfg.node_color,
                                   alpha=cfg.node_alpha, ax=ax)
        if cfg.show_labels and node_list:
            labels = {n: str(n) for n in node_list}
            nx.draw_networkx_labels(G, pos_bm, labels=labels,
                                    font_size=cfg.label_fontsize, font_color=cfg.label_color, ax=ax)

        if edges_draw:
            edgelist, edge_colors, widths = [], [], []
            for (u, v), w in zip(edges_draw, weights_draw):
                if u in pos_bm and v in pos_bm:
                    edgelist.append((u, v))
                    edge_colors.append(sm.to_rgba(w))
                    if np.isfinite(vmax) and (vmax > vmin):
                        t = (w - vmin) / (vmax - vmin + 1e-12)
                        t = max(0.0, min(1.0, t))
                    else:
                        t = 0.5
                    widths.append(cfg.edge_width_min + t * (cfg.edge_width_max - cfg.edge_width_min))
            if edgelist:
                use_arrows = cfg.edge_arrows and (len(edgelist) <= cfg.arrow_max_edges)
                if use_arrows:
                    nx.draw_networkx_edges(
                        G, pos_bm, ax=ax,
                        edgelist=edgelist,
                        edge_color=edge_colors,
                        alpha=cfg.edge_alpha,
                        width=widths,
                        arrows=True,
                        arrowstyle=cfg.arrowstyle,
                        arrowsize=cfg.arrowsize,
                        connectionstyle=cfg.connectionstyle,
                    )
                else:
                    nx.draw_networkx_edges(
                        G, pos_bm, ax=ax,
                        edgelist=edgelist,
                        edge_color=edge_colors,
                        alpha=cfg.edge_alpha,
                        width=widths,
                    )
            else:
                ax.text(0.5, 0.5, 'No edges', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(lbl, fontsize=max(10, ax.title.get_fontsize()))

    for ax in axes[len(panel_data):]:
        ax.axis('off')

    fig.suptitle(f'Explanation Graph ({vis_mode}): {model_name}', fontsize=cfg.fontsize)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Colorbar closer (smaller pad) and with larger fonts
    cbar = fig.colorbar(
        sm,
        ax=axes.tolist(),
        shrink=0.8,
        aspect=30,
        pad=0.02,        # bring colorbar closer to subfigures
        fraction=0.04    # reasonable width for the colorbar
    )
    cbar.set_label({
        "std": "Edge importance variability (σ)",
        "context": "Edge importance (mask × normalized edge_weight)",
    }.get(vis_mode, "Edge importance (mean edge_mask)"), fontsize=cfg.fontsize)
    cbar.ax.tick_params(labelsize=cfg.labelsize)  

    out_dir = os.path.join(output_root, dataset_name, 'explanation_graph', vis_mode)
    os.makedirs(out_dir, exist_ok=True)
    map_path = os.path.join(out_dir, f'{model_name}__{tag}.{cfg.file_ext}')
    plt.savefig(map_path, dpi=150)
    plt.close()
    print(f"Saved explanation grid for {model_name} ({vis_mode}, {dataset_name}) → {map_path}")

def get_group_feature_mats(group_name, data, graph_dataset_train, graph_dataset_test, expand_dummies=True):
    """
    Build feature matrices per node for a given group on the TEST split.

    Returns a mapping {feature_name: X} where X has shape [num_nodes, T] and
    columns are aligned to graph_dataset_test.nodes order. If a listed feature
    is missing but dummy-encoded columns exist (e.g., 'feat_X_*'), they are included.

    :param group_name: Name of the feature group to extract.
    :type group_name: str
    :param data: Data container with df_test and node identifiers (data.node_var).
    :type data: Any
    :param graph_dataset_train: Training graph dataset (used to read feature_groups).
    :type graph_dataset_train: Any
    :param graph_dataset_test: Test graph dataset (used for node order).
    :type graph_dataset_test: Any
    :param expand_dummies: Whether to include dummy one-hot expanded columns if base column is missing.
    :type expand_dummies: bool
    :returns: Mapping feature_name -> ndarray [num_nodes, T] of unscaled values.
    :rtype: dict[str, numpy.ndarray]
    :raises ValueError: If group_name is not present in feature_groups.
    :raises RuntimeWarning: If requested feature is missing in df_test (when not using dummies).
    """
    feature_groups = graph_dataset_train.dataset_kwargs.get('feature_groups', None)
    if feature_groups is None or group_name not in feature_groups:
        raise ValueError(f"No feature_groups mapping found for group '{group_name}'.")
    feat_names = feature_groups[group_name]

    node_var = data.node_var
    nodes_order = list(graph_dataset_test.nodes)
    mats = {}

    all_cols = set(data.df_test.columns)
    for f in feat_names:
        if f in all_cols:
            df = data.df_test.pivot(index='date', columns=node_var, values=f)
            df = df.reindex(columns=nodes_order)
            mats[f] = df.values.T.astype('float32')  # [num_nodes, T]
        else:
            # Try to expand dummy-encoded columns (e.g., 'Instant_00', 'Instant_01', ...)
            if expand_dummies:
                pref = f + "_"
                dummy_cols = [c for c in data.df_test.columns if c.startswith(pref)]
                if dummy_cols:
                    for dc in dummy_cols:
                        df = data.df_test.pivot(index='date', columns=node_var, values=dc)
                        df = df.reindex(columns=nodes_order)
                        mats[dc] = df.values.T.astype('float32')
                    continue
            warnings.warn(f"Feature '{f}' not found in df_test; skipping.", RuntimeWarning)
    return mats

def compute_ALE_avg_over_instants(group_values, group_contrib, n_bins=20, period=48, align='start'):
    """
    Compute ALE curves averaged over a periodic cycle (e.g., 48 half-hours).

    Steps:
    - Align/truncate to a multiple of period and (optionally) window start/end.
    - For each instant in the period, compute binned mean effects and center them.
    - Average centered curves across instants; report mean and std envelopes.

    :param group_values: Unscaled feature values, shape [num_nodes, T].
    :type group_values: torch.Tensor
    :param group_contrib: Group contributions/effects, shape [num_nodes, T].
    :type group_contrib: torch.Tensor
    :param n_bins: Number of bins for ALE discretization.
    :type n_bins: int
    :param period: Period length (e.g., 48 for half-hourly daily cycle).
    :type period: int
    :param align: 'start' to use the first Tm samples, 'end' to use the last Tm.
    :type align: str
    :returns: (x_mid, ale_mean, ale_std) arrays of length n_bins.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :raises ValueError: If not enough timesteps or no finite values.
    """
    gv = group_values.detach().cpu().numpy()
    gc = group_contrib.detach().cpu().numpy()
    N, T_gv = gv.shape
    _, T_gc = gc.shape
    Tm = min(T_gv, T_gc)
    if Tm < period:
        raise ValueError(f"Not enough timesteps ({Tm}) for period={period}.")
    # truncate to multiple of period
    Tm = (Tm // period) * period
    if align not in ('start', 'end'):
        align = 'start'
    if align == 'start':
        gv = gv[:, :Tm]
        gc = gc[:, :Tm]
    else:
        gv = gv[:, -Tm:]
        gc = gc[:, -Tm:]

    D = Tm // period  # number of complete days/windows
    # reshape to [N, D, period]
    gv = gv.reshape(N, D, period)
    gc = gc.reshape(N, D, period)

    # Global bins across all instants
    all_x = gv.reshape(-1)
    m_all = np.isfinite(all_x)
    all_x = all_x[m_all]
    if all_x.size == 0:
        raise ValueError("No finite feature values.")
    bins = np.linspace(np.min(all_x), np.max(all_x), n_bins + 1)
    x_mid = (bins[:-1] + bins[1:]) / 2.0

    # Per-instant ALE (mean by bin, centered), then average across instants
    ale_instants = np.full((period, n_bins), np.nan, dtype='float32')
    for h in range(period):
        x_h = gv[:, :, h].reshape(-1)
        y_h = gc[:, :, h].reshape(-1)
        m = np.isfinite(x_h) & np.isfinite(y_h)
        x_h = x_h[m]
        y_h = y_h[m]
        if x_h.size == 0:
            continue
        vals = np.full(n_bins, np.nan, dtype='float32')
        for k in range(n_bins):
            a, b = bins[k], bins[k+1]
            sel = (x_h >= a) & (x_h < b) if k < n_bins - 1 else (x_h >= a) & (x_h <= b)
            if sel.any():
                vals[k] = np.nanmean(y_h[sel])
        # center ALE for this instant
        mu = np.nanmean(vals)
        if np.isfinite(mu):
            ale_instants[h] = vals - mu

    # Use masked arrays to avoid warnings for bins with all-NaN across instants
    m_arr = np.ma.array(ale_instants, mask=~np.isfinite(ale_instants))
    ale_mean = m_arr.mean(axis=0).filled(np.nan)
    ale_std = m_arr.std(axis=0, ddof=0).filled(np.nan)
    return x_mid, ale_mean, ale_std

def plot_ALE_avg(x_mid, ale_mean, ale_std=None, period=24, label="feature", color="C0", smooth=True):
    """
    Plot averaged ALE curve (and optional ±1 std band).

    :param x_mid: Bin midpoints.
    :type x_mid: numpy.ndarray
    :param ale_mean: Mean centered ALE per bin.
    :type ale_mean: numpy.ndarray
    :param ale_std: Optional std per bin across instants.
    :type ale_std: numpy.ndarray | None
    :param label: Legend label and x-axis label for the feature.
    :type label: str
    :param color: Base color for line and band.
    :type color: str
    :param smooth: Use spline smoothing if enough valid points.
    :type smooth: bool
    :returns: None. Displays a Matplotlib figure.
    :rtype: None
    """
    plt.figure(figsize=(12, 6))
    if smooth and np.isfinite(ale_mean).sum() >= 4:
        try:
            spline = UnivariateSpline(x_mid[np.isfinite(ale_mean)], ale_mean[np.isfinite(ale_mean)],
                                      s=max(1e-6, 1e-4 * np.nanvar(ale_mean) * np.isfinite(ale_mean).sum()))
            x_smooth = np.linspace(float(np.nanmin(x_mid)), float(np.nanmax(x_mid)), 300)
            y_smooth = spline(x_smooth)
            plt.plot(x_smooth, y_smooth, color=color, linestyle="--", linewidth=2, label=f"ALE {label} (avg {period} instants)")
        except Exception:
            plt.plot(x_mid, ale_mean, color=color, linestyle="--", linewidth=2, marker="o", label=f"ALE {label} (avg)")
    else:
        plt.plot(x_mid, ale_mean, color=color, linestyle="--", linewidth=2, marker="o", label=f"ALE {label} (avg)")

    if ale_std is not None and np.isfinite(ale_std).any():
        y_lo = ale_mean - ale_std
        y_hi = ale_mean + ale_std
        plt.fill_between(x_mid, y_lo, y_hi, color=color, alpha=0.15, label="±1 std (across instants)")

    plt.xlabel(label)
    plt.ylabel(f"Effect on target")
    plt.title(f"ALE Plot (averaged over {period} instants): {label}")
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_ALE(group_values, group_contrib, n_bins=20):
    """
    Compute accumulated local effects (ALE) for a single feature/group.

    :param group_values: Unscaled feature values, shape [num_nodes, T].
    :type group_values: torch.Tensor
    :param group_contrib: Group contributions/effects, shape [num_nodes, T].
    :type group_contrib: torch.Tensor
    :param n_bins: Number of bins for ALE discretization.
    :type n_bins: int
    :returns: Flattened x values, y effects, bin midpoints, and centered ALE per bin.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :raises ValueError: If there are no finite values or all bins are empty.
    """
    """
    ALE for one additive group or a single feature.

    group_values: Tensor[num_nodes, T] (unscaled feature values)
    group_contrib: Tensor[num_nodes, T] (unscaled group effects)
    """
    # Flatten
    x = group_values.detach().cpu().flatten().numpy()
    y = group_contrib.detach().cpu().flatten().numpy()

    # Keep finite only
    mask = ~(~np.isfinite(x) | ~np.isfinite(y))
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        raise ValueError("No finite values to compute ALE.")

    # Bin edges
    bins = np.linspace(np.min(x), np.max(x), n_bins + 1)

    ale = []
    x_mid = []
    for i in range(n_bins):
        m = (x >= bins[i]) & (x < bins[i + 1]) if i < n_bins - 1 else (x >= bins[i]) & (x <= bins[i + 1])
        if m.sum() > 0:
            ale.append(y[m].mean())
            x_mid.append((bins[i] + bins[i + 1]) / 2.0)

    ale = np.asarray(ale, dtype='float32')
    x_mid = np.asarray(x_mid, dtype='float32')
    if ale.size == 0:
        raise ValueError("All ALE bins empty; check data distribution.")

    # Center ALE
    ale -= ale.mean()
    return x, y, x_mid, ale

def plot_ALE(x, y, x_mid, ale, label="feature", color="C0", align_to_mean=True, demean_scatter=False):
    """
    Scatter + ALE curve plot for a single feature/group.

    :param x: Flattened feature values.
    :type x: numpy.ndarray
    :param y: Flattened effects/contributions.
    :type y: numpy.ndarray
    :param x_mid: Bin midpoints.
    :type x_mid: numpy.ndarray
    :param ale: Centered ALE values per bin.
    :type ale: numpy.ndarray
    :param label: Legend label and x-axis label for the feature.
    :type label: str
    :param color: Base color for scatter and lines.
    :type color: str
    :param align_to_mean: Shift ALE curve by mean(y) to overlay the scatter trend.
    :type align_to_mean: bool
    :param demean_scatter: Center scatter around 0 by subtracting mean(y).
    :type demean_scatter: bool
    :returns: None. Displays a Matplotlib figure.
    :rtype: None
    """
    plt.figure(figsize=(12, 6))

    y_mean = np.nanmean(y) if np.isfinite(y).any() else 0.0
    y_scatter = y - y_mean if demean_scatter else y
    offset = 0.0 if demean_scatter else (y_mean if align_to_mean else 0.0)

    plt.scatter(x, y_scatter, s=8, alpha=0.25, color=color)

    if x_mid.size >= 4:  # need at least k+1 points for default k=3
        spline = UnivariateSpline(x_mid, ale, s=max(1e-6, 1e-4 * np.var(ale) * len(ale)))
        x_smooth = np.linspace(float(np.min(x_mid)), float(np.max(x_mid)), 300)
        y_smooth = spline(x_smooth) + offset
        plt.plot(x_smooth, y_smooth, color='black', linestyle="--", linewidth=2, label=f"ALE {label}")
    else:
        plt.plot(x_mid, ale + offset, color=color, linestyle="--", linewidth=2, marker="o", label=f"ALE {label}")

    plt.hist(x, bins=30, density=True, alpha=0.2, color="gray")
    plt.xlabel(label)
    plt.ylabel("Effect on load")
    plt.title(f"ALE Plot: {label}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def compute_ALE_per_node(group_values, group_contrib, n_bins=20, use_global_bins=True):
    """
    Compute per-node ALE curves.

    :param group_values: Unscaled feature values per node, shape [num_nodes, T].
    :type group_values: torch.Tensor
    :param group_contrib: Group contributions per node, shape [num_nodes, T].
    :type group_contrib: torch.Tensor
    :param n_bins: Number of bins per node.
    :type n_bins: int
    :param use_global_bins: Use shared global bins across nodes (recommended).
    :type use_global_bins: bool
    :returns: (x_mid, ale_mat, counts) where:
              - x_mid: midpoints (global if use_global_bins)
              - ale_mat: per-node centered ALE, shape [num_nodes, n_bins]
              - counts: samples per bin, shape [num_nodes, n_bins]
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :raises ValueError: If no finite feature values.
    """
    gv = group_values.detach().cpu().numpy()
    gc = group_contrib.detach().cpu().numpy()
    num_nodes, T = gv.shape
    ale_mat = np.full((num_nodes, n_bins), np.nan, dtype='float32')
    counts = np.zeros((num_nodes, n_bins), dtype='int32')

    if use_global_bins:
        all_x = gv.flatten()
        mask = np.isfinite(all_x)
        all_x = all_x[mask]
        if all_x.size == 0:
            raise ValueError("No finite feature values.")
        bins = np.linspace(np.min(all_x), np.max(all_x), n_bins + 1)
        x_mid = (bins[:-1] + bins[1:]) / 2.0
    else:
        bins = None
        x_mid = None  # variable per node (not implemented for plotting consistency)

    for i in range(num_nodes):
        x_i = gv[i]
        y_i = gc[i]
        m_i = np.isfinite(x_i) & np.isfinite(y_i)
        x_i = x_i[m_i]
        y_i = y_i[m_i]
        if x_i.size == 0:
            continue
        if not use_global_bins:
            bins_i = np.linspace(np.min(x_i), np.max(x_i), n_bins + 1)
            x_mid_i = (bins_i[:-1] + bins_i[1:]) / 2.0
        local_vals = []
        for b in range(n_bins):
            if use_global_bins:
                a, bnd = bins[b], bins[b + 1]
            else:
                a, bnd = bins_i[b], bins_i[b + 1]
            sel = (x_i >= a) & (x_i < bnd) if b < n_bins - 1 else (x_i >= a) & (x_i <= bnd)
            c = sel.sum()
            counts[i, b] = c
            if c > 0:
                local_vals.append(y_i[sel].mean())
            else:
                local_vals.append(np.nan)
        local_vals = np.asarray(local_vals, dtype='float32')
        mean_center = np.nanmean(local_vals)
        ale_mat[i] = local_vals - mean_center
    return x_mid, ale_mat, counts

def compute_ALE_per_node_avg_over_instants(group_values, group_contrib, n_bins=20, period=48, align='start', use_global_bins=True):
    """
    Compute per-node ALE curves averaged over a periodic cycle.

    :param group_values: Unscaled feature values per node, shape [num_nodes, T].
    :type group_values: torch.Tensor
    :param group_contrib: Group contributions per node, shape [num_nodes, T].
    :type group_contrib: torch.Tensor
    :param n_bins: Number of bins per node.
    :type n_bins: int
    :param period: Period length (e.g., 48 for half-hourly daily cycle).
    :type period: int
    :param align: 'start' or 'end' alignment when truncating to multiple of period.
    :type align: str
    :param use_global_bins: Only global bins are supported in this averaged mode.
    :type use_global_bins: bool
    :returns: (x_mid, ale_nodes, counts_nodes) where:
              - x_mid: bin midpoints (global)
              - ale_nodes: averaged centered ALE per node [N, n_bins]
              - counts_nodes: total sample counts per node/bin [N, n_bins]
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    :raises ValueError: If not enough timesteps or no finite values.
    :raises NotImplementedError: If use_global_bins is False.
    """
    gv = group_values.detach().cpu().numpy()
    gc = group_contrib.detach().cpu().numpy()
    N, T_gv = gv.shape
    _, T_gc = gc.shape
    Tm = min(T_gv, T_gc)
    if Tm < period:
        raise ValueError(f"Not enough timesteps ({Tm}) for period={period}.")
    Tm = (Tm // period) * period  # truncate to multiple of period
    if align not in ('start', 'end'):
        align = 'start'
    if align == 'start':
        gv = gv[:, :Tm]
        gc = gc[:, :Tm]
    else:
        gv = gv[:, -Tm:]
        gc = gc[:, -Tm:]
    D = Tm // period  # number of complete windows (e.g., days)

    # reshape to [N, D, period]
    gv = gv.reshape(N, D, period)
    gc = gc.reshape(N, D, period)

    # Bins (global recommended for consistent x across nodes)
    if use_global_bins:
        all_x = gv.reshape(-1)
        m_all = np.isfinite(all_x)
        all_x = all_x[m_all]
        if all_x.size == 0:
            raise ValueError("No finite feature values.")
        bins = np.linspace(np.min(all_x), np.max(all_x), n_bins + 1)
        x_mid = (bins[:-1] + bins[1:]) / 2.0
    else:
        raise NotImplementedError("Per-node bins not supported in averaged-per-instant mode.")

    ale_nodes = np.full((N, n_bins), np.nan, dtype='float32')
    counts_nodes = np.zeros((N, n_bins), dtype='int32')

    # Loop nodes and instants
    for i in range(N):
        ale_inst_i = np.full((period, n_bins), np.nan, dtype='float32')
        counts_i = np.zeros((period, n_bins), dtype='int32')
        for h in range(period):
            x_h = gv[i, :, h]  # [D]
            y_h = gc[i, :, h]  # [D]
            m = np.isfinite(x_h) & np.isfinite(y_h)
            x_h = x_h[m]
            y_h = y_h[m]
            if x_h.size == 0:
                continue
            vals = np.full(n_bins, np.nan, dtype='float32')
            cnts = np.zeros(n_bins, dtype='int32')
            for k in range(n_bins):
                a, b = bins[k], bins[k+1]
                sel = (x_h >= a) & (x_h < b) if k < n_bins - 1 else (x_h >= a) & (x_h <= b)
                c = int(sel.sum())
                cnts[k] = c
                if c > 0:
                    vals[k] = float(np.nanmean(y_h[sel]))
            mu = np.nanmean(vals)
            if np.isfinite(mu):
                ale_inst_i[h] = vals - mu
            counts_i[h] = cnts
        # Average across instants per bin without warnings (keeps NaN if no data in a bin)
        ale_nodes[i] = np.ma.array(ale_inst_i, mask=~np.isfinite(ale_inst_i)).mean(axis=0).filled(np.nan)
        counts_nodes[i] = counts_i.sum(axis=0)
    return x_mid, ale_nodes, counts_nodes

def plot_ALE_nodes(x_mid, ale_mat, counts=None, max_cols=4, smooth=True, min_points_spline=4, title_prefix="ALE", node_labels=None, target_name='Load'):
    """
    Plot small multiples of per-node ALE curves.

    :param x_mid: Bin midpoints (global).
    :type x_mid: numpy.ndarray
    :param ale_mat: Per-node centered ALE values, shape [num_nodes, n_bins].
    :type ale_mat: numpy.ndarray
    :param counts: Optional per-node sample counts per bin, same shape as ale_mat.
    :type counts: numpy.ndarray | None
    :param max_cols: Maximum number of subplot columns.
    :type max_cols: int
    :param smooth: Use spline smoothing when enough valid points.
    :type smooth: bool
    :param min_points_spline: Minimum valid points to enable smoothing.
    :type min_points_spline: int
    :param title_prefix: Title prefix for the figure.
    :type title_prefix: str
    :param node_labels: Optional labels for nodes; defaults to "Node i".
    :type node_labels: list[str] | None
    :returns: Matplotlib Figure with the grid of ALE plots.
    :rtype: matplotlib.figure.Figure
    """
    import math
    num_nodes = ale_mat.shape[0]
    labels = list(node_labels) if node_labels is not None else [f"Node {i}" for i in range(num_nodes)]
    cols = min(max_cols, num_nodes)
    rows = math.ceil(num_nodes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 2.4*rows), squeeze=False)
    axes = axes.flatten()
    for i in range(num_nodes):
        ax = axes[i]
        ale_i = ale_mat[i]
        valid = np.isfinite(ale_i)
        if valid.sum() == 0:
            ax.set_title(f"{labels[i]} (no data)")
            ax.axis('off')
            continue
        if smooth and valid.sum() >= min_points_spline:
            try:
                spline = UnivariateSpline(x_mid[valid], ale_i[valid],
                                          s=max(1e-6, 1e-4 * np.nanvar(ale_i[valid]) * valid.sum()))
                xs = np.linspace(float(np.min(x_mid[valid])), float(np.max(x_mid[valid])), 200)
                ax.plot(xs, spline(xs), color="C0", linewidth=1.5)
            except Exception:
                ax.plot(x_mid[valid], ale_i[valid], color="C0", linewidth=1.2)
        else:
            ax.plot(x_mid[valid], ale_i[valid], color="C0", linewidth=1.2)
        ax.axhline(0, color='gray', linewidth=0.6)
        ax.set_title(f"{labels[i]}")
        ax.tick_params(labelsize=8)
        if counts is not None:
            c_i = counts[i][valid].astype(float)
            if c_i.sum() > 0:
                bw = (np.nanmedian(np.diff(x_mid)) if x_mid.size > 1 else 1.0)
                yr = max(1e-12, np.nanmax(ale_i[valid]) - np.nanmin(ale_i[valid]))
                h = (c_i / max(1.0, c_i.max())) * yr * 0.2  # scale to 20% of ALE range
                ax.bar(x_mid[valid], h, bottom=0.0, width=bw * 0.85,
                       color='lightgray', alpha=0.35, align='center')
        if i % cols == 0:
            ax.set_ylabel(f"{target_name}")
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Feature value")
    for j in range(num_nodes, len(axes)):
        axes[j].remove()
    fig.suptitle(f"{title_prefix} per node", fontsize=12)
    fig.tight_layout()
    return fig

def _ale_weighted_rms(a, w=None, axis=-1):
    """
    Weighted root-mean-square along a given axis, with NaN-safe averaging.

    :param a: Input array.
    :type a: numpy.ndarray
    :param w: Optional weights array broadcastable to a.
    :type w: numpy.ndarray | None
    :param axis: Axis along which to compute the RMS.
    :type axis: int
    :returns: Weighted RMS value(s).
    :rtype: float | numpy.ndarray
    """
    a = np.asarray(a, dtype=float)
    if w is None:
        return np.sqrt(np.nanmean(a**2, axis=axis))
    w = np.asarray(w, dtype=float)
    num = np.nansum(w * (a**2), axis=axis)
    den = np.nansum(w, axis=axis)
    out = np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan), where=den>0))
    return out

def ale_scalar_importance(ale, counts=None, method='rms'):
    """
    Reduce ALE curve(s) to a scalar importance.

    Methods:
      - 'rms': root-mean-square (optionally weighted by counts)
      - 'range': max-min range
      - 'tv': total variation (L1 sum of absolute differences)

    :param ale: ALE curve(s), shape (n_bins,) or (n_nodes, n_bins).
    :type ale: numpy.ndarray
    :param counts: Optional weights per bin (same shape as ale or per row).
    :type counts: numpy.ndarray | None
    :param method: Reduction method ('rms'|'range'|'tv').
    :type method: str
    :returns: Scalar importance.
    :rtype: float
    :raises ValueError: If method is unknown or input rank is unsupported.
    """
    A = np.asarray(ale, dtype=float)
    if A.ndim == 1:
        if method == 'rms':
            return float(_ale_weighted_rms(A, counts))
        elif method == 'range':
            return float(np.nanmax(A) - np.nanmin(A))
        elif method == 'tv':
            return float(np.nansum(np.abs(np.diff(A))))
        else:
            raise ValueError(f"Unknown method: {method}")
    elif A.ndim == 2:
        # per-node -> aggregate across nodes
        if method == 'rms':
            if counts is None:
                per_node = _ale_weighted_rms(A, None, axis=1)  # RMS over bins per node
                return float(np.nanmean(per_node))
            else:
                C = np.asarray(counts, dtype=float)
                per_node = _ale_weighted_rms(A, C, axis=1)
                w_nodes = np.nansum(C, axis=1)
                return float(np.average(per_node, weights=np.where(np.isfinite(per_node), w_nodes, 0)))
        elif method == 'range':
            per_node = (np.nanmax(A, axis=1) - np.nanmin(A, axis=1))
            return float(np.nanmean(per_node))
        elif method == 'tv':
            per_node = np.nansum(np.abs(np.diff(A, axis=1)), axis=1)
            return float(np.nanmean(per_node))
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        raise ValueError("ale must be 1D or 2D.")

def compute_feature_importances_from_ALE(group_outputs, data, graph_dataset_train, graph_dataset_test,
                                         n_bins=20, mode='avg48', period=48, align='start',
                                         method='rms'):
    """
    Compute scalar feature importances from ALE for groups and features.

    Modes:
      - 'global': ALE over all nodes/times (compute_ALE)
      - 'avg48': average ALE over a periodic cycle (compute_ALE_avg_over_instants)
      - 'per_node_avg48': per-node ALE averaged over instants; aggregate across nodes

    :param group_outputs: Mapping group_name -> contributions tensor [N, T_pred].
    :type group_outputs: dict[str, torch.Tensor]
    :param data: Data container providing df_test, node_var, and date index.
    :type data: Any
    :param graph_dataset_train: Training dataset with dataset_kwargs['feature_groups'].
    :type graph_dataset_train: Any
    :param graph_dataset_test: Test dataset providing node ordering.
    :type graph_dataset_test: Any
    :param n_bins: Number of ALE bins.
    :type n_bins: int
    :param mode: One of {'global','avg48','per_node_avg48'}.
    :type mode: str
    :param period: Cycle length for the ``avg48`` and ``per_node_avg48`` modes.
    :type period: int
    :param align: 'start' or 'end' alignment for windowing.
    :type align: str
    :param method: Scalar reduction method for ALE ('rms'|'range'|'tv').
    :type method: str
    :returns: DataFrame with columns ['feature','group','importance'] sorted desc.
    :rtype: pandas.DataFrame
    """
    feature_groups = graph_dataset_train.dataset_kwargs.get('feature_groups', {})
    rows = []
    for group_name, feats in feature_groups.items():
        if group_name not in group_outputs:
            continue
        contrib = group_outputs[group_name]  # [num_nodes, T_pred]
        feat_mats = get_group_feature_mats(group_name, data, graph_dataset_train, graph_dataset_test)

        # Iterate over the actually available columns (handles missing and expanded dummies)
        for feat, X in feat_mats.items():
            Tm = min(X.shape[1], contrib.shape[1])
            if Tm <= 1:
                continue
            if mode == 'global':
                x, y, x_mid, ale = compute_ALE(torch.as_tensor(X[:, :Tm]), contrib[:, :Tm], n_bins=n_bins)
                imp = ale_scalar_importance(ale, counts=None, method=method)
            elif mode == 'avg48':
                if Tm < period:
                    continue
                x_mid, ale_mean, ale_std = compute_ALE_avg_over_instants(
                    torch.as_tensor(X[:, :Tm]), contrib[:, :Tm],
                    n_bins=n_bins, period=period, align=align
                )
                imp = ale_scalar_importance(ale_mean, counts=None, method=method)
            elif mode == 'per_node_avg48':
                if Tm < period:
                    continue
                x_mid, ale_nodes, counts = compute_ALE_per_node_avg_over_instants(
                    torch.as_tensor(X[:, :Tm]), contrib[:, :Tm],
                    n_bins=n_bins, period=period, align=align
                )
                imp = ale_scalar_importance(ale_nodes, counts=counts, method=method)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            if np.isfinite(imp):
                rows.append({'feature': f'{group_name}:{feat}', 'group': group_name, 'importance': float(imp)})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    return df

def plot_feature_importance_bar(df, top_k=None, normalize='sum', figsize=None, color='C0'):
    """
    Plot a horizontal bar chart of feature importances.

    :param df: DataFrame with columns ['feature','importance'].
    :type df: pandas.DataFrame
    :param top_k: Optional number of top features to display.
    :type top_k: int | None
    :param normalize: 'sum' to normalize by total, 'max' by maximum, or None.
    :type normalize: str | None
    :param figsize: Figure size (width, height) in inches.
    :type figsize: tuple[float, float] | None
    :param color: Matplotlib color for bars.
    :type color: str
    :returns: The DataFrame used for plotting (after normalization/filtering), or None if empty.
    :rtype: pandas.DataFrame | None
    """
    if df.empty:
        print("No feature importances to plot.")
        return None
    d = df.copy()
    vals = d['importance'].values.astype(float)
    if normalize == 'sum':
        s = np.nansum(vals)
        vals = vals / s if s > 0 else vals
        ylabel = "Relative importance (sum=1)"
    elif normalize == 'max':
        m = np.nanmax(vals)
        vals = vals / m if m > 0 else vals
        ylabel = "Relative importance (max=1)"
    else:
        ylabel = "Importance"
    d['importance_norm'] = vals
    if top_k is not None:
        d = d.head(top_k)
    d = d.sort_values('importance_norm', ascending=True)  # for horizontal bar
    h = max(3, 0.4 * len(d))
    if figsize is None:
        figsize = (10, h)
    plt.figure(figsize=figsize)
    plt.barh(d['feature'], d['importance_norm'], color=color, alpha=0.8)
    plt.xlabel(ylabel)
    plt.ylabel("Feature")
    plt.title("Feature importance from ALE")
    plt.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    return d


