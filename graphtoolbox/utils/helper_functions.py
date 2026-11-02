from math import asin, cos, radians, sin, sqrt
import numpy as np
import os
import re
import shutil
import torch
from typing import Any, Dict, Optional

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def clean_dir(directory: str) -> None:
    """
    Deletes and recreates a specified directory.

    Args:
        directory (str): The path to the directory to clean.
    """
    shutil.rmtree(directory)
    os.makedirs(directory)

def make_adam(params, lr: float, fused: bool = False, **kwargs) -> torch.optim.Optimizer:
    """
    Adam, optionally with the fused kernel.

    The default Adam updates the parameters one tensor at a time; on the small
    models of this package that loop costs as much as the backward pass. The
    fused kernel (CPU since torch 2.4, CUDA, MPS) does one update for all of
    them, which shortens a CPU epoch by up to a sixth. It rounds differently,
    so a fused run differs from an unfused one in the last digits: it is off by
    default to keep results identical to earlier versions.

    Args:
        params: Parameters to optimize.
        lr (float): Learning rate.
        fused (bool): Use the fused kernel. Falls back to the default
            implementation when the installed torch does not support it for
            these parameters.
        **kwargs: Other arguments of torch.optim.Adam.

    Returns:
        torch.optim.Adam: The optimizer.
    """
    params = list(params)
    if fused:
        try:
            return torch.optim.Adam(params, lr=lr, fused=True, **kwargs)
        except (RuntimeError, TypeError, ValueError):
            pass
    return torch.optim.Adam(params, lr=lr, **kwargs)

def epoch_mean(losses, weights) -> float:
    """
    Weighted mean of per-batch losses, read from the device once.

    The epoch loops keep each batch loss as a tensor and call this at the end:
    one synchronization per epoch instead of one ``.item()`` per batch. The sum
    runs in Python floats, in batch order, as the per-batch ``.item()`` loop did,
    so the value is the same to the last digit.

    Args:
        losses (list of torch.Tensor): Scalar loss of each batch.
        weights (list of int): Number of graphs in each batch.

    Returns:
        float: sum(loss * weight) / sum(weight), or 0 if there is no batch.
    """
    if not losses:
        return 0
    total = 0.0
    for loss, w in zip(torch.stack(losses).cpu().tolist(), weights):
        total += loss * w
    return total / sum(weights)

def batch_guards(dataset) -> "tuple[bool, bool]":
    """
    Whether any snapshot of a graph dataset holds a NaN feature or target, and
    whether any has no valid target.

    A training batch is skipped for these reasons only if one of its snapshots
    is, so the epoch loops test batches only when the dataset requires it:
    testing every batch costs a device synchronization per step. GraphDataset
    replaces NaNs by zeros, so both are usually False.

    Args:
        dataset: Iterable of torch_geometric Data with ``x``, and optionally
            ``y_scaled`` and ``mask_y``.

    Returns:
        tuple of bool: (has_nan, has_empty).
    """
    has_nan = has_empty = False
    for d in dataset:
        y_s = getattr(d, 'y_scaled', None)
        has_nan = has_nan or bool(torch.isnan(d.x).any()) or \
            (y_s is not None and bool(torch.isnan(y_s).any()))
        m = getattr(d, 'mask_y', None)
        has_empty = has_empty or (m is not None and bool(m.sum() == 0))
        if has_nan and has_empty:
            break
    return has_nan, has_empty

def batch_y_to_tensor(batch: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Converts a batch of data to a tensor, organizing elements by modulo num_nodes
    without detaching gradients.

    Args:
        batch (torch.Tensor): The batch of data containing the specified attribute.
        **kwargs:
            num_nodes (int): Number of nodes in the graph. Defaults to 12.

    Returns:
        torch.Tensor: A tensor where data is organized into num_nodes groups.
    """
    num_nodes = kwargs.get('num_nodes', 12)
    num_elements = batch.size(0)
    num_batches = num_elements // num_nodes
    y_tensor = torch.zeros(num_nodes, num_batches, dtype=batch.dtype, device=DEVICE)
    for idx in range(num_elements):
        y_tensor[idx % num_nodes, idx // num_nodes] = batch[idx]
    return y_tensor
    
def batch_x_to_tensor(batch: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Converts a batch of data to a tensor, organizing elements by modulo num_nodes
    without detaching gradients.

    Args:
        batch (torch.Tensor): The batch of data containing the specified attribute.
        **kwargs:
            num_nodes (int): Number of nodes in the graph. Defaults to 12.

    Returns:
        torch.Tensor: A tensor where data is organized into num_nodes groups.
    """
    num_nodes = kwargs.get('num_nodes', 12)
    num_elements, num_features = batch.shape
    num_batches = num_elements // num_nodes
    x_tensor = torch.zeros(num_nodes, num_batches, num_features, dtype=batch.dtype, device=DEVICE)
    for idx in range(num_elements):
        x_tensor[idx % num_nodes, idx // num_nodes] = batch[idx]
    return x_tensor      
  
def load_config(folder_config: str):
    """
    Loads configuration dictionaries.

    Args:
        folder_config (str): Path to the configuration file.
    
    Returns:
        Configuration file.
    """
    import sys
    sys.path.append(folder_config)
    import config
    return config

def load_kwargs(folder_config: str, kwargs: str) -> Dict[str, Any]:
    """
    Load keyword arguments.

    Args:
        kwargs (str): The name of the attribute (dictionary) to retrieve from the module.

    Returns:
        Dict[str, Any]: The dictionary retrieved from the module, or an empty dictionary if not found.
    """
    config = load_config(folder_config)
    new_kwargs = config.__getattribute__(kwargs) if hasattr(config, kwargs) else {}
    return new_kwargs

def save_config(folder_config: str, kwargs: str, new_values: Dict[str, Any]) -> None:
    """
    Save new values to a specified dictionary in a module.

    Args:
        kwargs (str): The name of the dictionary to update in the module.
        new_values (Dict[str, Any]): The new key-value pairs to add to the dictionary.

    Raises:
        ValueError: If the specified dictionary is not found in the module.
    """
    config_path = os.path.join(folder_config, 'config.py')
    with open(config_path, 'r') as file:
        lines = file.readlines()
    
    new_lines = []
    in_target_dict = False
    target_dict_found = False

    for line in lines:
        if line.strip().startswith(f'{kwargs} = {{'):
            in_target_dict = True
            target_dict_found = True
            new_lines.append(f'{kwargs} = {{\n')
            for key, value in new_values.items():
                if isinstance(value, str):
                    new_lines.append(f"    '{key}': '{value}',\n")
                else:
                    new_lines.append(f"    '{key}': {value},\n")
        elif in_target_dict and line.strip().startswith('}'):
            in_target_dict = False
            new_lines.append('}\n')
        elif not in_target_dict:
            new_lines.append(line)
    
    if not target_dict_found:
        raise ValueError(f"The dictionary '{kwargs}' does not exist in the config file.")
    
    with open(config_path, 'w') as file:
        file.writelines(new_lines)

def update_config(folder_config: str, kwargs: str, new_config: Dict[str, Any]) -> None:
    """
    Update a specified dictionary in a module with new configuration values.

    Args:
        kwargs (str): The name of the dictionary to update in the module.
        new_config (Dict[str, Any]): The new key-value pairs to update the dictionary with.

    Raises:
        ValueError: If the specified dictionary is not found in the module.
    """
    new_kwargs = load_kwargs(folder_config, kwargs)
    if new_kwargs != {}:
        new_kwargs.update(new_config)
    else:
        raise ValueError(f"The dictionary '{kwargs}' does not exist in the config file.")
    save_config(kwargs, new_kwargs)

def parser_config(config: str) -> Dict[str, Any]:
    """
    Parse a configuration string into a dictionary.

    Args:
        config (str): The configuration string, formatted as '{"key": "value", "key2": "value2",...}'.

    Returns:
        Dict[str, Any]: The parsed dictionary.

    Example:
        >>> parser_config('{"model_name": "GCN", "num_epochs": "100"})
        {'model_name': 'GCN', 'num_epochs': 100}
    """
    new_config = config.replace('{', '').replace('}', '').replace(' ', '').split(',')
    new_config_dict = {}
    for arg in new_config:
        key, value = arg.split(':')
        key = key.replace('"', '')
        value = value.replace('"', '')
        try:
            new_value = int(value)
        except ValueError:
            try:
                new_value = float(value)
            except ValueError:
                new_value = str(value)
        new_config_dict[key] = new_value
    return new_config_dict

def extract_parameters(model_string):
    pattern = re.compile(r'batch(\d+)_hidden(\d+)_layers(\d+)_epochs(\d+)')
    match = pattern.match(model_string)
    
    if match:
        batch_size = int(match.group(1))
        hidden_channels = int(match.group(2))
        layers = int(match.group(3))
        return {
            'batch_size': batch_size,
            'hidden_channels': hidden_channels,
            'num_layers': layers,
        }
    else:
        return None

def change_cwd(target_directory: Optional[str] = "GraphToolbox", verbose: bool = False) -> None:
    """
    Change the current working directory to the specified directory by traversing up 
    the directory tree until the target directory is found.

    Args:
        target_directory (Optional[str]): The target directory to change to. Defaults to "GraphToolbox".

    Raises:
        FileNotFoundError: If the target directory does not exist in the path hierarchy.
    """
    current_dir = os.getcwd()

    while True:
        if target_directory in os.listdir(current_dir):
            target_path = os.path.join(current_dir, target_directory)
            if os.path.isdir(target_path):
                os.chdir(target_path)
                if verbose:
                    print(f"Changed working directory to {os.getcwd()}")
                return
            else:
                raise NotADirectoryError(f"The path '{target_path}' is not a directory.")
        
        parent_dir = os.path.dirname(current_dir)
        
        if current_dir == parent_dir:
            raise FileNotFoundError(f"The directory '{target_directory}' does not exist in the path hierarchy.")
        
        current_dir = parent_dir

def get_geodesic_distance(point_1, point_2) -> float:
    """
    Calculate the great circle distance (in km) between two points
    on the earth (specified in decimal degrees)

    https://stackoverflow.com/a/4913653
    """

    lon1, lat1 = point_1
    lon2, lat2 = point_2

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def get_exponential_similarity(
    condensed_distance_matrix, bandwidth, threshold
):
    exp_similarity = np.exp(
        -(condensed_distance_matrix ** 2) / bandwidth / bandwidth
    )
    res_arr = np.where(exp_similarity > threshold, exp_similarity, 0.0)
    return res_arr

def build_adjacency_matrix(edge_index, edge_weight):
    num_nodes = edge_index.unique().size()[0]
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    adj_matrix[edge_index[0], edge_index[1]] = edge_weight.float()
    return adj_matrix