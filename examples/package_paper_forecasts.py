"""Package the paper checkpoints as compact, portable forecast archives.

This maintenance runner is intentionally separate from the analyses: it reads
the local ``.pt`` result directories and writes the ``forecasts_paper/*.npz``
files that are committed with the examples.  The archives contain predictions
and targets, never model weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
AGGREGATE_PREFIX = "aggregate__"


def load_tensor(path: Path) -> np.ndarray:
    """Load a CPU tensor as a portable float32 array."""
    tensor = torch.load(path, weights_only=False, map_location="cpu")
    return tensor.numpy().astype(np.float32)


def write_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"{path.relative_to(HERE)}: {len(arrays)} arrays")


def opera_mlpol(experts: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reproduce the block-online MLpol protocol used for the net-load table."""
    from opera import Mixture

    block_size = 48
    frame = pd.DataFrame(experts)
    prediction = np.empty(len(target), dtype=float)
    prediction[:block_size] = experts[:block_size].mean(axis=1)
    mixture = Mixture(
        y=target[:block_size], experts=frame.iloc[:block_size],
        model="MLpol", loss_type="mse",
    )
    for start in range(block_size, len(target), block_size):
        end = min(start + block_size, len(target))
        new_experts = frame.iloc[start:end]
        prediction[start:end] = np.ravel(
            mixture.predict(new_experts=new_experts)
        )[:end - start]
        mixture.update(new_experts=new_experts, new_y=target[start:end])
    return prediction.astype(np.float32)


def opera_aggregates(
    node_predictions: dict[str, np.ndarray],
    target: np.ndarray,
    target_nodes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the exact uniform, top-level, and bottom-up paper forecasts."""
    names = sorted(node_predictions)
    national = np.column_stack(
        [node_predictions[name].sum(axis=0) for name in names]
    )
    bottom_up = np.zeros(len(target), dtype=np.float32)
    for node in range(target_nodes.shape[0]):
        experts = np.column_stack(
            [node_predictions[name][node] for name in names]
        )
        bottom_up += opera_mlpol(experts, target_nodes[node])
    return {
        f"{AGGREGATE_PREFIX}uniform": national.mean(axis=1),
        f"{AGGREGATE_PREFIX}mlpol_top": opera_mlpol(national, target),
        f"{AGGREGATE_PREFIX}mlpol_bottom": bottom_up,
    }


def add_load_aggregates() -> None:
    """Add the aggregation outputs to the already packaged load sweep."""
    path = HERE / "load" / "forecasts_paper" / "load_sweep.npz"
    with np.load(path) as archive:
        arrays = {
            name: archive[name]
            for name in archive.files
            if not name.startswith(AGGREGATE_PREFIX)
        }
    node_predictions = {
        name: value for name, value in arrays.items()
        if name not in {"target", "target_nodes"}
    }
    # This is the native GraphToolbox MLpol implementation used by the load
    # notebook after removal of the reconciliation step.
    from graphtoolbox.aggregation import Aggregation

    names = sorted(node_predictions)
    national = np.column_stack(
        [node_predictions[name].sum(axis=0) for name in names]
    )
    arrays[f"{AGGREGATE_PREFIX}uniform"] = national.mean(axis=1)
    arrays[f"{AGGREGATE_PREFIX}mlpol_top"] = Aggregation(
        model="MLpol", loss="square"
    ).run(national, arrays["target"], block_size=48).prediction_.astype(np.float32)
    bottom_up = np.zeros(len(arrays["target"]), dtype=np.float32)
    for node in range(arrays["target_nodes"].shape[0]):
        experts = np.column_stack(
            [node_predictions[name][node] for name in names]
        )
        bottom_up += Aggregation(model="MLpol", loss="square").run(
            experts, arrays["target_nodes"][node], block_size=48
        ).prediction_.astype(np.float32)
    arrays[f"{AGGREGATE_PREFIX}mlpol_bottom"] = bottom_up
    write_archive(path, arrays)


def package_netload() -> None:
    root = HERE / "netload"
    direct_dir = root / "results_std_direct_netload"
    direct = {
        path.stem: load_tensor(path)
        for path in sorted(direct_dir.glob("*.pt"))
        if path.stem != "target"
    }
    direct["target"] = load_tensor(direct_dir / "target.pt")
    direct["target_nodes"] = load_tensor(
        root / "results_all_convolutions_netload" / "target_nodes.pt"
    )
    direct.update(opera_aggregates(
        {name: value for name, value in direct.items()
         if name not in {"target", "target_nodes"}},
        direct["target"], direct["target_nodes"],
    ))
    write_archive(root / "forecasts_paper" / "netload_direct.npz", direct)

    decomp_dir = root / "results_std_decomp_netload"
    decomp = {
        path.stem: load_tensor(path)
        for path in sorted(decomp_dir.glob("*.pt"))
    }
    for component in ("Load", "Wind_power", "Solar_power"):
        target = load_tensor(
            root / "results_decomp_netload" / f"{component}_target.pt"
        )
        decomp[f"target__{component}"] = target.sum(axis=0)
    decomp["target"] = direct["target"]
    decomp["target_nodes"] = direct["target_nodes"]
    components = {
        name.split("__", 1)[0]
        for name in decomp
        if "__" in name and not name.startswith("target__")
    }
    node_predictions = {
        name: (
            decomp[f"{name}__Load"]
            - decomp[f"{name}__Wind_power"]
            - decomp[f"{name}__Solar_power"]
        )
        for name in components
    }
    decomp.update(opera_aggregates(
        node_predictions, decomp["target"], decomp["target_nodes"]
    ))
    write_archive(
        root / "forecasts_paper" / "netload_decomposed.npz", decomp
    )


if __name__ == "__main__":
    add_load_aggregates()
    package_netload()
