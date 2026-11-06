"""Package the paper forecasts as compact, portable archives.

This maintenance runner is separate from the analyses: it reads the forecasts
written by the runners of ``load/hpo_v030`` and writes the ``forecasts_paper/*.npz``
files committed with the examples. The archives contain predictions and targets,
never model weights. Aggregates use ``graphtoolbox.aggregation.Aggregation``
(MLpol, square loss, weights updated once per day), as in the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from graphtoolbox.aggregation import Aggregation

HERE = Path(__file__).resolve().parent
RUNS = HERE / "load" / "hpo_v030"
AGGREGATE_PREFIX = "aggregate__"
COMPONENTS = ("Load", "Wind_power", "Solar_power")


def load_tensor(path: Path) -> np.ndarray:
    """Load a CPU tensor as a portable float32 array."""
    tensor = torch.load(path, weights_only=False, map_location="cpu")
    return tensor.numpy().astype(np.float32)


def write_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"{path.relative_to(HERE)}: {len(arrays)} arrays")


def mlpol(experts: np.ndarray, target: np.ndarray) -> np.ndarray:
    return Aggregation(model="MLpol", loss="square").run(
        experts, target, block_size=48).prediction_.astype(np.float32)


def aggregates(node_predictions: dict[str, np.ndarray], target: np.ndarray,
               target_nodes: np.ndarray) -> dict[str, np.ndarray]:
    """Uniform, top-level MLpol and bottom-up MLpol aggregates of regional forecasts."""
    names = sorted(node_predictions)
    national = np.column_stack([node_predictions[n].sum(axis=0) for n in names])
    bottom_up = np.zeros(len(target), dtype=np.float32)
    for node in range(target_nodes.shape[0]):
        bottom_up += mlpol(np.column_stack([node_predictions[n][node] for n in names]), target_nodes[node])
    return {f"{AGGREGATE_PREFIX}uniform": national.mean(axis=1).astype(np.float32),
            f"{AGGREGATE_PREFIX}mlpol_top": mlpol(national, target),
            f"{AGGREGATE_PREFIX}mlpol_bottom": bottom_up}


def package_load() -> None:
    """Static sweep (47 operators, with aggregates) and one-shot recurrent cells."""
    with np.load(RUNS / "load_sweep_v030.npz") as archive:
        write_archive(HERE / "load" / "forecasts_paper" / "load_sweep.npz",
                      {name: archive[name] for name in archive.files})
    cells = {path.stem.removeprefix("temporal_oneshot_"): load_tensor(path)
             for path in sorted((RUNS / "preds").glob("temporal_oneshot_*.pt"))}
    target = cells.pop("target")
    write_archive(HERE / "load" / "forecasts_paper" / "load_temporal_oneshot.npz",
                  {**cells, "target": target.sum(axis=0), "target_nodes": target})


def package_seeds() -> None:
    """Five-seed retraining of the leading load and net-load models (seed table)."""
    load = {f"GatedGraphConv_seed{s}": load_tensor(RUNS / "preds" / f"GatedGraphConv_seed{s}.pt")
            for s in (42, 0, 1, 2, 3)}
    write_archive(HERE / "load" / "forecasts_paper" / "load_seeds.npz", load)
    netload = {}
    for s in (0, 1, 2, 3, 4):
        for path in sorted((RUNS / "netload_v030" / f"leads_seed{s}").glob("*.pt")):
            if not path.stem.startswith("target"):
                netload[f"{path.stem}_seed{s}"] = load_tensor(path)
    write_archive(HERE / "netload" / "forecasts_paper" / "netload_seeds.npz", netload)


def package_netload(run: str = "leads_seed42") -> None:
    """Direct and decomposed net-load sweeps of the given run, with aggregates."""
    src = RUNS / "netload_v030" / run
    direct = {path.stem: load_tensor(path) for path in sorted(src.glob("*.pt"))
              if "__" not in path.stem}
    target_nodes = load_tensor(src / "target__NetLoad.pt")
    target = target_nodes.sum(axis=0)
    direct.update({"target": target, "target_nodes": target_nodes})
    direct.update(aggregates({k: v for k, v in direct.items() if not k.startswith("target")},
                             target, target_nodes))
    write_archive(HERE / "netload" / "forecasts_paper" / "netload_direct.npz", direct)

    decomp = {path.stem: load_tensor(path) for path in sorted(src.glob("*__*.pt"))
              if not path.stem.startswith("target__")}
    for component in COMPONENTS:
        decomp[f"target__{component}"] = load_tensor(src / f"target__{component}.pt").sum(axis=0)
    decomp.update({"target": target, "target_nodes": target_nodes})
    names = sorted({k.split("__", 1)[0] for k in decomp if "__" in k and not k.startswith("target__")})
    recombined = {n: decomp[f"{n}__Load"] - decomp[f"{n}__Wind_power"] - decomp[f"{n}__Solar_power"]
                  for n in names}
    decomp.update(aggregates(recombined, target, target_nodes))
    write_archive(HERE / "netload" / "forecasts_paper" / "netload_decomposed.npz", decomp)


if __name__ == "__main__":
    package_load()
    package_netload()
    package_seeds()
