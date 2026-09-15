"""Recompute the paper tables and load significance tests from cached forecasts.

No dataset or checkpoint is required. The default 2,000 moving-block resamples
match the paper; ``--n-boot 100`` is convenient for a quick smoke test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from graphtoolbox.evaluation import (
    bootstrap_metric,
    diebold_mariano,
    model_confidence_set,
    pairwise_dm,
)
from graphtoolbox.models.baseline import mape, rmse, smape


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results_cached_statistics"
AGGREGATE_PREFIX = "aggregate__"
LOAD_TABLE = [
    "LEConv", "GatedGraphConv", "GATv2Conv", "ARMAConv",
    "ResGatedGraphConv", "SGConv", "RGATConv", "RGCNConv", "GCNConv",
    "GATConv",
]


def read_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def static_predictions(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return regional static-model forecasts, excluding metadata/aggregates."""
    return {
        name: value for name, value in arrays.items()
        if name not in {"target", "target_nodes"}
        and not name.startswith((AGGREGATE_PREFIX, "target__"))
        and "__" not in name
    }


def aggregates(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    labels = {
        "aggregate__uniform": "Uniform average",
        "aggregate__mlpol_bottom": "MLpol (bottom-up)",
        "aggregate__mlpol_top": "MLpol (top-level)",
    }
    return {
        label: arrays[key]
        for key, label in labels.items()
        if key in arrays
    }


def decomposed_predictions(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    names = sorted({
        key.split("__", 1)[0]
        for key in arrays
        if "__" in key and not key.startswith(("target__", AGGREGATE_PREFIX))
    })
    return {
        name: (
            arrays[f"{name}__Load"]
            - arrays[f"{name}__Wind_power"]
            - arrays[f"{name}__Solar_power"]
        )
        for name in names
    }


def metric_rows(
    task: str,
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    percentage: str,
    n_boot: int,
) -> list[dict[str, float | str]]:
    rows = []
    for name, prediction in predictions.items():
        if prediction.ndim == 2:
            prediction = prediction.sum(axis=0)
        result = bootstrap_metric(
            prediction, target, metric="rmse", n_boot=n_boot,
            block_len=48, seed=0,
        )
        percentage_value = (
            mape(prediction, target)
            if percentage == "mape"
            else smape(prediction, target)
        )
        rows.append({
            "task": task,
            "model": name,
            percentage: percentage_value,
            "rmse_mw": rmse(prediction, target),
            "rmse_bootstrap_se_mw": result.se,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)

    load = read_archive(
        HERE / "load" / "forecasts_paper" / "load_sweep.npz"
    )
    load_nodes = static_predictions(load)
    load_national = {
        name: prediction.sum(axis=0)
        for name, prediction in load_nodes.items()
    }
    load_report = {
        name: load_nodes[name] for name in LOAD_TABLE
    }
    load_report.update(aggregates(load))

    direct = read_archive(
        HERE / "netload" / "forecasts_paper" / "netload_direct.npz"
    )
    direct_report = static_predictions(direct)
    direct_report.update(aggregates(direct))

    decomposed = read_archive(
        HERE / "netload" / "forecasts_paper" / "netload_decomposed.npz"
    )
    decomposed_report = decomposed_predictions(decomposed)
    decomposed_report.update(aggregates(decomposed))

    rows = metric_rows(
        "load", load_report, load["target"], "mape", args.n_boot
    )
    rows += metric_rows(
        "netload_direct", direct_report, direct["target"], "smape",
        args.n_boot,
    )
    rows += metric_rows(
        "netload_decomposed", decomposed_report, decomposed["target"],
        "smape", args.n_boot,
    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "metrics.csv", index=False)

    load_mcs = model_confidence_set(
        load_national, load["target"], loss="squared", alpha=0.10,
        n_boot=args.n_boot, block_len=48, seed=0,
    )
    table_dm = pairwise_dm(
        {name: load_national[name] for name in LOAD_TABLE},
        load["target"], loss="squared", h=48, correction="holm",
    )
    best = min(load_national, key=lambda name: rmse(
        load_national[name], load["target"]
    ))
    aggregate = load["aggregate__mlpol_top"]
    aggregate_dm = diebold_mariano(
        aggregate, load_national[best], load["target"], loss="squared", h=48,
        names=("MLpol (top-level)", best),
    )
    significance = {
        "load_mcs_level": 0.90,
        "load_mcs_total": len(load_national),
        "load_mcs_retained": len(load_mcs.included),
        "load_mcs_members": sorted(load_mcs.included),
        "load_table_best": best,
        "load_table_holm_dm_separates_from_best": [
            name for name in LOAD_TABLE
            if name != best
            and table_dm["pvalue_adjusted"].loc[name, best] < 0.05
        ],
        "load_mlpol_vs_best_dm_pvalue": aggregate_dm.pvalue,
        "load_mlpol_vs_best_dm_better": aggregate_dm.better,
    }
    with (args.output / "significance.json").open("w") as stream:
        json.dump(significance, stream, indent=2)

    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(
        f"\nLoad 90% MCS: {len(load_mcs.included)}/{len(load_national)} retained; "
        f"MLpol vs {best}: p={aggregate_dm.pvalue:.4g}."
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
