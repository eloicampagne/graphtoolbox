"""Render the paper's grouped-ALE panel from cached feature importances."""

from pathlib import Path

import pandas as pd

from graphtoolbox.interpretability import plot_ale_group_importance


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_additive_netload"


def main():
    table = pd.read_csv(RESULTS / "ale_importance.csv")
    _, grouped = plot_ale_group_importance(
        table, output_path=RESULTS / "ale_group_importance.pdf")
    grouped.to_csv(RESULTS / "ale_group_importance.csv", index=False)
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
