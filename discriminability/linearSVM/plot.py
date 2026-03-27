"""Plot linear-SVM discriminability heatmaps and boxplots from saved pairwise results."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discriminability.discriminability_analysis import (  # noqa: E402
    get_method_dir,
    plot_heatmaps,
    plot_natural_within_between_boxplots,
    plot_region_boxplots,
)


def main() -> None:
    """Create linear-SVM heatmaps and summary boxplots."""
    input_path = get_method_dir("linearSVM") / "pairwise_results.csv"
    results_df = pd.read_csv(input_path)
    plot_heatmaps(
        results_df,
        value_col="Accuracy",
        method_name="Linear SVM Accuracy",
        method_folder="linearSVM",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    plot_region_boxplots(
        results_df,
        value_col="Accuracy",
        method_name="Linear SVM Accuracy",
        method_folder="linearSVM",
        ylabel="Linear SVM Accuracy",
    )
    plot_natural_within_between_boxplots(
        results_df,
        value_col="Accuracy",
        method_name="Linear SVM Accuracy",
        method_folder="linearSVM",
        ylabel="Linear SVM Accuracy",
    )


if __name__ == "__main__":
    main()
