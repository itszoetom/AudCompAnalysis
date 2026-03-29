"""Plot LDA discriminability figures from saved pairwise results."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discriminability.discriminability_analysis import (  # noqa: E402
    load_method_results,
    plot_heatmaps,
    plot_natural_within_between_boxplots,
    plot_region_boxplots,
)

METHOD_KEY = "lda"
METHOD_LABEL = "LDA Accuracy"
VALUE_COL = "Accuracy"
YLABEL = "LDA Accuracy"


def main() -> None:
    """Create LDA discriminability figures."""
    results_df = load_method_results(METHOD_KEY)
    plot_heatmaps(
        results_df,
        method_key=METHOD_KEY,
        method_label=METHOD_LABEL,
        value_col=VALUE_COL,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    plot_region_boxplots(
        results_df,
        method_key=METHOD_KEY,
        method_label=METHOD_LABEL,
        value_col=VALUE_COL,
        ylabel=YLABEL,
    )
    plot_natural_within_between_boxplots(
        results_df,
        method_key=METHOD_KEY,
        method_label=METHOD_LABEL,
        value_col=VALUE_COL,
        ylabel=YLABEL,
    )


if __name__ == "__main__":
    main()
