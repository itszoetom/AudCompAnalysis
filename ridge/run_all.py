"""Run the full ridge-regression pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ridge.plot_ridge_boxplot import main as plot_population_boxplots  # noqa: E402
from ridge.plot_ridge_per_mouse import main as plot_per_mouse  # noqa: E402
from ridge.plot_ridge_per_session import main as plot_per_session  # noqa: E402
from ridge.ridge_population import main as plot_predicted_vs_actual  # noqa: E402


def main() -> None:
    """Run all ridge figures."""
    print("Running ridge-regression figures...")
    print("[1/4] Plotting population ridge boxplots...")
    plot_population_boxplots()
    print("[2/4] Plotting per-mouse ridge boxplots...")
    plot_per_mouse()
    print("[3/4] Plotting per-session ridge boxplots...")
    plot_per_session()
    print("[4/4] Plotting predicted-vs-actual ridge figures...")
    plot_predicted_vs_actual()


if __name__ == "__main__":
    main()
