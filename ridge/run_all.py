"""Run the full ridge-regression pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ridge.plot_ridge_per_session import main as plot_per_session  # noqa: E402
from ridge.plot_ridge_population import main as plot_population  # noqa: E402


def main() -> None:
    """Run all ridge figures."""
    print("Running ridge-regression figures...")
    print("[1/2] Plotting per-session ridge boxplots...")
    plot_per_session()
    print("[2/2] Plotting population ridge scatter and summary figures...")
    plot_population()


if __name__ == "__main__":
    main()
