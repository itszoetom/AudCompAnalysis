"""Run the full methods figure pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.plot_data_info import main as plot_data_info  # noqa: E402
from methods.plot_single_mouse_psth import main as plot_psth  # noqa: E402


def main() -> None:
    """Run all current methods figures."""
    print("Running methods figures...")
    print("[1/2] Plotting dataset summaries...")
    plot_data_info()
    print("[2/2] Plotting raster and PSTH examples...")
    plot_psth()


if __name__ == "__main__":
    main()
