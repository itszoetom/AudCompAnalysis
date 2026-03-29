"""Run the full PCA pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pca.plot_umap_population import main as plot_umap  # noqa: E402
from pca.plot_pca_population import main as plot_population_pca  # noqa: E402
from pca.plot_pca_population_avgs import main as plot_average_pca  # noqa: E402
from pca.plot_pca_speech import main as plot_speech_pca  # noqa: E402


def main() -> None:
    """Run all PCA figures."""
    print("Running PCA figures...")
    print("[1/4] Plotting population PCA figures...")
    plot_population_pca()
    print("[2/4] Plotting trial-averaged PCA figures...")
    plot_average_pca()
    print("[3/4] Plotting speech FT/VOT PCA figures...")
    plot_speech_pca()
    print("[4/4] Plotting UMAP figures...")
    plot_umap()


if __name__ == "__main__":
    main()
