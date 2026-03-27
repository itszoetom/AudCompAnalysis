"""Run session-subsampled Pearson discriminability analyses for all sounds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discriminability.discriminability_analysis import get_method_dir, run_pairwise_analysis  # noqa: E402


def pearson_metrics(resp1: np.ndarray, resp2: np.ndarray, _: int) -> dict[str, float]:
    """Return Pearson correlation and dissimilarity between two mean stimulus vectors."""
    mean_left = resp1.mean(axis=0)
    mean_right = resp2.mean(axis=0)
    correlation = float(np.corrcoef(mean_left, mean_right)[0, 1])
    return {
        "Correlation": correlation,
        "Dissimilarity": 1.0 - correlation,
    }


def main() -> None:
    """Run Pearson discriminability and save the long-form pairwise results."""
    output_dir = get_method_dir("pearson")
    results_df = run_pairwise_analysis(pearson_metrics)
    results_df.to_csv(output_dir / "pairwise_results.csv", index=False)


if __name__ == "__main__":
    main()
