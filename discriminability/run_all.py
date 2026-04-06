"""Run the full discriminability analysis and plotting pipeline
Takes 39 minutes to run"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discriminability.discriminability_analysis import run_method_analysis  # noqa: E402
from discriminability.plot_lda import main as plot_lda  # noqa: E402
from discriminability.plot_linear_svm import main as plot_svm  # noqa: E402
from discriminability.plot_pearson import main as plot_pearson  # noqa: E402


def main() -> None:
    """Run the full discriminability pipeline."""
    print("Running discriminability analyses...")
    print("[1/6] Running Pearson analysis...")
    run_method_analysis("pearson")
    print("[2/6] Running linear SVM analysis...")
    run_method_analysis("linearSVM")
    print("[3/6] Running LDA analysis...")
    run_method_analysis("lda")
    print("[4/6] Plotting Pearson figures...")
    plot_pearson()
    print("[5/6] Plotting linear SVM figures...")
    plot_svm()
    print("[6/6] Plotting LDA figures...")
    plot_lda()


if __name__ == "__main__":
    main()
