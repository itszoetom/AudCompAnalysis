"""Run all discriminability analyses and plots."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discriminability.run_all_analyses import main as run_analyses  # noqa: E402
from discriminability.lda.plot import main as plot_lda  # noqa: E402
from discriminability.linearSVM.plot import main as plot_svm  # noqa: E402
from discriminability.pearson.plot import main as plot_pearson  # noqa: E402


def main() -> None:
    """Run the full discriminability pipeline."""
    print("Running discriminability analyses...")
    run_analyses()
    print("[1/3] Plotting Pearson figures...")
    plot_pearson()
    print("[2/3] Plotting linear SVM figures...")
    plot_svm()
    print("[3/3] Plotting LDA figures...")
    plot_lda()


if __name__ == "__main__":
    main()
