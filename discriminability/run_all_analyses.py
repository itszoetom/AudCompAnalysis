"""Run all discriminability analysis scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discriminability.lda.analysis import main as run_lda  # noqa: E402
from discriminability.linearSVM.analysis import main as run_svm  # noqa: E402
from discriminability.pearson.analysis import main as run_pearson  # noqa: E402


def main() -> None:
    """Run Pearson, linear-SVM, and LDA analyses."""
    print("[1/3] Running Pearson analysis...")
    run_pearson()
    print("[2/3] Running linear SVM analysis...")
    run_svm()
    print("[3/3] Running LDA analysis...")
    run_lda()


if __name__ == "__main__":
    main()
