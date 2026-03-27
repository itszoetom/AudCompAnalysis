"""Run session-subsampled LDA discriminability analyses for all sounds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discriminability.discriminability_analysis import N_SPLITS, get_method_dir, run_pairwise_analysis  # noqa: E402


def lda_accuracy(resp1: np.ndarray, resp2: np.ndarray, seed: int) -> dict[str, float]:
    """Return 5-fold stratified LDA accuracy for one stimulus pair."""
    x_pair = np.vstack([resp1, resp2])
    y_pair = np.concatenate([np.zeros(len(resp1), dtype=int), np.ones(len(resp2), dtype=int)])
    n_splits = min(N_SPLITS, len(resp1), len(resp2))
    if n_splits < 2:
        return {"Accuracy": np.nan}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_index, test_index in cv.split(x_pair, y_pair):
        model = LinearDiscriminantAnalysis()
        model.fit(x_pair[train_index], y_pair[train_index])
        scores.append(float(model.score(x_pair[test_index], y_pair[test_index])))
    return {"Accuracy": float(np.mean(scores))}


def main() -> None:
    """Run LDA discriminability and save the long-form pairwise results."""
    output_dir = get_method_dir("lda")
    results_df = run_pairwise_analysis(lda_accuracy)
    results_df.to_csv(output_dir / "pairwise_results.csv", index=False)


if __name__ == "__main__":
    main()
