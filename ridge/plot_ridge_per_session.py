"""Create per-session ridge R2 distributions across brain regions for each spike window."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import params
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

try:
    from .ridge_analysis import (
        RIDGE_ALPHAS,
        apply_figure_style,
        available_mice,
        available_sessions,
        build_dataset,
        build_target_datasets,
        get_plot_brain_regions,
        list_available_sound_types,
        plot_ridge_summary,
        WINDOW_ORDER,
    )
except ImportError:
    from ridge_analysis import (
        RIDGE_ALPHAS,
        apply_figure_style,
        available_mice,
        available_sessions,
        build_dataset,
        build_target_datasets,
        get_plot_brain_regions,
        list_available_sound_types,
        plot_ridge_summary,
        WINDOW_ORDER,
    )


def build_session_frame(sound_type: str, n_neurons: int | None = None, n_subsamples: int = 100) -> pd.DataFrame:
    """Return one per-session ridge score per target after repeated neuron subsampling."""
    n_neurons = params.NEURONS_PER_SESSION[sound_type] if n_neurons is None else n_neurons
    records = []
    conditions = [
        (mouse_id, session_id, brain_area, window_name)
        for mouse_id in available_mice(sound_type)
        for session_id in available_sessions(sound_type, mouse_id=mouse_id)
        for brain_area in get_plot_brain_regions(sound_type)
        for window_name in WINDOW_ORDER
    ]
    for mouse_id, session_id, brain_area, window_name in tqdm(
        conditions,
        desc=f"Per-session ridge ({sound_type})",
        unit="dataset",
        dynamic_ncols=True,
    ):
        dataset = build_dataset(sound_type, window_name, brain_area, mouse_id=mouse_id, session_id=session_id)
        if dataset is None or dataset["X"].shape[1] < n_neurons:
            continue
        x_standardized = StandardScaler().fit_transform(dataset["X"])
        rng = np.random.default_rng(42)
        for target_dataset in build_target_datasets(dataset):
            y = target_dataset["Y"].copy()
            if bool(target_dataset["log_target"]):
                y = np.log(y + 1e-8)

            subsample_scores = []
            for iteration in range(n_subsamples):
                neuron_idx = rng.choice(dataset["X"].shape[1], size=n_neurons, replace=False)
                x_sampled = x_standardized[:, neuron_idx]
                kfold = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=42 + iteration)
                fold_scores = []
                for train_index, test_index in kfold.split(x_sampled):
                    ridge = RidgeCV(alphas=np.asarray(RIDGE_ALPHAS, dtype=float))
                    ridge.fit(x_sampled[train_index], y[train_index])
                    y_pred = ridge.predict(x_sampled[test_index])
                    fold_scores.append(float(r2_score(y[test_index], y_pred)))
                subsample_scores.append(float(np.mean(fold_scores)))
            records.append(
                {
                    "Mouse": mouse_id,
                    "Session": session_id,
                    "Brain Area": brain_area,
                    "Window": window_name,
                    "Target": target_dataset["target_name"],
                    "R2 Test": float(np.mean(subsample_scores)),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    """Run per-session ridge boxplot figures."""
    apply_figure_style()
    for sound_type in tqdm(list_available_sound_types(), desc="Per-session ridge plots", unit="sound", dynamic_ncols=True):
        print(f"Running per-session ridge plots for {sound_type}...")
        results_df = build_session_frame(sound_type)
        if results_df.empty:
            continue
        plot_ridge_summary(
            sound_type,
            results_df,
            title=f"{sound_type} per-session ridge $R^2$",
            filename=f"{sound_type}_ridge_per_session.png",
            pair_cols=["Mouse", "Session"],
        )


if __name__ == "__main__":
    main()
