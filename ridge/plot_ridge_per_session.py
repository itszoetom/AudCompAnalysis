"""Create per-session ridge R2 distributions across brain regions for each spike window."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from shared import params
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
        funcs,
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
        funcs,
        get_plot_brain_regions,
        list_available_sound_types,
        plot_ridge_summary,
        WINDOW_ORDER,
    )


def _eligible_sessions_for_area(
    sound_type: str,
    brain_area: str,
    n_neurons: int,
) -> list[tuple[str, str]]:
    """Return sessions in one area that have enough neurons for per-session ridge."""
    eligible_sessions = []
    for mouse_id in available_mice(sound_type):
        for session_id in available_sessions(sound_type, mouse_id=mouse_id):
            dataset = build_dataset(sound_type, WINDOW_ORDER[0], brain_area, mouse_id=mouse_id, session_id=session_id)
            if dataset is not None and dataset["X"].shape[1] >= n_neurons:
                eligible_sessions.append((mouse_id, session_id))
    return eligible_sessions


def collect_eligible_sessions(
    sound_type: str,
    n_neurons: int,
    *,
    seed: int | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Return all eligible sessions for each plotted brain area."""
    brain_regions = get_plot_brain_regions(sound_type)
    _ = seed
    return {
        brain_area: _eligible_sessions_for_area(sound_type, brain_area, n_neurons)
        for brain_area in brain_regions
    }


def build_session_frame(
    sound_type: str,
    n_neurons: int | None = None,
    n_subsamples: int = 100,
    selected_sessions: dict[str, Iterable[tuple[str, str]]] | None = None,
) -> pd.DataFrame:
    """Return one mean per-session ridge score after eligible-session and neuron subsampling."""
    n_neurons = params.NEURONS_PER_SESSION[sound_type] if n_neurons is None else n_neurons
    records = []
    eligible_sessions = selected_sessions or collect_eligible_sessions(sound_type, n_neurons)
    conditions = [
        (mouse_id, session_id, brain_area, window_name)
        for brain_area in get_plot_brain_regions(sound_type)
        for mouse_id, session_id in eligible_sessions.get(brain_area, [])
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
        if dataset["X"].shape[0] < 5:
            print(
                f"[per-session ridge] skipping {sound_type} {brain_area} {mouse_id} {session_id} {window_name}: "
                f"only {dataset['X'].shape[0]} trials available"
            )
            continue
        x_standardized = StandardScaler().fit_transform(dataset["X"])
        rng = np.random.default_rng(42)
        target_datasets = (
            [
                {
                    "target_name": "Speech Tuple",
                    "Y": funcs.labels_for_sound("speech", dataset["Y"]).astype(float),
                    "log_target": False,
                }
            ]
            if sound_type == "speech"
            else build_target_datasets(dataset)
        )
        for target_dataset in target_datasets:
            y = np.asarray(target_dataset["Y"], dtype=float).copy()

            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores_by_subsample: list[list[float]] = []
            fold_rmses_by_subsample: list[list[float]] = []
            fold_alphas_by_subsample: list[list[float]] = []
            for iteration in range(n_subsamples):
                neuron_idx = rng.choice(dataset["X"].shape[1], size=n_neurons, replace=False)
                x_sampled = x_standardized[:, neuron_idx]
                if len(y) < 5:
                    print(
                        f"[per-session ridge] warning: {sound_type} {brain_area} {mouse_id} {session_id} "
                        f"{window_name} target={target_dataset['target_name']} has only {len(y)} trials"
                    )
                fold_scores = []
                fold_rmses = []
                fold_alphas = []
                for fold_index, (train_index, test_index) in enumerate(kfold.split(x_sampled)):
                    if len(train_index) < 5:
                        print(
                            f"[per-session ridge] warning: inner fold below 5 training samples for "
                            f"{sound_type} {brain_area} {mouse_id} {session_id} {window_name}"
                        )
                    ridge = RidgeCV(
                        alphas=np.asarray(RIDGE_ALPHAS, dtype=float),
                        cv=funcs._ridge_inner_cv(len(train_index), 42 + fold_index),
                        scoring="r2",
                    )
                    ridge.fit(x_sampled[train_index], y[train_index])
                    y_pred = ridge.predict(x_sampled[test_index])
                    fold_scores.append(float(funcs.r2_score(y[test_index], y_pred)))
                    fold_rmses.append(float(np.sqrt(funcs.mean_squared_error(y[test_index], y_pred))))
                    fold_alphas.append(float(ridge.alpha_))
                fold_scores_by_subsample.append(fold_scores)
                fold_rmses_by_subsample.append(fold_rmses)
                fold_alphas_by_subsample.append(fold_alphas)

            mean_fold_scores = np.mean(np.asarray(fold_scores_by_subsample, dtype=float), axis=0)
            mean_fold_rmses = np.mean(np.asarray(fold_rmses_by_subsample, dtype=float), axis=0)
            mean_fold_alphas = np.mean(np.asarray(fold_alphas_by_subsample, dtype=float), axis=0)
            records.append(
                {
                    "Mouse": mouse_id,
                    "Session": session_id,
                    "Brain Area": brain_area,
                    "Window": window_name,
                    "Target": target_dataset["target_name"],
                    "R2 Test": float(np.mean(mean_fold_scores)),
                    "RMSE": float(np.mean(mean_fold_rmses)),
                    "Best Alpha": float(np.mean(mean_fold_alphas)),
                    "Neurons": int(n_neurons),
                    "Trials": int(len(y)),
                }
            )
    return pd.DataFrame(records)


def plot_ridge_alpha_summary(
    sound_type: str,
    results_df: pd.DataFrame,
    *,
    session_counts: dict[str, int],
) -> None:
    """Save one appendix-style alpha summary figure for per-session ridge."""
    if results_df.empty:
        return

    brain_regions = get_plot_brain_regions(sound_type)
    fig, axes = plt.subplots(
        1,
        len(WINDOW_ORDER),
        figsize=(3.8 * len(WINDOW_ORDER), 4.2),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    fig.suptitle(f"{sound_type} per-session ridge selected alpha", fontsize=26, fontweight="bold")
    region_palette = sns.color_palette("viridis", n_colors=len(brain_regions))

    for col_index, window_name in enumerate(WINDOW_ORDER):
        ax = axes[0, col_index]
        panel_df = results_df[results_df["Window"] == window_name].copy()
        if panel_df.empty:
            ax.axis("off")
            continue

        sns.boxplot(
            data=panel_df,
            x="Brain Area",
            y="Best Alpha",
            order=brain_regions,
            width=0.5,
            fliersize=2,
            linewidth=1,
            palette=region_palette,
            ax=ax,
        )
        sns.stripplot(
            data=panel_df,
            x="Brain Area",
            y="Best Alpha",
            order=brain_regions,
            color="black",
            alpha=0.35,
            size=3,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_title(window_name.capitalize(), fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Selected alpha" if col_index == 0 else "")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
        ax.set_xticklabels(
            [
                f"{params.short_names.get(region, region)}\n(n={session_counts.get(region, 0)})"
                for region in brain_regions
            ],
            rotation=20,
        )

    fig.savefig(funcs.get_figure_dir("decoding/ridge") / f"{sound_type}_ridge_per_session_alpha.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Run per-session ridge boxplot figures."""
    apply_figure_style()
    for sound_type in tqdm(list_available_sound_types(), desc="Per-session ridge plots", unit="sound", dynamic_ncols=True):
        print(f"Running per-session ridge plots for {sound_type}...")
        n_neurons = params.NEURONS_PER_SESSION[sound_type]
        selected_sessions = collect_eligible_sessions(sound_type, n_neurons)
        results_df = build_session_frame(sound_type, n_neurons=n_neurons, selected_sessions=selected_sessions)
        if results_df.empty:
            continue
        session_counts = {brain_area: len(list(sessions)) for brain_area, sessions in selected_sessions.items()}
        plot_ridge_summary(
            sound_type,
            results_df,
            title=f"{sound_type} per-session ridge $R^2$",
            filename=f"{sound_type}_ridge_per_session.png",
            pair_cols=["Mouse", "Session"],
            session_counts=session_counts,
        )
        plot_ridge_alpha_summary(sound_type, results_df, session_counts=session_counts)


if __name__ == "__main__":
    main()
