"""Create per-session ridge R2 distributions across brain regions for each spike window."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import params

try:
    from ..plot_stats import add_pairwise_annotations, box_centers, pairwise_group_tests
except ImportError:
    from plot_stats import add_pairwise_annotations, box_centers, pairwise_group_tests

try:
    from methods.methods_analysis import WINDOW_ORDER, available_mice, available_sessions, build_dataset, get_brain_regions
except ImportError:
    from methods_analysis import WINDOW_ORDER, available_mice, available_sessions, build_dataset, get_brain_regions

try:
    from .ridge_analysis import apply_figure_style, get_output_dir
except ImportError:
    from ridge_analysis import apply_figure_style, get_output_dir

ALPHAS = np.logspace(-3, 3, 20)


def score_session_dataset(x: np.ndarray, y: np.ndarray, log_target: bool) -> float:
    """Return the mean 5-fold ridge R2 for one neuron-subsampled session dataset."""
    target = np.log10(y + 1e-8) if log_target else y
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in splitter.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = target[train_index], target[test_index]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        ridge = RidgeCV(alphas=ALPHAS)
        ridge.fit(x_train, y_train)
        scores.append(r2_score(y_test, ridge.predict(x_test)))
    return float(np.mean(scores))


def build_session_frame(sound_type: str, n_neurons: int = 30, n_subsamples: int = 100) -> pd.DataFrame:
    """Return repeated per-session ridge scores for one sound type."""
    records = []
    for mouse_id in available_mice(sound_type):
        for session_id in available_sessions(sound_type, mouse_id=mouse_id):
            for brain_area in get_brain_regions(sound_type):
                if sound_type == "speech" and brain_area == "Dorsal auditory area":
                    continue
                for window_name in WINDOW_ORDER:
                    dataset = build_dataset(sound_type, window_name, brain_area, mouse_id=mouse_id, session_id=session_id)
                    if dataset is None or dataset["X"].shape[1] < n_neurons:
                        continue
                    if sound_type == "speech":
                        targets = [("FT", dataset["Y"][:, 0], False), ("VOT", dataset["Y"][:, 1], False)]
                    else:
                        targets = [(sound_type, dataset["Y"], sound_type in {"AM", "PT"})]
                    rng = np.random.default_rng(42)
                    for target_name, target_values, log_target in targets:
                        session_scores = []
                        for _ in range(n_subsamples):
                            neuron_idx = rng.choice(dataset["X"].shape[1], size=n_neurons, replace=False)
                            session_scores.append(score_session_dataset(dataset["X"][:, neuron_idx], target_values, log_target))
                        records.append(
                            {
                                "Mouse": mouse_id,
                                "Session": session_id,
                                "Brain Area": brain_area,
                                "Window": window_name,
                                "Target": target_name,
                                "R2 Test": float(np.mean(session_scores)),
                            }
                        )
    return pd.DataFrame(records)


def main() -> None:
    """Run per-session ridge boxplot figures."""
    apply_figure_style()
    for sound_type in ("speech", "AM", "PT", "naturalSound"):
        results_df = build_session_frame(sound_type)
        if results_df.empty:
            continue

        brain_regions = [region for region in params.targetSiteNames if region in results_df["Brain Area"].unique()]
        target_order = [target for target in ["FT", "VOT", sound_type] if target in results_df["Target"].unique()]
        use_hue = len(target_order) > 1
        fig, axes = plt.subplots(
            1,
            len(WINDOW_ORDER),
            figsize=(4.4 * len(WINDOW_ORDER), 4.8),
            squeeze=False,
            sharey=True,
            constrained_layout=True,
        )
        fig.suptitle(f"{sound_type} per-session ridge $R^2$", fontsize=16, fontweight="bold")
        y_min = float(results_df["R2 Test"].min())
        y_max = float(results_df["R2 Test"].max())
        max_annotations = len(target_order) * (len(brain_regions) * (len(brain_regions) - 1) // 2)
        y_step = 0.035 * ((y_max - y_min) if y_max > y_min else 1.0)

        for col_index, window_name in enumerate(WINDOW_ORDER):
            ax = axes[0, col_index]
            panel_df = results_df[results_df["Window"] == window_name].copy()
            if panel_df.empty:
                ax.axis("off")
                continue

            box_kwargs = dict(
                data=panel_df,
                x="Brain Area",
                y="R2 Test",
                order=brain_regions,
                width=0.5,
                fliersize=2,
                linewidth=1,
                ax=ax,
            )
            strip_kwargs = dict(
                data=panel_df,
                x="Brain Area",
                y="R2 Test",
                order=brain_regions,
                dodge=use_hue,
                size=3,
                alpha=0.35,
                ax=ax,
            )
            if use_hue:
                box_kwargs["hue"] = "Target"
                box_kwargs["hue_order"] = target_order
                strip_kwargs["hue"] = "Target"
                strip_kwargs["hue_order"] = target_order
            sns.boxplot(**box_kwargs)
            sns.stripplot(color="black", **strip_kwargs)
            if ax.legend_ is not None:
                ax.legend_.remove()

            stats_df = pairwise_group_tests(
                panel_df,
                group_col="Brain Area",
                value_col="R2 Test",
                group_order=brain_regions,
                hue_col="Target" if use_hue else None,
                hue_order=target_order if use_hue else None,
                pair_cols=["Mouse", "Session"],
            )
            add_pairwise_annotations(
                ax,
                stats_df,
                centers=box_centers(brain_regions, hue_levels=target_order if use_hue else None),
                data_max=y_max,
                data_min=y_min,
            )

            ax.set_title(window_name.capitalize(), fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("$R^2$" if col_index == 0 else "")
            ax.set_xticklabels([params.short_names.get(region, region) for region in brain_regions], rotation=20)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
            ax.set_ylim(y_min - y_step, y_max + y_step * (max_annotations + 2))

        if use_hue:
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles[: len(target_order)], labels[: len(target_order)], title="Target", loc="upper right", frameon=False)
        fig.savefig(get_output_dir() / f"{sound_type}_ridge_per_session.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
