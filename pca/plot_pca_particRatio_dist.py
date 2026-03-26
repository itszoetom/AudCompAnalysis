"""Create participation-ratio distributions for each sound type across brain regions and spike windows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

try:
    from .pca_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        build_sampled_dataset,
        compute_pca_summary,
        get_figure_dir,
        get_plot_brain_regions,
        get_target_neuron_count,
        list_available_sound_types,
    )
except ImportError:
    from pca_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        build_sampled_dataset,
        compute_pca_summary,
        get_figure_dir,
        get_plot_brain_regions,
        get_target_neuron_count,
        list_available_sound_types,
    )


def build_participation_frame(sound_type: str, iterations: int = 30) -> pd.DataFrame:
    """Return repeated participation-ratio measurements for one sound type."""
    records = []
    for iteration in range(iterations):
        for brain_area in get_plot_brain_regions(sound_type):
            for window_name in WINDOW_ORDER:
                dataset = build_sampled_dataset(
                    sound_type,
                    window_name,
                    brain_area,
                    n_neurons=get_target_neuron_count(sound_type),
                )
                if dataset is None:
                    continue
                summary = compute_pca_summary(dataset["X"])
                records.append(
                    {
                        "Brain Area": brain_area,
                        "Window": window_name,
                        "Participation Ratio": summary["participation_ratio"],
                        "Iteration": iteration,
                    }
                )
    return pd.DataFrame(records)


def add_significance_labels(ax, panel_df: pd.DataFrame) -> None:
    """Add simple Bonferroni-corrected significance markers above region comparisons."""
    brain_regions = list(panel_df["Brain Area"].unique())
    tests = []
    for left_index, left_region in enumerate(brain_regions):
        for right_index in range(left_index + 1, len(brain_regions)):
            right_region = brain_regions[right_index]
            left_values = panel_df.loc[panel_df["Brain Area"] == left_region, "Participation Ratio"]
            right_values = panel_df.loc[panel_df["Brain Area"] == right_region, "Participation Ratio"]
            if len(left_values) < 2 or len(right_values) < 2:
                continue
            _, p_value = mannwhitneyu(left_values, right_values, alternative="two-sided")
            tests.append((left_index, right_index, p_value))
    if not tests:
        return
    corrected = multipletests([p_value for _, _, p_value in tests], method="bonferroni")[1]
    y_max = panel_df["Participation Ratio"].max()
    for offset, ((left_index, right_index, _), p_value) in enumerate(zip(tests, corrected)):
        if p_value >= 0.05:
            continue
        star_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        height = y_max + 0.05 * (offset + 1)
        ax.plot([left_index, left_index, right_index, right_index], [height, height + 0.02, height + 0.02, height], c="black", lw=1)
        ax.text((left_index + right_index) / 2, height + 0.02, star_text, ha="center", va="bottom", fontsize=10)


def main() -> None:
    """Run participation-ratio summary figures for each available sound type."""
    apply_figure_style()
    for sound_type in list_available_sound_types():
        summary_df = build_participation_frame(sound_type)
        if summary_df.empty:
            continue
        fig, axes = plt.subplots(1, len(WINDOW_ORDER), figsize=(4.0 * len(WINDOW_ORDER), 4.0), squeeze=False, constrained_layout=True)
        fig.suptitle(f"{sound_type} participation ratio", fontsize=16, fontweight="bold")
        for col_index, window_name in enumerate(WINDOW_ORDER):
            ax = axes[0, col_index]
            panel_df = summary_df[summary_df["Window"] == window_name]
            sns.boxplot(data=panel_df, x="Brain Area", y="Participation Ratio", fliersize=2, linewidth=1, ax=ax)
            sns.stripplot(data=panel_df, x="Brain Area", y="Participation Ratio", color="black", size=3, alpha=0.35, ax=ax)
            ax.set_title(window_name.capitalize(), fontweight="bold")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
            add_significance_labels(ax, panel_df)
        fig.savefig(get_figure_dir() / f"{sound_type}_participation_ratio.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
