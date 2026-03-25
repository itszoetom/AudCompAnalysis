"""Participation-ratio summary plots from population PCA datasets."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from .pca_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        compute_pca_summary,
        get_figure_dir,
        get_condition_color,
        iter_population_datasets,
        list_available_sound_types,
    )
except ImportError:
    from pca_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        compute_pca_summary,
        get_figure_dir,
        get_condition_color,
        iter_population_datasets,
        list_available_sound_types,
    )


def build_participation_ratio_frame() -> pd.DataFrame:
    records = []
    for dataset in iter_population_datasets(sound_types=list_available_sound_types(), windows=WINDOW_ORDER):
        summary = compute_pca_summary(dataset["X"])
        records.append(
            {
                "Brain Area": dataset["brain_area"],
                "Sound Type": dataset["sound_type"],
                "Window": dataset["window_name"],
                "Participation Ratio": summary["participation_ratio"],
                "Neurons": summary["n_neurons"],
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    apply_figure_style()
    summary_df = build_participation_ratio_frame()
    if summary_df.empty:
        return

    fig, axes = plt.subplots(
        len(WINDOW_ORDER),
        1,
        figsize=(9, 3.3 * len(WINDOW_ORDER)),
        squeeze=False,
        constrained_layout=True,
    )
    palette = {
        sound: get_condition_color(sound, "Primary auditory area")
        for sound in summary_df["Sound Type"].unique()
    }
    for row_index, window_name in enumerate(WINDOW_ORDER):
        window_df = summary_df[summary_df["Window"] == window_name]
        sns.boxplot(
            data=window_df,
            x="Brain Area",
            y="Participation Ratio",
            hue="Sound Type",
            palette=palette,
            linewidth=1,
            fliersize=2,
            ax=axes[row_index, 0],
        )
        sns.stripplot(
            data=window_df,
            x="Brain Area",
            y="Participation Ratio",
            hue="Sound Type",
            dodge=True,
            palette=palette,
            size=3,
            alpha=0.45,
            legend=False,
            ax=axes[row_index, 0],
        )
        axes[row_index, 0].set_title(f"Participation ratio ({window_name.capitalize()})")
        axes[row_index, 0].set_xlabel("")
        axes[row_index, 0].tick_params(axis="x", rotation=20)
        axes[row_index, 0].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

    fig.savefig(get_figure_dir() / "participation_ratio_summary.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
