"""Create participation-ratio distributions across brain regions for each spike window."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable.
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

import params

try:
    from ..plot_stats import add_pairwise_annotations, box_centers, pairwise_group_tests
except ImportError:
    from plot_stats import add_pairwise_annotations, box_centers, pairwise_group_tests

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
    brain_regions = get_plot_brain_regions(sound_type)
    total_steps = iterations * len(brain_regions) * len(WINDOW_ORDER)
    progress = tqdm(total=total_steps, desc=f"PCA participation ratio ({sound_type})", unit="sample", dynamic_ncols=True)
    try:
        for iteration in range(iterations):
            for brain_area in brain_regions:
                for window_name in WINDOW_ORDER:
                    progress.set_postfix_str(
                        f"{params.short_names.get(brain_area, brain_area)} | {window_name}",
                        refresh=False,
                    )
                    dataset = build_sampled_dataset(
                        sound_type,
                        window_name,
                        brain_area,
                        n_neurons=get_target_neuron_count(sound_type),
                    )
                    progress.update(1)
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
    finally:
        close = getattr(progress, "close", None)
        if close is not None:
            close()
    return pd.DataFrame(records)


def main() -> None:
    """Run participation-ratio summary figures for each available sound type."""
    apply_figure_style()
    for sound_type in list_available_sound_types():
        summary_df = build_participation_frame(sound_type)
        if summary_df.empty:
            continue

        brain_regions = get_plot_brain_regions(sound_type)
        fig, axes = plt.subplots(
            1,
            len(WINDOW_ORDER),
            figsize=(4.4 * len(WINDOW_ORDER), 4.8),
            squeeze=False,
            sharey=True,
            constrained_layout=True,
        )
        fig.suptitle(f"{sound_type} participation ratio", fontsize=16, fontweight="bold")
        y_min = float(summary_df["Participation Ratio"].min())
        y_max = float(summary_df["Participation Ratio"].max())
        max_annotations = len(brain_regions) * (len(brain_regions) - 1) // 2
        y_step = 0.035 * ((y_max - y_min) if y_max > y_min else 1.0)

        for col_index, window_name in enumerate(WINDOW_ORDER):
            ax = axes[0, col_index]
            panel_df = summary_df[summary_df["Window"] == window_name].copy()
            sns.boxplot(
                data=panel_df,
                x="Brain Area",
                y="Participation Ratio",
                order=brain_regions,
                width=0.5,
                fliersize=2,
                linewidth=1,
                ax=ax,
            )
            sns.stripplot(
                data=panel_df,
                x="Brain Area",
                y="Participation Ratio",
                order=brain_regions,
                color="black",
                size=3,
                alpha=0.35,
                ax=ax,
            )

            stats_df = pairwise_group_tests(
                panel_df,
                group_col="Brain Area",
                value_col="Participation Ratio",
                group_order=brain_regions,
                pair_cols=["Iteration"],
            )
            add_pairwise_annotations(
                ax,
                stats_df,
                centers=box_centers(brain_regions),
                data_max=y_max,
                data_min=y_min,
            )

            ax.set_title(window_name.capitalize(), fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Participation Ratio" if col_index == 0 else "")
            ax.set_xticklabels([params.short_names.get(region, region) for region in brain_regions], rotation=20)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
            ax.set_ylim(y_min - y_step, y_max + y_step * (max_annotations + 2))

        fig.savefig(get_figure_dir() / f"{sound_type}_participation_ratio.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
