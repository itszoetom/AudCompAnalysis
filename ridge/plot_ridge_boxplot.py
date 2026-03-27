"""Create population ridge R2 distributions across brain regions for each spike window."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

import params

try:
    from ..plot_stats import add_pairwise_annotations, box_centers, pairwise_group_tests
except ImportError:
    from plot_stats import add_pairwise_annotations, box_centers, pairwise_group_tests

try:
    from .ridge_analysis import WINDOW_ORDER, apply_figure_style, get_output_dir, run_population_ridge
except ImportError:
    from ridge_analysis import WINDOW_ORDER, apply_figure_style, get_output_dir, run_population_ridge


def get_plot_brain_regions(sound_type: str, results_df) -> list[str]:
    """Return the ordered brain regions to include in one ridge figure."""
    regions = sorted(results_df.loc[results_df["Sound Type"] == sound_type, "Brain Area"].unique().tolist())
    if sound_type == "speech":
        regions = [region for region in regions if region != "Dorsal auditory area"]
    return [region for region in params.targetSiteNames if region in regions]


def main() -> None:
    """Run population ridge boxplot figures."""
    apply_figure_style()
    results_df = run_population_ridge(iterations=30)
    if results_df.empty:
        return

    for sound_type in results_df["Sound Type"].unique():
        sound_df = results_df[results_df["Sound Type"] == sound_type].copy()
        brain_regions = get_plot_brain_regions(sound_type, sound_df)
        target_order = [target for target in ["FT", "VOT", sound_type] if target in sound_df["Target"].unique()]
        use_hue = len(target_order) > 1
        fig, axes = plt.subplots(
            1,
            len(WINDOW_ORDER),
            figsize=(4.4 * len(WINDOW_ORDER), 4.8),
            squeeze=False,
            sharey=True,
            constrained_layout=True,
        )
        fig.suptitle(f"{sound_type} population ridge $R^2$", fontsize=16, fontweight="bold")
        y_min = float(sound_df["R2 Test"].min())
        y_max = float(sound_df["R2 Test"].max())
        max_annotations = len(target_order) * (len(brain_regions) * (len(brain_regions) - 1) // 2)
        y_step = 0.035 * ((y_max - y_min) if y_max > y_min else 1.0)

        for col_index, window_name in enumerate(WINDOW_ORDER):
            ax = axes[0, col_index]
            panel_df = sound_df[sound_df["Window"] == window_name].copy()
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
                pair_cols=["Iteration"],
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
        fig.savefig(get_output_dir() / f"{sound_type}_ridge_boxplots.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
