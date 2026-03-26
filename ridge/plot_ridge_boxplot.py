"""Create population ridge R2 distributions for each sound type across regions and spike windows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .ridge_analysis import WINDOW_ORDER, apply_figure_style, get_output_dir, run_population_ridge
except ImportError:
    from ridge_analysis import WINDOW_ORDER, apply_figure_style, get_output_dir, run_population_ridge

import params


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
        brain_regions = get_plot_brain_regions(sound_type, results_df)
        fig, axes = plt.subplots(
            len(brain_regions),
            len(WINDOW_ORDER),
            figsize=(4.0 * len(WINDOW_ORDER), 3.4 * len(brain_regions)),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(f"{sound_type} population ridge $R^2$", fontsize=16, fontweight="bold")
        sound_df = results_df[results_df["Sound Type"] == sound_type]
        for row_index, brain_area in enumerate(brain_regions):
            for col_index, window_name in enumerate(WINDOW_ORDER):
                ax = axes[row_index, col_index]
                panel_df = sound_df[(sound_df["Brain Area"] == brain_area) & (sound_df["Window"] == window_name)]
                if panel_df.empty:
                    ax.axis("off")
                    continue
                sns.boxplot(data=panel_df, x="Target", y="R2 Test", fliersize=2, linewidth=1, ax=ax)
                sns.stripplot(data=panel_df, x="Target", y="R2 Test", color="black", size=3, alpha=0.35, ax=ax)
                ax.set_title(f"{params.short_names[brain_area]}\n{window_name.capitalize()}", fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("$R^2$")
                ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
        fig.savefig(get_output_dir() / f"{sound_type}_ridge_boxplots.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
