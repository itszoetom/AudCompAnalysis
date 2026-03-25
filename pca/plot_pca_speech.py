"""Speech-specific PCA plots with FT and VOT colorings."""

import matplotlib.pyplot as plt

try:
    from .pca_analysis import (
        WINDOW_ORDER,
        DEFAULT_SCATTER_KWARGS,
        apply_figure_style,
        build_population_dataset,
        compute_pca_summary,
        format_panel_title,
        get_brain_regions,
        get_figure_dir,
        make_population_figure,
    )
except ImportError:
    from pca_analysis import (
        WINDOW_ORDER,
        DEFAULT_SCATTER_KWARGS,
        apply_figure_style,
        build_population_dataset,
        compute_pca_summary,
        format_panel_title,
        get_brain_regions,
        get_figure_dir,
        make_population_figure,
    )


def main() -> None:
    apply_figure_style()
    brain_regions = get_brain_regions("speech")
    fig, axes = make_population_figure(len(brain_regions), len(WINDOW_ORDER))
    fig.suptitle("speech PCA by FT and VOT", fontsize=16)

    plotted_panels = 0
    for row_index, brain_area in enumerate(brain_regions):
        for window_index, window_name in enumerate(WINDOW_ORDER):
            dataset = build_population_dataset("speech", window_name, brain_area)
            ft_ax = axes[row_index, 2 * window_index]
            vot_ax = axes[row_index, 2 * window_index + 1]
            if dataset is None:
                ft_ax.set_visible(False)
                vot_ax.set_visible(False)
                continue

            summary = compute_pca_summary(dataset["X"])
            scores = summary["scores"]
            explained = summary["explained_variance_ratio"]
            ft_labels = dataset["Y"][:, 0]
            vot_labels = dataset["Y"][:, 1]

            for ax, label_name, label_values in ((ft_ax, "FT", ft_labels), (vot_ax, "VOT", vot_labels)):
                scatter = ax.scatter(
                    scores[:, 0],
                    scores[:, 1],
                    c=label_values,
                    cmap="viridis",
                    **DEFAULT_SCATTER_KWARGS,
                )
                ax.set_title(f"{format_panel_title(brain_area, window_name)} {label_name}")
                ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
                ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
                plt.colorbar(scatter, ax=ax)
                plotted_panels += 1

    if plotted_panels == 0:
        plt.close(fig)
        return

    fig.savefig(get_figure_dir() / "speech_ft_vot_pca.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
