"""Population PCA plots from preprocessed `.npz` data."""

import matplotlib.pyplot as plt

try:
    from .pca_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        build_population_dataset,
        compute_pca_summary,
        format_panel_title,
        get_brain_regions,
        get_figure_dir,
        get_sound_min_neuron_count,
        get_sound_scree_ymax,
        list_available_sound_types,
        make_population_figure,
        plot_pca_scatter,
        plot_scree,
    )
except ImportError:
    from pca_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        build_population_dataset,
        compute_pca_summary,
        format_panel_title,
        get_brain_regions,
        get_figure_dir,
        get_sound_min_neuron_count,
        get_sound_scree_ymax,
        list_available_sound_types,
        make_population_figure,
        plot_pca_scatter,
        plot_scree,
    )


def main() -> None:
    apply_figure_style()
    for sound_type in list_available_sound_types():
        brain_regions = get_brain_regions(sound_type)
        fig, axes = make_population_figure(len(brain_regions), len(WINDOW_ORDER))
        min_neurons = get_sound_min_neuron_count(sound_type)
        fig.suptitle(
            f"{sound_type} population PCA (matched to {min_neurons} neurons per region)",
            fontsize=16,
        )
        scree_ymax = get_sound_scree_ymax(sound_type)

        plotted_panels = 0
        for row_index, brain_area in enumerate(brain_regions):
            for window_index, window_name in enumerate(WINDOW_ORDER):
                dataset = build_population_dataset(sound_type, window_name, brain_area)
                scree_ax = axes[row_index, 2 * window_index]
                scatter_ax = axes[row_index, 2 * window_index + 1]
                if dataset is None:
                    scree_ax.set_visible(False)
                    scatter_ax.set_visible(False)
                    continue

                summary = compute_pca_summary(dataset["X"])
                title = format_panel_title(brain_area, window_name)
                plot_scree(scree_ax, summary, title, y_max=scree_ymax)
                plot_pca_scatter(scatter_ax, dataset, summary, title)
                plotted_panels += 2

        if plotted_panels == 0:
            plt.close(fig)
            continue

        fig.savefig(get_figure_dir() / f"{sound_type}_population_pca.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
