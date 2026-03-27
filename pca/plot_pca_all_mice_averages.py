"""Create trial-averaged population PCA figures for each sound type across regions and windows."""

from __future__ import annotations

import matplotlib.pyplot as plt

try:
    from .pca_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        average_trials_by_stimulus,
        build_sampled_dataset,
        compute_pca_summary,
        format_panel_title,
        get_figure_dir,
        get_plot_brain_regions,
        get_target_neuron_count,
        labels_for_sound,
        list_available_sound_types,
        stimulus_tick_labels,
    )
except ImportError:
    from pca_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        average_trials_by_stimulus,
        build_sampled_dataset,
        compute_pca_summary,
        format_panel_title,
        get_figure_dir,
        get_plot_brain_regions,
        get_target_neuron_count,
        labels_for_sound,
        list_available_sound_types,
        stimulus_tick_labels,
    )


def main() -> None:
    """Run trial-averaged PCA scatter plots for each sound type."""
    apply_figure_style()
    for sound_type in list_available_sound_types():
        brain_regions = get_plot_brain_regions(sound_type)
        target_neurons = get_target_neuron_count(sound_type)
        fig, axes = plt.subplots(
            len(brain_regions),
            len(WINDOW_ORDER),
            figsize=(4.0 * len(WINDOW_ORDER), 3.5 * len(brain_regions)),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(f"{sound_type} averaged PCA (n={target_neurons} neurons per region)", fontsize=16, fontweight="bold")
        last_scatter = None

        for row_index, brain_area in enumerate(brain_regions):
            for col_index, window_name in enumerate(WINDOW_ORDER):
                ax = axes[row_index, col_index]
                dataset = build_sampled_dataset(sound_type, window_name, brain_area, n_neurons=target_neurons)
                if dataset is None:
                    ax.axis("off")
                    continue
                averaged_dataset = average_trials_by_stimulus(dataset)
                summary = compute_pca_summary(averaged_dataset["X"])
                scores = summary["scores"]
                explained = summary["explained_variance_ratio"]
                last_scatter = ax.scatter(
                    scores[:, 0],
                    scores[:, 1],
                    c=labels_for_sound(sound_type, averaged_dataset["Y"]),
                    cmap="viridis",
                    s=36,
                    alpha=0.9,
                    linewidths=0,
                )
                ax.set_title(format_panel_title(brain_area, window_name), fontweight="bold")
                ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
                ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

        if last_scatter is not None:
            colorbar = fig.colorbar(last_scatter, ax=fig.axes, location="bottom", fraction=0.03, pad=0.04)
            stim_array = averaged_dataset["Y"]
            colorbar.set_ticks(labels_for_sound(sound_type, stim_array))
            colorbar.set_ticklabels(stimulus_tick_labels(sound_type, stim_array))
            colorbar.ax.tick_params(labelsize=8, rotation=35)
            colorbar.set_label("Stimulus", fontsize=12)

        fig.savefig(get_figure_dir() / f"{sound_type}_population_pca_averaged.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
