"""Create population PCA scatter and scree figures for each sound type across regions and windows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
    from .pca_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        build_sampled_dataset,
        compute_pca_summary,
        format_panel_title,
        get_figure_dir,
        get_plot_brain_regions,
        get_target_neuron_count,
        labels_for_sound,
        list_available_sound_types,
        plot_scree,
        stimulus_tick_labels,
    )
except ImportError:
    from pca_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        build_sampled_dataset,
        compute_pca_summary,
        format_panel_title,
        get_figure_dir,
        get_plot_brain_regions,
        get_target_neuron_count,
        labels_for_sound,
        list_available_sound_types,
        plot_scree,
        stimulus_tick_labels,
    )


def collect_sound_results(sound_type: str) -> tuple[dict[tuple[str, str], dict[str, np.ndarray]], int]:
    """Collect sampled datasets and PCA summaries for one sound figure."""
    target_neurons = get_target_neuron_count(sound_type)
    results = {}
    for brain_area in get_plot_brain_regions(sound_type):
        for window_name in WINDOW_ORDER:
            dataset = build_sampled_dataset(sound_type, window_name, brain_area, n_neurons=target_neurons)
            if dataset is None:
                continue
            results[(brain_area, window_name)] = {
                "dataset": dataset,
                "summary": compute_pca_summary(dataset["X"]),
            }
    return results, target_neurons


def shared_scatter_limits(results: dict[tuple[str, str], dict[str, np.ndarray]]) -> tuple[float, float, float, float]:
    """Compute common PC axis limits for one figure."""
    x_values = []
    y_values = []
    for panel in results.values():
        scores = panel["summary"]["scores"]
        x_values.append(scores[:, 0])
        y_values.append(scores[:, 1])
    x_min = min(values.min() for values in x_values)
    x_max = max(values.max() for values in x_values)
    y_min = min(values.min() for values in y_values)
    y_max = max(values.max() for values in y_values)
    return x_min, x_max, y_min, y_max


def add_shared_colorbar(fig: plt.Figure, scatter, sound_type: str, stim_array: np.ndarray) -> None:
    """Add one shared stimulus colorbar for a sound figure."""
    colorbar = fig.colorbar(scatter, ax=fig.axes, location="bottom", fraction=0.03, pad=0.04)
    color_values = labels_for_sound(sound_type, stim_array)
    unique_values = np.unique(color_values)
    colorbar.set_ticks(unique_values)
    colorbar.set_ticklabels(stimulus_tick_labels(sound_type, stim_array))
    colorbar.ax.tick_params(labelsize=8, rotation=35)
    colorbar.set_label("Stimulus", fontsize=12)


def save_scatter_figure(sound_type: str, results: dict[tuple[str, str], dict[str, np.ndarray]], target_neurons: int) -> None:
    """Save the PC1-versus-PC2 scatter figure for one sound type."""
    brain_regions = get_plot_brain_regions(sound_type)
    fig, axes = plt.subplots(
        len(brain_regions),
        len(WINDOW_ORDER),
        figsize=(4.0 * len(WINDOW_ORDER), 3.5 * len(brain_regions)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(f"{sound_type} population PCA (n={target_neurons} neurons per region)", fontsize=16, fontweight="bold")
    x_min, x_max, y_min, y_max = shared_scatter_limits(results)
    last_scatter = None

    for row_index, brain_area in enumerate(brain_regions):
        for col_index, window_name in enumerate(WINDOW_ORDER):
            ax = axes[row_index, col_index]
            panel = results.get((brain_area, window_name))
            if panel is None:
                ax.axis("off")
                continue
            dataset = panel["dataset"]
            summary = panel["summary"]
            scores = summary["scores"]
            explained = summary["explained_variance_ratio"]
            last_scatter = ax.scatter(
                scores[:, 0],
                scores[:, 1],
                c=labels_for_sound(sound_type, dataset["Y"]),
                cmap="viridis",
                s=24,
                alpha=0.85,
                linewidths=0,
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(format_panel_title(brain_area, window_name), fontweight="bold")
            ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
            ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

    if last_scatter is not None:
        add_shared_colorbar(fig, last_scatter, sound_type, next(iter(results.values()))["dataset"]["Y"])
    fig.savefig(get_figure_dir() / f"{sound_type}_population_pca.png", dpi=300)
    plt.close(fig)


def save_scree_figure(sound_type: str, results: dict[tuple[str, str], dict[str, np.ndarray]], target_neurons: int) -> None:
    """Save the scree-plot figure for one sound type."""
    brain_regions = get_plot_brain_regions(sound_type)
    fig, axes = plt.subplots(
        len(brain_regions),
        len(WINDOW_ORDER),
        figsize=(4.0 * len(WINDOW_ORDER), 3.5 * len(brain_regions)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(f"{sound_type} scree plots (n={target_neurons} neurons per region)", fontsize=16, fontweight="bold")
    y_max = max(float(panel["summary"]["explained_variance_ratio"][0]) * 100 for panel in results.values()) * 1.05
    for row_index, brain_area in enumerate(brain_regions):
        for col_index, window_name in enumerate(WINDOW_ORDER):
            ax = axes[row_index, col_index]
            panel = results.get((brain_area, window_name))
            if panel is None:
                ax.axis("off")
                continue
            plot_scree(ax, panel["summary"], format_panel_title(brain_area, window_name), y_max=y_max)
    fig.savefig(get_figure_dir() / f"{sound_type}_pca_scree.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Run PCA population plots for each available sound type."""
    apply_figure_style()
    for sound_type in list_available_sound_types():
        results, target_neurons = collect_sound_results(sound_type)
        if not results:
            continue
        save_scatter_figure(sound_type, results, target_neurons)
        save_scree_figure(sound_type, results, target_neurons)


if __name__ == "__main__":
    main()
