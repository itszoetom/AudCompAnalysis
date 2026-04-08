"""Create population PCA scatter and scree figures for each sound type across regions and windows."""

from __future__ import annotations

import matplotlib.pyplot as plt
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

try:
    from .pca_analysis import (
        add_stimulus_colorbar,
        apply_figure_style,
        build_sampled_dataset,
        collect_sound_results,
        compute_pca_summary,
        format_panel_title,
        get_figure_dir,
        get_target_neuron_count,
        labels_for_sound,
        list_available_sound_types,
        panel_conditions,
        plot_scree,
        shared_scatter_limits,
    )
except ImportError:
    from pca_analysis import (
        add_stimulus_colorbar,
        apply_figure_style,
        build_sampled_dataset,
        collect_sound_results,
        compute_pca_summary,
        format_panel_title,
        get_figure_dir,
        get_target_neuron_count,
        labels_for_sound,
        list_available_sound_types,
        panel_conditions,
        plot_scree,
        shared_scatter_limits,
    )


def save_projection_figure(sound_type: str, results, target_neurons: int) -> None:
    """Save one PCA projection figure for one sound type."""
    panels = panel_conditions(sound_type)
    n_rows = max(row_index for row_index, *_ in panels) + 1
    n_windows = max(col_index for *_, col_index, _ in panels) + 1
    fig, axes = plt.subplots(
        n_rows,
        n_windows,
        figsize=(3.8 * n_windows, 3.6 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(f"{sound_type} population PCA projections (n={target_neurons} neurons per region)", fontsize=26, fontweight="bold")
    x_min, x_max, y_min, y_max = shared_scatter_limits(results)
    last_scatter = None

    for row_index, brain_area, col_index, window_name in panels:
        scatter_ax = axes[row_index, col_index]
        panel = results.get((brain_area, window_name))
        if panel is None:
            scatter_ax.axis("off")
            continue
        dataset = panel["dataset"]
        summary = panel["summary"]
        scores = summary["scores"]
        explained = summary["explained_variance_ratio"]
        last_scatter = scatter_ax.scatter(
            scores[:, 0],
            scores[:, 1],
            c=labels_for_sound(sound_type, dataset["Y"]),
            cmap="viridis",
            s=24,
            alpha=0.85,
            linewidths=0,
        )
        scatter_ax.set_xlim(x_min, x_max)
        scatter_ax.set_ylim(y_min, y_max)
        scatter_ax.set_title(format_panel_title(brain_area, window_name), fontweight="bold")
        scatter_ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        scatter_ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        scatter_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

    if last_scatter is not None:
        add_stimulus_colorbar(fig, last_scatter, sound_type, next(iter(results.values()))["dataset"]["Y"])
    fig.savefig(get_figure_dir() / f"pca/{sound_type}_population_pca_projections.png", dpi=300)
    plt.close(fig)


def save_scree_figure(sound_type: str, results, target_neurons: int) -> None:
    """Save one scree-only PCA figure for one sound type."""
    panels = panel_conditions(sound_type)
    n_rows = max(row_index for row_index, *_ in panels) + 1
    n_windows = max(col_index for *_, col_index, _ in panels) + 1
    fig, axes = plt.subplots(
        n_rows,
        n_windows,
        figsize=(3.8 * n_windows, 3.6 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(f"{sound_type} population PCA scree plots (n={target_neurons} neurons per region)", fontsize=26, fontweight="bold")
    scree_ymax = max(float(panel["summary"]["explained_variance_ratio"][0]) * 100 for panel in results.values()) * 1.05

    for row_index, brain_area, col_index, window_name in panels:
        scree_ax = axes[row_index, col_index]
        panel = results.get((brain_area, window_name))
        if panel is None:
            scree_ax.axis("off")
            continue
        summary = panel["summary"]
        plot_scree(scree_ax, summary, format_panel_title(brain_area, window_name), y_max=scree_ymax)
    fig.savefig(get_figure_dir() / f"pca/{sound_type}_population_pca_scree.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Run PCA population plots for each available sound type."""
    apply_figure_style()
    sound_types = list_available_sound_types()
    for sound_type in tqdm(sound_types, desc="PCA population figures", unit="sound", dynamic_ncols=True):
        print(f"Building PCA population projections and scree plots for {sound_type}...")
        target_neurons = get_target_neuron_count(sound_type)
        results = collect_sound_results(
            sound_type,
            lambda brain_area, window_name: build_sampled_dataset(
                sound_type,
                window_name,
                brain_area,
                n_neurons=target_neurons,
            ),
            lambda dataset: compute_pca_summary(dataset["X"]),
            desc=f"PCA population panels ({sound_type})",
        )
        if not results:
            continue
        save_projection_figure(sound_type, results, target_neurons)
        save_scree_figure(sound_type, results, target_neurons)


if __name__ == "__main__":
    main()
