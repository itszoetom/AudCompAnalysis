"""Create trial-averaged population PCA figures for each sound type across regions and windows."""

from __future__ import annotations

import matplotlib.pyplot as plt
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

try:
    from .pca_analysis import (
        WINDOW_ORDER,
        add_stimulus_colorbar,
        apply_figure_style,
        average_trials_by_stimulus,
        build_sampled_dataset,
        compute_pca_summary,
        collect_sound_results,
        format_panel_title,
        get_figure_dir,
        get_plot_brain_regions,
        get_target_neuron_count,
        labels_for_sound,
        list_available_sound_types,
        make_sound_figure,
        panel_conditions,
    )
except ImportError:
    from pca_analysis import (
        WINDOW_ORDER,
        add_stimulus_colorbar,
        apply_figure_style,
        average_trials_by_stimulus,
        build_sampled_dataset,
        compute_pca_summary,
        collect_sound_results,
        format_panel_title,
        get_figure_dir,
        get_plot_brain_regions,
        get_target_neuron_count,
        labels_for_sound,
        list_available_sound_types,
        make_sound_figure,
        panel_conditions,
    )


def main() -> None:
    """Run trial-averaged PCA scatter plots for each sound type."""
    apply_figure_style()
    sound_types = list_available_sound_types()
    for sound_type in tqdm(sound_types, desc="PCA averaged figures", unit="sound", dynamic_ncols=True):
        print(f"Building trial-averaged PCA plots for {sound_type}...")
        target_neurons = get_target_neuron_count(sound_type)
        fig, axes = make_sound_figure(sound_type, "")
        fig.suptitle(f"{sound_type} averaged PCA (n={target_neurons} neurons per region)", fontsize=16, fontweight="bold")

        def build_panel(brain_area: str, window_name: str):
            dataset = build_sampled_dataset(sound_type, window_name, brain_area, n_neurons=target_neurons)
            return None if dataset is None else average_trials_by_stimulus(dataset)

        results = collect_sound_results(
            sound_type,
            build_panel,
            lambda dataset: compute_pca_summary(dataset["X"]),
            desc=f"PCA averaged panels ({sound_type})",
        )
        last_scatter = None
        for row_index, brain_area, col_index, window_name in panel_conditions(sound_type):
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
                s=36,
                alpha=0.9,
                linewidths=0,
            )
            ax.set_title(format_panel_title(brain_area, window_name), fontweight="bold")
            ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
            ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)

        if last_scatter is not None and results:
            add_stimulus_colorbar(fig, last_scatter, sound_type, next(iter(results.values()))["dataset"]["Y"])

        fig.savefig(get_figure_dir() / f"{sound_type}_population_pca_averaged.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
