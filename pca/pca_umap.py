"""Create population UMAP figures for each sound type across regions and spike windows."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import umap

try:
    from .pca_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        build_sampled_dataset,
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
        build_sampled_dataset,
        format_panel_title,
        get_figure_dir,
        get_plot_brain_regions,
        get_target_neuron_count,
        labels_for_sound,
        list_available_sound_types,
        stimulus_tick_labels,
    )


def main() -> None:
    """Run the population UMAP figures for each available sound type."""
    apply_figure_style()
    sound_types = list_available_sound_types()
    for sound_type in tqdm(sound_types, desc="PCA UMAP figures", unit="sound", dynamic_ncols=True):
        print(f"Building UMAP figures for {sound_type}...")
        brain_regions = get_plot_brain_regions(sound_type)
        target_neurons = get_target_neuron_count(sound_type)
        fig, axes = plt.subplots(
            len(brain_regions),
            len(WINDOW_ORDER),
            figsize=(4.0 * len(WINDOW_ORDER), 3.5 * len(brain_regions)),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(f"{sound_type} population UMAP (n={target_neurons} neurons per region)", fontsize=16, fontweight="bold")
        last_scatter = None
        last_dataset = None
        panel_conditions = [(row_index, brain_area, col_index, window_name) for row_index, brain_area in enumerate(brain_regions) for col_index, window_name in enumerate(WINDOW_ORDER)]
        for row_index, brain_area, col_index, window_name in tqdm(
            panel_conditions,
            desc=f"UMAP panels ({sound_type})",
            unit="panel",
            dynamic_ncols=True,
        ):
            ax = axes[row_index, col_index]
            dataset = build_sampled_dataset(sound_type, window_name, brain_area, n_neurons=target_neurons)
            if dataset is None:
                ax.axis("off")
                continue
            reducer = umap.UMAP(
                n_neighbors=min(15, dataset["X"].shape[0] - 1),
                min_dist=0.1,
                random_state=42,
                n_jobs=1,
            )
            embedding = reducer.fit_transform(dataset["X"])
            last_scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels_for_sound(sound_type, dataset["Y"]),
                cmap="viridis",
                s=24,
                alpha=0.85,
                linewidths=0,
            )
            last_dataset = dataset
            ax.set_title(format_panel_title(brain_area, window_name), fontweight="bold")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
        if last_scatter is not None and last_dataset is not None:
            colorbar = fig.colorbar(last_scatter, ax=fig.axes, location="bottom", fraction=0.03, pad=0.04)
            color_values = labels_for_sound(sound_type, last_dataset["Y"])
            tick_labels = stimulus_tick_labels(sound_type, last_dataset["Y"])
            unique_values, first_indices = np.unique(color_values, return_index=True)
            colorbar.set_ticks(unique_values)
            colorbar.set_ticklabels([tick_labels[index] for index in first_indices])
            colorbar.ax.tick_params(labelsize=8, rotation=35)
            colorbar.set_label("Stimulus", fontsize=12)
        fig.savefig(get_figure_dir() / f"{sound_type}_population_umap.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
