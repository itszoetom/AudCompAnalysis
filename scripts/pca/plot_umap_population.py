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
        add_stimulus_colorbar,
        apply_figure_style,
        build_sampled_dataset,
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
        build_sampled_dataset,
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
    """Run the population UMAP figures for each available sound type."""
    apply_figure_style()
    sound_types = list_available_sound_types()
    for sound_type in tqdm(sound_types, desc="PCA UMAP figures", unit="sound", dynamic_ncols=True):
        print(f"Building UMAP figures for {sound_type}...")
        target_neurons = get_target_neuron_count(sound_type)
        fig, axes = make_sound_figure(sound_type, "")
        fig.suptitle(f"{sound_type} population UMAP (n={target_neurons} neurons per region)", fontsize=26, fontweight="bold")
        last_scatter = None
        last_dataset = None
        for row_index, brain_area, col_index, window_name in tqdm(
            panel_conditions(sound_type),
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
            add_stimulus_colorbar(fig, last_scatter, sound_type, last_dataset["Y"])
        fig.savefig(get_figure_dir() / f"{sound_type}_population_umap.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
