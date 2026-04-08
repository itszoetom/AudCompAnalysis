"""Create speech-only PCA figures colored separately by FT and VOT across regions and windows."""

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
        apply_figure_style,
        build_sampled_dataset,
        compute_pca_summary,
        format_panel_title,
        get_figure_dir,
        get_plot_brain_regions,
        get_target_neuron_count,
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
    )


def save_feature_figure(feature_name: str, feature_index: int) -> None:
    """Save one speech PCA figure colored by a single speech feature."""
    brain_regions = get_plot_brain_regions("speech")
    target_neurons = get_target_neuron_count("speech")
    fig, axes = plt.subplots(
        len(brain_regions),
        len(WINDOW_ORDER),
        figsize=(3.6 * len(WINDOW_ORDER), 3.1 * len(brain_regions)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(f"speech PCA colored by {feature_name} (n={target_neurons} neurons per region)", fontsize=26, fontweight="bold")
    last_scatter = None
    panel_conditions = [(row_index, brain_area, col_index, window_name) for row_index, brain_area in enumerate(brain_regions) for col_index, window_name in enumerate(WINDOW_ORDER)]
    for row_index, brain_area, col_index, window_name in tqdm(
        panel_conditions,
        desc=f"Speech PCA {feature_name}",
        unit="panel",
        dynamic_ncols=True,
    ):
        ax = axes[row_index, col_index]
        dataset = build_sampled_dataset("speech", window_name, brain_area, n_neurons=target_neurons)
        if dataset is None:
            ax.axis("off")
            continue
        summary = compute_pca_summary(dataset["X"])
        scores = summary["scores"]
        explained = summary["explained_variance_ratio"]
        color_values = dataset["Y"][:, feature_index]
        last_scatter = ax.scatter(scores[:, 0], scores[:, 1], c=color_values, cmap="viridis", s=24, alpha=0.85)
        ax.set_title(format_panel_title(brain_area, window_name), fontweight="bold")
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
    if last_scatter is not None:
        colorbar = fig.colorbar(last_scatter, ax=fig.axes, location="bottom", fraction=0.03, pad=0.04)
        colorbar.set_label(f"{feature_name} value", fontsize=22)
        colorbar.ax.tick_params(labelsize=22)
    fig.savefig(get_figure_dir() / "pca/speech_separate" / f"speech_pca_{feature_name.lower()}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the speech PCA FT and VOT figures."""
    apply_figure_style()
    print("Building speech PCA figures colored by FT and VOT...")
    save_feature_figure("FT", 0)
    save_feature_figure("VOT", 1)


if __name__ == "__main__":
    main()
