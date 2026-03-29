"""PCA-specific analysis helpers built on shared project utilities."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import params  # noqa: E402
import funcs  # noqa: E402

WINDOW_ORDER = params.WINDOW_ORDER
apply_figure_style = funcs.apply_figure_style
build_population_dataset = funcs.build_population_dataset
build_sampled_dataset = funcs.build_sampled_dataset
get_plot_brain_regions = funcs.get_plot_brain_regions
get_target_neuron_count = funcs.get_target_neuron_count
labels_for_sound = funcs.labels_for_sound
list_available_sound_types = funcs.list_available_sound_types
stimulus_tick_labels = funcs.stimulus_tick_labels

DEFAULT_SCATTER_KWARGS = {"s": 24, "alpha": 0.85, "linewidths": 0}


def get_figure_dir() -> Path:
    """Return the PCA figure output directory."""
    return funcs.get_figure_dir("pca")


def panel_conditions(sound_type: str) -> list[tuple[int, str, int, str]]:
    """Return row/column panel conditions for one sound type."""
    return [
        (row_index, brain_area, col_index, window_name)
        for row_index, brain_area in enumerate(get_plot_brain_regions(sound_type))
        for col_index, window_name in enumerate(WINDOW_ORDER)
    ]


def make_sound_figure(
    sound_type: str,
    title: str,
    *,
    width_scale: float = 4.0,
    height_scale: float = 3.5,
    sharey: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a standard sound-by-region-by-window figure grid."""
    brain_regions = get_plot_brain_regions(sound_type)
    return plt.subplots(
        len(brain_regions),
        len(WINDOW_ORDER),
        figsize=(width_scale * len(WINDOW_ORDER), height_scale * len(brain_regions)),
        squeeze=False,
        sharey=sharey,
        constrained_layout=True,
    )


def format_panel_title(brain_area: str, window_name: str) -> str:
    """Return a compact subplot title using short brain-region labels."""
    return f"{params.short_names.get(brain_area, brain_area)}\n{window_name.capitalize()}"


def calculate_participation_ratio(explained_variance_ratio: np.ndarray) -> float:
    """Return the participation ratio for one PCA spectrum."""
    return float((np.sum(explained_variance_ratio) ** 2) / np.sum(explained_variance_ratio ** 2))


def average_trials_by_stimulus(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Average repeated trials within each stimulus identity."""
    y = np.asarray(dataset["Y"])
    if dataset["sound_type"] == "speech":
        unique_y = np.asarray(params.unique_labels)
        inverse = labels_for_sound("speech", y)
    elif y.ndim == 1:
        unique_y, inverse = np.unique(y, return_inverse=True)
    else:
        unique_y, inverse = np.unique(y, axis=0, return_inverse=True)

    averaged_x = np.vstack(
        [dataset["X"][inverse == stim_index].mean(axis=0) for stim_index in range(len(unique_y))]
    )
    averaged = dict(dataset)
    averaged["X"] = averaged_x
    averaged["Y"] = unique_y
    return averaged


def compute_pca_summary(x: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Standardize one population matrix and compute PCA summary statistics."""
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA()
    scores = pca.fit_transform(x_scaled)
    explained = pca.explained_variance_ratio_
    return {
        "scores": scores,
        "explained_variance_ratio": explained,
        "participation_ratio": calculate_participation_ratio(explained),
        "n_neurons": x.shape[1],
        "n_trials": x.shape[0],
    }


def iter_population_datasets(
    sound_types: Iterable[str] | None = None,
    windows: Iterable[str] = params.WINDOW_ORDER,
) -> list[dict[str, np.ndarray]]:
    """Iterate through all available equal-neuron population datasets."""
    datasets = []
    for sound_type in sound_types or funcs.list_available_sound_types():
        for window_name in windows:
            for brain_area in funcs.get_brain_regions(sound_type):
                dataset = build_population_dataset(sound_type, window_name, brain_area)
                if dataset is not None:
                    datasets.append(dataset)
    return datasets


def plot_scree(
    ax: plt.Axes,
    pca_summary: dict[str, np.ndarray | float | int],
    title: str,
    y_max: float | None = None,
) -> None:
    """Draw a scree plot for the first principal components."""
    explained = np.asarray(pca_summary["explained_variance_ratio"])
    n_components = min(len(explained), 12)
    ax.bar(np.arange(n_components), explained[:n_components] * 100, color="black")
    ax.set_title(title)
    ax.set_xlabel("PC")
    ax.set_ylabel("% Explained Variance")
    ax.set_xticks(np.arange(n_components))
    if y_max is not None:
        ax.set_ylim(0, y_max)
    ax.text(
        0.98,
        0.95,
        f"PR = {pca_summary['participation_ratio']:.2f}\nn = {pca_summary['n_neurons']}",
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    sns.despine(ax=ax)


def shared_scatter_limits(results: dict[tuple[str, str], dict[str, np.ndarray]]) -> tuple[float, float, float, float]:
    """Compute common PC axis limits for one figure."""
    x_values = [panel["summary"]["scores"][:, 0] for panel in results.values()]
    y_values = [panel["summary"]["scores"][:, 1] for panel in results.values()]
    return (
        min(values.min() for values in x_values),
        max(values.max() for values in x_values),
        min(values.min() for values in y_values),
        max(values.max() for values in y_values),
    )


def add_stimulus_colorbar(fig: plt.Figure, scatter, sound_type: str, stim_array: np.ndarray) -> None:
    """Add one shared stimulus colorbar for a sound figure."""
    colorbar = fig.colorbar(scatter, ax=fig.axes, location="bottom", fraction=0.03, pad=0.04)
    color_values = labels_for_sound(sound_type, stim_array)
    tick_labels = stimulus_tick_labels(sound_type, stim_array)
    unique_values, first_indices = np.unique(color_values, return_index=True)
    colorbar.set_ticks(unique_values)
    colorbar.set_ticklabels([tick_labels[index] for index in first_indices])
    colorbar.ax.tick_params(labelsize=8, rotation=35)
    colorbar.set_label("Stimulus", fontsize=12)


def collect_sound_results(
    sound_type: str,
    build_panel: Callable[[str, str], dict[str, np.ndarray] | None],
    summarize_panel: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray | float | int]],
    *,
    desc: str,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Collect datasets and summaries for one sound figure."""
    results = {}
    for _, brain_area, _, window_name in tqdm(
        panel_conditions(sound_type),
        desc=desc,
        unit="panel",
        dynamic_ncols=True,
    ):
        dataset = build_panel(brain_area, window_name)
        if dataset is None:
            continue
        results[(brain_area, window_name)] = {"dataset": dataset, "summary": summarize_panel(dataset)}
    return results
