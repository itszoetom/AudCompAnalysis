"""Shared PCA and UMAP helpers for population auditory-cortex analyses across both datasets."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import funcs  # noqa: E402
import params  # noqa: E402

WINDOW_ORDER = ("onset", "sustained", "offset")
WINDOW_TO_KEY = {
    "onset": "onsetfr",
    "sustained": "sustainedfr",
    "offset": "offsetfr",
}
SOUND_ORDER = ("speech", "AM", "PT", "naturalSound")
SOUND_FILE_KEYS = {
    "speech": "speech",
    "AM": "AM",
    "PT": "pureTones",
    "naturalSound": "naturalSound",
}
DEFAULT_SCATTER_KWARGS = {"s": 24, "alpha": 0.85, "linewidths": 0}
SPEECH_SYLLABLE_MAP = {
    (0, 0): "/ba/",
    (100, 0): "/da/",
    (0, 100): "/pa/",
    (100, 100): "/ta/",
}


def get_data_dir() -> Path:
    """Return the directory containing saved firing-rate arrays."""
    return Path(params.dbSavePath)


def get_figure_dir() -> Path:
    """Return the PCA figure output directory."""
    figure_dir = Path(params.figSavePath) / "pca"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def apply_figure_style() -> None:
    """Apply the shared paper-style plotting defaults."""
    sns.set_theme(
        style="ticks",
        context="paper",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        },
    )


def get_condition_color(sound_type: str, brain_area: str) -> tuple[float, ...]:
    """Return the configured region color for one sound type."""
    return params.color_palette.get(f"{brain_area} - {sound_type}", (0.25, 0.25, 0.25, 1.0))


def format_panel_title(brain_area: str, window_name: str) -> str:
    """Return a compact subplot title using short brain-region labels."""
    return f"{params.short_names.get(brain_area, brain_area)}\n{window_name.capitalize()}"


def get_plot_brain_regions(sound_type: str) -> list[str]:
    """Return the ordered brain regions to include in paper figures."""
    regions = get_brain_regions(sound_type)
    if sound_type == "speech":
        regions = [region for region in regions if region != "Dorsal auditory area"]
    return [region for region in params.targetSiteNames if region in regions]


def get_target_neuron_count(sound_type: str) -> int:
    """Return the fixed per-figure neuron count used in thesis figures."""
    return 99 if sound_type == "speech" else 278


def sample_neuron_indices(sound_type: str, brain_area: str, n_neurons: int | None = None, seed: int = 42) -> np.ndarray:
    """Return reproducibly sampled neuron indices for one sound and brain area."""
    sound_data = load_sound_npz(sound_type)
    brain_indices = np.flatnonzero(sound_data["brainRegionArray"] == brain_area)
    target = n_neurons if n_neurons is not None else min(get_target_neuron_count(sound_type), len(brain_indices))
    if len(brain_indices) <= target:
        return brain_indices
    rng = np.random.default_rng(seed + sum(f"{sound_type}-{brain_area}".encode("utf-8")))
    return np.sort(rng.choice(brain_indices, size=target, replace=False))


def build_sampled_dataset(sound_type: str, window_name: str, brain_area: str, n_neurons: int | None = None) -> dict[str, np.ndarray] | None:
    """Return one population dataset after per-figure neuron subsampling."""
    sound_data = load_sound_npz(sound_type)
    cell_indices = sample_neuron_indices(sound_type, brain_area, n_neurons=n_neurons)
    if len(cell_indices) == 0:
        return None
    x = sound_data[WINDOW_TO_KEY[window_name]][cell_indices].T
    return {
        "sound_type": sound_type,
        "window_name": window_name,
        "brain_area": brain_area,
        "X": x,
        "Y": sound_data["stimArray"],
        "mouse_ids": sound_data["mouseIDArray"][cell_indices],
        "session_ids": sound_data["sessionIDArray"][cell_indices],
    }


def format_speech_label(label: tuple[int, int] | np.ndarray) -> str:
    """Format one speech tuple as VOT/FT text with a syllable tag when available."""
    ft_value, vot_value = tuple(np.asarray(label).tolist())
    suffix = f" ({SPEECH_SYLLABLE_MAP[(ft_value, vot_value)]})" if (ft_value, vot_value) in SPEECH_SYLLABLE_MAP else ""
    return f"VOT={vot_value} FT={ft_value}{suffix}"


def natural_sound_labels() -> list[str]:
    """Return written natural-sound labels in the stored presentation order."""
    return [f"{category}_{index}" for category in params.SOUND_CATEGORIES for index in range(1, 5)]


def stimulus_tick_labels(sound_type: str, stim_array: np.ndarray) -> list[str]:
    """Return human-readable stimulus labels for one sound type."""
    if sound_type == "speech":
        return [format_speech_label(label) for label in np.asarray(stim_array)]
    if sound_type == "naturalSound":
        labels = natural_sound_labels()
        return [labels[int(value)] for value in np.asarray(stim_array, dtype=int)]
    if sound_type == "AM":
        return [f"AM White Noise \u2014 {int(value)} Hz" for value in np.asarray(stim_array, dtype=float)]
    return [f"Pure Tones \u2014 {int(value)} Hz" for value in np.asarray(stim_array, dtype=float)]


def list_available_sound_types() -> list[str]:
    """List the sound types with saved `.npz` arrays available."""
    data_dir = get_data_dir()
    available = []
    for sound_type in SOUND_ORDER:
        file_key = SOUND_FILE_KEYS[sound_type]
        if (data_dir / f"fr_arrays_{file_key}.npz").exists():
            available.append(sound_type)
    return available


def _collapse_stimulus_array(stim_array: np.ndarray, n_trials: int) -> np.ndarray:
    """Collapse redundant stimulus arrays to one shared trial-wise label vector."""
    if stim_array.ndim == 1:
        return stim_array

    if stim_array.ndim == 2 and stim_array.shape[1] == n_trials:
        first_row = stim_array[0]
        if not np.all(stim_array == first_row):
            raise ValueError("Stimulus labels are not consistent across cells.")
        return first_row

    if stim_array.shape[0] == n_trials:
        return stim_array

    if stim_array.shape[0] % n_trials != 0:
        raise ValueError(
            f"Stimulus array with shape {stim_array.shape} is not compatible with {n_trials} trials."
        )

    n_repeats = stim_array.shape[0] // n_trials
    reshaped = stim_array.reshape((n_repeats, n_trials) + stim_array.shape[1:])
    first_repeat = reshaped[0]
    if not np.all(reshaped == first_repeat):
        raise ValueError("Stimulus labels are not consistent across sessions.")
    return first_repeat


def load_sound_npz(sound_type: str) -> dict[str, np.ndarray]:
    """Load one saved firing-rate array file and normalize its label shapes."""
    file_key = SOUND_FILE_KEYS.get(sound_type, sound_type)
    filepath = get_data_dir() / f"fr_arrays_{file_key}.npz"
    if not filepath.exists():
        raise FileNotFoundError(f"No preprocessed file found for {sound_type}: {filepath}")

    raw = np.load(filepath, allow_pickle=True)
    onset = raw["onsetfr"]
    stim = _collapse_stimulus_array(raw["stimArray"], onset.shape[1])
    return {
        "onsetfr": onset,
        "sustainedfr": raw["sustainedfr"],
        "offsetfr": raw["offsetfr"],
        "brainRegionArray": raw["brainRegionArray"],
        "stimArray": stim,
        "mouseIDArray": raw["mouseIDArray"],
        "sessionIDArray": raw["sessionIDArray"],
    }


def get_brain_regions(sound_type: str) -> list[str]:
    """Return the sorted brain regions present for one sound type."""
    sound_data = load_sound_npz(sound_type)
    return sorted(np.unique(sound_data["brainRegionArray"]).tolist())


@lru_cache(maxsize=None)
def get_sound_min_neuron_count(sound_type: str) -> int:
    """Return the minimum neuron count across brain regions for one sound type."""
    sound_data = load_sound_npz(sound_type)
    counts = []
    for brain_area in get_brain_regions(sound_type):
        counts.append(np.count_nonzero(sound_data["brainRegionArray"] == brain_area))
    if not counts:
        raise ValueError(f"No brain regions found for {sound_type}.")
    return min(counts)


@lru_cache(maxsize=None)
def get_sampled_cell_indices(sound_type: str, brain_area: str) -> np.ndarray:
    """Return cached equal-neuron indices for one sound and brain area."""
    sound_data = load_sound_npz(sound_type)
    brain_indices = np.flatnonzero(sound_data["brainRegionArray"] == brain_area)
    if len(brain_indices) == 0:
        return brain_indices

    target_count = get_sound_min_neuron_count(sound_type)
    if len(brain_indices) <= target_count:
        return brain_indices

    seed = sum(f"{sound_type}-{brain_area}".encode("utf-8"))
    rng = np.random.default_rng(seed)
    sampled = np.sort(rng.choice(brain_indices, size=target_count, replace=False))
    return sampled


def build_population_dataset(
    sound_type: str,
    window_name: str,
    brain_area: str,
) -> dict[str, np.ndarray] | None:
    """Build one population dataset for a sound, time window, and brain region."""
    sound_data = load_sound_npz(sound_type)
    cell_indices = get_sampled_cell_indices(sound_type, brain_area)
    if len(cell_indices) == 0:
        return None

    x_key = WINDOW_TO_KEY[window_name]
    x = sound_data[x_key][cell_indices].T
    return {
        "sound_type": sound_type,
        "window_name": window_name,
        "brain_area": brain_area,
        "X": x,
        "Y": sound_data["stimArray"],
        "mouse_ids": sound_data["mouseIDArray"][cell_indices],
        "session_ids": sound_data["sessionIDArray"][cell_indices],
    }


def iter_population_datasets(
    sound_types: Iterable[str] | None = None,
    windows: Iterable[str] = WINDOW_ORDER,
) -> list[dict[str, np.ndarray]]:
    """Iterate through all available population datasets."""
    datasets = []
    for sound_type in sound_types or list_available_sound_types():
        for window_name in windows:
            for brain_area in get_brain_regions(sound_type):
                dataset = build_population_dataset(sound_type, window_name, brain_area)
                if dataset is not None:
                    datasets.append(dataset)
    return datasets


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


def compute_pca_summary(x: np.ndarray) -> dict[str, np.ndarray]:
    """Standardize one population matrix and compute PCA summary statistics."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA()
    scores = pca.fit_transform(x_scaled)
    explained = pca.explained_variance_ratio_
    return {
        "scores": scores,
        "explained_variance_ratio": explained,
        "participation_ratio": funcs.calculate_participation_ratio(explained),
        "n_neurons": x.shape[1],
        "n_trials": x.shape[0],
    }


def get_sound_scree_ymax(sound_type: str, average_trials: bool = False) -> float:
    """Return a shared scree y-limit for one sound figure."""
    max_explained = 0.0
    for window_name in WINDOW_ORDER:
        for brain_area in get_brain_regions(sound_type):
            dataset = build_population_dataset(sound_type, window_name, brain_area)
            if dataset is None:
                continue
            if average_trials:
                dataset = average_trials_by_stimulus(dataset)
            summary = compute_pca_summary(dataset["X"])
            if len(summary["explained_variance_ratio"]) == 0:
                continue
            max_explained = max(max_explained, float(summary["explained_variance_ratio"][0]) * 100)
    return max_explained * 1.05 if max_explained else 1.0


def labels_for_sound(sound_type: str, stim_array: np.ndarray) -> np.ndarray:
    """Convert stimulus identities into numeric color values."""
    if sound_type == "speech":
        label_to_number = {tuple(label): idx for idx, label in enumerate(params.unique_labels)}
        return np.array([label_to_number[tuple(label)] for label in np.asarray(stim_array)])

    stim_values = np.asarray(stim_array)
    if stim_values.ndim > 1:
        if stim_values.dtype.kind in {"O", "U", "S"}:
            unique_labels, inverse = np.unique(stim_values, axis=0, return_inverse=True)
            return inverse
        first_row = stim_values[0]
        if np.allclose(stim_values, first_row):
            stim_values = first_row

    stim_values = np.asarray(stim_values, dtype=float)
    if np.any(stim_values <= 0):
        _, inverse = np.unique(stim_values, return_inverse=True)
        return inverse
    return np.log10(stim_values)


def plot_scree(ax: plt.Axes, pca_summary: dict[str, np.ndarray], title: str, y_max: float | None = None) -> None:
    """Draw a scree plot for the first principal components."""
    explained = pca_summary["explained_variance_ratio"]
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


def plot_pca_scatter(
    ax: plt.Axes,
    dataset: dict[str, np.ndarray],
    pca_summary: dict[str, np.ndarray],
    title: str,
    labels: np.ndarray | None = None,
) -> None:
    """Draw a PC1-versus-PC2 scatter plot."""
    scores = pca_summary["scores"]
    color_values = labels if labels is not None else labels_for_sound(dataset["sound_type"], dataset["Y"])
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=color_values, cmap="viridis", **DEFAULT_SCATTER_KWARGS)
    explained = pca_summary["explained_variance_ratio"]
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
    sns.despine(ax=ax)
    plt.colorbar(scatter, ax=ax)


def make_population_figure(n_rows: int, n_windows: int, *, square_scale: float = 3.2) -> tuple[plt.Figure, np.ndarray]:
    """Create a square-ish grid for population figures."""
    return plt.subplots(
        n_rows,
        2 * n_windows,
        figsize=(square_scale * 2 * n_windows, square_scale * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
