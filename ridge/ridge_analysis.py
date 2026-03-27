"""Shared ridge-regression helpers for population decoding across both auditory datasets."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable.
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

def get_data_dir() -> Path:
    """Return the directory containing saved firing-rate arrays."""
    return Path(params.dbSavePath)


def get_output_dir() -> Path:
    """Return the ridge figure and table output directory."""
    output_dir = Path(params.figSavePath) / "ridge"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
        raise ValueError(f"Incompatible stimulus array shape {stim_array.shape} for {n_trials} trials.")

    n_repeats = stim_array.shape[0] // n_trials
    reshaped = stim_array.reshape((n_repeats, n_trials) + stim_array.shape[1:])
    first_repeat = reshaped[0]
    if not np.all(reshaped == first_repeat):
        raise ValueError("Stimulus labels are not consistent across sessions.")
    return first_repeat


def list_available_sound_types() -> list[str]:
    """List the sound types with saved `.npz` arrays available."""
    data_dir = get_data_dir()
    return [
        sound_type
        for sound_type in SOUND_ORDER
        if (data_dir / f"fr_arrays_{SOUND_FILE_KEYS[sound_type]}.npz").exists()
    ]


def load_sound_npz(sound_type: str) -> dict[str, np.ndarray]:
    """Load one saved firing-rate array file and normalize its label shapes."""
    filepath = get_data_dir() / f"fr_arrays_{SOUND_FILE_KEYS[sound_type]}.npz"
    raw = np.load(filepath, allow_pickle=True)
    onset = raw["onsetfr"]
    stim_numeric_source = raw["stimNumericArray"] if "stimNumericArray" in raw else raw["stimArray"]
    return {
        "onsetfr": onset,
        "sustainedfr": raw["sustainedfr"],
        "offsetfr": raw["offsetfr"],
        "brainRegionArray": raw["brainRegionArray"],
        "stimArray": _collapse_stimulus_array(stim_numeric_source, onset.shape[1]),
    }


def get_brain_regions(sound_type: str) -> list[str]:
    """Return the sorted brain regions present for one sound type."""
    return sorted(np.unique(load_sound_npz(sound_type)["brainRegionArray"]).tolist())


@lru_cache(maxsize=None)
def get_sound_min_neuron_count(sound_type: str) -> int:
    """Return the minimum neuron count across brain regions for one sound type."""
    sound_data = load_sound_npz(sound_type)
    counts = [np.count_nonzero(sound_data["brainRegionArray"] == brain_area) for brain_area in get_brain_regions(sound_type)]
    return min(counts)


@lru_cache(maxsize=None)
def get_sampled_cell_indices(sound_type: str, brain_area: str) -> np.ndarray:
    """Return cached equal-neuron indices for one sound and brain area."""
    sound_data = load_sound_npz(sound_type)
    cell_indices = np.flatnonzero(sound_data["brainRegionArray"] == brain_area)
    if len(cell_indices) <= get_sound_min_neuron_count(sound_type):
        return cell_indices
    seed = sum(f"ridge-{sound_type}-{brain_area}".encode("utf-8"))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(cell_indices, size=get_sound_min_neuron_count(sound_type), replace=False))


def build_population_dataset(sound_type: str, window_name: str, brain_area: str) -> dict[str, np.ndarray] | None:
    """Build one ridge dataset for a sound, time window, and brain region."""
    sound_data = load_sound_npz(sound_type)
    cell_indices = get_sampled_cell_indices(sound_type, brain_area)
    if len(cell_indices) == 0:
        return None

    x = sound_data[WINDOW_TO_KEY[window_name]][cell_indices].T
    y = sound_data["stimArray"]
    datasets = []
    if sound_type == "speech":
        datasets.append(
            {
                "sound_type": sound_type,
                "target_name": "FT",
                "window_name": window_name,
                "brain_area": brain_area,
                "X": x,
                "Y": y[:, 0],
                "log_target": False,
            }
        )
        datasets.append(
            {
                "sound_type": sound_type,
                "target_name": "VOT",
                "window_name": window_name,
                "brain_area": brain_area,
                "X": x,
                "Y": y[:, 1],
                "log_target": False,
            }
        )
    else:
        datasets.append(
            {
                "sound_type": sound_type,
                "target_name": sound_type,
                "window_name": window_name,
                "brain_area": brain_area,
                "X": x,
                "Y": y.astype(float),
                "log_target": sound_type in {"AM", "PT"},
            }
        )
    return datasets


def get_target_labels(dataset: dict[str, np.ndarray]) -> np.ndarray:
    """Return the regression targets in their plotted scale."""
    if dataset["sound_type"] == "speech":
        return dataset["Y"]
    if dataset["log_target"]:
        return np.log10(dataset["Y"])
    return dataset["Y"]


def apply_figure_style() -> None:
    """Apply the shared paper-style plotting defaults."""
    sns.set_theme(
        style="ticks",
        context="paper",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        },
    )


def iter_population_datasets(
    sound_types: Iterable[str] | None = None,
    windows: Iterable[str] = WINDOW_ORDER,
) -> list[dict[str, np.ndarray]]:
    """Iterate through all available ridge datasets."""
    datasets = []
    for sound_type in sound_types or list_available_sound_types():
        for window_name in windows:
            for brain_area in get_brain_regions(sound_type):
                sound_datasets = build_population_dataset(sound_type, window_name, brain_area)
                if sound_datasets is not None:
                    datasets.extend(sound_datasets)
    return datasets


def fit_best_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alphas: np.ndarray | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    log_target: bool = False,
) -> dict[str, np.ndarray | float]:
    """Fit one ridge model after tuning alpha on a held-out split."""
    transformed_y = np.log10(y) if log_target else y
    x_train, x_test, y_train, y_test = train_test_split(
        x, transformed_y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    alphas = alphas if alphas is not None else np.logspace(-10, 5, 200)

    best_alpha = None
    best_r2 = -np.inf
    best_model = None
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(x_train_scaled, y_train)
        r2 = model.score(x_test_scaled, y_test)
        if r2 > best_r2:
            best_alpha = alpha
            best_r2 = r2
            best_model = model

    y_pred = best_model.predict(x_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    corr = float(pearsonr(y_test, y_pred)[0]) if len(y_test) > 1 else np.nan
    tolerance = 0.05 * max(np.ptp(y_test), 1e-12)
    within_tolerance = float(np.mean(np.abs(y_pred - y_test) <= tolerance) * 100)

    return {
        "best_alpha": float(best_alpha),
        "r2_test": float(best_r2),
        "rmse": rmse,
        "pearson_r": corr,
        "percent_within_tolerance": within_tolerance,
        "n_neurons": int(x.shape[1]),
        "y_test": y_test,
        "y_pred": y_pred,
    }


def run_population_ridge(
    sound_types: Iterable[str] | None = None,
    windows: Iterable[str] = WINDOW_ORDER,
    iterations: int = 30,
) -> pd.DataFrame:
    """Run repeated population ridge fits across all sound, region, and window conditions."""
    records = []
    datasets = iter_population_datasets(sound_types=sound_types, windows=windows)
    progress = tqdm(total=len(datasets) * iterations, desc="Ridge regression", unit="fit", dynamic_ncols=True)
    try:
        for dataset in datasets:
            progress.set_postfix_str(
                f"{dataset['sound_type']} | {params.short_names.get(dataset['brain_area'], dataset['brain_area'])} | {dataset['window_name']} | {dataset['target_name']}",
                refresh=False,
            )
            for iteration in range(iterations):
                fit = fit_best_ridge(
                    dataset["X"],
                    dataset["Y"],
                    log_target=dataset["log_target"],
                    random_state=42 + iteration,
                )
                progress.update(1)
                records.append(
                    {
                        "Brain Area": dataset["brain_area"],
                        "Sound Type": dataset["sound_type"],
                        "Target": dataset["target_name"],
                        "Window": dataset["window_name"],
                        "Iteration": iteration,
                        "Neurons": fit["n_neurons"],
                        "Best Alpha": fit["best_alpha"],
                        "R2 Test": fit["r2_test"],
                        "RMSE": fit["rmse"],
                        "Pearson r": fit["pearson_r"],
                        "Percent Within Tolerance": fit["percent_within_tolerance"],
                    }
                )
    finally:
        close = getattr(progress, "close", None)
        if close is not None:
            close()
    return pd.DataFrame(records)


def minimum_common_neuron_count(
    sound_types: Iterable[str] | None = None,
    windows: Iterable[str] = WINDOW_ORDER,
) -> int:
    """Return the smallest neuron count across the selected ridge datasets."""
    datasets = iter_population_datasets(sound_types=sound_types, windows=windows)
    return min(dataset["X"].shape[1] for dataset in datasets)


def run_subset_ridge(
    subset_size: int,
    *,
    iterations: int = 30,
    sound_types: Iterable[str] | None = None,
    windows: Iterable[str] = WINDOW_ORDER,
    seed: int = 42,
) -> pd.DataFrame:
    """Run repeated ridge fits after random equal-neuron subsampling."""
    rng = np.random.default_rng(seed)
    records = []
    datasets = [dataset for dataset in iter_population_datasets(sound_types=sound_types, windows=windows) if dataset["X"].shape[1] >= subset_size]
    progress = tqdm(total=len(datasets) * iterations, desc="Subset ridge", unit="fit", dynamic_ncols=True)
    try:
        for dataset in datasets:
            progress.set_postfix_str(
                f"{dataset['sound_type']} | {params.short_names.get(dataset['brain_area'], dataset['brain_area'])} | {dataset['window_name']} | {dataset['target_name']}",
                refresh=False,
            )
            for iteration in range(iterations):
                neuron_idx = rng.choice(dataset["X"].shape[1], subset_size, replace=False)
                fit = fit_best_ridge(
                    dataset["X"][:, neuron_idx],
                    dataset["Y"],
                    log_target=dataset["log_target"],
                    random_state=seed + iteration,
                )
                progress.update(1)
                records.append(
                    {
                        "Brain Area": dataset["brain_area"],
                        "Sound Type": dataset["sound_type"],
                        "Target": dataset["target_name"],
                        "Window": dataset["window_name"],
                        "Iteration": iteration,
                        "Neurons": subset_size,
                        "Best Alpha": fit["best_alpha"],
                        "R2 Test": fit["r2_test"],
                    }
                )
    finally:
        close = getattr(progress, "close", None)
        if close is not None:
            close()
    return pd.DataFrame(records)
