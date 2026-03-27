"""Shared helpers for methods figures using saved firing-rate arrays and per-mouse subsets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def get_figure_dir() -> Path:
    """Return the methods figure output directory."""
    figure_dir = Path(params.figSavePath) / "methods"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


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
    stim_labels = _collapse_stimulus_array(raw["stimArray"], onset.shape[1])
    stim_numeric_source = raw["stimNumericArray"] if "stimNumericArray" in raw else raw["stimArray"]
    return {
        "onsetfr": onset,
        "sustainedfr": raw["sustainedfr"],
        "offsetfr": raw["offsetfr"],
        "brainRegionArray": raw["brainRegionArray"],
        "stimArray": _collapse_stimulus_array(stim_numeric_source, onset.shape[1]),
        "stimLabelArray": stim_labels,
        "mouseIDArray": raw["mouseIDArray"],
        "sessionIDArray": raw["sessionIDArray"],
    }


def available_mice(sound_type: str) -> list[str]:
    """Return the mice present for one sound type."""
    return sorted(np.unique(load_sound_npz(sound_type)["mouseIDArray"]).tolist())


def available_sessions(sound_type: str, mouse_id: str | None = None) -> list[str]:
    """Return the sessions present for one sound type, optionally for one mouse."""
    sound_data = load_sound_npz(sound_type)
    session_ids = sound_data["sessionIDArray"]
    if mouse_id is not None:
        session_ids = session_ids[sound_data["mouseIDArray"] == mouse_id]
    return sorted(np.unique(session_ids).tolist())


def get_brain_regions(sound_type: str) -> list[str]:
    """Return the sorted brain regions present for one sound type."""
    return sorted(np.unique(load_sound_npz(sound_type)["brainRegionArray"]).tolist())


def build_dataset(
    sound_type: str,
    window_name: str,
    brain_area: str,
    *,
    mouse_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, np.ndarray] | None:
    """Build one per-mouse or per-session dataset for a sound, time window, and region."""
    sound_data = load_sound_npz(sound_type)
    mask = sound_data["brainRegionArray"] == brain_area
    if mouse_id is not None:
        mask &= sound_data["mouseIDArray"] == mouse_id
    if session_id is not None:
        mask &= sound_data["sessionIDArray"] == session_id
    if not np.any(mask):
        return None

    x = sound_data[WINDOW_TO_KEY[window_name]][mask].T
    return {
        "sound_type": sound_type,
        "window_name": window_name,
        "brain_area": brain_area,
        "X": x,
        "Y": sound_data["stimArray"],
        "Y_labels": sound_data["stimLabelArray"],
        "mouse_ids": sound_data["mouseIDArray"][mask],
        "session_ids": sound_data["sessionIDArray"][mask],
    }


def calculate_participation_ratio(explained_variance_ratio):
    return ((np.sum(explained_variance_ratio)) ** 2) / np.sum(explained_variance_ratio ** 2)


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
        "participation_ratio": calculate_participation_ratio(explained),
        "n_neurons": x.shape[1],
    }


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


def labels_for_sound(sound_type: str, stim_array: np.ndarray) -> np.ndarray:
    """Convert stimulus identities into numeric color values."""
    if sound_type == "speech":
        label_to_number = {tuple(label): idx for idx, label in enumerate(params.unique_labels)}
        return np.array([label_to_number[tuple(label)] for label in np.asarray(stim_array)])

    stim_values = np.asarray(stim_array)
    if stim_values.ndim > 1:
        first_row = stim_values[0]
        if np.allclose(stim_values, first_row):
            stim_values = first_row
    stim_values = np.asarray(stim_values, dtype=float)
    if np.any(stim_values <= 0):
        _, inverse = np.unique(stim_values, return_inverse=True)
        return inverse
    return np.log10(stim_values)


def fit_best_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    log_target: bool = False,
    random_state: int = 42,
    alphas: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Fit one ridge model for a per-mouse or per-session dataset."""
    transformed_y = np.log10(y) if log_target else y
    x_train, x_test, y_train, y_test = train_test_split(
        x, transformed_y, test_size=0.2, random_state=random_state
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
            best_r2 = r2
            best_alpha = alpha
            best_model = model

    y_pred = best_model.predict(x_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    corr = float(pearsonr(y_test, y_pred)[0]) if len(y_test) > 1 else np.nan
    return {
        "best_alpha": float(best_alpha),
        "r2_test": float(best_r2),
        "rmse": rmse,
        "pearson_r": corr,
        "y_test": y_test,
        "y_pred": y_pred,
        "n_neurons": int(x.shape[1]),
    }


def run_mouse_ridge(sound_types: Iterable[str] | None = None, windows: Iterable[str] = WINDOW_ORDER) -> pd.DataFrame:
    """Run ridge fits across all mice, brain regions, and spike windows."""
    records = []
    for sound_type in sound_types or list_available_sound_types():
        for mouse_id in available_mice(sound_type):
            for window_name in windows:
                for brain_area in get_brain_regions(sound_type):
                    dataset = build_dataset(sound_type, window_name, brain_area, mouse_id=mouse_id)
                    if dataset is None:
                        continue
                    if sound_type == "speech":
                        targets = [("FT", dataset["Y"][:, 0], False), ("VOT", dataset["Y"][:, 1], False)]
                    else:
                        targets = [(sound_type, dataset["Y"], sound_type in {"AM", "PT"})]

                    for target_name, target_values, log_target in targets:
                        fit = fit_best_ridge(dataset["X"], target_values, log_target=log_target)
                        records.append(
                            {
                                "Mouse": mouse_id,
                                "Brain Area": brain_area,
                                "Sound Type": sound_type,
                                "Target": target_name,
                                "Window": window_name,
                                "Neurons": fit["n_neurons"],
                                "Best Alpha": fit["best_alpha"],
                                "R2 Test": fit["r2_test"],
                                "RMSE": fit["rmse"],
                                "Pearson r": fit["pearson_r"],
                            }
                        )
    return pd.DataFrame(records)
