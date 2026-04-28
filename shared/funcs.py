"""Shared data loading, dataset building, subsampling, and cross-validation helpers.

All analysis modules import from this file rather than reimplementing these utilities.
Key responsibilities:
  - load_sound_npz: cached loading of .npz firing-rate arrays built by build_firing_rate_arrays.py
  - build_dataset / build_population_dataset / build_sampled_dataset: construct trial × neuron
    matrices for a given sound type, spike window, and brain region
  - run_ridge_cv: standardized 5-fold ridge regression pipeline with inner RidgeCV alpha tuning
  - run_loo_classifier_cv: leave-one-out classifier pipeline used by discriminability analyses
  - apply_figure_style: sets the shared serif/paper-style matplotlib theme used across all figures
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler

from shared import params

RIDGE_ALPHAS = np.logspace(-5, 10, 200)
RIDGE_N_SPLITS = 5
DISCRIMINABILITY_SVM_C_VALUES = np.logspace(-5, 4, 20)


def get_data_dir() -> Path:
    """Return the directory containing saved firing-rate arrays."""
    return Path(params.dbSavePath)


def get_figure_dir(method_name: str) -> Path:
    """Return one method-specific figure output directory."""
    figure_dir = Path(params.figSavePath) / method_name
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def get_nested_figure_dir(method_name: str, *parts: str) -> Path:
    """Return a nested figure output directory below one method root."""
    figure_dir = get_figure_dir(method_name).joinpath(*parts)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def get_results_path(method_name: str, result_name: str) -> Path:
    """Return one top-level CSV path inside a method figure directory."""
    return get_figure_dir(method_name) / result_name


def collapse_stimulus_array(stim_array: np.ndarray, n_trials: int) -> np.ndarray:
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


def list_available_sound_types() -> list[str]:
    """List the sound types with saved `.npz` arrays available."""
    data_dir = get_data_dir()
    available = []
    for sound_type in params.SOUND_ORDER:
        file_key = params.SOUND_FILE_KEYS[sound_type]
        if (data_dir / f"fr_arrays_{file_key}.npz").exists():
            available.append(sound_type)
    return available


@lru_cache(maxsize=None)
def load_sound_npz(sound_type: str) -> dict[str, np.ndarray]:
    """Load one firing-rate array file and normalize its label shapes."""
    filepath = get_data_dir() / f"fr_arrays_{params.SOUND_FILE_KEYS[sound_type]}.npz"
    raw = np.load(filepath, allow_pickle=True)
    onset = raw["onsetfr"]
    stim_labels = collapse_stimulus_array(raw["stimArray"], onset.shape[1])
    stim_numeric_source = raw["stimNumericArray"] if "stimNumericArray" in raw else raw["stimArray"]
    return {
        "onsetfr": onset,
        "sustainedfr": raw["sustainedfr"],
        "offsetfr": raw["offsetfr"],
        "brainRegionArray": raw["brainRegionArray"],
        "stimArray": collapse_stimulus_array(stim_numeric_source, onset.shape[1]),
        "stimLabelArray": stim_labels,
        "mouseIDArray": raw["mouseIDArray"],
        "sessionIDArray": raw["sessionIDArray"],
    }


def get_brain_regions(sound_type: str) -> list[str]:
    """Return the sorted brain regions present for one sound type."""
    return sorted(np.unique(load_sound_npz(sound_type)["brainRegionArray"]).tolist())


def get_plot_brain_regions(sound_type: str) -> list[str]:
    """Return the ordered brain regions to analyze for one sound."""
    regions = get_brain_regions(sound_type)
    if sound_type == "speech":
        regions = [region for region in regions if region != "Dorsal auditory area"]
    return [region for region in params.targetSiteNames if region in regions]


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
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "legend.title_fontsize": 22,
            "figure.titlesize": 26,
        },
    )
    # Explicitly reinforce Times New Roman after seaborn theme so it is never
    # silently overridden by a subsequent rcParams reset elsewhere.
    import matplotlib as _mpl
    _mpl.rcParams["font.family"] = "serif"
    _mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]


def trial_indices_from_selector(selector: np.ndarray, n_trials: int) -> np.ndarray:
    """Convert a trial selector into explicit integer trial indices."""
    selector_array = np.asarray(selector)
    if selector_array.dtype == bool:
        return np.flatnonzero(selector_array)
    selector_array = selector_array.astype(int).ravel()
    if selector_array.size == n_trials and np.all(np.isin(selector_array, [0, 1])):
        return np.flatnonzero(selector_array)
    return selector_array


def normalize_index_limits(index_limits: np.ndarray) -> np.ndarray:
    """Ensure event-locked index limits are shaped as trials x 2."""
    limits = np.asarray(index_limits)
    if limits.ndim != 2:
        raise ValueError(f"Unexpected index_limits shape {limits.shape}.")
    if limits.shape[1] == 2:
        return limits
    if limits.shape[0] == 2:
        return limits.T
    raise ValueError(f"Unexpected index_limits shape {limits.shape}.")


def get_target_neuron_count(sound_type: str) -> int:
    """Return the fixed population neuron count used in thesis figures."""
    return 99 if sound_type == "speech" else 278


@lru_cache(maxsize=None)
def get_sound_min_neuron_count(sound_type: str) -> int:
    """Return the minimum neuron count across brain regions for one sound type."""
    sound_data = load_sound_npz(sound_type)
    counts = [
        np.count_nonzero(sound_data["brainRegionArray"] == brain_area)
        for brain_area in get_brain_regions(sound_type)
    ]
    if not counts:
        raise ValueError(f"No brain regions found for {sound_type}.")
    return min(counts)


def sample_neuron_indices(
    sound_type: str,
    brain_area: str,
    n_neurons: int | None = None,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Return reproducibly sampled neuron indices for one sound and brain area."""
    sound_data = load_sound_npz(sound_type)
    brain_indices = np.flatnonzero(sound_data["brainRegionArray"] == brain_area)
    if len(brain_indices) == 0:
        return brain_indices

    target = n_neurons if n_neurons is not None else min(get_target_neuron_count(sound_type), len(brain_indices))
    if len(brain_indices) <= target:
        return brain_indices

    rng = np.random.default_rng(seed + sum(f"{sound_type}-{brain_area}".encode("utf-8")))
    return np.sort(rng.choice(brain_indices, size=target, replace=False))


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


def build_dataset(
    sound_type: str,
    window_name: str,
    brain_area: str,
    *,
    mouse_id: str | None = None,
    session_id: str | None = None,
    cell_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray] | None:
    """Build one shared dataset for a sound, time window, and region subset."""
    sound_data = load_sound_npz(sound_type)
    if cell_indices is None:
        mask = sound_data["brainRegionArray"] == brain_area
        if mouse_id is not None:
            mask &= sound_data["mouseIDArray"] == mouse_id
        if session_id is not None:
            mask &= sound_data["sessionIDArray"] == session_id
        if not np.any(mask):
            return None
        cell_indices = np.flatnonzero(mask)
    else:
        cell_indices = np.asarray(cell_indices, dtype=int)
        if len(cell_indices) == 0:
            return None

    return {
        "sound_type": sound_type,
        "window_name": window_name,
        "brain_area": brain_area,
        "X": sound_data[params.WINDOW_TO_KEY[window_name]][cell_indices].T,
        "Y": sound_data["stimArray"],
        "Y_labels": sound_data["stimLabelArray"],
        "mouse_ids": sound_data["mouseIDArray"][cell_indices],
        "session_ids": sound_data["sessionIDArray"][cell_indices],
        "cell_indices": cell_indices,
    }


def build_sampled_dataset(
    sound_type: str,
    window_name: str,
    brain_area: str,
    *,
    n_neurons: int | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray] | None:
    """Build one dataset after reproducible neuron subsampling."""
    return build_dataset(
        sound_type,
        window_name,
        brain_area,
        cell_indices=sample_neuron_indices(sound_type, brain_area, n_neurons=n_neurons, seed=seed),
    )


def build_population_dataset(sound_type: str, window_name: str, brain_area: str) -> dict[str, np.ndarray] | None:
    """Build one equal-neuron population dataset for shared analyses."""
    return build_dataset(
        sound_type,
        window_name,
        brain_area,
        cell_indices=get_sampled_cell_indices(sound_type, brain_area),
    )


def labels_for_sound(sound_type: str, stim_array: np.ndarray) -> np.ndarray:
    """Convert stimulus identities into numeric color values."""
    if sound_type == "speech":
        label_to_number = {tuple(label): idx for idx, label in enumerate(params.unique_labels)}
        return np.array([label_to_number[tuple(label)] for label in np.asarray(stim_array)])

    stim_values = np.asarray(stim_array)
    if stim_values.ndim > 1:
        if stim_values.dtype.kind in {"O", "U", "S"}:
            _, inverse = np.unique(stim_values, axis=0, return_inverse=True)
            return inverse
        first_row = stim_values[0]
        if np.allclose(stim_values, first_row):
            stim_values = first_row

    stim_values = np.asarray(stim_values, dtype=float)
    if np.any(stim_values <= 0):
        _, inverse = np.unique(stim_values, return_inverse=True)
        return inverse
    return np.log10(stim_values)


def format_speech_label(label: tuple[int, int] | np.ndarray) -> str:
    """Format one speech tuple as `"(FT, VOT)"`."""
    ft_value, vot_value = tuple(np.asarray(label).tolist())
    return str((ft_value, vot_value))


def speech_label(label: tuple[int, int] | np.ndarray, *, include_syllable: bool = False) -> str:
    """Return one speech tuple label, optionally with the syllable endpoint tag."""
    key = tuple(np.asarray(label).tolist())
    suffix = f" ({params.SPEECH_SYLLABLES[key]})" if include_syllable and key in params.SPEECH_SYLLABLES else ""
    return f"{format_speech_label(key)}{suffix}"


def stimulus_display_labels(
    sound_type: str,
    stim_numeric: np.ndarray,
    stim_label_array: np.ndarray | None = None,
    *,
    include_speech_syllables: bool = False,
) -> np.ndarray:
    """Return readable trial-wise stimulus labels for one sound type."""
    if sound_type == "speech":
        return np.asarray(
            [speech_label(label, include_syllable=include_speech_syllables) for label in np.asarray(stim_numeric)],
            dtype=object,
        )
    if sound_type == "naturalSound":
        if stim_label_array is not None:
            return np.asarray(stim_label_array, dtype=object)
        label_array = np.empty(np.asarray(stim_numeric).shape, dtype=object)
        for index, value in np.ndenumerate(np.asarray(stim_numeric, dtype=float)):
            label_array[index] = "" if np.isnan(value) else params.NAT_SOUND_LABEL_MAP[int(value)]
        return label_array
    if sound_type in {"AM", "PT"}:
        return np.asarray([f"{int(float(value))} Hz" for value in np.asarray(stim_numeric, dtype=float)], dtype=object)
    return np.asarray(stim_numeric, dtype=object)


def select_speech_trials(labels: np.ndarray, max_repeats: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first `max_repeats` repeats for each speech token and sort by `(FT, VOT)`."""
    counts: dict[tuple[int, int], int] = {}
    keep_indices = []
    for trial_index, label in enumerate(map(tuple, labels.tolist())):
        repeat_count = counts.get(label, 0)
        if repeat_count < max_repeats:
            keep_indices.append(trial_index)
            counts[label] = repeat_count + 1

    keep_indices = np.asarray(keep_indices, dtype=int)
    filtered_labels = labels[keep_indices]
    sort_order = np.lexsort((filtered_labels[:, 1], filtered_labels[:, 0]))
    return keep_indices[sort_order], filtered_labels[sort_order]


def stimulus_order(
    sound_type: str,
    stim_numeric: np.ndarray | None = None,
    *,
    include_speech_syllables: bool = False,
) -> list[str]:
    """Return the preferred stimulus order for plotting."""
    if sound_type == "speech":
        return [speech_label(label, include_syllable=include_speech_syllables) for label in params.unique_labels]
    if sound_type == "naturalSound":
        return params.NAT_SOUND_LABELS
    if stim_numeric is None:
        stim_numeric = load_sound_npz(sound_type)["stimArray"]
    unique_values = np.unique(np.asarray(stim_numeric, dtype=float))
    return [f"{int(float(value))} Hz" for value in unique_values]


def stimulus_tick_labels(sound_type: str, stim_array: np.ndarray) -> list[str]:
    """Return human-readable stimulus labels for one sound type."""
    if sound_type == "speech":
        return [format_speech_label(label) for label in np.asarray(stim_array)]
    if sound_type == "naturalSound":
        return [params.NAT_SOUND_LABELS[int(value)] for value in np.asarray(stim_array, dtype=int)]
    if sound_type == "AM":
        return [f"{int(value)} Hz" for value in np.asarray(stim_array, dtype=float)]
    return [f"{int(value)} Hz" for value in np.asarray(stim_array, dtype=float)]


def ridge_target_values(y: np.ndarray, *, log_target: bool) -> np.ndarray:
    """Return regression targets in the fitted scale."""
    y = np.asarray(y, dtype=float)
    if log_target:
        return np.log10(np.clip(y, np.finfo(float).tiny, None))
    return y


def deterministic_seed(*parts: object, base: int = 42) -> int:
    """Return a reproducible integer seed from one or more key parts."""
    key = "|".join(map(str, parts))
    return base + sum(key.encode("utf-8"))


def build_subsampled_session_dataset(
    sound_type: str,
    window_name: str,
    brain_area: str,
    mouse_id: str,
    session_id: str,
    *,
    n_neurons: int,
    seed: int,
) -> dict[str, np.ndarray] | None:
    """Build one session dataset after deterministic neuron subsampling."""
    dataset = build_dataset(
        sound_type,
        window_name,
        brain_area,
        mouse_id=mouse_id,
        session_id=session_id,
    )
    if dataset is None:
        return None

    x_by_neuron = dataset["X"].T
    if x_by_neuron.shape[0] < n_neurons:
        return None
    if x_by_neuron.shape[0] > n_neurons:
        rng = np.random.default_rng(seed)
        neuron_indices = np.sort(rng.choice(x_by_neuron.shape[0], size=n_neurons, replace=False))
        x_by_neuron = x_by_neuron[neuron_indices]

    return {
        "X": x_by_neuron.T,
        "Y": dataset["Y"],
        "Y_labels": dataset["Y_labels"],
    }


def _ridge_inner_cv(n_samples: int, random_state: int) -> KFold:
    """Return the inner ridge CV splitter for one training fold."""
    return KFold(
        n_splits=max(2, min(RIDGE_N_SPLITS, n_samples)),
        shuffle=True,
        random_state=random_state,
    )


def run_ridge_cv(
    x: np.ndarray,
    y: np.ndarray,
    *,
    log_target: bool = False,
    random_state: int = 42,
    alphas: np.ndarray | None = None,
    n_splits: int = RIDGE_N_SPLITS,
) -> dict[str, np.ndarray | float]:
    """Run the shared standardized outer-CV ridge pipeline."""
    x = np.asarray(x, dtype=float)
    target = ridge_target_values(y, log_target=log_target)
    n_trials = x.shape[0]
    outer_splits = min(n_splits, n_trials)
    if outer_splits < 2:
        raise ValueError(f"Need at least 2 samples for ridge CV, got {n_trials}.")

    alphas = RIDGE_ALPHAS if alphas is None else np.asarray(alphas, dtype=float)
    splitter = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    fold_r2 = []
    fold_rmse = []
    fold_alphas = []
    all_true = []
    all_pred = []

    for fold_index, (train_index, test_index) in enumerate(splitter.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = target[train_index], target[test_index]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = RidgeCV(
            alphas=alphas,
            cv=_ridge_inner_cv(len(train_index), random_state + fold_index),
            scoring="r2",
        )
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)

        fold_r2.append(float(r2_score(y_test, y_pred)))
        fold_rmse.append(float(np.sqrt(mean_squared_error(y_test, y_pred))))
        fold_alphas.append(float(model.alpha_))
        all_true.append(y_test)
        all_pred.append(y_pred)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    pearson_value = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else np.nan

    return {
        "mean_r2": float(np.mean(fold_r2)),
        "mean_rmse": float(np.mean(fold_rmse)),
        "mean_alpha": float(np.mean(fold_alphas)),
        "pearson_r": pearson_value,
        "fold_r2": np.asarray(fold_r2, dtype=float),
        "fold_rmse": np.asarray(fold_rmse, dtype=float),
        "fold_alphas": np.asarray(fold_alphas, dtype=float),
        "y_true": y_true,
        "y_pred": y_pred,
        "n_neurons": int(x.shape[1]),
        "n_trials": int(n_trials),
    }


def run_loo_classifier_cv(
    model_factory,
    x: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 42,
    n_splits: int = 5,
    standardize: bool = True,
) -> dict[str, np.ndarray | float]:
    """Run the shared fold-wise standardized leave-one-out classifier CV pipeline."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    class_counts = np.bincount(y.astype(int))
    available_classes = class_counts[class_counts > 0]
    if len(available_classes) == 0 or int(available_classes.min()) < 2:
        return {
            "mean_accuracy": np.nan,
            "fold_accuracy": np.asarray([], dtype=float),
            "y_true": np.asarray([], dtype=y.dtype),
            "y_pred": np.asarray([], dtype=y.dtype),
        }

    splitter = LeaveOneOut()
    fold_scores = []
    all_true = []
    all_pred = []

    for train_index, test_index in splitter.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if standardize:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
        model = model_factory()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        fold_scores.append(float(np.mean(y_pred == y_test)))
        all_true.append(y_test)
        all_pred.append(y_pred)

    return {
        "mean_accuracy": float(np.mean(fold_scores)),
        "fold_accuracy": np.asarray(fold_scores, dtype=float),
        "y_true": np.concatenate(all_true),
        "y_pred": np.concatenate(all_pred),
    }
