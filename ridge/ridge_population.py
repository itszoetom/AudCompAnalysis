"""Create population ridge predicted-versus-actual figures for each sound type across regions and windows."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import params

try:
    from .ridge_analysis import SOUND_FILE_KEYS, WINDOW_ORDER, WINDOW_TO_KEY, apply_figure_style, get_output_dir
except ImportError:
    from ridge_analysis import SOUND_FILE_KEYS, WINDOW_ORDER, WINDOW_TO_KEY, apply_figure_style, get_output_dir

ALPHAS = np.logspace(-3, 3, 20)


def load_sound_npz(sound_type: str) -> dict[str, np.ndarray]:
    """Load one saved firing-rate array file."""
    file_key = SOUND_FILE_KEYS[sound_type]
    raw = np.load(os.path.join(params.dbSavePath, f"fr_arrays_{file_key}.npz"), allow_pickle=True)
    stim_numeric_source = raw["stimNumericArray"] if "stimNumericArray" in raw else raw["stimArray"]
    return {
        "onsetfr": raw["onsetfr"],
        "sustainedfr": raw["sustainedfr"],
        "offsetfr": raw["offsetfr"],
        "stimArray": stim_numeric_source[0] if sound_type != "speech" and stim_numeric_source.ndim > 1 else stim_numeric_source,
        "brainRegionArray": raw["brainRegionArray"],
    }


def get_plot_brain_regions(sound_type: str) -> list[str]:
    """Return the ordered brain regions to include in paper figures."""
    regions = sorted(np.unique(load_sound_npz(sound_type)["brainRegionArray"]).tolist())
    if sound_type == "speech":
        regions = [region for region in regions if region != "Dorsal auditory area"]
    return [region for region in params.targetSiteNames if region in regions]


def get_target_neuron_count(sound_type: str) -> int:
    """Return the fixed per-figure neuron count."""
    return 99 if sound_type == "speech" else 278


def sample_neurons(sound_data: dict[str, np.ndarray], sound_type: str, brain_area: str) -> np.ndarray:
    """Return reproducibly sampled neurons for one region."""
    brain_indices = np.flatnonzero(sound_data["brainRegionArray"] == brain_area)
    target = min(get_target_neuron_count(sound_type), len(brain_indices))
    if len(brain_indices) <= target:
        return brain_indices
    rng = np.random.default_rng(42 + sum(f"{sound_type}-{brain_area}".encode("utf-8")))
    return np.sort(rng.choice(brain_indices, size=target, replace=False))


def fit_population_cv(x: np.ndarray, y: np.ndarray, log_target: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """Run 5-fold ridge CV and return mean R2 with all held-out predictions."""
    target = np.log10(y + 1e-8) if log_target else y
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    all_true = []
    all_pred = []
    for train_index, test_index in splitter.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = target[train_index], target[test_index]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        ridge = RidgeCV(alphas=ALPHAS)
        ridge.fit(x_train, y_train)
        y_pred = ridge.predict(x_test)
        r2_scores.append(r2_score(y_test, y_pred))
        all_true.append(y_test)
        all_pred.append(y_pred)
    return float(np.mean(r2_scores)), np.concatenate(all_true), np.concatenate(all_pred)


def plot_panel(ax: plt.Axes, x: np.ndarray, y: np.ndarray, title: str, log_target: bool) -> None:
    """Plot one predicted-versus-actual ridge panel."""
    mean_r2, y_true, y_pred = fit_population_cv(x, y, log_target=log_target)
    if log_target:
        y_true = 10 ** y_true
        y_pred = 10 ** y_pred
    ax.scatter(y_true, y_pred, s=28, alpha=0.75, color="black")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.5, color="tab:red")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\n$R^2$={mean_r2:.2f}", fontweight="bold")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)


def main() -> None:
    """Run population ridge predicted-versus-actual figures."""
    apply_figure_style()
    for sound_type in ("speech", "AM", "PT", "naturalSound"):
        sound_data = load_sound_npz(sound_type)
        brain_regions = get_plot_brain_regions(sound_type)
        fig, axes = plt.subplots(
            len(brain_regions),
            len(WINDOW_ORDER),
            figsize=(4.0 * len(WINDOW_ORDER), 3.5 * len(brain_regions)),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(
            f"{sound_type} population ridge (n={get_target_neuron_count(sound_type)} neurons per region)",
            fontsize=16,
            fontweight="bold",
        )
        for row_index, brain_area in enumerate(brain_regions):
            neuron_indices = sample_neurons(sound_data, sound_type, brain_area)
            for col_index, window_name in enumerate(WINDOW_ORDER):
                ax = axes[row_index, col_index]
                if len(neuron_indices) == 0:
                    ax.axis("off")
                    continue
                x = sound_data[WINDOW_TO_KEY[window_name]][neuron_indices].T
                if sound_type == "speech":
                    y = sound_data["stimArray"][:, 0]
                    title = f"{params.short_names[brain_area]} {window_name.capitalize()} FT"
                else:
                    y = sound_data["stimArray"].astype(float)
                    title = f"{params.short_names[brain_area]} {window_name.capitalize()}"
                plot_panel(ax, x, y, title, log_target=sound_type in {"AM", "PT"})
        fig.savefig(get_output_dir() / f"{sound_type}_ridge_population.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
