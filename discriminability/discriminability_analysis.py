"""Shared loading, session subsampling, and plotting helpers for discriminability analyses."""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from methods.methods_analysis import WINDOW_ORDER, load_sound_npz  # noqa: E402
from plot_stats import (  # noqa: E402
    add_pairwise_annotations,
    add_within_group_hue_annotations,
    box_centers,
    pairwise_group_tests,
    pairwise_hue_tests_within_group,
)

SOUND_ORDER = ("speech", "AM", "PT", "naturalSound")
SOUND_FILE_KEYS = {
    "speech": "speech",
    "AM": "AM",
    "PT": "pureTones",
    "naturalSound": "naturalSound",
}
SOUND_DISPLAY_NAMES = {
    "speech": "Speech",
    "AM": "AM",
    "PT": "Pure Tones",
    "naturalSound": "Natural Sounds",
}
NEURONS_PER_SESSION = {
    "speech": 10,
    "AM": 30,
    "PT": 30,
    "naturalSound": 30,
}
MAX_SESSIONS_PER_REGION = 5
N_SUBSAMPLINGS = 100
N_SPLITS = 5


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


def get_method_dir(method_folder: str) -> Path:
    """Return the output directory for one discriminability method."""
    output_dir = Path(params.figSavePath) / "discriminability" / method_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_plot_brain_regions(sound_type: str) -> list[str]:
    """Return the ordered brain regions to analyze for one sound."""
    regions = sorted(np.unique(load_sound_npz(sound_type)["brainRegionArray"]).tolist())
    if sound_type == "speech":
        regions = [region for region in regions if region != "Dorsal auditory area"]
    return [region for region in params.targetSiteNames if region in regions]


def speech_label(label: tuple[int, int] | np.ndarray) -> str:
    """Return one speech tuple label with a syllable tag when available."""
    key = tuple(np.asarray(label).tolist())
    suffix = f" ({params.SPEECH_SYLLABLES[key]})" if key in params.SPEECH_SYLLABLES else ""
    return f"{key}{suffix}"


def stimulus_display_labels(sound_type: str, stim_numeric: np.ndarray, stim_label_array: np.ndarray | None = None) -> np.ndarray:
    """Return readable trial-wise stimulus labels for one sound type."""
    if sound_type == "speech":
        return np.asarray([speech_label(label) for label in np.asarray(stim_numeric)], dtype=object)
    if sound_type == "naturalSound" and stim_label_array is not None:
        return np.asarray(stim_label_array, dtype=object)
    if sound_type == "AM":
        return np.asarray([f"{int(float(value))} Hz" for value in np.asarray(stim_numeric, dtype=float)], dtype=object)
    if sound_type == "PT":
        return np.asarray([f"{int(float(value))} Hz" for value in np.asarray(stim_numeric, dtype=float)], dtype=object)
    return np.asarray(stim_numeric, dtype=object)


def stimulus_order(sound_type: str, stim_numeric: np.ndarray | None = None) -> list[str]:
    """Return the preferred stimulus order for plotting."""
    if sound_type == "speech":
        return [speech_label(label) for label in params.unique_labels]
    if sound_type == "naturalSound":
        return params.NAT_SOUND_LABELS

    if stim_numeric is None:
        stim_numeric = load_sound_npz(sound_type)["stimArray"]
    unique_values = np.unique(np.asarray(stim_numeric, dtype=float))
    return [f"{int(float(value))} Hz" for value in unique_values]


def session_keys_for_region(sound_type: str, brain_area: str) -> list[tuple[str, str]]:
    """Return valid `(mouse_id, session_id)` keys for one sound and brain region."""
    sound_data = load_sound_npz(sound_type)
    mask = sound_data["brainRegionArray"] == brain_area
    counts: dict[tuple[str, str], int] = {}
    for mouse_id, session_id in zip(sound_data["mouseIDArray"][mask], sound_data["sessionIDArray"][mask]):
        key = (str(mouse_id), str(session_id))
        counts[key] = counts.get(key, 0) + 1

    min_neurons = NEURONS_PER_SESSION[sound_type]
    valid_sessions = [key for key, count in counts.items() if count >= min_neurons]
    valid_sessions.sort(key=lambda item: (item[1], item[0]))
    return valid_sessions[:MAX_SESSIONS_PER_REGION]


def session_seed(sound_type: str, brain_area: str, mouse_id: str, session_id: str, iteration: int) -> int:
    """Return a deterministic RNG seed for one session subsampling run."""
    key = f"{sound_type}|{brain_area}|{mouse_id}|{session_id}|{iteration}"
    return 42 + sum(key.encode("utf-8"))


def build_session_dataset(
    sound_type: str,
    window_name: str,
    brain_area: str,
    mouse_id: str,
    session_id: str,
    iteration: int,
) -> dict[str, np.ndarray] | None:
    """Return one neuron-subsampled, z-scored session dataset."""
    sound_data = load_sound_npz(sound_type)
    mask = (
        (sound_data["brainRegionArray"] == brain_area)
        & (sound_data["mouseIDArray"] == mouse_id)
        & (sound_data["sessionIDArray"] == session_id)
    )
    if np.count_nonzero(mask) < NEURONS_PER_SESSION[sound_type]:
        return None

    x_raw = sound_data[f"{window_name}fr"][mask]
    n_neurons = NEURONS_PER_SESSION[sound_type]
    if x_raw.shape[0] > n_neurons:
        rng = np.random.default_rng(session_seed(sound_type, brain_area, mouse_id, session_id, iteration))
        neuron_indices = np.sort(rng.choice(x_raw.shape[0], size=n_neurons, replace=False))
        x_raw = x_raw[neuron_indices]

    x = StandardScaler().fit_transform(x_raw.T)
    y_numeric = sound_data["stimArray"]
    y_labels = stimulus_display_labels(sound_type, y_numeric, sound_data.get("stimLabelArray"))
    return {
        "X": x,
        "Y_numeric": y_numeric,
        "Y_labels": y_labels,
    }


def ordered_trial_labels(sound_type: str, y_numeric: np.ndarray, y_labels: np.ndarray) -> list[str]:
    """Return the ordered stimulus labels present in one dataset."""
    preferred_order = stimulus_order(sound_type, stim_numeric=y_numeric)
    present_labels = set(np.asarray(y_labels, dtype=object).tolist())
    return [label for label in preferred_order if label in present_labels]


def natural_pair_type(label_left: str, label_right: str) -> str | None:
    """Return whether one natural-sound pair is within-category or between-category."""
    left_category = label_left.rsplit(" ", 1)[0]
    right_category = label_right.rsplit(" ", 1)[0]
    return "Within" if left_category == right_category else "Between"


def analysis_conditions() -> list[tuple[str, str, str, list[tuple[str, str]]]]:
    """Return the valid sound, region, window, and session combinations to analyze."""
    conditions = []
    for sound_type in SOUND_ORDER:
        for brain_area in get_plot_brain_regions(sound_type):
            sessions = session_keys_for_region(sound_type, brain_area)
            if not sessions:
                continue
            for window_name in WINDOW_ORDER:
                conditions.append((sound_type, brain_area, window_name, sessions))
    return conditions


def run_pairwise_analysis(
    metric_fn: Callable[[np.ndarray, np.ndarray, int], dict[str, float]],
) -> pd.DataFrame:
    """Run one session-subsampled pairwise discriminability analysis across all sounds."""
    records: list[dict[str, object]] = []
    conditions = analysis_conditions()
    total_steps = sum(len(sessions) * N_SUBSAMPLINGS for _, _, _, sessions in conditions)
    progress = tqdm(total=total_steps, desc="Discriminability analysis", unit="subsample", dynamic_ncols=True)
    try:
        for sound_type, brain_area, window_name, sessions in conditions:
            progress.set_postfix_str(
                f"{sound_type} | {params.short_names.get(brain_area, brain_area)} | {window_name}",
                refresh=False,
            )
            for mouse_id, session_id in sessions:
                for iteration in range(N_SUBSAMPLINGS):
                    session_data = build_session_dataset(sound_type, window_name, brain_area, mouse_id, session_id, iteration)
                    progress.update(1)
                    if session_data is None:
                        continue

                    x = session_data["X"]
                    y_labels = session_data["Y_labels"]
                    labels = ordered_trial_labels(sound_type, session_data["Y_numeric"], y_labels)
                    for label_left, label_right in combinations(labels, 2):
                        left_mask = y_labels == label_left
                        right_mask = y_labels == label_right
                        metrics = metric_fn(x[left_mask], x[right_mask], session_seed(sound_type, brain_area, mouse_id, session_id, iteration))
                        record = {
                            "Sound Type": sound_type,
                            "Brain Area": brain_area,
                            "Window": window_name,
                            "Mouse ID": mouse_id,
                            "Session ID": session_id,
                            "Iteration": iteration,
                            "Stim 1": label_left,
                            "Stim 2": label_right,
                            "Pair Type": natural_pair_type(label_left, label_right) if sound_type == "naturalSound" else None,
                        }
                        record.update(metrics)
                        records.append(record)
    finally:
        close = getattr(progress, "close", None)
        if close is not None:
            close()
    return pd.DataFrame(records)


def aggregate_session_scores(results_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Collapse pairwise rows to one mean score per session and subsampling run."""
    return (
        results_df.groupby(
            ["Sound Type", "Brain Area", "Window", "Mouse ID", "Session ID", "Iteration"],
            as_index=False,
        )[value_col]
        .mean()
        .rename(columns={value_col: "Score"})
    )


def aggregate_natural_pair_type_scores(results_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Collapse natural-sound rows to one mean within/between score per session and subsampling run."""
    natural_df = results_df[results_df["Sound Type"] == "naturalSound"].dropna(subset=["Pair Type"]).copy()
    return (
        natural_df.groupby(
            ["Sound Type", "Brain Area", "Window", "Mouse ID", "Session ID", "Iteration", "Pair Type"],
            as_index=False,
        )[value_col]
        .mean()
        .rename(columns={value_col: "Score"})
    )


def plot_heatmaps(
    results_df: pd.DataFrame,
    *,
    value_col: str,
    method_name: str,
    method_folder: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    """Create one heatmap grid per sound type."""
    apply_figure_style()
    output_dir = get_method_dir(method_folder)
    for sound_type in SOUND_ORDER:
        sound_df = results_df[results_df["Sound Type"] == sound_type].copy()
        if sound_df.empty:
            continue
        brain_regions = get_plot_brain_regions(sound_type)
        labels = stimulus_order(sound_type)
        fig, axes = plt.subplots(
            len(brain_regions),
            len(WINDOW_ORDER),
            figsize=(4.6 * len(WINDOW_ORDER), 3.8 * len(brain_regions)),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(f"{SOUND_DISPLAY_NAMES[sound_type]} {method_name} heatmaps", fontsize=16, fontweight="bold")

        for row_index, brain_area in enumerate(brain_regions):
            for col_index, window_name in enumerate(WINDOW_ORDER):
                ax = axes[row_index, col_index]
                panel_df = sound_df[(sound_df["Brain Area"] == brain_area) & (sound_df["Window"] == window_name)].copy()
                if panel_df.empty:
                    ax.axis("off")
                    continue

                matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
                grouped = (
                    panel_df.groupby(["Stim 1", "Stim 2"], as_index=False)[value_col]
                    .mean()
                    .rename(columns={value_col: "Value"})
                )
                for row in grouped.itertuples(index=False):
                    matrix.loc[row[0], row[1]] = row[2]
                    matrix.loc[row[1], row[0]] = row[2]

                sns.heatmap(
                    matrix,
                    ax=ax,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    square=True,
                    cbar=(row_index == 0 and col_index == len(WINDOW_ORDER) - 1),
                    cbar_kws={"label": method_name},
                )
                ax.set_title(f"{params.short_names.get(brain_area, brain_area)}\n{window_name.capitalize()}", fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis="x", rotation=60, labelsize=7)
                ax.tick_params(axis="y", rotation=0, labelsize=7)

        fig.savefig(output_dir / f"{sound_type}_heatmaps.png", dpi=300)
        plt.close(fig)


def plot_region_boxplots(
    results_df: pd.DataFrame,
    *,
    value_col: str,
    method_name: str,
    method_folder: str,
    ylabel: str,
) -> None:
    """Create one 1x3 region-comparison boxplot figure per sound type."""
    apply_figure_style()
    output_dir = get_method_dir(method_folder)
    summary_df = aggregate_session_scores(results_df, value_col=value_col)
    for sound_type in SOUND_ORDER:
        sound_df = summary_df[summary_df["Sound Type"] == sound_type].copy()
        if sound_df.empty:
            continue
        brain_regions = get_plot_brain_regions(sound_type)
        fig, axes = plt.subplots(1, len(WINDOW_ORDER), figsize=(4.6 * len(WINDOW_ORDER), 4.8), sharey=True, constrained_layout=True)
        fig.suptitle(f"{SOUND_DISPLAY_NAMES[sound_type]} {method_name}", fontsize=16, fontweight="bold")
        y_min = float(sound_df["Score"].min())
        y_max = float(sound_df["Score"].max())
        max_annotations = len(brain_regions) * (len(brain_regions) - 1) // 2
        y_step = 0.035 * ((y_max - y_min) if y_max > y_min else 1.0)

        for ax, window_name in zip(np.ravel(axes), WINDOW_ORDER):
            panel_df = sound_df[sound_df["Window"] == window_name].copy()
            sns.boxplot(
                data=panel_df,
                x="Brain Area",
                y="Score",
                order=brain_regions,
                width=0.5,
                fliersize=2,
                linewidth=1,
                ax=ax,
            )
            sns.stripplot(
                data=panel_df,
                x="Brain Area",
                y="Score",
                order=brain_regions,
                color="black",
                alpha=0.35,
                size=3,
                ax=ax,
            )
            stats_df = pairwise_group_tests(
                panel_df,
                group_col="Brain Area",
                value_col="Score",
                group_order=brain_regions,
                pair_cols=["Mouse ID", "Session ID", "Iteration"],
            )
            add_pairwise_annotations(
                ax,
                stats_df,
                centers=box_centers(brain_regions),
                data_max=y_max,
                data_min=y_min,
            )
            ax.set_title(window_name.capitalize(), fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(ylabel if window_name == WINDOW_ORDER[0] else "")
            ax.set_xticklabels([params.short_names.get(region, region) for region in brain_regions], rotation=20)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
            ax.set_ylim(y_min - y_step, y_max + y_step * (max_annotations + 2))

        fig.savefig(output_dir / f"{sound_type}_region_boxplots.png", dpi=300)
        plt.close(fig)


def plot_natural_within_between_boxplots(
    results_df: pd.DataFrame,
    *,
    value_col: str,
    method_name: str,
    method_folder: str,
    ylabel: str,
) -> None:
    """Create the natural-sound within-vs-between boxplot figure."""
    apply_figure_style()
    output_dir = get_method_dir(method_folder)
    summary_df = aggregate_natural_pair_type_scores(results_df, value_col=value_col)
    if summary_df.empty:
        return

    brain_regions = get_plot_brain_regions("naturalSound")
    hue_order = ["Within", "Between"]
    fig, axes = plt.subplots(1, len(WINDOW_ORDER), figsize=(4.8 * len(WINDOW_ORDER), 4.8), sharey=True, constrained_layout=True)
    fig.suptitle(f"Natural Sounds {method_name} within vs between category", fontsize=16, fontweight="bold")
    y_min = float(summary_df["Score"].min())
    y_max = float(summary_df["Score"].max())
    max_annotations = len(hue_order)
    y_step = 0.035 * ((y_max - y_min) if y_max > y_min else 1.0)

    for ax, window_name in zip(np.ravel(axes), WINDOW_ORDER):
        panel_df = summary_df[summary_df["Window"] == window_name].copy()
        sns.boxplot(
            data=panel_df,
            x="Brain Area",
            y="Score",
            hue="Pair Type",
            order=brain_regions,
            hue_order=hue_order,
            width=0.5,
            palette={"Within": "skyblue", "Between": "salmon"},
            showfliers=False,
            linewidth=1,
            ax=ax,
        )
        sns.stripplot(
            data=panel_df,
            x="Brain Area",
            y="Score",
            hue="Pair Type",
            order=brain_regions,
            hue_order=hue_order,
            palette={"Within": "skyblue", "Between": "salmon"},
            dodge=True,
            alpha=0.35,
            size=3,
            linewidth=0,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        stats_df = pairwise_hue_tests_within_group(
            panel_df,
            group_col="Brain Area",
            group_order=brain_regions,
            hue_col="Pair Type",
            hue_order=hue_order,
            value_col="Score",
            pair_cols=["Mouse ID", "Session ID", "Iteration"],
        )
        add_within_group_hue_annotations(
            ax,
            stats_df,
            centers=box_centers(brain_regions, hue_levels=hue_order),
            data_max=y_max,
            data_min=y_min,
        )
        ax.set_title(window_name.capitalize(), fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if window_name == WINDOW_ORDER[0] else "")
        ax.set_xticklabels([params.short_names.get(region, region) for region in brain_regions], rotation=20)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
        ax.set_ylim(y_min - y_step, y_max + y_step * (max_annotations + 2))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[: len(hue_order)], labels[: len(hue_order)], title="Pair Type", loc="upper right", frameon=False)
    fig.savefig(output_dir / "naturalSound_within_between_boxplots.png", dpi=300)
    plt.close(fig)
