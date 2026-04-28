"""Ridge-specific analysis helpers built on the shared project pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import funcs, params  # noqa: E402
from shared.plot_stats import add_pairwise_annotations, box_centers, pairwise_group_tests  # noqa: E402

WINDOW_ORDER = params.WINDOW_ORDER
SOUND_DISPLAY_NAMES = params.SOUND_DISPLAY_NAMES

# Font size hierarchy matching project-wide style
FONTSIZE_SUPTITLE = 38
FONTSIZE_TITLE = 34
FONTSIZE_LABEL = 34
RIDGE_ALPHAS = funcs.RIDGE_ALPHAS
apply_figure_style = funcs.apply_figure_style
build_dataset = funcs.build_dataset
build_sampled_dataset = funcs.build_sampled_dataset
get_plot_brain_regions = funcs.get_plot_brain_regions
get_target_neuron_count = funcs.get_target_neuron_count
list_available_sound_types = funcs.list_available_sound_types
available_mice = funcs.available_mice
available_sessions = funcs.available_sessions


def get_output_dir() -> Path:
    """Return the ridge figure output directory."""
    return funcs.get_figure_dir("decoding/ridge")


def build_target_datasets(dataset: dict[str, np.ndarray]) -> list[dict[str, np.ndarray | str | bool]]:
    """Split one shared dataset into one or two ridge regression targets."""
    sound_type = dataset["sound_type"]
    if sound_type == "speech":
        return [
            {
                "sound_type": sound_type,
                "target_name": "FT",
                "window_name": dataset["window_name"],
                "brain_area": dataset["brain_area"],
                "X": dataset["X"],
                "Y": dataset["Y"][:, 0],
                "log_target": False,
            },
            {
                "sound_type": sound_type,
                "target_name": "VOT",
                "window_name": dataset["window_name"],
                "brain_area": dataset["brain_area"],
                "X": dataset["X"],
                "Y": dataset["Y"][:, 1],
                "log_target": False,
            },
        ]

    return [
        {
            "sound_type": sound_type,
            "target_name": sound_type,
            "window_name": dataset["window_name"],
            "brain_area": dataset["brain_area"],
            "X": dataset["X"],
            "Y": dataset["Y"].astype(float),
            "log_target": sound_type in {"AM", "PT"},
        }
    ]


def build_population_target_datasets(
    sound_type: str,
    window_name: str,
    brain_area: str,
) -> list[dict[str, np.ndarray | str | bool]]:
    """Build one ridge target dataset list for a population condition."""
    dataset = funcs.build_population_dataset(sound_type, window_name, brain_area)
    if dataset is None:
        return []
    return build_target_datasets(dataset)


def compute_ridge_alpha_tuning(
    sound_type: str,
    window_name: str,
    brain_area: str,
) -> list[dict[str, object]]:
    """Sweep every RIDGE_ALPHAS value on the population dataset and record mean CV R².

    For each alpha the full 5-fold ridge pipeline is run with that single alpha
    fixed (no internal alpha search), giving a tuning curve analogous to the SVM
    C-parameter tuning.  Speech uses a single tuple-form target for consistency
    with the per-session analysis.  Returns one record per (target × alpha).
    """
    dataset = funcs.build_population_dataset(sound_type, window_name, brain_area)
    if dataset is None:
        return []

    if sound_type == "speech":
        target_datasets = [
            {
                "target_name": "Speech Tuple",
                "X": dataset["X"],
                "Y": funcs.labels_for_sound("speech", dataset["Y"]).astype(float),
                "log_target": False,
            }
        ]
    else:
        target_datasets = build_target_datasets(dataset)

    records = []
    for td in target_datasets:
        for alpha in RIDGE_ALPHAS:
            fit = fit_best_ridge(
                td["X"],
                td["Y"],
                log_target=bool(td["log_target"]),
                alphas=np.array([float(alpha)]),
            )
            records.append(
                {
                    "Sound Type": sound_type,
                    "Brain Area": brain_area,
                    "Window": window_name,
                    "Target": td["target_name"],
                    "Alpha": float(alpha),
                    "R2": float(fit["r2_test"]),
                }
            )
    return records


def iter_population_datasets(
    sound_types: Iterable[str] | None = None,
    windows: Iterable[str] = params.WINDOW_ORDER,
) -> list[dict[str, np.ndarray | str | bool]]:
    """Iterate through all available ridge population datasets."""
    datasets = []
    for sound_type in sound_types or funcs.list_available_sound_types():
        for window_name in windows:
            for brain_area in funcs.get_brain_regions(sound_type):
                datasets.extend(build_population_target_datasets(sound_type, window_name, brain_area))
    return datasets


def fit_best_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    log_target: bool = False,
    random_state: int = 42,
    alphas: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Run the shared 5-fold standardized ridge pipeline."""
    fit = funcs.run_ridge_cv(
        x,
        y,
        log_target=log_target,
        random_state=random_state,
        alphas=alphas,
    )
    return {
        "best_alpha": float(fit["mean_alpha"]),
        "r2_test": float(fit["mean_r2"]),
        "rmse": float(fit["mean_rmse"]),
        "pearson_r": float(fit["pearson_r"]),
        "y_test": fit["y_true"],
        "y_pred": fit["y_pred"],
        "n_neurons": int(fit["n_neurons"]),
        "fold_r2": fit["fold_r2"],
        "fold_alphas": fit["fold_alphas"],
    }


def run_population_ridge(
    sound_types: Iterable[str] | None = None,
    windows: Iterable[str] = params.WINDOW_ORDER,
    iterations: int = 30,
) -> pd.DataFrame:
    """Run repeated ridge fits across all sound, region, and window conditions."""
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
                    log_target=bool(dataset["log_target"]),
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
                    }
                )
    finally:
        close = getattr(progress, "close", None)
        if close is not None:
            close()
    return pd.DataFrame(records)


def target_order_for_sound(sound_type: str, results_df: pd.DataFrame) -> list[str]:
    """Return the plotted ridge target order for one sound."""
    canonical_order = ["FT", "VOT", sound_type, "Speech Tuple"]
    present_targets = results_df["Target"].unique().tolist()
    ordered = [target for target in canonical_order if target in present_targets]
    return ordered or present_targets


def plot_ridge_summary(
    sound_type: str,
    results_df: pd.DataFrame,
    *,
    title: str,
    filename: str,
    pair_cols: list[str],
    session_counts: dict[str, int] | None = None,
) -> None:
    """Plot one standard ridge region-comparison summary figure."""
    if results_df.empty:
        return

    brain_regions = get_plot_brain_regions(sound_type)
    target_order = target_order_for_sound(sound_type, results_df)
    use_hue = len(target_order) > 1
    fig, axes = plt.subplots(
        1,
        len(WINDOW_ORDER),
        figsize=(6.5 * len(WINDOW_ORDER), 7.0),
        squeeze=False,
        sharey=True,
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.84, bottom=0.20)
    fig.suptitle(title, fontsize=FONTSIZE_SUPTITLE, fontweight="bold", y=0.97)
    y_min = float(results_df["R2 Test"].min())
    y_max = float(results_df["R2 Test"].max())
    max_annotations = len(target_order) * (len(brain_regions) * (len(brain_regions) - 1) // 2)
    y_step = 0.035 * ((y_max - y_min) if y_max > y_min else 1.0)

    for col_index, window_name in enumerate(WINDOW_ORDER):
        ax = axes[0, col_index]
        panel_df = results_df[results_df["Window"] == window_name].copy()
        if panel_df.empty:
            ax.axis("off")
            continue
        region_palette = sns.color_palette("viridis", n_colors=len(brain_regions))

        box_kwargs = {
            "data": panel_df,
            "x": "Brain Area",
            "y": "R2 Test",
            "order": brain_regions,
            "width": 0.24,
            "showfliers": False,
            "linewidth": 1.2,
            "ax": ax,
        }
        strip_kwargs = {
            "data": panel_df,
            "x": "Brain Area",
            "y": "R2 Test",
            "order": brain_regions,
            "dodge": use_hue,
            "size": 4,
            "alpha": 0.70,
            "ax": ax,
        }
        if not use_hue:
            box_kwargs["palette"] = region_palette
        if use_hue:
            box_kwargs.update({"hue": "Target", "hue_order": target_order})
            strip_kwargs.update({"hue": "Target", "hue_order": target_order})

        sns.boxplot(**box_kwargs)
        sns.stripplot(color="black", **strip_kwargs)
        if ax.legend_ is not None:
            ax.legend_.remove()

        stats_df = pairwise_group_tests(
            panel_df,
            group_col="Brain Area",
            value_col="R2 Test",
            group_order=brain_regions,
            hue_col="Target" if use_hue else None,
            hue_order=target_order if use_hue else None,
            pair_cols=pair_cols,
            test_mode="unpaired",
        )
        add_pairwise_annotations(
            ax,
            stats_df,
            centers=box_centers(brain_regions, hue_levels=target_order if use_hue else None),
            data_max=y_max,
            data_min=y_min,
        )
        ax.set_title(window_name.capitalize(), fontsize=FONTSIZE_TITLE, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("$R^2$" if col_index == 0 else "", fontsize=FONTSIZE_LABEL)
        ax.set_xticks(range(len(brain_regions)))
        if use_hue:
            ax.set_xticklabels(
                [params.short_names.get(r, r) for r in brain_regions],
                rotation=20, fontsize=FONTSIZE_LABEL,
            )
        else:
            ax.set_xticklabels([""] * len(brain_regions))
        ax.tick_params(axis="y", labelsize=FONTSIZE_LABEL)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
        ax.set_ylim(y_min - y_step, y_max + y_step * (max_annotations + 2))

    if use_hue:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles[: len(target_order)], labels[: len(target_order)],
                title="Target", loc="upper right", frameon=True,
                facecolor="white", edgecolor="0.75", fontsize=FONTSIZE_LABEL - 8,
            )
    else:
        _pal = sns.color_palette("viridis", n_colors=len(brain_regions))
        legend_handles = []
        for i, r in enumerate(brain_regions):
            lbl = params.short_names.get(r, r)
            if session_counts and r in session_counts:
                lbl = f"{lbl} (n={session_counts[r]})"
            legend_handles.append(mpatches.Patch(facecolor=_pal[i], edgecolor="black", linewidth=0.8, label=lbl))
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.54, 0.01),
            ncol=len(brain_regions),
            fontsize=FONTSIZE_LABEL - 8,
            frameon=True,
            facecolor="white",
            edgecolor="0.4",
        )
    fig.savefig(get_output_dir() / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
