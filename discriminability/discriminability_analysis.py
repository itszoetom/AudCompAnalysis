"""Shared discriminability analysis and plotting helpers."""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import funcs, params  # noqa: E402
from shared.plot_stats import (  # noqa: E402
    add_pairwise_annotations,
    box_centers,
    pairwise_group_tests,
)

SVM_C_VALUES = funcs.DISCRIMINABILITY_SVM_C_VALUES

# Font size hierarchy matching project-wide style
FONTSIZE_SUPTITLE = 38
FONTSIZE_TITLE = 32
FONTSIZE_LABEL = 28

VIRIDIS_PAIR_PALETTE = {
    "Within": plt.cm.viridis(0.25),
    "Between": plt.cm.viridis(0.8),
}


def get_figure_dir(*parts: str) -> Path:
    """Return a nested discriminability output directory."""
    return funcs.get_nested_figure_dir("decoding/discriminability", *parts)


def figure_output_dir(sound_type: str, method_key: str) -> Path:
    """Return the figure directory for one sound type and discriminability method."""
    if method_key == "lda":
        return get_figure_dir("adish/lda", sound_type)
    return get_figure_dir(sound_type)


def example_output_dir() -> Path:
    """Return the directory for illustrative discriminability figures."""
    return get_figure_dir("examples")


def get_results_path(method_key: str) -> Path:
    """Return the pairwise-results CSV path for one discriminability method."""
    return funcs.get_results_path("decoding/discriminability/adish", f"{method_key}_pairwise_results.csv")


def get_tuning_path() -> Path:
    """Return the linear-SVM hyperparameter-tuning CSV path."""
    return funcs.get_results_path("discriminability", "linearSVM_hyperparameter_tuning.csv")


def build_population_analysis_dataset(
    sound_type: str,
    window_name: str,
    brain_area: str,
) -> dict[str, np.ndarray] | None:
    """Return one equal-neuron, z-scored population dataset."""
    dataset = funcs.build_population_dataset(sound_type, window_name, brain_area)
    if dataset is None:
        return None

    return {
        "X": StandardScaler().fit_transform(dataset["X"]),
        "Y_numeric": dataset["Y"],
        "Y_labels": funcs.stimulus_display_labels(
            sound_type,
            dataset["Y"],
            dataset["Y_labels"],
            include_speech_syllables=True,
        ),
    }


def ordered_trial_labels(sound_type: str, y_numeric: np.ndarray, y_labels: np.ndarray) -> list[str]:
    """Return the ordered stimulus labels present in one dataset."""
    preferred_order = funcs.stimulus_order(sound_type, stim_numeric=y_numeric, include_speech_syllables=True)
    present_labels = set(np.asarray(y_labels, dtype=object).tolist())
    return [label for label in preferred_order if label in present_labels]


def natural_pair_type(label_left: str, label_right: str) -> str | None:
    """Return whether one natural-sound pair is within-category or between-category."""
    left_category = label_left.rsplit(" ", 1)[0]
    right_category = label_right.rsplit(" ", 1)[0]
    return "Within" if left_category == right_category else "Between"


def analysis_conditions() -> list[tuple[str, str, str]]:
    """Return the valid sound, region, and window combinations to analyze."""
    return [
        (sound_type, brain_area, window_name)
        for sound_type in params.SOUND_ORDER
        for brain_area in funcs.get_plot_brain_regions(sound_type)
        for window_name in params.WINDOW_ORDER
    ]


def collect_condition_records(
    sound_type: str,
    brain_area: str,
    window_name: str,
    metric_fn: Callable[[np.ndarray, np.ndarray, int], dict[str, float]],
    *,
    extra_fields: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Collect all pairwise records for one population condition."""
    records = []
    dataset = build_population_analysis_dataset(sound_type, window_name, brain_area)
    if dataset is None:
        return records

    x = dataset["X"]
    y_labels = dataset["Y_labels"]
    labels = ordered_trial_labels(sound_type, dataset["Y_numeric"], y_labels)
    seed = funcs.deterministic_seed(sound_type, brain_area, window_name)
    base_record = {"Sound Type": sound_type, "Brain Area": brain_area, "Window": window_name}
    for label_left, label_right in combinations(labels, 2):
        record = {
            **base_record,
            "Stim 1": label_left,
            "Stim 2": label_right,
            "Pair Type": natural_pair_type(label_left, label_right) if sound_type == "naturalSound" else None,
        }
        if extra_fields is not None:
            record.update(extra_fields)
        record.update(metric_fn(x[y_labels == label_left], x[y_labels == label_right], seed))
        records.append(record)
    return records


def pearson_metrics(resp1: np.ndarray, resp2: np.ndarray, _: int) -> dict[str, float]:
    """Return Pearson correlation and dissimilarity between two mean stimulus vectors."""
    mean_left = resp1.mean(axis=0)
    mean_right = resp2.mean(axis=0)
    correlation = float(np.corrcoef(mean_left, mean_right)[0, 1])
    return {"Correlation": correlation, "Dissimilarity": 1.0 - correlation}


def _shuffled_pair_dataset(resp1: np.ndarray, resp2: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return one deterministically shuffled binary stimulus-pair dataset."""
    x_pair = np.vstack([resp1, resp2])
    y_pair = np.concatenate([np.zeros(len(resp1), dtype=int), np.ones(len(resp2), dtype=int)])
    shuffle_index = np.random.default_rng(seed).permutation(len(y_pair))
    return x_pair[shuffle_index], y_pair[shuffle_index]


def _cv_accuracy(model_factory: Callable[[], object], resp1: np.ndarray, resp2: np.ndarray, seed: int) -> float:
    """Return a leave-one-out CV accuracy for one stimulus pair."""
    x_pair, y_pair = _shuffled_pair_dataset(resp1, resp2, seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        return float(
            funcs.run_loo_classifier_cv(
                model_factory,
                x_pair,
                y_pair,
                random_state=seed,
                standardize=False,
            )["mean_accuracy"]
        )


def svm_accuracy(resp1: np.ndarray, resp2: np.ndarray, seed: int, c_value: float = 1.0) -> dict[str, float]:
    """Return leave-one-out linear-SVM accuracy for one stimulus pair."""
    return {
        "Accuracy": _cv_accuracy(
            lambda: LinearSVC(C=c_value, max_iter=200000, dual="auto", tol=1e-3),
            resp1,
            resp2,
            seed,
        )
    }


def lda_accuracy(resp1: np.ndarray, resp2: np.ndarray, seed: int) -> dict[str, float]:
    """Return leave-one-out LDA accuracy for one stimulus pair."""
    return {"Accuracy": _cv_accuracy(LinearDiscriminantAnalysis, resp1, resp2, seed)}


ANALYSIS_METHODS = {
    "pearson": pearson_metrics,
    "linearSVM": svm_accuracy,
    "lda": lda_accuracy,
}


def tune_linear_svm_c(
    sound_type: str,
    brain_area: str,
    window_name: str,
    *,
    return_trace: bool = False,
) -> float | tuple[float, list[dict[str, object]]]:
    """Return the best linear-SVM `C` for one sound, region, and window."""
    mean_accuracies = []
    trace_records = []
    for c_value in SVM_C_VALUES:
        pair_accuracies = [
            float(record["Accuracy"])
            for record in collect_condition_records(
                sound_type,
                brain_area,
                window_name,
                lambda resp1, resp2, seed, c_value=c_value: svm_accuracy(resp1, resp2, seed, c_value=c_value),
            )
        ]
        mean_accuracy = float(np.nanmean(pair_accuracies)) if pair_accuracies else np.nan
        mean_accuracies.append(mean_accuracy)
        trace_records.append(
            {
                "Sound Type": sound_type,
                "Brain Area": brain_area,
                "Window": window_name,
                "C": float(c_value),
                "Mean Accuracy": mean_accuracy,
            }
        )

    if np.all(np.isnan(mean_accuracies)):
        return (1.0, trace_records) if return_trace else 1.0
    best_index = int(np.nanargmax(mean_accuracies))
    best_c = float(SVM_C_VALUES[best_index])
    return (best_c, trace_records) if return_trace else best_c


def plot_svm_hyperparameter_tuning(tuning_df: pd.DataFrame) -> None:
    """Plot the linear-SVM `C` tuning traces for each sound type."""
    funcs.apply_figure_style()
    for sound_type in params.SOUND_ORDER:
        sound_df = tuning_df[tuning_df["Sound Type"] == sound_type].copy()
        if sound_df.empty:
            continue
        output_dir = figure_output_dir(sound_type, "linearSVM")
        brain_regions = funcs.get_plot_brain_regions(sound_type)
        fig, axes = make_sound_figure(sound_type, width_scale=5.5, height_scale=5.2)
        fig.suptitle(
            f"Linear SVM Hyperparameter Tuning for {params.SOUND_DISPLAY_NAMES[sound_type]}",
            fontsize=FONTSIZE_SUPTITLE,
            fontweight="bold",
        )
        for row_index, brain_area in enumerate(brain_regions):
            for col_index, window_name in enumerate(params.WINDOW_ORDER):
                ax = axes[row_index, col_index]
                panel_df = sound_df[
                    (sound_df["Brain Area"] == brain_area) & (sound_df["Window"] == window_name)
                ].copy()
                if panel_df.empty:
                    ax.axis("off")
                    continue

                ax.plot(panel_df["C"], panel_df["Mean Accuracy"],
                        marker="o", markersize=7, linewidth=2.0,
                        color=plt.cm.viridis(0.45))
                best_row = panel_df.loc[panel_df["Mean Accuracy"].idxmax()]
                ax.scatter([best_row["C"]], [best_row["Mean Accuracy"]],
                           color=plt.cm.viridis(0.85), s=80, zorder=4,
                           edgecolor="white", linewidth=1.2)
                ax.annotate(
                    f"C = {best_row['C']:.2g}",
                    xy=(best_row["C"], best_row["Mean Accuracy"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=FONTSIZE_LABEL, color=plt.cm.viridis(0.85),
                )
                ax.set_xscale("log")
                ax.set_title(
                    f"{params.short_names.get(brain_area, brain_area)} - {window_name.capitalize()}",
                    fontsize=FONTSIZE_TITLE,
                    fontweight="bold",
                )
                ax.set_xlabel("Regularization Parameter C", fontsize=FONTSIZE_LABEL)
                ax.set_ylabel("Mean Pairwise Accuracy", fontsize=FONTSIZE_LABEL)
                ax.tick_params(labelsize=FONTSIZE_LABEL)
                ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
                sns.despine(ax=ax)

        fig.savefig(output_dir / f"linearSVM_{sound_type}_hyperparameter_tuning.png", dpi=300)
        plt.close(fig)


def choose_example_svm_row(results_df: pd.DataFrame) -> pd.Series | None:
    """Return one representative linear-SVM pair for an example boundary visualization."""
    preferred_conditions = [
        ("speech", "Primary auditory area", "onset"),
        ("speech", "Ventral auditory area", "onset"),
        ("speech", "Posterior auditory area", "onset"),
        ("AM", "Primary auditory area", "onset"),
    ]
    for sound_type, brain_area, window_name in preferred_conditions:
        subset = results_df[
            (results_df["Sound Type"] == sound_type)
            & (results_df["Brain Area"] == brain_area)
            & (results_df["Window"] == window_name)
        ].copy()
        if not subset.empty:
            return subset.sort_values("Accuracy", ascending=False).iloc[0]
    if results_df.empty:
        return None
    return results_df.sort_values("Accuracy", ascending=False).iloc[0]


def plot_linear_svm_example(results_df: pd.DataFrame) -> None:
    """Plot one illustrative linear-SVM decision boundary in a 2D PCA projection."""
    funcs.apply_figure_style()
    example_row = choose_example_svm_row(results_df)
    if example_row is None:
        return

    sound_type = str(example_row["Sound Type"])
    brain_area = str(example_row["Brain Area"])
    window_name = str(example_row["Window"])
    stim_left = str(example_row["Stim 1"])
    stim_right = str(example_row["Stim 2"])
    c_value = float(example_row["C"]) if "C" in example_row and not pd.isna(example_row["C"]) else 1.0

    dataset = build_population_analysis_dataset(sound_type, window_name, brain_area)
    if dataset is None:
        return

    y_labels = np.asarray(dataset["Y_labels"], dtype=object)
    keep_mask = np.isin(y_labels, [stim_left, stim_right])
    if int(np.sum(keep_mask)) < 4:
        return

    x_pair = np.asarray(dataset["X"][keep_mask], dtype=float)
    y_pair = np.where(y_labels[keep_mask] == stim_right, 1, 0)
    projected = PCA(n_components=2).fit_transform(x_pair)

    model = LinearSVC(C=c_value, max_iter=200000, dual="auto", tol=1e-3)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(projected, y_pair)

    x_pad = 1.5 * np.std(projected[:, 0]) if np.std(projected[:, 0]) > 0 else 1.5
    y_pad = 1.5 * np.std(projected[:, 1]) if np.std(projected[:, 1]) > 0 else 1.5
    xx, yy = np.meshgrid(
        np.linspace(projected[:, 0].min() - x_pad, projected[:, 0].max() + x_pad, 300),
        np.linspace(projected[:, 1].min() - y_pad, projected[:, 1].max() + y_pad, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    decision = model.decision_function(grid).reshape(xx.shape)
    class_colors = [plt.cm.viridis(0.2), plt.cm.viridis(0.8)]

    fig, ax = plt.subplots(figsize=(9.0, 7.2), constrained_layout=True)
    ax.contourf(xx, yy, decision > 0, levels=1, alpha=0.08, colors=class_colors)
    visible_margin_levels = [level for level in (-1, 0, 1) if decision.min() <= level <= decision.max()]
    if visible_margin_levels:
        style_map = {-1: ("#6c757d", "--", 1.3), 0: ("#111111", "-", 2.0), 1: ("#6c757d", "--", 1.3)}
        ax.contour(
            xx,
            yy,
            decision,
            levels=visible_margin_levels,
            colors=[style_map[level][0] for level in visible_margin_levels],
            linestyles=[style_map[level][1] for level in visible_margin_levels],
            linewidths=[style_map[level][2] for level in visible_margin_levels],
        )
    ax.scatter(
        projected[y_pair == 0, 0],
        projected[y_pair == 0, 1],
        s=42,
        color=class_colors[0],
        edgecolor="white",
        linewidth=0.5,
        label=stim_left,
    )
    ax.scatter(
        projected[y_pair == 1, 0],
        projected[y_pair == 1, 1],
        s=42,
        color=class_colors[1],
        edgecolor="white",
        linewidth=0.5,
        label=stim_right,
    )
    short_area = params.short_names.get(brain_area, brain_area)
    ax.set_title(
        f"Linear SVM Decision Boundary — {params.SOUND_DISPLAY_NAMES[sound_type]}, {short_area}, {window_name.capitalize()}",
        fontsize=FONTSIZE_TITLE,
        fontweight="bold",
    )
    ax.set_xlabel("PC 1 of Pairwise Response Matrix", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("PC 2 of Pairwise Response Matrix", fontsize=FONTSIZE_LABEL)
    ax.tick_params(labelsize=FONTSIZE_LABEL)
    ax.legend(frameon=True, loc="upper left", fontsize=18, title_fontsize=18)
    sns.despine(ax=ax)
    margin_note = ""
    if -1 not in visible_margin_levels or 1 not in visible_margin_levels:
        margin_note = "\nOne or both margins fall outside this 2D projection"
    ax.text(
        0.02,
        0.02,
        f"Solid line: decision boundary  |  Dashed: margins\n"
        f"C = {c_value:.3g}   LOO accuracy = {float(example_row['Accuracy']):.2f}"
        f"{margin_note}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.92, "pad": 4, "boxstyle": "round,pad=0.4"},
    )
    fig.savefig(example_output_dir() / "linearSVM_example_boundary.png", dpi=300)
    plt.close(fig)


def run_pairwise_analysis(metric_fn: Callable[[np.ndarray, np.ndarray, int], dict[str, float]]) -> pd.DataFrame:
    """Run one population pairwise discriminability analysis across all sounds."""
    records: list[dict[str, object]] = []
    conditions = analysis_conditions()
    progress = tqdm(total=len(conditions), desc="Discriminability analysis", unit="condition", dynamic_ncols=True)
    try:
        for sound_type, brain_area, window_name in conditions:
            progress.set_postfix_str(
                f"{sound_type} | {params.short_names.get(brain_area, brain_area)} | {window_name}",
                refresh=False,
            )
            records.extend(collect_condition_records(sound_type, brain_area, window_name, metric_fn))
            progress.update(1)
    finally:
        close = getattr(progress, "close", None)
        if close is not None:
            close()
    return pd.DataFrame(records)


def run_method_analysis(method_key: str) -> pd.DataFrame:
    """Run and save one discriminability analysis."""
    if method_key != "linearSVM":
        results_df = run_pairwise_analysis(ANALYSIS_METHODS[method_key])
    else:
        records: list[dict[str, object]] = []
        tuning_records: list[dict[str, object]] = []
        conditions = analysis_conditions()
        progress = tqdm(total=len(conditions), desc="Discriminability analysis", unit="condition", dynamic_ncols=True)
        try:
            for sound_type, brain_area, window_name in conditions:
                best_c, condition_tuning_records = tune_linear_svm_c(
                    sound_type,
                    brain_area,
                    window_name,
                    return_trace=True,
                )
                tuning_records.extend(condition_tuning_records)
                progress.set_postfix_str(
                    f"{sound_type} | {params.short_names.get(brain_area, brain_area)} | {window_name} | C={best_c:.3g}",
                    refresh=False,
                )
                records.extend(
                    collect_condition_records(
                        sound_type,
                        brain_area,
                        window_name,
                        lambda resp1, resp2, seed, best_c=best_c: svm_accuracy(resp1, resp2, seed, c_value=best_c),
                        extra_fields={"C": best_c},
                    )
                )
                progress.update(1)
        finally:
            close = getattr(progress, "close", None)
            if close is not None:
                close()
        results_df = pd.DataFrame(records)
        tuning_df = pd.DataFrame(tuning_records)
        tuning_df.to_csv(get_tuning_path(), index=False)
        plot_svm_hyperparameter_tuning(tuning_df)
    results_df.to_csv(get_results_path(method_key), index=False)
    return results_df


def load_method_results(method_key: str) -> pd.DataFrame:
    """Load saved pairwise results for one discriminability method."""
    return pd.read_csv(get_results_path(method_key))


def make_sound_figure(
    sound_type: str,
    *,
    width_scale: float,
    height_scale: float,
    sharey: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a standard discriminability figure grid for one sound type."""
    brain_regions = funcs.get_plot_brain_regions(sound_type)
    return plt.subplots(
        len(brain_regions),
        len(params.WINDOW_ORDER),
        figsize=(width_scale * len(params.WINDOW_ORDER), height_scale * len(brain_regions)),
        squeeze=False,
        sharey=sharey,
        constrained_layout=True,
    )


def heatmap_matrix(panel_df: pd.DataFrame, labels: list[str], value_col: str) -> pd.DataFrame:
    """Build one symmetric heatmap matrix from long-form pairwise results."""
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    grouped = panel_df.groupby(["Stim 1", "Stim 2"], as_index=False)[value_col].mean()
    for row in grouped.itertuples(index=False):
        matrix.loc[row[0], row[1]] = row[2]
        matrix.loc[row[1], row[0]] = row[2]
    return matrix


def score_axis_limits(score_df: pd.DataFrame) -> tuple[float, float, float]:
    """Return shared y-axis components for one summary figure."""
    y_min = float(score_df.min())
    y_max = float(score_df.max())
    y_step = 0.035 * ((y_max - y_min) if y_max > y_min else 1.0)
    return y_min, y_max, y_step


def format_boxplot_axis(
    ax: plt.Axes,
    window_name: str,
    ylabel: str,
    brain_regions: list[str],
    *,
    show_ylabel: bool,
    y_min: float,
    y_max: float,
) -> None:
    """Apply shared axis formatting for discriminability boxplots."""
    ax.set_title(window_name.capitalize(), fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=FONTSIZE_LABEL)
    ax.set_xticks(range(len(brain_regions)))
    ax.set_xticklabels([params.short_names.get(region, region) for region in brain_regions], rotation=20, fontsize=FONTSIZE_LABEL)
    ax.tick_params(axis="y", labelsize=FONTSIZE_LABEL)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
    ax.set_ylim(y_min, y_max)


def plot_heatmaps(
    results_df: pd.DataFrame,
    *,
    method_key: str,
    method_label: str,
    value_col: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    """Create one heatmap grid per sound type."""
    funcs.apply_figure_style()
    for sound_type in params.SOUND_ORDER:
        sound_df = results_df[results_df["Sound Type"] == sound_type].copy()
        if sound_df.empty:
            continue
        output_dir = figure_output_dir(sound_type, method_key)
        brain_regions = funcs.get_plot_brain_regions(sound_type)
        labels = funcs.stimulus_order(sound_type, include_speech_syllables=True)
        fig, axes = make_sound_figure(sound_type, width_scale=5.5, height_scale=5.2)
        fig.suptitle(
            f"Pairwise {method_label} Heatmaps — {params.SOUND_DISPLAY_NAMES[sound_type]}",
            fontsize=FONTSIZE_SUPTITLE,
            fontweight="bold",
        )
        colorbar_source = None

        for row_index, brain_area in enumerate(brain_regions):
            for col_index, window_name in enumerate(params.WINDOW_ORDER):
                ax = axes[row_index, col_index]
                panel_df = sound_df[
                    (sound_df["Brain Area"] == brain_area) & (sound_df["Window"] == window_name)
                ].copy()
                if panel_df.empty:
                    ax.axis("off")
                    continue

                heatmap = sns.heatmap(
                    heatmap_matrix(panel_df, labels, value_col),
                    ax=ax,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    square=True,
                    cbar=False,
                )
                if colorbar_source is None:
                    colorbar_source = heatmap.collections[0]
                ax.set_title(
                    f"{params.short_names.get(brain_area, brain_area)} - {window_name.capitalize()}",
                    fontsize=FONTSIZE_TITLE,
                    fontweight="bold",
                )
                ax.set_xlabel("")
                ax.set_ylabel("")
                # Every other x-axis label at 45° for all sound types; strip Hz for AM/PT
                x_lbls = [t.get_text() for t in ax.get_xticklabels()]
                y_lbls = [t.get_text() for t in ax.get_yticklabels()]
                if sound_type in {"AM", "PT"}:
                    x_lbls = [lbl.replace(" Hz", "") for lbl in x_lbls]
                    y_lbls = [lbl.replace(" Hz", "") for lbl in y_lbls]
                    ax.set_yticklabels(y_lbls, rotation=0, fontsize=FONTSIZE_LABEL)
                else:
                    ax.tick_params(axis="y", rotation=0, labelsize=FONTSIZE_LABEL)
                x_lbls = [lbl if i % 2 == 0 else "" for i, lbl in enumerate(x_lbls)]
                ax.set_xticklabels(x_lbls, rotation=45, ha="right", fontsize=FONTSIZE_LABEL)

        if colorbar_source is not None:
            colorbar = fig.colorbar(colorbar_source, ax=axes, location="right", fraction=0.03, pad=0.02)
            colorbar.set_label(method_label, fontsize=FONTSIZE_LABEL)
            colorbar.ax.tick_params(labelsize=FONTSIZE_LABEL)
        fig.savefig(output_dir / f"{method_key}_{sound_type}_heatmaps.png", dpi=300)
        plt.close(fig)


def plot_region_boxplots(
    results_df: pd.DataFrame,
    *,
    method_key: str,
    method_label: str,
    value_col: str,
    ylabel: str,
) -> None:
    """Create one 1x3 region-comparison boxplot figure per sound type."""
    funcs.apply_figure_style()
    for sound_type in params.SOUND_ORDER:
        sound_df = results_df[results_df["Sound Type"] == sound_type].copy()
        if sound_df.empty:
            continue
        output_dir = figure_output_dir(sound_type, method_key)
        brain_regions = funcs.get_plot_brain_regions(sound_type)
        fig, axes = plt.subplots(1, len(params.WINDOW_ORDER), figsize=(5.5 * len(params.WINDOW_ORDER), 5.2), sharey=True, constrained_layout=True)
        fig.suptitle(f"Pairwise {method_label} — {params.SOUND_DISPLAY_NAMES[sound_type]}", fontsize=FONTSIZE_SUPTITLE, fontweight="bold")
        max_annotations = len(brain_regions) * (len(brain_regions) - 1) // 2
        y_min, y_max, y_step = score_axis_limits(sound_df[value_col])
        region_palette = sns.color_palette("viridis", n_colors=len(brain_regions))

        for ax, window_name in zip(np.ravel(axes), params.WINDOW_ORDER):
            panel_df = sound_df[sound_df["Window"] == window_name].copy()
            sns.boxplot(
                data=panel_df,
                x="Brain Area",
                y=value_col,
                hue="Brain Area",
                order=brain_regions,
                hue_order=brain_regions,
                width=0.5,
                fliersize=2,
                linewidth=1,
                palette=region_palette,
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=panel_df,
                x="Brain Area",
                y=value_col,
                order=brain_regions,
                color="black",
                alpha=0.35,
                size=3,
                ax=ax,
            )
            stats_df = pairwise_group_tests(
                panel_df,
                group_col="Brain Area",
                value_col=value_col,
                group_order=brain_regions,
                test_mode="unpaired",
            )
            add_pairwise_annotations(
                ax,
                stats_df,
                centers=box_centers(brain_regions),
                data_max=y_max,
                data_min=y_min,
            )
            format_boxplot_axis(
                ax,
                window_name,
                ylabel,
                brain_regions,
                show_ylabel=(window_name == params.WINDOW_ORDER[0]),
                y_min=y_min - y_step,
                y_max=y_max + y_step * (max_annotations + 2),
            )

        fig.savefig(output_dir / f"{method_key}_{sound_type}_region_boxplots.png", dpi=300)
        plt.close(fig)


def plot_natural_within_between_boxplots(
    results_df: pd.DataFrame,
    *,
    method_key: str,
    method_label: str,
    value_col: str,
    ylabel: str,
) -> None:
    """Create the natural-sound within-vs-between boxplot figure."""
    funcs.apply_figure_style()
    output_dir = figure_output_dir("naturalSound", method_key)
    natural_df = results_df[results_df["Sound Type"] == "naturalSound"].dropna(subset=["Pair Type"]).copy()
    if natural_df.empty:
        return

    brain_regions = funcs.get_plot_brain_regions("naturalSound")
    hue_order = ["Within", "Between"]
    fig, axes = plt.subplots(1, len(params.WINDOW_ORDER), figsize=(5.5 * len(params.WINDOW_ORDER), 5.2), sharey=True, constrained_layout=True)
    fig.suptitle(
        f"Natural Sounds {method_label} — Within vs. Between Category",
        fontsize=FONTSIZE_SUPTITLE,
        fontweight="bold",
    )
    y_min, y_max, y_step = score_axis_limits(natural_df[value_col])
    max_annotations = len(hue_order) * (len(brain_regions) * (len(brain_regions) - 1) // 2)

    for ax, window_name in zip(np.ravel(axes), params.WINDOW_ORDER):
        panel_df = natural_df[natural_df["Window"] == window_name].copy()
        sns.boxplot(
            data=panel_df,
            x="Brain Area",
            y=value_col,
            hue="Pair Type",
            order=brain_regions,
            hue_order=hue_order,
            width=0.5,
            palette=VIRIDIS_PAIR_PALETTE,
            showfliers=False,
            linewidth=1,
            ax=ax,
        )
        sns.stripplot(
            data=panel_df,
            x="Brain Area",
            y=value_col,
            hue="Pair Type",
            order=brain_regions,
            hue_order=hue_order,
            palette=VIRIDIS_PAIR_PALETTE,
            dodge=True,
            alpha=0.35,
            size=3,
            linewidth=0,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        stats_df = pairwise_group_tests(
            panel_df,
            group_col="Brain Area",
            value_col=value_col,
            group_order=brain_regions,
            hue_col="Pair Type",
            hue_order=hue_order,
            test_mode="unpaired",
        )
        add_pairwise_annotations(
            ax,
            stats_df,
            centers=box_centers(brain_regions, hue_levels=hue_order),
            data_max=y_max,
            data_min=y_min,
        )
        format_boxplot_axis(
            ax,
            window_name,
            ylabel,
            brain_regions,
            show_ylabel=(window_name == params.WINDOW_ORDER[0]),
            y_min=y_min - y_step,
            y_max=y_max + y_step * (max_annotations + 2),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles[: len(hue_order)],
            labels[: len(hue_order)],
            title="Pair Type",
            loc="outside right center",
            frameon=True,
            fontsize=FONTSIZE_LABEL,
            title_fontsize=FONTSIZE_LABEL,
        )
    fig.savefig(output_dir / f"{method_key}_naturalSound_within_between_boxplots.png", dpi=300)
    plt.close(fig)
