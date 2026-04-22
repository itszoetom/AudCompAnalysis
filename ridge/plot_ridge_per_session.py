"""Create per-session ridge R² distributions and diagnostic figures.

Three figures are produced per sound type:
  1. Per-session R² boxplot (one box per brain region, stats annotations).
  2. Alpha tuning curves (brain-area rows × window columns, peak marked).
  3. Good-fit vs poor-fit predicted-vs-actual scatter (PT and AM only).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from shared import params

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

try:
    from .ridge_analysis import (
        FONTSIZE_LABEL,
        FONTSIZE_SUPTITLE,
        FONTSIZE_TITLE,
        RIDGE_ALPHAS,
        SOUND_DISPLAY_NAMES,
        apply_figure_style,
        available_mice,
        available_sessions,
        build_dataset,
        build_population_target_datasets,
        build_target_datasets,
        compute_ridge_alpha_tuning,
        fit_best_ridge,
        funcs,
        get_plot_brain_regions,
        list_available_sound_types,
        plot_ridge_summary,
        WINDOW_ORDER,
    )
except ImportError:
    from ridge_analysis import (
        FONTSIZE_LABEL,
        FONTSIZE_SUPTITLE,
        FONTSIZE_TITLE,
        RIDGE_ALPHAS,
        SOUND_DISPLAY_NAMES,
        apply_figure_style,
        available_mice,
        available_sessions,
        build_dataset,
        build_population_target_datasets,
        build_target_datasets,
        compute_ridge_alpha_tuning,
        fit_best_ridge,
        funcs,
        get_plot_brain_regions,
        list_available_sound_types,
        plot_ridge_summary,
        WINDOW_ORDER,
    )


# ---------------------------------------------------------------------------
# Session eligibility helpers
# ---------------------------------------------------------------------------

def _eligible_sessions_for_area(
    sound_type: str,
    brain_area: str,
    n_neurons: int,
) -> list[tuple[str, str]]:
    """Return sessions in one area that have enough neurons for per-session ridge."""
    eligible_sessions = []
    for mouse_id in available_mice(sound_type):
        for session_id in available_sessions(sound_type, mouse_id=mouse_id):
            dataset = build_dataset(sound_type, WINDOW_ORDER[0], brain_area, mouse_id=mouse_id, session_id=session_id)
            if dataset is not None and dataset["X"].shape[1] >= n_neurons:
                eligible_sessions.append((mouse_id, session_id))
    return eligible_sessions


def collect_eligible_sessions(
    sound_type: str,
    n_neurons: int,
    *,
    seed: int | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Return all eligible sessions for each plotted brain area."""
    brain_regions = get_plot_brain_regions(sound_type)
    _ = seed
    return {
        brain_area: _eligible_sessions_for_area(sound_type, brain_area, n_neurons)
        for brain_area in brain_regions
    }


# ---------------------------------------------------------------------------
# Figure 1: per-session R² frame
# ---------------------------------------------------------------------------

def build_session_frame(
    sound_type: str,
    n_neurons: int | None = None,
    n_subsamples: int = 50,
    selected_sessions: dict[str, Iterable[tuple[str, str]]] | None = None,
) -> pd.DataFrame:
    """Return one mean per-session ridge score after eligible-session and neuron subsampling."""
    n_neurons = params.NEURONS_PER_SESSION[sound_type] if n_neurons is None else n_neurons
    records = []
    eligible_sessions = selected_sessions or collect_eligible_sessions(sound_type, n_neurons)
    conditions = [
        (mouse_id, session_id, brain_area, window_name)
        for brain_area in get_plot_brain_regions(sound_type)
        for mouse_id, session_id in eligible_sessions.get(brain_area, [])
        for window_name in WINDOW_ORDER
    ]
    for mouse_id, session_id, brain_area, window_name in tqdm(
        conditions,
        desc=f"Per-session ridge ({sound_type})",
        unit="dataset",
        dynamic_ncols=True,
    ):
        dataset = build_dataset(sound_type, window_name, brain_area, mouse_id=mouse_id, session_id=session_id)
        if dataset is None or dataset["X"].shape[1] < n_neurons:
            continue
        if dataset["X"].shape[0] < 5:
            print(
                f"[per-session ridge] skipping {sound_type} {brain_area} {mouse_id} {session_id} {window_name}: "
                f"only {dataset['X'].shape[0]} trials available"
            )
            continue
        x_standardized = StandardScaler().fit_transform(dataset["X"])
        rng = np.random.default_rng(42)
        target_datasets = (
            [
                {
                    "target_name": "Speech Tuple",
                    "Y": funcs.labels_for_sound("speech", dataset["Y"]).astype(float),
                    "log_target": False,
                }
            ]
            if sound_type == "speech"
            else build_target_datasets(dataset)
        )
        for target_dataset in target_datasets:
            y = np.asarray(target_dataset["Y"], dtype=float).copy()

            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores_by_subsample: list[list[float]] = []
            fold_rmses_by_subsample: list[list[float]] = []
            fold_alphas_by_subsample: list[list[float]] = []
            for iteration in range(n_subsamples):
                neuron_idx = rng.choice(dataset["X"].shape[1], size=n_neurons, replace=False)
                x_sampled = x_standardized[:, neuron_idx]
                if len(y) < 5:
                    print(
                        f"[per-session ridge] warning: {sound_type} {brain_area} {mouse_id} {session_id} "
                        f"{window_name} target={target_dataset['target_name']} has only {len(y)} trials"
                    )
                fold_scores = []
                fold_rmses = []
                fold_alphas = []
                for fold_index, (train_index, test_index) in enumerate(kfold.split(x_sampled)):
                    if len(train_index) < 5:
                        print(
                            f"[per-session ridge] warning: inner fold below 5 training samples for "
                            f"{sound_type} {brain_area} {mouse_id} {session_id} {window_name}"
                        )
                    ridge = RidgeCV(
                        alphas=np.asarray(RIDGE_ALPHAS, dtype=float),
                        cv=funcs._ridge_inner_cv(len(train_index), 42 + fold_index),
                        scoring="r2",
                    )
                    ridge.fit(x_sampled[train_index], y[train_index])
                    y_pred = ridge.predict(x_sampled[test_index])
                    fold_scores.append(float(funcs.r2_score(y[test_index], y_pred)))
                    fold_rmses.append(float(np.sqrt(funcs.mean_squared_error(y[test_index], y_pred))))
                    fold_alphas.append(float(ridge.alpha_))
                fold_scores_by_subsample.append(fold_scores)
                fold_rmses_by_subsample.append(fold_rmses)
                fold_alphas_by_subsample.append(fold_alphas)

            mean_fold_scores = np.mean(np.asarray(fold_scores_by_subsample, dtype=float), axis=0)
            mean_fold_rmses = np.mean(np.asarray(fold_rmses_by_subsample, dtype=float), axis=0)
            mean_fold_alphas = np.mean(np.asarray(fold_alphas_by_subsample, dtype=float), axis=0)
            records.append(
                {
                    "Mouse": mouse_id,
                    "Session": session_id,
                    "Brain Area": brain_area,
                    "Window": window_name,
                    "Target": target_dataset["target_name"],
                    "R2 Test": float(np.mean(mean_fold_scores)),
                    "RMSE": float(np.mean(mean_fold_rmses)),
                    "Best Alpha": float(np.mean(mean_fold_alphas)),
                    "Neurons": int(n_neurons),
                    "Trials": int(len(y)),
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure 2: alpha tuning grid
# ---------------------------------------------------------------------------

def plot_ridge_alpha_tuning_grid(sound_type: str) -> None:
    """Plot alpha tuning curves in a brain-area-rows × window-columns grid.

    Matches the SVM C-tuning figure style: one panel per (brain area, window),
    viridis line color, peak alpha marked with a red dot and annotation.
    """
    brain_regions = get_plot_brain_regions(sound_type)
    if not brain_regions:
        return
    display_name = SOUND_DISPLAY_NAMES.get(sound_type, sound_type)
    n_rows = len(brain_regions)
    n_cols = len(WINDOW_ORDER)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 5.2 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        f"Ridge Regularization Tuning — {display_name}",
        fontsize=FONTSIZE_SUPTITLE,
        fontweight="bold",
    )

    for row_index, brain_area in enumerate(brain_regions):
        for col_index, window_name in enumerate(WINDOW_ORDER):
            ax = axes[row_index, col_index]
            records = compute_ridge_alpha_tuning(sound_type, window_name, brain_area)
            if not records:
                ax.axis("off")
                continue
            df = pd.DataFrame(records)
            # One target per panel after the speech-tuple fix in compute_ridge_alpha_tuning
            for _, target_df in df.groupby("Target"):
                mean_r2_by_alpha = target_df.groupby("Alpha")["R2"].mean()
                log_alphas = np.log10(mean_r2_by_alpha.index.values.astype(float))
                ax.plot(
                    log_alphas,
                    mean_r2_by_alpha.values,
                    marker="o",
                    markersize=7,
                    linewidth=2.0,
                    color=plt.cm.viridis(0.45),
                )
                best_idx = int(np.argmax(mean_r2_by_alpha.values))
                best_alpha = float(mean_r2_by_alpha.index.values[best_idx])
                best_r2 = float(mean_r2_by_alpha.values[best_idx])
                ax.scatter(
                    [np.log10(best_alpha)],
                    [best_r2],
                    color="tab:red",
                    s=120,
                    zorder=4,
                    edgecolor="white",
                    linewidth=1.2,
                )
                # Fixed right-middle position so label never overlaps the curve
                ax.text(
                    0.97, 0.5,
                    f"α = {best_alpha:.2g}",
                    ha="right",
                    va="center",
                    fontsize=FONTSIZE_LABEL,
                    color="tab:red",
                    transform=ax.transAxes,
                )

            # Column header: window name only on top row
            if row_index == 0:
                ax.set_title(window_name.capitalize(), fontsize=FONTSIZE_TITLE, fontweight="bold")

            ax.set_xlabel(r"$\log_{10}(\alpha)$", fontsize=FONTSIZE_LABEL)

            # Row label: brain region only on left column
            if col_index == 0:
                short_area = params.short_names.get(brain_area, brain_area)
                ax.set_ylabel(
                    f"{short_area}\nMean $R^2$ (CV)",
                    fontsize=FONTSIZE_LABEL,
                    fontweight="bold",
                )
            else:
                ax.set_ylabel("")

            ax.tick_params(labelsize=FONTSIZE_LABEL)
            ax.grid(linestyle="--", linewidth=0.5, alpha=0.35)
            sns.despine(ax=ax)

    fig.savefig(
        funcs.get_figure_dir("decoding/ridge") / f"{sound_type}_ridge_alpha_tuning.png",
        dpi=300,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: good-fit vs poor-fit scatter (PT and AM)
# ---------------------------------------------------------------------------

def _collect_fit_examples(sound_types: list[str], window_name: str = "sustained") -> list[dict]:
    """Run population ridge for the given sound types (one window) and collect all fits."""
    results = []
    for sound_type in tqdm(sound_types, desc="Collecting fit examples", unit="sound", dynamic_ncols=True):
        brain_regions = get_plot_brain_regions(sound_type)
        for brain_area in brain_regions:
            target_datasets = build_population_target_datasets(sound_type, window_name, brain_area)
            for td in target_datasets:
                fit = fit_best_ridge(
                    td["X"],
                    td["Y"],
                    log_target=bool(td["log_target"]),
                )
                results.append({
                    "sound_type": sound_type,
                    "brain_area": brain_area,
                    "window_name": window_name,
                    "target_name": td["target_name"],
                    "r2": fit["r2_test"],
                    "y_true": fit["y_test"],
                    "y_pred": fit["y_pred"],
                })
    return results


def _select_good_bad_pair(fits: list[dict]) -> tuple[dict, dict] | None:
    """Select one poor-fit and one good-fit example from different brain regions if possible."""
    if len(fits) < 2:
        return None
    sorted_fits = sorted(fits, key=lambda x: x["r2"])
    # Prefer examples from different brain areas
    for bad in sorted_fits[: max(1, len(sorted_fits) // 2 + 1)]:
        for good in reversed(sorted_fits):
            if good["brain_area"] != bad["brain_area"]:
                return bad, good
    # Fallback: just use worst and best regardless of area
    return sorted_fits[0], sorted_fits[-1]


def plot_ridge_fit_examples(sound_types: list[str] | None = None) -> None:
    """Plot a 2x2 predicted-vs-actual grid: PT (top) and AM (bottom), poor left and good right.

    Both examples for each sound type come from the sustained window, preferably different
    brain regions.  X and Y axis limits are matched within each panel so the identity line
    spans the full square.
    """
    sound_types = sound_types or ["PT", "AM"]
    examples = _collect_fit_examples(sound_types, window_name="sustained")
    if not examples:
        return

    # Group by sound_type
    by_sound: dict[str, list[dict]] = {}
    for ex in examples:
        by_sound.setdefault(ex["sound_type"], []).append(ex)

    rows: list[tuple[dict, dict]] = []
    row_sound_types: list[str] = []
    for st in sound_types:
        pair = _select_good_bad_pair(by_sound.get(st, []))
        if pair is not None:
            rows.append(pair)
            row_sound_types.append(st)

    if not rows:
        return

    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(13.0, 6.0 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        "Ridge Regression: Predicted vs Actual (Sustained)",
        fontsize=FONTSIZE_SUPTITLE,
        fontweight="bold",
    )

    for row_idx, ((bad_example, good_example), sound_type) in enumerate(zip(rows, row_sound_types)):
        display_name = SOUND_DISPLAY_NAMES.get(sound_type, sound_type)
        for col_idx, (example, panel_label) in enumerate(
            [(bad_example, "Poor Fit"), (good_example, "Good Fit")]
        ):
            ax = axes[row_idx, col_idx]
            y_true = np.asarray(example["y_true"], dtype=float)
            y_pred = np.asarray(example["y_pred"], dtype=float)
            r2 = float(example["r2"])

            ax.scatter(
                y_true,
                y_pred,
                color=plt.cm.viridis(0.45),
                alpha=0.7,
                s=55,
                edgecolor="white",
                linewidth=0.5,
            )
            # Equal axis limits so the identity line is a true diagonal
            lim_min = min(float(y_true.min()), float(y_pred.min()))
            lim_max = max(float(y_true.max()), float(y_pred.max()))
            pad = (lim_max - lim_min) * 0.05
            ax.set_xlim(lim_min - pad, lim_max + pad)
            ax.set_ylim(lim_min - pad, lim_max + pad)
            ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad],
                    "k--", linewidth=1.5, alpha=0.7)

            short_area = params.short_names.get(example["brain_area"], example["brain_area"])
            ax.set_title(
                f"{display_name} — {panel_label} ({short_area})",
                fontsize=FONTSIZE_TITLE,
                fontweight="bold",
            )
            ax.set_xlabel("Actual", fontsize=FONTSIZE_LABEL)
            ax.set_ylabel("Predicted" if col_idx == 0 else "", fontsize=FONTSIZE_LABEL)
            ax.text(
                0.05, 0.95,
                f"$R^2$ = {r2:.2f}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=FONTSIZE_LABEL,
            )
            ax.tick_params(labelsize=FONTSIZE_LABEL)
            ax.set_aspect("equal", adjustable="box")
            sns.despine(ax=ax)

    fig.savefig(
        funcs.get_figure_dir("decoding/ridge") / "ridge_fit_examples_PT_AM.png",
        dpi=300,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the three ridge regression figures."""
    apply_figure_style()

    for sound_type in tqdm(list_available_sound_types(), desc="Ridge plots", unit="sound", dynamic_ncols=True):
        print(f"\n--- {sound_type} ---")
        n_neurons = params.NEURONS_PER_SESSION[sound_type]
        selected_sessions = collect_eligible_sessions(sound_type, n_neurons)

        # Figure 1: per-session R² boxplot
        results_df = build_session_frame(sound_type, n_neurons=n_neurons, selected_sessions=selected_sessions)
        if not results_df.empty:
            session_counts = {
                brain_area: len(list(sessions))
                for brain_area, sessions in selected_sessions.items()
            }
            display_name = SOUND_DISPLAY_NAMES.get(sound_type, sound_type)
            plot_ridge_summary(
                sound_type,
                results_df,
                title=f"Per-Session Ridge Regression — {display_name}",
                filename=f"{sound_type}_ridge_per_session.png",
                pair_cols=["Mouse", "Session"],
                session_counts=session_counts,
            )

        # Figure 2: alpha tuning grid
        plot_ridge_alpha_tuning_grid(sound_type)

    # Figure 3: good/bad fit scatter for PT and AM
    plot_ridge_fit_examples(["PT", "AM"])


if __name__ == "__main__":
    main()
