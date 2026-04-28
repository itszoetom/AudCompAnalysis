"""Create per-session ridge R² distributions and diagnostic figures for PT and AM.

Three figures per sound type:
  1. Per-session R² boxplot (one box per brain region, stats annotations).
  2. Alpha tuning curves (brain-area rows × window columns, peak marked).
  3. Good-fit vs poor-fit predicted-vs-actual scatter (searches all windows).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared import params

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

try:
    from .ridge_analysis import (
        FONTSIZE_LABEL, FONTSIZE_SUPTITLE, FONTSIZE_TITLE,
        RIDGE_ALPHAS, SOUND_DISPLAY_NAMES, apply_figure_style,
        available_mice, available_sessions, build_dataset,
        build_target_datasets, compute_ridge_alpha_tuning,
        fit_best_ridge, funcs, get_plot_brain_regions,
        plot_ridge_summary, WINDOW_ORDER,
    )
except ImportError:
    from ridge_analysis import (
        FONTSIZE_LABEL, FONTSIZE_SUPTITLE, FONTSIZE_TITLE,
        RIDGE_ALPHAS, SOUND_DISPLAY_NAMES, apply_figure_style,
        available_mice, available_sessions, build_dataset,
        build_target_datasets, compute_ridge_alpha_tuning,
        fit_best_ridge, funcs, get_plot_brain_regions,
        plot_ridge_summary, WINDOW_ORDER,
    )


# ---------------------------------------------------------------------------
# Session eligibility
# ---------------------------------------------------------------------------

def _eligible_sessions_for_area(sound_type: str, brain_area: str, n_neurons: int) -> list[tuple[str, str]]:
    eligible = []
    for mouse_id in available_mice(sound_type):
        for session_id in available_sessions(sound_type, mouse_id=mouse_id):
            ds = build_dataset(sound_type, WINDOW_ORDER[0], brain_area, mouse_id=mouse_id, session_id=session_id)
            if ds is not None and ds["X"].shape[1] >= n_neurons:
                eligible.append((mouse_id, session_id))
    return eligible


def collect_eligible_sessions(sound_type: str, n_neurons: int) -> dict[str, list[tuple[str, str]]]:
    return {
        ba: _eligible_sessions_for_area(sound_type, ba, n_neurons)
        for ba in get_plot_brain_regions(sound_type)
    }


# ---------------------------------------------------------------------------
# Figure 1: per-session R² — parallel worker
# ---------------------------------------------------------------------------

def _fit_one_condition(args: tuple) -> list[dict]:
    """Worker: fit one (session × brain area × window) condition."""
    sound_type, mouse_id, session_id, brain_area, window_name, n_neurons, n_subsamples = args
    dataset = build_dataset(sound_type, window_name, brain_area, mouse_id=mouse_id, session_id=session_id)
    if dataset is None or dataset["X"].shape[1] < n_neurons or dataset["X"].shape[0] < 5:
        return []

    x_std = StandardScaler().fit_transform(dataset["X"])
    seed = abs(hash((sound_type, brain_area, mouse_id, session_id, window_name))) % (2 ** 31)
    rng = np.random.default_rng(seed)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    records = []

    for td in build_target_datasets(dataset):
        y = np.asarray(td["Y"], dtype=float)
        scores_all, rmses_all, alphas_all = [], [], []
        for _ in range(n_subsamples):
            idx = rng.choice(dataset["X"].shape[1], size=n_neurons, replace=False)
            xs = x_std[:, idx]
            fold_scores, fold_rmses, fold_alphas = [], [], []
            for fi, (tr, te) in enumerate(kfold.split(xs)):
                ridge = RidgeCV(
                    alphas=np.asarray(RIDGE_ALPHAS, dtype=float),
                    cv=funcs._ridge_inner_cv(len(tr), 42 + fi),
                    scoring="r2",
                )
                ridge.fit(xs[tr], y[tr])
                yp = ridge.predict(xs[te])
                fold_scores.append(float(funcs.r2_score(y[te], yp)))
                fold_rmses.append(float(np.sqrt(funcs.mean_squared_error(y[te], yp))))
                fold_alphas.append(float(ridge.alpha_))
            scores_all.append(fold_scores)
            rmses_all.append(fold_rmses)
            alphas_all.append(fold_alphas)

        records.append({
            "Mouse": mouse_id, "Session": session_id,
            "Brain Area": brain_area, "Window": window_name,
            "Target": td["target_name"],
            "R2 Test": float(np.mean(scores_all)),
            "RMSE": float(np.mean(rmses_all)),
            "Best Alpha": float(np.mean(alphas_all)),
            "Neurons": n_neurons, "Trials": len(y),
        })
    return records


def build_session_frame(
    sound_type: str,
    n_neurons: int | None = None,
    n_subsamples: int = 100,
    selected_sessions: dict[str, Iterable[tuple[str, str]]] | None = None,
) -> pd.DataFrame:
    """Return per-session ridge scores using parallel processing."""
    n_neurons = params.NEURONS_PER_SESSION[sound_type] if n_neurons is None else n_neurons
    eligible = selected_sessions or collect_eligible_sessions(sound_type, n_neurons)
    conditions = [
        (sound_type, mouse_id, session_id, brain_area, window_name, n_neurons, n_subsamples)
        for brain_area in get_plot_brain_regions(sound_type)
        for mouse_id, session_id in eligible.get(brain_area, [])
        for window_name in WINDOW_ORDER
    ]
    records: list[dict] = []
    n_workers = min(os.cpu_count() or 4, max(len(conditions), 1))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fit_one_condition, args): args for args in conditions}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Per-session ridge ({sound_type})", unit="dataset", dynamic_ncols=True):
            records.extend(future.result())
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure 2: alpha tuning grid
# ---------------------------------------------------------------------------

def plot_ridge_alpha_tuning_grid(sound_type: str) -> None:
    brain_regions = get_plot_brain_regions(sound_type)
    if not brain_regions:
        return
    n_rows, n_cols = len(brain_regions), len(WINDOW_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 5.5 * n_rows),
                             squeeze=False, constrained_layout=True)
    fig.suptitle(f"Ridge Regularization Tuning: {SOUND_DISPLAY_NAMES.get(sound_type, sound_type)}",
                 fontsize=FONTSIZE_SUPTITLE, fontweight="bold")

    for ri, brain_area in enumerate(brain_regions):
        for ci, window_name in enumerate(WINDOW_ORDER):
            ax = axes[ri, ci]
            records = compute_ridge_alpha_tuning(sound_type, window_name, brain_area)
            if not records:
                ax.axis("off")
                continue
            for _, tdf in pd.DataFrame(records).groupby("Target"):
                mean_r2 = tdf.groupby("Alpha")["R2"].mean()
                log_alphas = np.log10(mean_r2.index.values.astype(float))
                ax.plot(log_alphas, mean_r2.values, marker="o", markersize=7,
                        linewidth=2.0, color=plt.cm.viridis(0.45))
                best_idx = int(np.argmax(mean_r2.values))
                best_alpha = float(mean_r2.index.values[best_idx])
                ax.scatter([np.log10(best_alpha)], [mean_r2.values[best_idx]],
                           color="tab:red", s=120, zorder=4, edgecolor="white", linewidth=1.2)
                alpha_str = str(int(best_alpha)) if best_alpha == int(best_alpha) else f"{best_alpha:.2f}"
                ax.text(0.97, 0.5, f"α = {alpha_str}", ha="right", va="center",
                        fontsize=FONTSIZE_LABEL - 6, color="tab:red", transform=ax.transAxes)

            if ri == 0:
                ax.set_title(window_name.capitalize(), fontsize=FONTSIZE_TITLE, fontweight="bold")
            ax.set_xlabel(r"$\log_{10}(\alpha)$", fontsize=FONTSIZE_LABEL)
            if ci == 0:
                short = params.short_names.get(brain_area, brain_area)
                ax.set_ylabel(rf"$\bf{{{short}}}$" + "\nMean $R^2$ (CV)", fontsize=FONTSIZE_LABEL)
            else:
                ax.set_ylabel("")
            ax.tick_params(labelsize=FONTSIZE_LABEL)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
            sns.despine(ax=ax)

    fig.savefig(funcs.get_figure_dir("decoding/ridge") / f"{sound_type}_ridge_alpha_tuning.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: good-fit vs poor-fit scatter
# ---------------------------------------------------------------------------

def _collect_fit_examples(sound_types: list[str]) -> list[dict]:
    """Collect per-session ridge fits across all brain areas and windows for good/poor contrast."""
    results = []
    for sound_type in tqdm(sound_types, desc="Collecting fit examples", unit="sound", dynamic_ncols=True):
        for brain_area in get_plot_brain_regions(sound_type):
            for mouse_id in available_mice(sound_type):
                for session_id in available_sessions(sound_type, mouse_id=mouse_id):
                    for window_name in WINDOW_ORDER:
                        dataset = build_dataset(sound_type, window_name, brain_area,
                                                mouse_id=mouse_id, session_id=session_id)
                        if dataset is None or dataset["X"].shape[0] < 5:
                            continue
                        for td in build_target_datasets(dataset):
                            y = np.asarray(td["Y"], dtype=float)
                            fit = fit_best_ridge(td["X"], y, log_target=bool(td["log_target"]))
                            results.append({
                                "sound_type": sound_type,
                                "brain_area": brain_area,
                                "window_name": window_name,
                                "r2": fit["r2_test"],
                                "y_true": fit["y_test"],
                                "y_pred": fit["y_pred"],
                            })
    return results


def _select_good_bad_pair(fits: list[dict], min_gap: float = 0.3) -> tuple[dict, dict] | None:
    if len(fits) < 2:
        return None
    sorted_fits = sorted(fits, key=lambda x: x["r2"])
    poor_candidates = [f for f in sorted_fits if 0.1 <= f["r2"] <= 0.3]
    good_candidates = [f for f in reversed(sorted_fits) if f["r2"] >= 0.6]
    for bad in poor_candidates:
        for good in good_candidates:
            if good["r2"] - bad["r2"] >= min_gap and good["brain_area"] != bad["brain_area"]:
                return bad, good
    for bad in poor_candidates:
        for good in good_candidates:
            if good["r2"] - bad["r2"] >= min_gap:
                return bad, good
    # Fallback: relax poor-fit constraint but still avoid negatives
    poor_candidates = [f for f in sorted_fits if f["r2"] >= 0.0]
    for bad in poor_candidates:
        for good in good_candidates:
            if good["r2"] - bad["r2"] >= min_gap:
                return bad, good
    return sorted_fits[0], sorted_fits[-1]


def plot_ridge_fit_examples(sound_types: list[str] | None = None) -> None:
    """Plot a 2×2 predicted-vs-actual grid: PT (top) and AM (bottom), poor left and good right."""
    sound_types = sound_types or ["PT", "AM"]
    examples = _collect_fit_examples(sound_types)
    if not examples:
        return

    by_sound: dict[str, list[dict]] = {}
    for ex in examples:
        by_sound.setdefault(ex["sound_type"], []).append(ex)

    rows, row_labels = [], []
    for st in sound_types:
        pair = _select_good_bad_pair(by_sound.get(st, []))
        if pair is not None:
            rows.append(pair)
            row_labels.append(st)
    if not rows:
        return

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13.0, 6.0 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Ridge Regression: Predicted vs Actual", fontsize=FONTSIZE_SUPTITLE, fontweight="bold")

    for ri, ((bad, good), st) in enumerate(zip(rows, row_labels)):
        display_name = SOUND_DISPLAY_NAMES.get(st, st)
        for ci, (ex, label) in enumerate([(bad, "Poor Fit"), (good, "Good Fit")]):
            ax = axes[ri, ci]
            y_true = np.asarray(ex["y_true"], dtype=float)
            y_pred = np.asarray(ex["y_pred"], dtype=float)

            ax.scatter(y_true, y_pred, color=plt.cm.viridis(0.45), alpha=0.7,
                       s=55, edgecolor="white", linewidth=0.5)
            lim_min = min(float(y_true.min()), float(y_pred.min()))
            lim_max = max(float(y_true.max()), float(y_pred.max()))
            pad = (lim_max - lim_min) * 0.05
            ax.set_xlim(lim_min - pad, lim_max + pad)
            ax.set_ylim(lim_min - pad, lim_max + pad)
            ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad],
                    "k--", linewidth=1.5, alpha=0.7)

            short_area = params.short_names.get(ex["brain_area"], ex["brain_area"])
            window_label = ex["window_name"].capitalize()
            ax.set_title(f"{label}: {display_name} ({short_area}, {window_label})",
                         fontsize=FONTSIZE_TITLE - 6)
            ax.set_xlabel("Actual", fontsize=FONTSIZE_LABEL - 10)
            ax.set_ylabel("Predicted" if ci == 0 else "", fontsize=FONTSIZE_LABEL - 10)
            ax.text(0.05, 0.95, f"$R^2$ = {ex['r2']:.2f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=FONTSIZE_LABEL - 10)
            ax.tick_params(labelsize=FONTSIZE_LABEL - 10)
            ax.set_aspect("equal", adjustable="box")
            sns.despine(ax=ax)

    fig.savefig(funcs.get_figure_dir("decoding/ridge") / "ridge_fit_examples_PT_AM.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    apply_figure_style()
    for sound_type in tqdm(["PT", "AM"], desc="Ridge plots", unit="sound", dynamic_ncols=True):
        print(f"\n--- {sound_type} ---")
        n_neurons = params.NEURONS_PER_SESSION[sound_type]
        selected_sessions = collect_eligible_sessions(sound_type, n_neurons)

        results_df = build_session_frame(sound_type, n_neurons=n_neurons, selected_sessions=selected_sessions)
        if not results_df.empty:
            session_counts = {ba: len(list(ss)) for ba, ss in selected_sessions.items()}
            plot_ridge_summary(
                sound_type, results_df,
                title=f"Per-Session Ridge Regression for {SOUND_DISPLAY_NAMES.get(sound_type, sound_type)}",
                filename=f"{sound_type}_ridge_per_session.png",
                pair_cols=["Mouse", "Session"],
                session_counts=session_counts,
            )
        plot_ridge_alpha_tuning_grid(sound_type)

    plot_ridge_fit_examples(["PT", "AM"])


if __name__ == "__main__":
    main()
