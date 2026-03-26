"""Create per-mouse ridge R2 distributions for each sound type across regions and spike windows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import params

try:
    from ..methods.methods_analysis import WINDOW_ORDER, available_mice, build_dataset, fit_best_ridge, get_brain_regions
except ImportError:
    from methods.methods_analysis import WINDOW_ORDER, available_mice, build_dataset, fit_best_ridge, get_brain_regions

try:
    from .ridge_analysis import apply_figure_style, get_output_dir
except ImportError:
    from ridge_analysis import apply_figure_style, get_output_dir


def build_mouse_frame(sound_type: str) -> pd.DataFrame:
    """Return repeated per-mouse ridge scores for one sound type."""
    records = []
    for mouse_id in available_mice(sound_type):
        for brain_area in get_brain_regions(sound_type):
            if sound_type == "speech" and brain_area == "Dorsal auditory area":
                continue
            for window_name in WINDOW_ORDER:
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
                            "Window": window_name,
                            "Target": target_name,
                            "R2 Test": fit["r2_test"],
                        }
                    )
    return pd.DataFrame(records)


def main() -> None:
    """Run per-mouse ridge boxplot figures."""
    apply_figure_style()
    for sound_type in ("speech", "AM", "PT", "naturalSound"):
        results_df = build_mouse_frame(sound_type)
        if results_df.empty:
            continue
        brain_regions = [region for region in params.targetSiteNames if region in results_df["Brain Area"].unique()]
        fig, axes = plt.subplots(
            len(brain_regions),
            len(WINDOW_ORDER),
            figsize=(4.0 * len(WINDOW_ORDER), 3.4 * len(brain_regions)),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(f"{sound_type} per-mouse ridge $R^2$", fontsize=16, fontweight="bold")
        for row_index, brain_area in enumerate(brain_regions):
            for col_index, window_name in enumerate(WINDOW_ORDER):
                ax = axes[row_index, col_index]
                panel_df = results_df[(results_df["Brain Area"] == brain_area) & (results_df["Window"] == window_name)]
                if panel_df.empty:
                    ax.axis("off")
                    continue
                sns.boxplot(data=panel_df, x="Target", y="R2 Test", fliersize=2, linewidth=1, ax=ax)
                sns.stripplot(data=panel_df, x="Target", y="R2 Test", hue="Mouse", dodge=False, size=3, alpha=0.55, ax=ax)
                if ax.legend_ is not None:
                    ax.legend_.remove()
                ax.set_title(f"{params.short_names[brain_area]}\n{window_name.capitalize()}", fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("$R^2$")
                ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
        fig.savefig(get_output_dir() / f"{sound_type}_ridge_per_mouse.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
