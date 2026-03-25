"""Ridge-decoding summary figures arranged by sound, brain area, and spike window."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .ridge_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        get_brain_regions,
        get_output_dir,
        list_available_sound_types,
        run_population_ridge,
    )
except ImportError:
    from ridge_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        get_brain_regions,
        get_output_dir,
        list_available_sound_types,
        run_population_ridge,
    )


def main() -> None:
    apply_figure_style()
    results_df = run_population_ridge(iterations=30)
    if results_df.empty:
        return

    for sound_type in list_available_sound_types():
        sound_df = results_df[results_df["Sound Type"] == sound_type]
        brain_regions = get_brain_regions(sound_type)
        fig, axes = plt.subplots(
            len(brain_regions),
            len(WINDOW_ORDER),
            figsize=(3.2 * len(WINDOW_ORDER), 2.8 * len(brain_regions)),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(f"{sound_type} ridge decoding", fontsize=16)

        for row_index, brain_area in enumerate(brain_regions):
            for col_index, window_name in enumerate(WINDOW_ORDER):
                ax = axes[row_index, col_index]
                panel_df = sound_df[
                    (sound_df["Brain Area"] == brain_area) & (sound_df["Window"] == window_name)
                ].copy()
                if panel_df.empty:
                    ax.axis("off")
                    continue

                if sound_type == "speech":
                    sns.boxplot(data=panel_df, x="Target", y="R2 Test", linewidth=1, fliersize=2, ax=ax)
                    sns.stripplot(data=panel_df, x="Target", y="R2 Test", dodge=True, size=3, alpha=0.45, ax=ax)
                else:
                    panel_df["Target"] = sound_type
                    sns.boxplot(data=panel_df, x="Target", y="R2 Test", linewidth=1, fliersize=2, ax=ax)
                    sns.stripplot(data=panel_df, x="Target", y="R2 Test", size=3, alpha=0.45, ax=ax)

                ax.set_title(f"{brain_area}\n{window_name.capitalize()}")
                ax.set_xlabel("")
                ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

        figure_path = get_output_dir() / f"{sound_type}_ridge_boxplots.png"
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)
        print(f"Saved ridge boxplots to {figure_path}")


if __name__ == "__main__":
    main()
