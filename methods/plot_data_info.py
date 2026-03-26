"""Plot neuron-count and recording-summary figures for each sound type using sustained-window metadata."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import params

try:
    from .methods_analysis import apply_figure_style, get_figure_dir, list_available_sound_types, load_sound_npz
except ImportError:
    from methods_analysis import apply_figure_style, get_figure_dir, list_available_sound_types, load_sound_npz


def build_count_frame(sound_type: str) -> pd.DataFrame:
    """Return one metadata table of neuron counts by mouse, session, and brain region."""
    sound_data = load_sound_npz(sound_type)
    frame = pd.DataFrame(
        {
            "Brain Area": sound_data["brainRegionArray"],
            "Mouse": sound_data["mouseIDArray"],
            "Session": sound_data["sessionIDArray"],
        }
    )
    return frame


def main() -> None:
    """Run neuron-count summary figures for each sound type."""
    apply_figure_style()
    for sound_type in list_available_sound_types():
        count_frame = build_count_frame(sound_type)
        if count_frame.empty:
            continue
        session_counts = count_frame.groupby(["Brain Area", "Session"]).size().reset_index(name="Neurons per Session")
        mouse_counts = count_frame.groupby(["Brain Area", "Mouse"]).size().reset_index(name="Neurons per Mouse")
        total_counts = count_frame.groupby("Brain Area").size().reset_index(name="Total Neurons")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
        fig.suptitle(f"{sound_type} data summary", fontsize=16, fontweight="bold")

        sns.histplot(data=session_counts, x="Neurons per Session", hue="Brain Area", multiple="stack", ax=axes[0])
        axes[0].set_title("Session counts", fontweight="bold")
        axes[0].set_xlabel("Neurons per session")

        sns.histplot(data=mouse_counts, x="Neurons per Mouse", hue="Brain Area", multiple="stack", ax=axes[1])
        axes[1].set_title("Mouse counts", fontweight="bold")
        axes[1].set_xlabel("Neurons per mouse")

        sns.barplot(data=total_counts, x="Brain Area", y="Total Neurons", ax=axes[2])
        axes[2].set_title("Total neurons", fontweight="bold")
        axes[2].set_xlabel("")
        axes[2].tick_params(axis="x", rotation=20)

        for ax in axes:
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)

        fig.savefig(get_figure_dir() / f"{sound_type}_data_summary.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
