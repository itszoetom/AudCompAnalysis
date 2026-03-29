"""Plot dataset summary figures for speech and combined non-speech recordings."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import params
import funcs


def build_count_frame(sound_type: str) -> pd.DataFrame:
    """Return one metadata table of neuron counts by mouse, session, and brain region."""
    sound_data = funcs.load_sound_npz(sound_type)
    return pd.DataFrame(
        {
            "Brain Area": sound_data["brainRegionArray"],
            "Mouse": sound_data["mouseIDArray"],
            "Session": sound_data["sessionIDArray"],
        }
    )


def viridis_region_palette(brain_areas: list[str]) -> dict[str, tuple[float, ...]]:
    """Return a viridis palette keyed by brain area."""
    colors = plt.cm.viridis([0.15, 0.4, 0.65, 0.9])
    return {brain_area: colors[index] for index, brain_area in enumerate(brain_areas)}


def plot_summary_figure(count_frame: pd.DataFrame, title: str, filename: str, session_cutoff: int) -> None:
    """Plot one combined data-summary figure."""
    if count_frame.empty:
        return
    brain_areas = [region for region in params.targetSiteNames if region in count_frame["Brain Area"].unique()]
    palette = viridis_region_palette(brain_areas)
    session_counts = count_frame.groupby(["Brain Area", "Session"]).size().reset_index(name="Neurons per Session")
    mouse_counts = count_frame.groupby(["Brain Area", "Mouse"]).size().reset_index(name="Neurons per Mouse")
    total_counts = count_frame.groupby("Brain Area").size().reset_index(name="Total Neurons")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    sns.histplot(
        data=session_counts,
        x="Neurons per Session",
        hue="Brain Area",
        multiple="stack",
        palette=palette,
        hue_order=brain_areas,
        ax=axes[0],
    )
    axes[0].axvline(session_cutoff, color="black", linestyle=":", linewidth=1.5)
    axes[0].set_title("Session counts", fontweight="bold")
    axes[0].set_xlabel("Neurons per session")

    sns.histplot(
        data=mouse_counts,
        x="Neurons per Mouse",
        hue="Brain Area",
        multiple="stack",
        palette=palette,
        hue_order=brain_areas,
        ax=axes[1],
    )
    axes[1].set_title("Mouse counts", fontweight="bold")
    axes[1].set_xlabel("Neurons per mouse")

    sns.barplot(
        data=total_counts,
        x="Brain Area",
        y="Total Neurons",
        palette=palette,
        order=brain_areas,
        hue="Brain Area",
        dodge=False,
        legend=False,
        ax=axes[2],
    )
    axes[2].set_title("Total neurons", fontweight="bold")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=20)

    for ax in axes:
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)

    fig.savefig(funcs.get_figure_dir("methods") / filename, dpi=300)
    plt.close(fig)


def main() -> None:
    """Run combined non-speech and speech data-summary figures."""
    funcs.apply_figure_style()
    print("Building combined non-speech and speech data summaries...")
    speech_frame = build_count_frame("speech")
    nonspeech_frame = build_count_frame("PT")
    summary_jobs = [
        (nonspeech_frame, "Non-speech data summary (PT, AM, natural sounds)", "nonspeech_data_summary.png", 30),
        (speech_frame, "Speech data summary", "speech_data_summary.png", 10),
    ]
    for frame, title, filename, session_cutoff in tqdm(summary_jobs, desc="Methods data summaries", unit="figure", dynamic_ncols=True):
        plot_summary_figure(frame, title=title, filename=filename, session_cutoff=session_cutoff)


if __name__ == "__main__":
    main()
