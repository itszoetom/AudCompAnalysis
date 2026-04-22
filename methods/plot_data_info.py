"""Plot per-subregion neuron-count distributions for the Natural Sounds and Speech datasets.

Each figure shows histograms of neurons recorded per session, one panel per auditory cortical
subregion.  Two vertical markers indicate:
  - thin dashed  : per-session minimum required for per-session ridge regression (30 or 10 n)
  - thick solid  : fixed-count population subsampling target for encoding / discriminability (278 or 99 n)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

from shared import funcs, params

# Font sizes for medium-sized multi-panel summary figures
FONTSIZE_SUPTITLE = 24
FONTSIZE_TITLE = 20
FONTSIZE_LABEL = 18
FONTSIZE_TICK = 16
FONTSIZE_LEGEND = 16

# Minimum per-session neuron count for per-session ridge regression
SESSION_NEURON_CUTOFFS: dict[str, int] = {
    "nonspeech": 30,
    "speech": 10,
}

# Fixed neuron count for population encoding and discriminability analyses
POPULATION_NEURON_TARGETS: dict[str, int] = {
    "nonspeech": 278,
    "speech": 99,
}


def build_session_neuron_counts(sound_type: str) -> pd.DataFrame:
    """Return neurons per session per brain region for one sound type."""
    sound_data = funcs.load_sound_npz(sound_type)
    df = pd.DataFrame({
        "Brain Area": sound_data["brainRegionArray"],
        "Mouse": sound_data["mouseIDArray"],
        "Session": sound_data["sessionIDArray"],
    })
    return (
        df.groupby(["Brain Area", "Mouse", "Session"])
        .size()
        .reset_index(name="Neurons")
    )


def _viridis_colors(n: int) -> list[tuple[float, ...]]:
    """Return n evenly spaced viridis colors."""
    return [plt.cm.viridis(p) for p in np.linspace(0.15, 0.90, max(n, 1))]


def plot_neuron_count_histograms(
    counts_df: pd.DataFrame,
    brain_areas: list[str],
    *,
    title: str,
    filename: str,
    session_cutoff: int,
    population_target: int,
) -> None:
    """Save a two-panel figure: overlaid per-region histogram + total-neuron bar chart."""
    if counts_df.empty:
        return

    n_regions = len(brain_areas)
    colors = _viridis_colors(n_regions)

    fig, (ax_hist, ax_bar) = plt.subplots(
        1, 2,
        figsize=(11.0, 4.8),
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=FONTSIZE_SUPTITLE, fontweight="bold")

    # ---- Panel 1: overlaid histogram of neurons per session ----
    all_counts = counts_df[counts_df["Brain Area"].isin(brain_areas)]["Neurons"].values
    if len(all_counts) > 0:
        x_max = max(int(all_counts.max()), session_cutoff) * 1.12
    else:
        x_max = session_cutoff * 1.5
    bins = np.linspace(0, x_max, 22)

    for brain_area, color in zip(brain_areas, colors):
        region_df = counts_df[counts_df["Brain Area"] == brain_area]
        neuron_counts = region_df["Neurons"].values
        if len(neuron_counts) == 0:
            continue
        short_name = params.short_names.get(brain_area, brain_area)
        ax_hist.hist(
            neuron_counts,
            bins=bins,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.70,
            label=short_name,
        )

    ax_hist.axvline(
        session_cutoff,
        color="#222222",
        linestyle="--",
        linewidth=2.0,
        label=f"n = {session_cutoff}",
    )
    ax_hist.set_xlabel("Neurons per session", fontsize=FONTSIZE_LABEL)
    ax_hist.set_ylabel("Number of sessions", fontsize=FONTSIZE_LABEL)
    ax_hist.set_title("Session neuron counts", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax_hist.tick_params(labelsize=FONTSIZE_TICK)
    ax_hist.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.30)
    ax_hist.legend(fontsize=FONTSIZE_LEGEND, frameon=False)
    sns.despine(ax=ax_hist)

    # ---- Panel 2: total neurons per brain region + red threshold line ----
    total_neurons = [
        int(counts_df[counts_df["Brain Area"] == ba]["Neurons"].sum())
        for ba in brain_areas
    ]
    short_names = [params.short_names.get(ba, ba) for ba in brain_areas]
    x_pos = np.arange(n_regions)

    ax_bar.bar(x_pos, total_neurons, color=colors, edgecolor="white", linewidth=0.6, alpha=0.88)
    ax_bar.axhline(
        population_target,
        color="tab:red",
        linestyle="-",
        linewidth=2.2,
        label=f"n = {population_target}",
    )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(short_names, fontsize=FONTSIZE_TICK)
    ax_bar.set_ylabel("Total neurons", fontsize=FONTSIZE_LABEL)
    ax_bar.set_title("Total neurons per region", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax_bar.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    ax_bar.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.30)
    ax_bar.legend(fontsize=FONTSIZE_LEGEND, frameon=False)
    sns.despine(ax=ax_bar)

    fig.savefig(funcs.get_figure_dir("methods") / filename, dpi=300)
    plt.close(fig)


def main() -> None:
    """Build neuron-count summary figures for the Natural Sounds and Speech datasets."""
    funcs.apply_figure_style()
    print("Building data info summary figures...")

    # Natural Sounds Dataset — use PT as the representative sound type since PT, AM, and
    # naturalSound all share the same recording database and session structure.
    nonspeech_counts = build_session_neuron_counts("PT")
    nonspeech_areas = [r for r in params.targetSiteNames
                       if r in nonspeech_counts["Brain Area"].unique()]
    plot_neuron_count_histograms(
        nonspeech_counts,
        nonspeech_areas,
        title="Natural Sounds Dataset: Neurons Recorded Per Session",
        filename="nonspeech_data_summary.png",
        session_cutoff=SESSION_NEURON_CUTOFFS["nonspeech"],
        population_target=POPULATION_NEURON_TARGETS["nonspeech"],
    )

    # Speech Dataset — AuD excluded (insufficient neuron count for subsampling)
    speech_counts = build_session_neuron_counts("speech")
    speech_areas = [
        r for r in params.targetSiteNames
        if r in speech_counts["Brain Area"].unique() and r != "Dorsal auditory area"
    ]
    plot_neuron_count_histograms(
        speech_counts,
        speech_areas,
        title="Speech Dataset: Neurons Recorded Per Session",
        filename="speech_data_summary.png",
        session_cutoff=SESSION_NEURON_CUTOFFS["speech"],
        population_target=POPULATION_NEURON_TARGETS["speech"],
    )


if __name__ == "__main__":
    main()
