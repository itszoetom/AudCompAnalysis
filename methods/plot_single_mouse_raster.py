"""Plot example single-cell rasters with spike-window markers for representative sessions."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jaratoolbox import behavioranalysis, celldatabase, ephyscore, extraplots, spikesanalysis, settings

import params

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})

SUBJECT = "feat005"
SESSION_DATE = "2022-02-07"
PROBE_DEPTH = 3020
N_CELLS_TO_PLOT = 12
PLOT_TYPES = ("pureTones", "AM", "FT", "VOT")
PLOT_WINDOWS = {
    "pureTones": {"time_range": np.array([-0.1, 0.3]), "markers": (0.0, 0.03, 0.1, 0.13)},
    "AM": {"time_range": np.array([-0.5, 1.5]), "markers": (0.0, 0.2, 0.5, 0.7)},
    "FT": {"time_range": np.array([-0.5, 1.5]), "markers": (0.0, 0.2, 0.5, 0.7)},
    "VOT": {"time_range": np.array([-0.5, 1.5]), "markers": (0.0, 0.2, 0.5, 0.7)},
}


def get_output_dir() -> Path:
    """Return the raster figure output directory."""
    output_dir = Path(params.figSavePath) / "methods" / "raster"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_session_dataframe():
    """Load one example recording session from the speech dataset."""
    db_path = os.path.join(settings.DATABASE_PATH, params.SPEECH_STUDY_NAME, f"celldb_{SUBJECT}.h5")
    mouse_df = celldatabase.load_hdf(db_path)
    return mouse_df[(mouse_df.date == SESSION_DATE) & (mouse_df.pdepth == PROBE_DEPTH)].reset_index(drop=True)


def get_trial_groups(plot_type: str, behavior_data):
    """Return grouped trials and labels for one plot type."""
    if plot_type == "FT":
        trial_param = behavior_data["targetFTpercent"]
        possible_params = np.unique(trial_param)
        labels = [f"FT {int(value)}%" for value in possible_params]
    elif plot_type == "VOT":
        trial_param = behavior_data["targetVOTpercent"]
        possible_params = np.unique(trial_param)
        labels = [f"VOT {int(value)}%" for value in possible_params]
    else:
        trial_param = behavior_data["currentFreq"]
        possible_params = np.unique(trial_param)
        labels = [f"{int(value)}" for value in possible_params]
    return behavioranalysis.find_trials_each_type(trial_param, possible_params), labels


def load_plot_data(one_cell, plot_type: str):
    """Load spike-aligned data for one cell and plot type."""
    session_type = "FTVOTBorders" if plot_type in {"FT", "VOT"} else plot_type
    ephys_data, behavior_data = one_cell.load(sessiontype=session_type, behavClass=None)
    time_range = PLOT_WINDOWS[plot_type]["time_range"]
    spike_times_from_onset, _, index_limits = spikesanalysis.eventlocked_spiketimes(
        ephys_data["spikeTimes"],
        ephys_data["events"]["stimOn"],
        time_range,
        spikeindex=False,
    )
    trials_each_type, labels = get_trial_groups(plot_type, behavior_data)
    return spike_times_from_onset, index_limits, time_range, trials_each_type, labels


def add_window_markers(ax, plot_type: str):
    """Draw the spike-window boundaries on one axis."""
    for marker in PLOT_WINDOWS[plot_type]["markers"]:
        ax.axvline(marker, color="black", linestyle="--", linewidth=1)


def main() -> None:
    """Run the example single-cell raster figure generation."""
    session_df = load_session_dataframe()
    if session_df.empty:
        return
    for cell_index in range(min(N_CELLS_TO_PLOT, len(session_df))):
        one_cell = ephyscore.Cell(session_df.iloc[cell_index])
        fig, axes = plt.subplots(1, len(PLOT_TYPES), figsize=(4.2 * len(PLOT_TYPES), 3.8), constrained_layout=True)
        fig.suptitle(f"{SUBJECT} {SESSION_DATE} cell {cell_index}", fontsize=14)
        for col_index, plot_type in enumerate(PLOT_TYPES):
            try:
                spike_times_from_onset, index_limits, time_range, trials_each_type, labels = load_plot_data(one_cell, plot_type)
            except IndexError:
                axes[col_index].axis("off")
                continue
            plt.sca(axes[col_index])
            extraplots.raster_plot(
                spike_times_from_onset,
                index_limits,
                timeRange=time_range,
                trialsEachCond=trials_each_type,
                labels=labels,
                rasterized=True,
            )
            axes[col_index].set_title(f"{plot_type} raster", fontweight="bold")
            axes[col_index].set_xlabel("Time from onset (s)")
            axes[col_index].set_ylabel("Trial")
            add_window_markers(axes[col_index], plot_type)
        fig.savefig(get_output_dir() / f"{SUBJECT}_{SESSION_DATE}_cell_{cell_index}_raster.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
