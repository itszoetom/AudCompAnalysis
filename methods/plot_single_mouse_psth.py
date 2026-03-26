"""Plot example single-cell PSTHs with spike-window markers for representative sessions."""

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
BIN_WIDTH = 0.01
PLOT_TYPES = ("pureTones", "AM", "FT", "VOT")
PLOT_WINDOWS = {
    "pureTones": {"time_range": np.array([-0.1, 0.3]), "markers": (0.0, 0.03, 0.1, 0.13)},
    "AM": {"time_range": np.array([-0.5, 1.5]), "markers": (0.0, 0.2, 0.5, 0.7)},
    "FT": {"time_range": np.array([-0.5, 1.5]), "markers": (0.0, 0.2, 0.5, 0.7)},
    "VOT": {"time_range": np.array([-0.5, 1.5]), "markers": (0.0, 0.2, 0.5, 0.7)},
}


def get_output_dir() -> Path:
    """Return the PSTH figure output directory."""
    output_dir = Path(params.figSavePath) / "methods" / "psth"
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
        labels = [f"{ft}%" for ft in possible_params]
    elif plot_type == "VOT":
        trial_param = behavior_data["targetVOTpercent"]
        possible_params = np.unique(trial_param)
        labels = [f"{vot}%" for vot in possible_params]
    else:
        trial_param = behavior_data["currentFreq"]
        possible_params = np.unique(trial_param)
        if plot_type == "pureTones":
            labels = [f"{freq / 1000:.0f}" for freq in possible_params]
        else:
            labels = [f"{freq:.0f}" for freq in possible_params]

    return behavioranalysis.find_trials_each_type(trial_param, possible_params), labels


def load_plot_data(one_cell, plot_type: str):
    """Load spike-aligned data for one cell and plot type."""
    session_type = "FTVOTBorders" if plot_type in {"FT", "VOT"} else plot_type
    ephys_data, behavior_data = one_cell.load(sessiontype=session_type, behavClass=None)
    time_range = PLOT_WINDOWS[plot_type]["time_range"]
    bins_start = np.arange(time_range[0], time_range[1], BIN_WIDTH)
    spike_times = ephys_data["spikeTimes"]
    event_onsets = ephys_data["events"]["stimOn"]
    spike_times_from_onset, _, index_limits = spikesanalysis.eventlocked_spiketimes(
        spike_times,
        event_onsets,
        time_range,
        spikeindex=False,
    )
    spike_count_matrix = np.asarray(
        spikesanalysis.spiketimes_to_spikecounts(spike_times_from_onset, index_limits, bins_start)
    )
    trials_each_type, labels = get_trial_groups(plot_type, behavior_data)
    return spike_times_from_onset, index_limits, spike_count_matrix, bins_start, time_range, trials_each_type, labels


def add_window_markers(ax, plot_type: str):
    """Draw the spike-window boundaries on one axis."""
    colors = ("#d9a404", "#2a9d8f", "#c0392b", "#c0392b")
    for marker, color in zip(PLOT_WINDOWS[plot_type]["markers"], colors):
        ax.axvline(marker, color=color, linestyle="--", linewidth=1)


def main() -> None:
    """Run the example single-cell PSTH figure generation."""
    session_df = load_session_dataframe()
    if session_df.empty:
        return

    n_cells = min(N_CELLS_TO_PLOT, len(session_df))
    for cell_index in range(n_cells):
        one_cell = ephyscore.Cell(session_df.iloc[cell_index])
        fig, axes = plt.subplots(2, len(PLOT_TYPES), figsize=(4.2 * len(PLOT_TYPES), 6), constrained_layout=True)
        fig.suptitle(f"{SUBJECT} {SESSION_DATE} cell {cell_index}", fontsize=14)

        for col_index, plot_type in enumerate(PLOT_TYPES):
            try:
                spike_times_from_onset, index_limits, spike_count_matrix, bins_start, time_range, trials_each_type, labels = (
                    load_plot_data(one_cell, plot_type)
                )
            except IndexError:
                axes[0, col_index].axis("off")
                axes[1, col_index].axis("off")
                continue

            plt.sca(axes[0, col_index])
            extraplots.raster_plot(
                spike_times_from_onset,
                index_limits,
                timeRange=time_range,
                trialsEachCond=trials_each_type,
                labels=labels,
                rasterized=True,
            )
            axes[0, col_index].set_title(f"{plot_type} raster")
            axes[0, col_index].set_xlabel("Time from onset (s)")
            add_window_markers(axes[0, col_index], plot_type)

            plt.sca(axes[1, col_index])
            extraplots.plot_psth(
                spike_count_matrix / BIN_WIDTH,
                smoothWinSize=6,
                binsStartTime=bins_start,
                trialsEachCond=trials_each_type,
                linewidth=2,
                downsamplefactor=1,
            )
            axes[1, col_index].set_title(f"{plot_type} PSTH")
            axes[1, col_index].set_xlabel("Time from onset (s)")
            axes[1, col_index].set_ylabel("Firing rate (spk/s)")
            add_window_markers(axes[1, col_index], plot_type)

        fig.savefig(get_output_dir() / f"{SUBJECT}_{SESSION_DATE}_cell_{cell_index}_psth.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
