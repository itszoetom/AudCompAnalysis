"""Plot example single-cell rasters and PSTHs with viridis stimulus coloring."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from jaratoolbox import behavioranalysis, celldatabase, ephyscore, spikesanalysis

import params
import funcs

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})

N_CELLS_TO_PLOT = 1
MAX_CELLS_TO_SCORE = 24
BIN_WIDTH = 0.01
EXAMPLE_SESSIONS = (
    {
        "figure_key": "speech",
        "subject": "feat005",
        "session_date": "2022-02-07",
        "pdepth": 3020,
        "database_path": os.path.join(params.DATABASE_PATH, params.SPEECH_STUDY_NAME, "celldb_feat005.h5"),
        "plot_types": ("FT", "VOT"),
    },
    {
        "figure_key": "nonspeech",
        "subject": "feat014",
        "session_date": "2024-02-22",
        "pdepth": None,
        "database_path": params.fullPath,
        "plot_types": ("pureTones", "AM", "naturalSound"),
    },
)
PLOT_WINDOWS = {
    "pureTones": {"time_range": np.array([-0.1, 0.3]), "markers": (0.0, 0.03, 0.1, 0.13)},
    "AM": {"time_range": np.array([-0.5, 1.5]), "markers": (0.0, 0.2, 0.5, 0.7)},
    "naturalSound": {"time_range": np.array([-2.0, 6.0]), "markers": (0.0, 0.5, 4.0, 4.5)},
    "FT": {"time_range": np.array([-0.5, 1.5]), "markers": (0.0, 0.2, 0.5, 0.7)},
    "VOT": {"time_range": np.array([-0.5, 1.5]), "markers": (0.0, 0.2, 0.5, 0.7)},
}


def get_output_dir() -> Path:
    """Return the PSTH figure output directory."""
    return funcs.get_nested_figure_dir("methods", "psth")


def load_session_dataframe(config: dict[str, object]):
    """Load one example recording session."""
    mouse_df = celldatabase.load_hdf(config["database_path"])
    if "subject" in mouse_df.columns:
        mouse_df = mouse_df[mouse_df.subject == config["subject"]]
    mouse_df = mouse_df[mouse_df.date == config["session_date"]]
    if config["pdepth"] is not None:
        mouse_df = mouse_df[mouse_df.pdepth == config["pdepth"]]
    elif not mouse_df.empty:
        mouse_df = mouse_df[mouse_df.pdepth == np.sort(mouse_df.pdepth.unique())[0]]
    return mouse_df.reset_index(drop=True)


def get_trial_groups(plot_type: str, behavior_data):
    """Return grouped trials and readable labels for one plot type."""
    if plot_type in {"FT", "VOT"}:
        speech_labels = np.column_stack((behavior_data["targetFTpercent"], behavior_data["targetVOTpercent"])).astype(int)
        keep_indices, sorted_labels = funcs.select_speech_trials(
            speech_labels,
            max_repeats=params.SPEECH_REPEATS_PER_TOKEN,
        )
        feature_index = 0 if plot_type == "FT" else 1
        feature_name = "FT" if plot_type == "FT" else "VOT"
        possible_params = np.unique(sorted_labels[:, feature_index])
        selectors = [keep_indices[sorted_labels[:, feature_index] == value] for value in possible_params]
        labels = [f"{feature_name} {int(value)}%" for value in possible_params]
        return selectors, labels

    if plot_type == "naturalSound":
        trial_param = np.asarray(behavior_data["soundID"], dtype=int)
        possible_params = np.unique(trial_param)
        selectors = [np.flatnonzero(trial_param == value) for value in possible_params]
        labels = [params.NAT_SOUND_LABEL_MAP[int(value)] for value in possible_params]
        return selectors, labels

    trial_param = np.asarray(behavior_data["currentFreq"])
    possible_params = np.unique(trial_param)
    selectors = [np.flatnonzero(trial_param == value) for value in possible_params]
    unit = "kHz" if plot_type == "pureTones" else "Hz"
    labels = [f"{value / 1000:.0f} {unit}" if plot_type == "pureTones" else f"{int(value)} {unit}" for value in possible_params]
    return selectors, labels

def load_plot_data(one_cell, plot_type: str):
    """Load spike-aligned data for one cell and plot type."""
    session_type = "FTVOTBorders" if plot_type in {"FT", "VOT"} else plot_type
    ephys_data, behavior_data = one_cell.load(sessiontype=session_type, behavClass=None)
    time_range = PLOT_WINDOWS[plot_type]["time_range"]
    bins_start = np.arange(time_range[0], time_range[1], BIN_WIDTH)
    spike_times_from_onset, _, index_limits = spikesanalysis.eventlocked_spiketimes(
        ephys_data["spikeTimes"],
        ephys_data["events"]["stimOn"],
        time_range,
        spikeindex=False,
    )
    index_limits = funcs.normalize_index_limits(index_limits)
    trials_each_type, labels = get_trial_groups(plot_type, behavior_data)
    return spike_times_from_onset, index_limits, bins_start, time_range, trials_each_type, labels


def condition_colors(n_conditions: int) -> np.ndarray:
    """Return one viridis color per stimulus condition."""
    return plt.cm.viridis(np.linspace(0.08, 0.95, max(n_conditions, 2)))[:n_conditions]


def raster_ylabel(plot_type: str) -> str:
    """Return the label used on the raster y-axis."""
    return {
        "pureTones": "Frequency (kHz)",
        "AM": "AM Rate (Hz)",
        "FT": "FT (%)",
        "VOT": "VOT (%)",
        "naturalSound": "Sound",
    }.get(plot_type, "Condition")


def add_window_markers(ax, plot_type: str) -> None:
    """Draw spike-window boundaries on one axis."""
    for marker, color in zip(PLOT_WINDOWS[plot_type]["markers"], ("y", "c", "r", "r")):
        ax.axvline(marker, color=color, linestyle="--", linewidth=1.5)


def plot_colored_raster(ax, spike_times_from_onset, index_limits, trials_each_type, labels, time_range) -> None:
    """Plot one dot raster with grouped colored sidebars for stimulus conditions."""
    colors = condition_colors(len(labels))
    n_trials = index_limits.shape[0]
    sidebar_frac = 0.055
    y_position = 1
    y_ticks = []
    y_labels = []
    for condition_index, (selector, label) in enumerate(zip(trials_each_type, labels)):
        trial_numbers = funcs.trial_indices_from_selector(selector, n_trials)
        if len(trial_numbers) == 0:
            continue
        start_y = y_position
        for trial_number in trial_numbers:
            spikes = spike_times_from_onset[index_limits[trial_number, 0] : index_limits[trial_number, 1]]
            if len(spikes):
                ax.scatter(spikes, np.full(spikes.shape, y_position), s=14, color="black", linewidths=0)
            y_position += 1
        stop_y = y_position - 1
        y_ticks.append((start_y + stop_y) / 2)
        y_labels.append(label)
        ax.axhspan(start_y - 0.5, stop_y + 0.5, xmin=0.0, xmax=sidebar_frac, color=colors[condition_index], ec=None)
        ax.axhspan(start_y - 0.5, stop_y + 0.5, xmin=1.0 - sidebar_frac, xmax=1.0, color=colors[condition_index], ec=None)
    ax.set_xlim(time_range[0], time_range[1])
    ax.set_ylim(0.5, max(y_position - 0.5, 1.5))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)


def trial_spike_times(spike_times_from_onset: np.ndarray, index_limits: np.ndarray, trial_number: int) -> np.ndarray:
    """Return event-locked spike times for one trial."""
    return spike_times_from_onset[index_limits[trial_number, 0] : index_limits[trial_number, 1]]


def plot_colored_psth(ax, spike_times_from_onset: np.ndarray, index_limits: np.ndarray, bins_start: np.ndarray, trials_each_type, labels) -> None:
    """Plot one PSTH axis with one viridis line per stimulus condition."""
    colors = condition_colors(len(labels))
    n_trials = index_limits.shape[0]
    bin_edges = np.append(bins_start, bins_start[-1] + BIN_WIDTH)
    for condition_index, (selector, label) in enumerate(zip(trials_each_type, labels)):
        trial_numbers = funcs.trial_indices_from_selector(selector, n_trials)
        if len(trial_numbers) == 0:
            continue
        trial_histograms = [np.histogram(trial_spike_times(spike_times_from_onset, index_limits, trial_number), bins=bin_edges)[0] for trial_number in trial_numbers]
        mean_rate = np.mean(np.vstack(trial_histograms), axis=0) / BIN_WIDTH
        smooth_kernel = np.ones(5) / 5
        smooth_rate = np.convolve(mean_rate, smooth_kernel, mode="same")
        ax.plot(bins_start, smooth_rate, color=colors[condition_index], linewidth=3, alpha=0.98)


def condition_response_strength(spike_times_from_onset: np.ndarray, index_limits: np.ndarray, bins_start: np.ndarray, trials_each_type) -> float:
    """Return the strongest mean onset response across stimulus conditions."""
    onset_stop = 0.5 if bins_start[0] <= -1 else 0.2
    n_trials = index_limits.shape[0]
    scores = []
    for selector in trials_each_type:
        trial_numbers = funcs.trial_indices_from_selector(selector, n_trials)
        if len(trial_numbers) == 0:
            continue
        onset_rates = [
            np.count_nonzero((trial_spike_times(spike_times_from_onset, index_limits, trial_number) >= 0.0) & (trial_spike_times(spike_times_from_onset, index_limits, trial_number) < onset_stop)) / onset_stop
            for trial_number in trial_numbers
        ]
        scores.append(float(np.mean(onset_rates)))
    return max(scores) if scores else -np.inf


def find_best_cell_indices(session_df, config: dict[str, object]) -> list[int]:
    """Return the most stimulus-responsive example cells for one session."""
    scored_cells = []
    error_messages: list[str] = []
    n_candidates = min(len(session_df), MAX_CELLS_TO_SCORE)
    for cell_index in range(n_candidates):
        one_cell = ephyscore.Cell(session_df.iloc[cell_index])
        total_score = 0.0
        valid_plot = False
        for plot_type in config["plot_types"]:
            try:
                spike_times_from_onset, index_limits, bins_start, _, trials_each_type, _ = load_plot_data(one_cell, plot_type)
            except Exception as exc:
                error_messages.append(f"cell {cell_index} {plot_type}: {type(exc).__name__}: {exc}")
                continue
            total_score += condition_response_strength(spike_times_from_onset, index_limits, bins_start, trials_each_type)
            valid_plot = True
        if valid_plot:
            scored_cells.append((total_score, cell_index))
    scored_cells.sort(reverse=True)
    if scored_cells:
        return [cell_index for _, cell_index in scored_cells[:N_CELLS_TO_PLOT]]

    print(
        f"No scored example cells found for {config['figure_key']} "
        f"({config['subject']} {config['session_date']}). Falling back to the first available cells."
    )
    for message in error_messages[:6]:
        print(f"  {message}")
    return list(range(min(N_CELLS_TO_PLOT, len(session_df))))


def main() -> None:
    """Run the example single-cell PSTH figure generation."""
    for config in EXAMPLE_SESSIONS:
        session_df = load_session_dataframe(config)
        if session_df.empty:
            continue
        best_cell_indices = find_best_cell_indices(session_df, config)
        if not best_cell_indices:
            continue
        print(f"Building example PSTHs for {config['subject']} {config['session_date']} ({config['figure_key']})...")
        for cell_index in tqdm(best_cell_indices, desc=f"Methods PSTHs ({config['figure_key']})", unit="cell", dynamic_ncols=True):
            one_cell = ephyscore.Cell(session_df.iloc[cell_index])
            plot_types = config["plot_types"]
            fig, axes = plt.subplots(
                1,
                2 * len(plot_types),
                figsize=(5.0 * len(plot_types), 6.4),
                constrained_layout=True,
            )
            fig.suptitle(f"Cell {cell_index}", fontsize=18)

            for col_index, plot_type in enumerate(plot_types):
                try:
                    spike_times_from_onset, index_limits, bins_start, time_range, trials_each_type, labels = load_plot_data(one_cell, plot_type)
                except Exception as exc:
                    print(f"Skipping cell {cell_index} {plot_type}: {type(exc).__name__}: {exc}")
                    axes[2 * col_index].axis("off")
                    axes[2 * col_index + 1].axis("off")
                    continue

                raster_ax = axes[2 * col_index]
                psth_ax = axes[2 * col_index + 1]

                plot_colored_raster(raster_ax, spike_times_from_onset, index_limits, trials_each_type, labels, time_range)
                raster_ax.set_title(f"{plot_type} - Raster Plot", fontsize=17)
                raster_ax.set_xlabel("Time from onset (s)")
                raster_ax.set_ylabel(raster_ylabel(plot_type))
                add_window_markers(raster_ax, plot_type)

                plot_colored_psth(psth_ax, spike_times_from_onset, index_limits, bins_start, trials_each_type, labels)
                psth_ax.set_title(f"{plot_type} - PSTH", fontsize=17)
                psth_ax.set_xlabel("Time from onset (s)")
                psth_ax.set_ylabel("Firing Rate (spikes/s)")
                psth_ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
                add_window_markers(psth_ax, plot_type)

            fig.savefig(get_output_dir() / f"{config['subject']}_{config['session_date']}_{config['figure_key']}_cell_{cell_index}_psth.png", dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    main()
