"""Plot one example single-neuron firing-rate figure for methods explanations."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from jaratoolbox import behavioranalysis, celldatabase, ephyscore, spikesanalysis, settings

import params

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})

SUBJECT = "feat005"
SESSION_DATE = "2022-02-07"
PROBE_DEPTH = 3020
MAX_CELLS_TO_SCORE = 24
PLOT_TYPE = "FT"
BIN_WIDTH = 0.01
PLOT_WINDOW = {"time_range": np.array([-0.5, 1.0]), "markers": (0.0, 0.2, 0.5, 0.7)}
WINDOW_LABELS = ("Onset", "Sustained", "Offset")


def get_output_dir() -> Path:
    """Return the spike-rate figure output directory."""
    output_dir = Path(params.figSavePath) / "methods" / "spike_rate"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_session_dataframe():
    """Load the example recording session used for the methods figures."""
    db_path = os.path.join(settings.DATABASE_PATH, params.SPEECH_STUDY_NAME, f"celldb_{SUBJECT}.h5")
    mouse_df = celldatabase.load_hdf(db_path)
    return mouse_df[(mouse_df.date == SESSION_DATE) & (mouse_df.pdepth == PROBE_DEPTH)].reset_index(drop=True)


def get_trial_groups(behavior_data):
    """Return FT-grouped trials and readable labels."""
    trial_param = behavior_data["targetFTpercent"]
    possible_params = np.unique(trial_param)
    labels = [f"FT {int(value)}%" for value in possible_params]
    return behavioranalysis.find_trials_each_type(trial_param, possible_params), possible_params, labels


def trial_indices_from_selector(selector: np.ndarray, n_trials: int) -> np.ndarray:
    """Convert one trial selector into integer trial indices."""
    selector_array = np.asarray(selector)
    if selector_array.dtype == bool:
        return np.flatnonzero(selector_array)
    selector_array = selector_array.astype(int).ravel()
    if selector_array.size == n_trials and np.all(np.isin(selector_array, [0, 1])):
        return np.flatnonzero(selector_array)
    return selector_array


def normalize_index_limits(index_limits: np.ndarray) -> np.ndarray:
    """Ensure event-locked index limits are shaped as trials x 2."""
    limits = np.asarray(index_limits)
    if limits.ndim != 2:
        raise ValueError(f"Unexpected index_limits shape {limits.shape}.")
    if limits.shape[1] == 2:
        return limits
    if limits.shape[0] == 2:
        return limits.T
    raise ValueError(f"Unexpected index_limits shape {limits.shape}.")


def load_plot_data(one_cell):
    """Load event-locked spike data for the example cell."""
    ephys_data, behavior_data = one_cell.load(sessiontype="FTVOTBorders", behavClass=None)
    bins_start = np.arange(PLOT_WINDOW["time_range"][0], PLOT_WINDOW["time_range"][1], BIN_WIDTH)
    spike_times_from_onset, _, index_limits = spikesanalysis.eventlocked_spiketimes(
        ephys_data["spikeTimes"],
        ephys_data["events"]["stimOn"],
        PLOT_WINDOW["time_range"],
        spikeindex=False,
    )
    index_limits = normalize_index_limits(index_limits)
    trials_each_type, possible_params, labels = get_trial_groups(behavior_data)
    return spike_times_from_onset, index_limits, bins_start, trials_each_type, possible_params, labels


def trial_spike_times(spike_times_from_onset: np.ndarray, index_limits: np.ndarray, trial_number: int) -> np.ndarray:
    """Return the event-locked spike times for one trial."""
    return spike_times_from_onset[index_limits[trial_number, 0] : index_limits[trial_number, 1]]


def select_example_condition(
    spike_times_from_onset: np.ndarray,
    index_limits: np.ndarray,
    trials_each_type: list[np.ndarray],
) -> int:
    """Pick the condition with the strongest onset response."""
    onset_rates = []
    n_trials = index_limits.shape[0]
    for trial_indices in trials_each_type:
        condition_trial_numbers = trial_indices_from_selector(trial_indices, n_trials)
        if len(condition_trial_numbers) == 0:
            onset_rates.append(-np.inf)
            continue
        condition_rates = [
            window_rate(trial_spike_times(spike_times_from_onset, index_limits, trial_number), 0.0, 0.2)
            for trial_number in condition_trial_numbers
        ]
        onset_rates.append(float(np.mean(condition_rates)))
    return int(np.argmax(onset_rates))


def window_rate(spike_times: np.ndarray, start_time: float, stop_time: float) -> float:
    """Return firing rate in spikes/s for one analysis window."""
    return float(np.count_nonzero((spike_times >= start_time) & (spike_times < stop_time)) / (stop_time - start_time))


def add_window_shading(ax) -> None:
    """Draw shaded response windows on a PSTH axis."""
    window_edges = PLOT_WINDOW["markers"]
    colors = ("#e9c46a", "#2a9d8f", "#e76f51")
    for color, start_time, stop_time in zip(colors, window_edges[:-1], window_edges[1:]):
        ax.axvspan(start_time, stop_time, color=color, alpha=0.18, linewidth=0)
    for marker in window_edges:
        ax.axvline(marker, color="black", linestyle="--", linewidth=1)


def plot_single_condition_raster(ax: plt.Axes, selected_spike_times: list[np.ndarray]) -> None:
    """Plot a simple raster for one selected stimulus condition."""
    for trial_index, spikes in enumerate(selected_spike_times, start=1):
        if len(spikes):
            ax.vlines(spikes, trial_index - 0.4, trial_index + 0.4, color=plt.cm.viridis(0.7), linewidth=0.8)
    ax.set_ylim(0.5, max(len(selected_spike_times) + 0.5, 1.5))


def psth_from_trials(selected_spike_times: list[np.ndarray], bins_start: np.ndarray, bin_width: float) -> np.ndarray:
    """Return the mean PSTH across selected trials."""
    if not selected_spike_times:
        return np.zeros_like(bins_start, dtype=float)
    bin_edges = np.append(bins_start, bins_start[-1] + bin_width)
    binned = [np.histogram(spikes, bins=bin_edges)[0] for spikes in selected_spike_times]
    return np.mean(np.vstack(binned), axis=0) / bin_width


def find_best_cell_index(session_df) -> int | None:
    """Return the most onset-responsive example cell in the session."""
    best_index = None
    best_score = -np.inf
    n_candidates = min(len(session_df), MAX_CELLS_TO_SCORE)
    for cell_index in range(n_candidates):
        one_cell = ephyscore.Cell(session_df.iloc[cell_index])
        try:
            spike_times_from_onset, index_limits, _, trials_each_type, _, _ = load_plot_data(one_cell)
            condition_index = select_example_condition(spike_times_from_onset, index_limits, trials_each_type)
            trial_indices = trial_indices_from_selector(trials_each_type[condition_index], index_limits.shape[0])
            if len(trial_indices) == 0:
                continue
            condition_rates = [
                window_rate(trial_spike_times(spike_times_from_onset, index_limits, trial_number), 0.0, 0.2)
                for trial_number in trial_indices
            ]
            score = float(np.mean(condition_rates))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_index = cell_index
    return best_index


def plot_window_bar_chart(ax: plt.Axes, firing_rates: list[float]) -> None:
    """Plot a simple bar chart for onset, sustained, and offset firing rates."""
    ax.bar(WINDOW_LABELS, firing_rates, color=["#e9c46a", "#2a9d8f", "#e76f51"], edgecolor="black", linewidth=1)
    ax.set_ylabel("Firing rate (spk/s)")
    ax.set_title("Windowed firing rate", fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)


def main() -> None:
    """Create one single-neuron explanatory firing-rate figure."""
    session_df = load_session_dataframe()
    best_cell_index = find_best_cell_index(session_df) if not session_df.empty else None
    if best_cell_index is None:
        return

    print(f"Building single-neuron firing-rate methods figure for {SUBJECT} {SESSION_DATE}...")
    for _ in tqdm(range(1), desc="Methods spike-rate figure", unit="figure", dynamic_ncols=True):
        one_cell = ephyscore.Cell(session_df.iloc[best_cell_index])
        spike_times_from_onset, index_limits, bins_start, trials_each_type, possible_params, labels = load_plot_data(one_cell)
        condition_index = select_example_condition(spike_times_from_onset, index_limits, trials_each_type)
        trial_indices = trial_indices_from_selector(trials_each_type[condition_index], index_limits.shape[0])
        if len(trial_indices) == 0:
            return
        condition_label = labels[condition_index]

        selected_spike_times = [trial_spike_times(spike_times_from_onset, index_limits, trial_number) for trial_number in trial_indices]
        if not selected_spike_times:
            return
        psth = psth_from_trials(selected_spike_times, bins_start, BIN_WIDTH)
        exemplar_trial = selected_spike_times[0]
        firing_rates = [
            window_rate(exemplar_trial, 0.0, 0.2),
            window_rate(exemplar_trial, 0.2, 0.5),
            window_rate(exemplar_trial, 0.5, 0.7),
        ]

        fig = plt.figure(figsize=(11, 6.5), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=[2.4, 1.0], height_ratios=[1.0, 1.15])
        raster_ax = fig.add_subplot(grid[0, 0])
        psth_ax = fig.add_subplot(grid[1, 0], sharex=raster_ax)
        bar_ax = fig.add_subplot(grid[:, 1])
        fig.suptitle(f"Example single-neuron firing rate ({SUBJECT}, {SESSION_DATE}, {condition_label})", fontsize=15, fontweight="bold")

        plot_single_condition_raster(raster_ax, selected_spike_times)
        raster_ax.set_xlim(PLOT_WINDOW["time_range"][0], PLOT_WINDOW["time_range"][1])
        raster_ax.set_title("Single-stimulus raster", fontweight="bold")
        raster_ax.set_xlabel("")
        raster_ax.set_ylabel("Trial")
        add_window_shading(raster_ax)

        psth_ax.plot(bins_start, psth, color="black", linewidth=2)
        add_window_shading(psth_ax)
        psth_ax.set_title("PSTH with spike windows", fontweight="bold")
        psth_ax.set_xlabel("Time from stimulus onset (s)")
        psth_ax.set_ylabel("Firing rate (spk/s)")
        psth_ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

        plot_window_bar_chart(bar_ax, firing_rates)
        bar_ax.text(
            0.02,
            0.98,
            "Bars show spikes counted\nwithin each analysis window\nfor one example trial.",
            transform=bar_ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

        fig.savefig(get_output_dir() / f"{SUBJECT}_{SESSION_DATE}_cell_{best_cell_index}_{PLOT_TYPE}_single_neuron_firing_rate.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
