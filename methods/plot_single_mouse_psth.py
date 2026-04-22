"""Plot one clean raster-plus-PSTH example figure for each thesis sound type."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from jaratoolbox import celldatabase, ephyscore, spikesanalysis

from shared import funcs, params

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

FONTSIZE_SUPTITLE = 26
FONTSIZE_TITLE = 22
FONTSIZE_LABEL = 20
FONTSIZE_TICK = 18
FONTSIZE_LEGEND = 20

N_CELLS_TO_PLOT = 1
MAX_CELLS_TO_SCORE = 128
BIN_WIDTH = 0.01
LABEL_SUBSAMPLE_LIMIT = 8
PLOT_CONDITION_LIMITS = {
    "pureTones": 6,
    "AM": 6,
    "naturalSound": 8,
}
MIN_TOTAL_EVOKED_SPIKES = {
    "AM": 100,
    "naturalSound": 100,
}
EXAMPLE_FIGURES = (
    {
        "figure_key": "speechCombined",
        "title": "Speech syllables",
        "subject": "feat005",
        "session_date": "2022-02-07",
        "pdepth": 3020,
        "database_path": os.path.join(params.DATABASE_PATH, params.SPEECH_STUDY_NAME, "celldb_feat005.h5"),
        "plot_types": ("speechTuples",),
    },
    {
        "figure_key": "pureTones",
        "title": "Pure tones",
        "subject": "feat014",
        "session_date": "2024-03-18",
        "pdepth": None,
        "database_path": params.fullPath,
        "plot_types": ("pureTones",),
    },
    {
        "figure_key": "AM",
        "title": "AM white noise",
        "subject": "feat014",
        "session_date": "2024-03-18",
        "pdepth": None,
        "database_path": params.fullPath,
        "plot_types": ("AM",),
    },
    {
        "figure_key": "naturalSound",
        "title": "Natural sounds",
        "subject": "feat014",
        "session_date": "2024-03-18",
        "pdepth": None,
        "database_path": params.fullPath,
        "plot_types": ("naturalSound",),
    },
)


def window_plot_config(plot_type: str) -> dict[str, np.ndarray | tuple[float, ...]]:
    """Return plotting limits plus onset/sustained/offset markers for one sound type."""
    if plot_type in {"FT", "VOT", "speechTuples"}:
        periods = params.speech_allPeriods[1:]
        return {"time_range": np.array([-0.5, 1.0]), "windows": periods}
    if plot_type == "pureTones":
        periods = params.PT_allPeriods[1:]
        return {"time_range": np.array([-0.1, 0.4]), "windows": periods}
    if plot_type == "AM":
        periods = params.AM_allPeriods[1:]
        return {"time_range": np.array([-0.5, 1.0]), "windows": periods}
    periods = params.natSounds_allPeriods[1:]
    return {"time_range": np.array([-2.0, 6.0]), "windows": periods}


def get_output_dir() -> Path:
    """Return the PSTH figure output directory."""
    return funcs.get_nested_figure_dir("methods", "psth")


def load_session_dataframe(config: dict[str, object]):
    """Load one example recording session."""
    mouse_df = celldatabase.load_hdf(config["database_path"])
    print(
        f"[{config['figure_key']}] loaded database {config['database_path']} "
        f"with {len(mouse_df)} rows before filtering"
    )
    if "subject" in mouse_df.columns:
        mouse_df = mouse_df[mouse_df.subject == config["subject"]]
        print(f"[{config['figure_key']}] rows after subject filter ({config['subject']}): {len(mouse_df)}")
    mouse_df = mouse_df[mouse_df.date == config["session_date"]]
    print(f"[{config['figure_key']}] rows after date filter ({config['session_date']}): {len(mouse_df)}")
    if config["pdepth"] is not None:
        mouse_df = mouse_df[mouse_df.pdepth == config["pdepth"]]
        print(f"[{config['figure_key']}] rows after pdepth filter ({config['pdepth']}): {len(mouse_df)}")
    elif not mouse_df.empty:
        chosen_depth = np.sort(mouse_df.pdepth.unique())[0]
        mouse_df = mouse_df[mouse_df.pdepth == chosen_depth]
        print(f"[{config['figure_key']}] no pdepth provided, using first available depth {chosen_depth}: {len(mouse_df)} rows")
    else:
        print(f"[{config['figure_key']}] no rows remain after subject/date filtering")
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

    if plot_type == "speechTuples":
        speech_labels = np.column_stack((behavior_data["targetFTpercent"], behavior_data["targetVOTpercent"])).astype(int)
        keep_indices, sorted_labels = funcs.select_speech_trials(
            speech_labels,
            max_repeats=params.SPEECH_REPEATS_PER_TOKEN,
        )
        selectors = []
        labels = []
        for label in params.unique_labels:
            mask = np.all(sorted_labels == np.asarray(label), axis=1)
            selectors.append(keep_indices[mask])
            labels.append(funcs.format_speech_label(label))
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
    if plot_type == "pureTones":
        labels = [f"{value / 1000:.0f} kHz" for value in possible_params]
    else:
        labels = [f"{int(value)} Hz" for value in possible_params]
    return selectors, labels


def load_plot_data(one_cell, plot_type: str):
    """Load spike-aligned data for one cell and plot type."""
    session_type = "FTVOTBorders" if plot_type in {"FT", "VOT", "speechTuples"} else plot_type
    ephys_data, behavior_data = one_cell.load(sessiontype=session_type, behavClass=None)
    plot_config = window_plot_config(plot_type)
    bins_start = np.arange(plot_config["time_range"][0], plot_config["time_range"][1], BIN_WIDTH)
    spike_times_from_onset, _, index_limits = spikesanalysis.eventlocked_spiketimes(
        ephys_data["spikeTimes"],
        ephys_data["events"]["stimOn"],
        plot_config["time_range"],
        spikeindex=False,
    )
    index_limits = funcs.normalize_index_limits(index_limits)
    trials_each_type, labels = get_trial_groups(plot_type, behavior_data)
    nonempty_conditions = sum(len(funcs.trial_indices_from_selector(selector, index_limits.shape[0])) > 0 for selector in trials_each_type)
    print(
        f"[{plot_type}] loaded {len(labels)} conditions, "
        f"{nonempty_conditions} non-empty, {index_limits.shape[0]} total trials"
    )
    return spike_times_from_onset, index_limits, bins_start, plot_config, trials_each_type, labels


def maybe_subset_conditions(plot_type: str, trials_each_type, labels):
    """Reduce very large condition sets to a representative subset for cleaner example plots."""
    limit = PLOT_CONDITION_LIMITS.get(plot_type)
    if limit is None or len(labels) <= limit:
        return trials_each_type, labels
    keep = np.unique(np.linspace(0, len(labels) - 1, limit, dtype=int))
    return [trials_each_type[index] for index in keep], [labels[index] for index in keep]


def condition_colors(n_conditions: int) -> np.ndarray:
    """Return one viridis color per stimulus condition."""
    return plt.cm.viridis(np.linspace(0.08, 0.95, max(n_conditions, 2)))[:n_conditions]


def raster_ylabel(plot_type: str) -> str:
    """Return the label used on the raster y-axis."""
    return {
        "pureTones": "Frequency",
        "AM": "AM rate",
        "FT": "FT (%)",
        "VOT": "VOT (%)",
        "speechTuples": "(FT, VOT)",
        "naturalSound": "Stimulus",
    }.get(plot_type, "Condition")


def add_window_shading(ax, plot_type: str) -> None:
    """Draw onset/sustained/offset windows as subtle shaded regions."""
    windows = window_plot_config(plot_type)["windows"]
    colors = ("#e9c46a", "#2a9d8f", "#e76f51")
    for (start, stop), color in zip(windows, colors):
        ax.axvspan(start, stop, color=color, alpha=0.08, zorder=0)

def add_figure_window_legend(fig, legend_y: float = 0.96) -> None:
    """Add one readable figure-level legend for onset/sustained/offset windows."""
    handles = [
        Patch(facecolor="#e9c46a", edgecolor="#e9c46a", alpha=0.22, label="Onset"),
        Patch(facecolor="#2a9d8f", edgecolor="#2a9d8f", alpha=0.22, label="Sustained"),
        Patch(facecolor="#e76f51", edgecolor="#e76f51", alpha=0.22, label="Offset"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        facecolor="white",
        fontsize=FONTSIZE_LEGEND,
        ncol=3,
        handlelength=1.5,
        columnspacing=1.4,
    )


def display_tick_mask(n_labels: int, plot_type: str) -> list[bool]:
    """Return which raster y tick labels should be shown for readability."""
    if plot_type == "speechTuples":
        return [True] * n_labels
    if n_labels <= LABEL_SUBSAMPLE_LIMIT:
        return [True] * n_labels
    keep = np.unique(np.linspace(0, n_labels - 1, LABEL_SUBSAMPLE_LIMIT, dtype=int))
    return [index in keep for index in range(n_labels)]


def plot_colored_raster(ax, spike_times_from_onset, index_limits, trials_each_type, labels, time_range, plot_type: str) -> None:
    """Plot one dot raster with grouped colored sidebars for stimulus conditions."""
    colors = condition_colors(len(labels))
    n_trials = index_limits.shape[0]
    sidebar_frac = 0.05
    y_position = 1
    y_ticks = []
    y_labels = []
    tick_mask = display_tick_mask(len(labels), plot_type)
    for condition_index, (selector, label) in enumerate(zip(trials_each_type, labels)):
        trial_numbers = funcs.trial_indices_from_selector(selector, n_trials)
        if len(trial_numbers) == 0:
            continue
        start_y = y_position
        for trial_number in trial_numbers:
            spikes = spike_times_from_onset[index_limits[trial_number, 0] : index_limits[trial_number, 1]]
            if len(spikes):
                ax.scatter(spikes, np.full(spikes.shape, y_position), s=10, color="black", linewidths=0)
            y_position += 1
        stop_y = y_position - 1
        y_ticks.append((start_y + stop_y) / 2)
        y_labels.append(label if tick_mask[condition_index] else "")
        ax.axhspan(start_y - 0.5, stop_y + 0.5, xmin=0.0, xmax=sidebar_frac, color=colors[condition_index], ec=None)
        ax.axhspan(start_y - 0.5, stop_y + 0.5, xmin=1.0 - sidebar_frac, xmax=1.0, color=colors[condition_index], ec=None)
    ax.set_xlim(time_range[0], time_range[1])
    ax.set_ylim(0.5, max(y_position - 0.5, 1.5))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=FONTSIZE_TICK)


def trial_spike_times(spike_times_from_onset: np.ndarray, index_limits: np.ndarray, trial_number: int) -> np.ndarray:
    """Return event-locked spike times for one trial."""
    return spike_times_from_onset[index_limits[trial_number, 0] : index_limits[trial_number, 1]]


def plot_colored_psth(ax, spike_times_from_onset: np.ndarray, index_limits: np.ndarray, bins_start: np.ndarray, trials_each_type, labels) -> None:
    """Plot one PSTH axis with one viridis line per stimulus condition."""
    colors = condition_colors(len(labels))
    n_trials = index_limits.shape[0]
    bin_edges = np.append(bins_start, bins_start[-1] + BIN_WIDTH)
    for condition_index, selector in enumerate(trials_each_type):
        trial_numbers = funcs.trial_indices_from_selector(selector, n_trials)
        if len(trial_numbers) == 0:
            continue
        trial_histograms = [
            np.histogram(trial_spike_times(spike_times_from_onset, index_limits, trial_number), bins=bin_edges)[0]
            for trial_number in trial_numbers
        ]
        mean_rate = np.mean(np.vstack(trial_histograms), axis=0) / BIN_WIDTH
        smooth_rate = np.convolve(mean_rate, np.ones(5) / 5, mode="same")
        ax.plot(
            bins_start,
            smooth_rate,
            color=colors[condition_index],
            linewidth=2.1 if len(labels) > 8 else 2.7,
            alpha=0.9,
        )


def condition_response_strength(
    spike_times_from_onset: np.ndarray,
    index_limits: np.ndarray,
    bins_start: np.ndarray,
    trials_each_type,
    response_stop: float,
    plot_type: str,
) -> float:
    """Return a score favoring reliable evoked responses with some condition structure."""
    n_trials = index_limits.shape[0]
    bin_edges = np.append(bins_start, bins_start[-1] + BIN_WIDTH)
    condition_scores = []
    evoked_means = []
    total_evoked_spikes = 0
    for selector in trials_each_type:
        trial_numbers = funcs.trial_indices_from_selector(selector, n_trials)
        if len(trial_numbers) == 0:
            continue
        selector_evoked_spikes = 0
        trial_histograms = [
            np.histogram(trial_spike_times(spike_times_from_onset, index_limits, trial_number), bins=bin_edges)[0]
            for trial_number in trial_numbers
        ]
        for trial_number in trial_numbers:
            spikes = trial_spike_times(spike_times_from_onset, index_limits, trial_number)
            selector_evoked_spikes += int(np.count_nonzero((spikes >= 0.0) & (spikes < response_stop)))
        mean_rate = np.mean(np.vstack(trial_histograms), axis=0) / BIN_WIDTH
        baseline_mask = bins_start < 0.0
        evoked_mask = (bins_start >= 0.0) & (bins_start < response_stop)
        if not np.any(evoked_mask):
            continue
        baseline_mean = float(np.mean(mean_rate[baseline_mask])) if np.any(baseline_mask) else 0.0
        baseline_std = float(np.std(mean_rate[baseline_mask])) if np.any(baseline_mask) else 0.0
        peak_evoked = float(np.max(mean_rate[evoked_mask]))
        mean_evoked = float(np.mean(mean_rate[evoked_mask]))
        snr_like = (peak_evoked - baseline_mean) / max(baseline_std, 1.0)
        condition_scores.append(snr_like)
        evoked_means.append(mean_evoked)
        total_evoked_spikes += selector_evoked_spikes
    if not condition_scores:
        return -np.inf
    min_required_spikes = MIN_TOTAL_EVOKED_SPIKES.get(plot_type, 0)
    if total_evoked_spikes < min_required_spikes:
        return -np.inf
    tuning_spread = float(np.std(evoked_means)) if len(evoked_means) > 1 else 0.0
    mean_evoked = float(np.mean(evoked_means))
    return max(condition_scores) + 0.2 * tuning_spread + 0.12 * mean_evoked + 0.01 * total_evoked_spikes


def find_best_cell_indices(session_df, config: dict[str, object]) -> list[int]:
    """Return the most stimulus-responsive example cells for one session."""
    scored_cells = []
    failed_cells: list[str] = []
    n_candidates = min(len(session_df), MAX_CELLS_TO_SCORE)
    print(f"[{config['figure_key']}] scoring up to {n_candidates} candidate cells")
    for cell_index in range(n_candidates):
        one_cell = ephyscore.Cell(session_df.iloc[cell_index])
        total_score = 0.0
        valid_plot = False
        for plot_type in config["plot_types"]:
            try:
                spike_times_from_onset, index_limits, bins_start, plot_config, trials_each_type, _ = load_plot_data(one_cell, plot_type)
            except Exception as exc:
                failed_cells.append(f"cell {cell_index} {plot_type}: {type(exc).__name__}: {exc}")
                continue
            total_score += condition_response_strength(
                spike_times_from_onset,
                index_limits,
                bins_start,
                trials_each_type,
                response_stop=float(plot_config["windows"][-1][1]),
                plot_type=plot_type,
            )
            valid_plot = True
        if valid_plot:
            scored_cells.append((total_score, cell_index))
            print(f"[{config['figure_key']}] cell {cell_index} score = {total_score:.3f}")
        else:
            print(f"[{config['figure_key']}] cell {cell_index} had no valid plot types")
    scored_cells.sort(reverse=True)
    if failed_cells:
        print(f"[{config['figure_key']}] sample load failures:")
        for message in failed_cells[:8]:
            print(f"  {message}")
    print(f"[{config['figure_key']}] scored {len(scored_cells)} usable cells")
    return [cell_index for _, cell_index in scored_cells[:N_CELLS_TO_PLOT]]


def create_figure_axes(plot_types: tuple[str, ...]) -> tuple[plt.Figure, np.ndarray]:
    """Return a clean axes layout for one sound-specific PSTH figure."""
    if len(plot_types) == 1:
        return plt.subplots(1, 2, figsize=(10.8, 6.0), constrained_layout=True, squeeze=False)
    return plt.subplots(2, 2, figsize=(11.2, 8.6), constrained_layout=True, squeeze=False)


_COMBINED_PANEL_ORDER = ["pureTones", "AM", "naturalSound", "speechCombined"]
_COMBINED_PLOT_TYPES = {
    "pureTones": "pureTones",
    "AM": "AM",
    "naturalSound": "naturalSound",
    "speechCombined": "speechTuples",
}
_COMBINED_PANEL_LABELS = ["A", "B", "C", "D"]
# For each panel, pick this rank from the sorted candidate list (0 = best scorer).
# Speech uses rank 1 to avoid the most hyper-reactive cell.
_COMBINED_CELL_RANK = {
    "pureTones": 0,
    "AM": 0,
    "naturalSound": 0,
    "speechCombined": 1,
}


def build_combined_psth_figure() -> None:
    """Create a combined 2×2 panel figure (PT/AM top row, NS/Speech bottom row).

    Each panel occupies two columns of the underlying 2×4 axes grid:
    col 0-1 = left panel (raster, PSTH), col 2-3 = right panel (raster, PSTH).
    """
    panel_configs = {c["figure_key"]: c for c in EXAMPLE_FIGURES if c["figure_key"] in _COMBINED_PANEL_ORDER}
    ordered_configs = [panel_configs[key] for key in _COMBINED_PANEL_ORDER if key in panel_configs]

    # 2 grid rows × 4 cols: each sound type owns one (raster, PSTH) column pair
    # Extra figure height (13.5 vs 11.6) gives headroom at the top for the
    # suptitle + legend without cramping the subplots.
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(22.0, 13.5),
        squeeze=False,
    )
    # Reserve the top 14 % of the figure for suptitle + legend; subplots fill the rest.
    fig.subplots_adjust(top=0.86, bottom=0.05, left=0.05, right=0.98, hspace=0.45, wspace=0.35)
    fig.suptitle(
        "Example auditory cortex single-neuron responses",
        fontsize=FONTSIZE_SUPTITLE,
        fontweight="bold",
        y=0.98,
    )
    add_figure_window_legend(fig, legend_y=0.95)

    for panel_index, config in enumerate(ordered_configs):
        plot_type = _COMBINED_PLOT_TYPES[config["figure_key"]]
        panel_label = _COMBINED_PANEL_LABELS[panel_index]
        grid_row = panel_index // 2
        raster_col = (panel_index % 2) * 2
        psth_col = raster_col + 1
        raster_ax = axes[grid_row, raster_col]
        psth_ax = axes[grid_row, psth_col]

        # Panel letter in upper-left corner of the raster axis
        raster_ax.text(
            -0.10, 1.04, panel_label,
            transform=raster_ax.transAxes,
            fontsize=FONTSIZE_SUPTITLE,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

        session_df = load_session_dataframe(config)
        if session_df.empty:
            raster_ax.axis("off")
            psth_ax.axis("off")
            continue

        best_cell_indices = find_best_cell_indices(session_df, config)
        if not best_cell_indices:
            raster_ax.axis("off")
            psth_ax.axis("off")
            continue

        desired_rank = _COMBINED_CELL_RANK.get(config["figure_key"], 0)
        cell_rank = min(desired_rank, len(best_cell_indices) - 1)
        one_cell = ephyscore.Cell(session_df.iloc[best_cell_indices[cell_rank]])

        try:
            (
                spike_times_from_onset,
                index_limits,
                bins_start,
                plot_config,
                trials_each_type,
                labels,
            ) = load_plot_data(one_cell, plot_type)
        except Exception as exc:
            print(f"[combined figure] skipping {config['figure_key']} {plot_type}: {type(exc).__name__}: {exc}")
            raster_ax.axis("off")
            psth_ax.axis("off")
            continue

        plot_trials_each_type, plot_labels = maybe_subset_conditions(plot_type, trials_each_type, labels)

        if plot_type == "speechTuples":
            raster_title = "Speech syllable raster"
            psth_title = "Speech syllable PSTH"
        else:
            raster_title = f"{config['title']} raster"
            psth_title = f"{config['title']} PSTH"

        plot_colored_raster(
            raster_ax,
            spike_times_from_onset,
            index_limits,
            plot_trials_each_type,
            plot_labels,
            plot_config["time_range"],
            plot_type,
        )
        raster_ax.set_title(raster_title, fontsize=FONTSIZE_TITLE, fontweight="bold")
        raster_ax.set_xlabel("Time from stimulus onset (s)", fontsize=FONTSIZE_LABEL)
        raster_ax.set_ylabel(raster_ylabel(plot_type), fontsize=FONTSIZE_LABEL)
        raster_ax.tick_params(labelsize=FONTSIZE_TICK)
        add_window_shading(raster_ax, plot_type)

        plot_colored_psth(psth_ax, spike_times_from_onset, index_limits, bins_start, plot_trials_each_type, plot_labels)
        psth_ax.set_title(psth_title, fontsize=FONTSIZE_TITLE, fontweight="bold")
        psth_ax.set_xlabel("Time from stimulus onset (s)", fontsize=FONTSIZE_LABEL)
        psth_ax.set_ylabel("Firing rate (spikes/s)", fontsize=FONTSIZE_LABEL)
        psth_ax.tick_params(labelsize=FONTSIZE_TICK)
        psth_ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
        add_window_shading(psth_ax, plot_type)

    fig.savefig(get_output_dir() / "combined_psth_figure2.png", dpi=250)
    print("[combined figure] saved combined_psth_figure2.png")
    plt.close(fig)


def main() -> None:
    """Run the example single-cell PSTH figure generation."""
    funcs.apply_figure_style()
    for config in EXAMPLE_FIGURES:
        print(
            f"Preparing {config['figure_key']} example "
            f"({config['subject']} {config['session_date']}, plot types={config['plot_types']})"
        )
        session_df = load_session_dataframe(config)
        if session_df.empty:
            print(f"[{config['figure_key']}] session dataframe is empty, skipping")
            continue
        best_cell_indices = find_best_cell_indices(session_df, config)
        if not best_cell_indices:
            print(f"[{config['figure_key']}] no valid example cells found, skipping")
            continue
        print(f"Building example PSTHs for {config['title']} ({config['subject']} {config['session_date']})...")
        for cell_index in tqdm(best_cell_indices, desc=f"Methods PSTHs ({config['figure_key']})", unit="cell", dynamic_ncols=True):
            one_cell = ephyscore.Cell(session_df.iloc[cell_index])
            fig, axes = create_figure_axes(config["plot_types"])
            fig.suptitle(
                f"{config['title']} example raster and PSTH (cell {cell_index})",
                fontsize=FONTSIZE_SUPTITLE,
                fontweight="bold",
            )
            add_figure_window_legend(fig)

            for plot_index, plot_type in enumerate(config["plot_types"]):
                row_index = plot_index if len(config["plot_types"]) > 1 else 0
                raster_ax = axes[row_index, 0]
                psth_ax = axes[row_index, 1]
                try:
                    spike_times_from_onset, index_limits, bins_start, plot_config, trials_each_type, labels = load_plot_data(one_cell, plot_type)
                except Exception as exc:
                    print(f"Skipping cell {cell_index} {plot_type}: {type(exc).__name__}: {exc}")
                    raster_ax.axis("off")
                    psth_ax.axis("off")
                    continue
                plot_trials_each_type, plot_labels = maybe_subset_conditions(plot_type, trials_each_type, labels)

                plot_colored_raster(
                    raster_ax,
                    spike_times_from_onset,
                    index_limits,
                    plot_trials_each_type,
                    plot_labels,
                    plot_config["time_range"],
                    plot_type,
                )
                raster_title = f"{plot_type} raster" if plot_type in {"FT", "VOT"} else f"{config['title']} raster"
                psth_title = f"{plot_type} PSTH" if plot_type in {"FT", "VOT"} else f"{config['title']} PSTH"
                if plot_type == "speechTuples":
                    raster_title = "Speech tuple-order raster"
                    psth_title = "Speech tuple-order PSTH"
                raster_ax.set_title(raster_title, fontsize=FONTSIZE_TITLE, fontweight="bold")
                raster_ax.set_xlabel("Time from stimulus onset (s)", fontsize=FONTSIZE_LABEL)
                raster_ax.set_ylabel(raster_ylabel(plot_type), fontsize=FONTSIZE_LABEL)
                raster_ax.tick_params(labelsize=FONTSIZE_TICK)
                add_window_shading(raster_ax, plot_type)

                plot_colored_psth(psth_ax, spike_times_from_onset, index_limits, bins_start, plot_trials_each_type, plot_labels)
                psth_ax.set_title(psth_title, fontsize=FONTSIZE_TITLE, fontweight="bold")
                psth_ax.set_xlabel("Time from stimulus onset (s)", fontsize=FONTSIZE_LABEL)
                psth_ax.set_ylabel("Firing rate (spikes/s)", fontsize=FONTSIZE_LABEL)
                psth_ax.tick_params(labelsize=FONTSIZE_TICK)
                psth_ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
                add_window_shading(psth_ax, plot_type)

            fig.savefig(
                get_output_dir() / f"{config['subject']}_{config['session_date']}_{config['figure_key']}_cell_{cell_index}_psth.png",
                dpi=250,
            )
            print(
                f"[{config['figure_key']}] saved "
                f"{config['subject']}_{config['session_date']}_{config['figure_key']}_cell_{cell_index}_psth.png"
            )
            plt.close(fig)

    print("Building combined Figure 2 (A/B/C/D)...")
    build_combined_psth_figure()


if __name__ == "__main__":
    main()
