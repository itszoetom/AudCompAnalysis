"""Build firing-rate arrays from the speech and non-speech auditory datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from jaratoolbox import celldatabase, ephyscore, spikesanalysis
from tqdm import tqdm

import funcs
import params

WINDOW_NAMES = params.WINDOW_NAMES


def simplify_recording_sites(celldb: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first recording-site label for each cell."""
    celldb = celldb.copy()
    celldb["simpleSiteName"] = celldb["recordingSiteName"].str.split(",").str[0]
    return celldb


def mean_center_trials(firing_rates: np.ndarray) -> np.ndarray:
    """Subtract each neuron's across-trial mean firing rate."""
    return firing_rates - firing_rates.mean(axis=1, keepdims=True)


def save_npz(filename: Path, **arrays: np.ndarray) -> None:
    """Save one processed dataset and print a compact summary."""
    print(f"Saving {filename.name} to {filename}")
    for key in ("basefr", "onsetfr", "sustainedfr", "offsetfr", "brainRegionArray", "stimArray", "mouseIDArray", "sessionIDArray"):
        if key in arrays:
            print(f"  {key}: {np.shape(arrays[key])}")
    np.savez(filename, **arrays)
    print("Saved!")


def speech_trial_labels(bdata: dict) -> np.ndarray:
    """Return trial labels as a two-column `(FT, VOT)` array."""
    return np.column_stack((bdata["targetFTpercent"], bdata["targetVOTpercent"])).astype(int)


def load_speech_session(
    speech_db: pd.DataFrame, subject: str, date: str, brain_area: str
) -> tuple[ephyscore.CellEnsemble | None, dict | None, dict | None]:
    """Load one speech session for one mouse, date, and brain area."""
    session_db = speech_db[
        (speech_db.subject == subject)
        & (speech_db.date == date)
        & (speech_db.simpleSiteName == brain_area)
    ]
    if session_db.empty:
        print(f"No speech data for {subject} {date} {brain_area}")
        return None, None, None

    ensemble = ephyscore.CellEnsemble(session_db)
    try:
        ephys_data, bdata = ensemble.load("FTVOTBorders")
    except IndexError:
        print(f"No speech sound data for {subject} {date} {brain_area}")
        return None, None, None
    return ensemble, ephys_data, bdata


def speech_window_rates(ensemble: ephyscore.CellEnsemble, window: list[float]) -> np.ndarray:
    """Compute speech firing rates for one time window."""
    spike_counts = ensemble.spiketimes_to_spikecounts([window[0], window[1]])
    return spike_counts.sum(axis=2) / (window[1] - window[0])


def speech_analysis_window(window_name: str) -> list[float]:
    """Return the speech onset/sustained/offset window from the shared period definition."""
    period_index = WINDOW_NAMES.index(window_name) + 1
    return params.speech_allPeriods[period_index]


def build_speech_arrays() -> None:
    """Build and save session-concatenated firing-rate arrays for speech."""
    speech_db = simplify_recording_sites(celldatabase.load_hdf(params.fullPath_Speech))
    session_store = {
        window_name: {"X": [], "brain": [], "mouse": [], "session": []}
        for window_name in WINDOW_NAMES
    }
    reference_labels = None

    print("Loading speech data sessions...")
    for subject in params.SPEECH_SUBJECTS:
        for date in params.recordingDate_list[subject]:
            for brain_area in params.targetSiteNames:
                ensemble, ephys_data, bdata = load_speech_session(speech_db, subject, date, brain_area)
                if ensemble is None:
                    continue

                ensemble.eventlocked_spiketimes(ephys_data["events"]["stimOn"], params.speech_time_range)
                trial_indices, sorted_labels = funcs.select_speech_trials(
                    speech_trial_labels(bdata),
                    max_repeats=params.SPEECH_REPEATS_PER_TOKEN,
                )
                if reference_labels is None:
                    reference_labels = sorted_labels
                elif not np.array_equal(reference_labels, sorted_labels):
                    raise ValueError(f"Speech trial labels do not match reference order for {subject} {date} {brain_area}.")

                session_id = date
                print(f"Processing speech session {session_id}")

                for window_name in WINDOW_NAMES:
                    window = speech_analysis_window(window_name)
                    firing_rates = mean_center_trials(speech_window_rates(ensemble, window))[:, trial_indices]
                    n_neurons = firing_rates.shape[0]
                    session_store[window_name]["X"].append(firing_rates)
                    session_store[window_name]["brain"].extend([brain_area] * n_neurons)
                    session_store[window_name]["mouse"].extend([subject] * n_neurons)
                    session_store[window_name]["session"].extend([session_id] * n_neurons)
                    print(f"  {window_name}: {firing_rates.shape}")

    if reference_labels is None:
        raise RuntimeError("No speech sessions were loaded.")

    onset_data = np.concatenate(session_store["onset"]["X"], axis=0)
    save_npz(
        Path(params.dbSavePath) / "fr_arrays_speech.npz",
        onsetfr=onset_data,
        sustainedfr=np.concatenate(session_store["sustained"]["X"], axis=0),
        offsetfr=np.concatenate(session_store["offset"]["X"], axis=0),
        brainRegionArray=np.asarray(session_store["onset"]["brain"], dtype=object),
        stimArray=funcs.stimulus_display_labels("speech", reference_labels),
        stimNumericArray=reference_labels,
        mouseIDArray=np.asarray(session_store["onset"]["mouse"], dtype=object),
        sessionIDArray=np.asarray(session_store["onset"]["session"], dtype=object),
    )


def align_trial_counts(current_stim: np.ndarray, event_onsets: np.ndarray, row_index: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Match behavior and ephys trials, allowing the known off-by-one case."""
    if len(current_stim) > len(event_onsets) or len(current_stim) < len(event_onsets) - 1:
        print(f"[{row_index}] Warning: BehavTrials ({len(current_stim)}) and EphysTrials ({len(event_onsets)})")
        return None, None
    if len(current_stim) == len(event_onsets) - 1:
        event_onsets = event_onsets[: len(current_stim)]
    return current_stim, event_onsets


def eventlocked_period_rates(
    spike_times: np.ndarray,
    event_onsets: np.ndarray,
    time_range: list[float],
    periods: list[list[float]],
) -> list[np.ndarray]:
    """Compute one firing-rate array per period for a single cell."""
    spike_times_from_event, _, index_limits = spikesanalysis.eventlocked_spiketimes(
        spike_times,
        event_onsets,
        time_range,
    )
    period_durations = [stop - start for start, stop in periods]
    rates = []
    for period, duration in zip(periods, period_durations):
        spike_count_mat = spikesanalysis.spiketimes_to_spikecounts(
            spike_times_from_event,
            index_limits,
            period,
        )
        rates.append(spike_count_mat[:, 0] / duration)
    return rates


def initialize_non_speech_arrays(n_cells: int, n_trials: int) -> dict[str, np.ndarray]:
    """Allocate output arrays for one non-speech stimulus set."""
    return {
        "basefr": np.full((n_cells, n_trials), np.nan),
        "onsetfr": np.full((n_cells, n_trials), np.nan),
        "sustainedfr": np.full((n_cells, n_trials), np.nan),
        "offsetfr": np.full((n_cells, n_trials), np.nan),
        "stimArray": np.full((n_cells, n_trials), np.nan),
        "brainRegionArray": np.empty(n_cells, dtype=object),
        "mouseIDArray": np.empty(n_cells, dtype=object),
        "sessionIDArray": np.empty(n_cells, dtype=object),
    }


def build_non_speech_arrays(celldb: pd.DataFrame, stim_type: str) -> dict[str, np.ndarray]:
    """Build one non-speech firing-rate array set."""
    config = params.STIMULUS_BUILD_CONFIGS[stim_type]
    arrays = initialize_non_speech_arrays(len(celldb), config["n_trials"])

    for cell_index, (row_index, db_row) in enumerate(
        tqdm(celldb.iterrows(), total=len(celldb), desc=f"Calculating firing rates for {stim_type}")
    ):
        one_cell = ephyscore.Cell(db_row)
        ephys_data, bdata = one_cell.load(stim_type)

        current_stim = np.asarray(bdata[config["stim_var"]][: config["n_trials"]])
        event_onsets = np.asarray(ephys_data["events"]["stimOn"][: config["n_trials"]])
        current_stim, event_onsets = align_trial_counts(current_stim, event_onsets, row_index)
        if current_stim is None:
            continue

        firing_rates = eventlocked_period_rates(
            ephys_data["spikeTimes"],
            event_onsets,
            config["time_range"],
            config["periods"],
        )
        sort_order = np.argsort(current_stim)
        n_valid_trials = len(sort_order)
        for key, rate_values in zip(("basefr", "onsetfr", "sustainedfr", "offsetfr"), firing_rates):
            arrays[key][cell_index, :n_valid_trials] = rate_values[sort_order]
        arrays["stimArray"][cell_index, :n_valid_trials] = current_stim[sort_order]
        arrays["brainRegionArray"][cell_index] = db_row["simpleSiteName"]
        arrays["mouseIDArray"][cell_index] = db_row["subject"]
        arrays["sessionIDArray"][cell_index] = db_row["date"]

    return arrays


def build_all_non_speech_arrays() -> None:
    """Build and save firing-rate arrays for natural sounds, AM, and pure tones."""
    celldb = simplify_recording_sites(celldatabase.load_hdf(params.fullPath))
    celldb = celldb[celldb["simpleSiteName"].isin(params.targetSiteNames)].reset_index()

    for stim_type in ("naturalSound", "AM", "pureTones"):
        arrays = build_non_speech_arrays(celldb, stim_type)
        arrays_to_save = dict(arrays)
        arrays_to_save["stimNumericArray"] = arrays["stimArray"][0].copy()
        if stim_type == "naturalSound":
            arrays_to_save["stimArray"] = funcs.stimulus_display_labels("naturalSound", arrays["stimArray"][0])
        save_npz(Path(params.dbSavePath) / f"fr_arrays_{stim_type}.npz", **arrays_to_save)


def main() -> None:
    """Build every saved firing-rate array used by the analysis code."""
    build_speech_arrays()
    build_all_non_speech_arrays()


if __name__ == "__main__":
    main()
