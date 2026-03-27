"""Load the raw HDF datasets, compute firing-rate arrays, and save unified `.npz` files."""

import os
import params
import numpy as np
from copy import deepcopy
from jaratoolbox import celldatabase, ephyscore, spikesanalysis
import pandas as pd
from tqdm import tqdm

WINDOW_NAMES = ('onset', 'sustained', 'offset')

data = {}
for window_name in WINDOW_NAMES:
    data[window_name] = {
        'X': [],
        'Y_brain': [],
        'Y_freq': [],
        'mouseID': [],
        'sessionID': []
    }

# Initialize dictionaries for population-level arrays
X_all = {'onset': [], 'sustained': [], 'offset': []}
Y_brain_all = []
Y_freq = None
indices = None
previous_freq = None

def load_speech_data(subject, date, targetSiteName):
    """Load one speech session for a subject, date, and brain area."""
    fullDb = celldatabase.load_hdf(params.fullPath_Speech)
    fullDb["recordingSiteName"] = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])
    celldb = fullDb[(fullDb.subject == subject)]
    celldbSubset = celldb[(celldb.date == date)]
    celldbSubset = celldbSubset[(celldbSubset.recordingSiteName == targetSiteName)]

    if celldbSubset.empty:
        print(f"No data in {targetSiteName} on {date} for Speech")
        return None, None, None

    ensemble = ephyscore.CellEnsemble(celldbSubset)
    try:
        ephysData, bdata = ensemble.load("FTVOTBorders")
    except IndexError:
        print(f"No sound data for {targetSiteName} on {date} for {subject}")
        return None, None, None
    return ensemble, ephysData, bdata


def calculate_speech_firing_rate(window, ensemble, bdata):
    """Return speech firing rates and FT/VOT labels for one time window."""
    binEdges = [window[0], window[1]]
    spikeCounts = ensemble.spiketimes_to_spikecounts(binEdges)
    sumEvokedFR = spikeCounts.sum(axis=2)
    spikesPerSecEvoked = sumEvokedFR / (window[1] - window[0])

    FTParamsEachTrial = bdata['targetFTpercent']
    VOTParamsEachTrial = bdata['targetVOTpercent']
    nTrials = len(bdata['targetFTpercent'])
    Y_frequency = np.array([(FTParamsEachTrial[i], VOTParamsEachTrial[i]) for i in range(nTrials)])

    return spikesPerSecEvoked, Y_frequency


def normalize_speech_firing_rate(spikesPerSecEvoked, targetSiteName):
    """Mean-center trials and attach one brain-area label per neuron."""
    trialMeans = spikesPerSecEvoked.mean(axis=1)
    spikesPerSecEvokedNormalized = spikesPerSecEvoked.T - trialMeans
    spikesPerSecEvokedNormalized = spikesPerSecEvokedNormalized.T
    Y_brain_area_array = [targetSiteName] * spikesPerSecEvokedNormalized.shape[0]

    return spikesPerSecEvokedNormalized, Y_brain_area_array


def clean_speech_data(X, Y_freq, n=20):
    """Keep the first `n` repetitions per speech token and sort trials consistently."""
    Y_freq_array = Y_freq[0] if isinstance(Y_freq, list) else Y_freq

    df = pd.DataFrame(Y_freq_array)
    valid_indices = df.groupby(list(df.columns)).head(n).index.tolist()

    X_filtered = X[:, valid_indices]
    Y_filtered = Y_freq_array[valid_indices]

    sort_indices = np.lexsort((Y_filtered[:, 1], Y_filtered[:, 0]))

    return X_filtered[:, sort_indices], Y_filtered[sort_indices], deepcopy(Y_filtered[sort_indices]), sort_indices


def sort_speech_arrays(X_list, indices):
    """Apply one shared trial order to each speech session array."""
    sorted_x_list = []
    for x in X_list:
        sorted_x = x[:, indices]
        sorted_x_list.append(sorted_x)
    return sorted_x_list


def save_speech_data():
    """Save the processed speech arrays to disk."""
    onset_X = []
    sustained_X = []
    offset_X = []
    brain_regions = []
    stim_array = []
    mouseIDs = []
    sessionIDs = []

    for window_name in WINDOW_NAMES:
        X_array = np.concatenate(data[window_name]['X'], axis=0)

        if window_name == 'onset':
            onset_X = X_array
            brain_regions = np.array(data[window_name]['Y_brain'])
            stim_array = data[window_name]['Y_freq'][0][:, :2]
            mouseIDs = np.array(data[window_name]['mouseID'])
            sessionIDs = np.array(data[window_name]['sessionID'])
        elif window_name == 'sustained':
            sustained_X = X_array
        elif window_name == 'offset':
            offset_X = X_array

    fr_arrays_filename = os.path.join(params.dbSavePath, f'fr_arrays_speech.npz')
    print(f"Saving speech data to {fr_arrays_filename}")
    print(f"  Onset shape: {onset_X.shape}")
    print(f"  Sustained shape: {sustained_X.shape}")
    print(f"  Offset shape: {offset_X.shape}")
    print(f"  Brain regions: {brain_regions.shape}")
    print(f"  Stim: {stim_array.shape}")
    print(f"  Unique mice: {np.unique(mouseIDs)}")
    print(f"  Unique sessions: {len(np.unique(sessionIDs))}")

    np.savez(
        fr_arrays_filename,
        onsetfr=onset_X,
        sustainedfr=sustained_X,
        offsetfr=offset_X,
        brainRegionArray=brain_regions,
        stimArray=stim_array,
        mouseIDArray=mouseIDs,
        sessionIDArray=sessionIDs,
    )
    print(f"Saved speech!")


def calculate_fr_arrays(celldb: pd.DataFrame, stimType: str, stimVar: str, timeRange: list, allPeriods: list) -> list:
    """Compute firing-rate arrays for one pre-aligned non-speech stimulus set."""
    n_cells = len(celldb)
    period_duration = [window[1] - window[0] for window in allPeriods]

    if stimType == 'AM':
        nTrials = 220
    elif stimType == 'naturalSound':
        nTrials = 200
    elif stimType == 'pureTones':
        nTrials = 320
    else:
        raise ValueError(f"Unrecognized stimulus type: {stimType}. Should be in ['AM', 'naturalSound', 'pureTones']")

    basefr = np.full((n_cells, nTrials), np.nan)
    onsetfr = np.full((n_cells, nTrials), np.nan)
    sustainedfr = np.full((n_cells, nTrials), np.nan)
    offsetfr = np.full((n_cells, nTrials), np.nan)
    stimArray = np.full((n_cells, nTrials), np.nan)
    brainRegion = np.empty(n_cells, object)
    mouseID = np.empty(n_cells, object)
    sessionID = np.empty(n_cells, object)

    for indCell, (indRow, dbRow) in enumerate(
        tqdm(celldb.iterrows(), total=len(celldb), desc=f"Calculating firing rates for {stimType}")):
        oneCell = ephyscore.Cell(dbRow)
        ephysData, bdata = oneCell.load(stimType)

        spikeTimes = ephysData['spikeTimes']
        eventOnsetTimes = ephysData['events']['stimOn'][:nTrials]
        currentStim = bdata[stimVar][:nTrials]

        if (len(currentStim) > len(eventOnsetTimes)) or \
                (len(currentStim) < len(eventOnsetTimes) - 1):
            print(f'[{indRow}] Warning! BevahTrials ({len(currentStim)}) and ' +
                  f'EphysTrials ({len(eventOnsetTimes)})')
            continue
        if len(currentStim) == len(eventOnsetTimes) - 1:
            eventOnsetTimes = eventOnsetTimes[:len(currentStim)]

        (spikeTimesFromEventOnset, trialIndexForEachSpike, indexLimitsEachTrial) = \
            spikesanalysis.eventlocked_spiketimes(spikeTimes, eventOnsetTimes, timeRange)

        spikesEachTrialEachPeriod = []
        for period in allPeriods:
            spikeCountMat = spikesanalysis.spiketimes_to_spikecounts(spikeTimesFromEventOnset,
                                                                     indexLimitsEachTrial, period)
            spikesEachTrial = spikeCountMat[:, 0]
            spikesEachTrialEachPeriod.append(spikesEachTrial)

        firingRateBase = spikesEachTrialEachPeriod[0] / period_duration[0]
        firingRateOnset = spikesEachTrialEachPeriod[1] / period_duration[1]
        firingRateSustain = spikesEachTrialEachPeriod[2] / period_duration[2]
        firingRateOffset = spikesEachTrialEachPeriod[3] / period_duration[3]

        sorted_stim_ind = np.argsort(currentStim)
        sorted_stim_array = currentStim[sorted_stim_ind]
        sorted_fr_base = firingRateBase[sorted_stim_ind]
        sorted_fr_onset = firingRateOnset[sorted_stim_ind]
        sorted_fr_sustained = firingRateSustain[sorted_stim_ind]
        sorted_fr_offset = firingRateOffset[sorted_stim_ind]

        basefr[indCell, :] = sorted_fr_base
        onsetfr[indCell, :] = sorted_fr_onset
        sustainedfr[indCell, :] = sorted_fr_sustained
        offsetfr[indCell, :] = sorted_fr_offset
        stimArray[indCell, :] = sorted_stim_array
        brainRegion[indCell] = dbRow['simpleSiteName']
        mouseID[indCell] = dbRow['subject']
        sessionID[indCell] = dbRow['date']

    return [basefr, onsetfr, sustainedfr, offsetfr, stimArray, brainRegion, mouseID, sessionID]


def save_non_speech_arrays(stimType, fr_arrays):
    """Save one non-speech firing-rate array set to disk."""
    fr_arrays_filename = os.path.join(params.dbSavePath, f'fr_arrays_{stimType}.npz')

    print(f"Saving {stimType} data to {fr_arrays_filename}")
    print(f"  Onset shape: {fr_arrays[0].shape}")
    print(f"  Sustained shape: {fr_arrays[1].shape}")
    print(f"  Offset shape: {fr_arrays[2].shape}")
    print(f"  Brain regions: {fr_arrays[3].shape}")
    print(f"  Stim: {fr_arrays[4].shape}")
    print(f"  Unique mice: {np.unique(fr_arrays[5])}")
    print(f"  Unique sessions: {len(np.unique(fr_arrays[6]))}")

    np.savez(
        fr_arrays_filename,
        basefr=fr_arrays[0],
        onsetfr=fr_arrays[1],
        sustainedfr=fr_arrays[2],
        offsetfr=fr_arrays[3],
        stimArray=fr_arrays[4],
        brainRegionArray=fr_arrays[5],
        mouseIDArray=fr_arrays[6],
        sessionIDArray=fr_arrays[7],
    )
    print("Saved!")


# Main processing loop
print("Loading speech data sessions...")
for subject in params.SPEECH_SUBJECTS:
    print(f"\nProcessing subject: {subject}")
    for date in params.recordingDate_list[subject]:
        for targetSiteName in params.targetSiteNames:
            print(f"  Date: {date}, Brain area: {targetSiteName}")
            ensemble, ephys, bdata = load_speech_data(subject, date, targetSiteName)
            if ensemble is None:
                print(f"      No data for {subject}, {date}, {targetSiteName}, speech")
                continue

            session_data = {'onset': None, 'sustained': None, 'offset': None}
            eventOnsetTimes = ephys['events']['stimOn']
            ensemble.eventlocked_spiketimes(eventOnsetTimes, params.speech_time_range)

            speech_windows = {k: v for k, v in params.spike_windows.items() if k.startswith('speech')}

            for window_key, window in speech_windows.items():
                window_name = window_key.split(' - ')[1]  # now always 'onset', 'sustained', 'offset'

                print(f"      Window: {window_name}")
                spikesPerSecEvoked, Y_frequency = calculate_speech_firing_rate(window, ensemble, bdata)
                X, y_brain = normalize_speech_firing_rate(spikesPerSecEvoked, targetSiteName)
                X_clean, Y_freq, previous_freq, indices =  clean_speech_data(X, Y_frequency)
                session_data[window_name] = X_clean

                print(f"        Processed {X_clean.shape[0]} neurons × {X_clean.shape[1]} trials")

            if session_data['onset'] is not None:
                n_neurons = session_data['onset'].shape[0]
                for window_name in WINDOW_NAMES:
                    if session_data[window_name] is not None:
                        data[window_name]['X'].append(session_data[window_name])
                        data[window_name]['Y_brain'].extend(y_brain)
                        data[window_name]['Y_freq'].append(Y_freq)
                        data[window_name]['mouseID'].extend([subject] * n_neurons)
                        data[window_name]['sessionID'].extend(
                            [f"{subject}_{date}_{targetSiteName}"] * n_neurons)

                        X_all[window_name].append(session_data[window_name])

                Y_brain_all.extend(y_brain)

print("\nBuilding speech population arrays...")
population_data = {}
for window_name in WINDOW_NAMES:
    population_data[window_name] = {}

    X_sorted = sort_speech_arrays(X_all[window_name], indices)
    X_array = np.concatenate(X_sorted, axis=0)
    Y_brain_array = np.array(Y_brain_all)
    Y_freq_array = Y_freq

    print(f"\nspeech - {window_name}: Combined shape {X_array.shape}, Brain areas: {len(Y_brain_array)}")

    for brain_area in params.targetSiteNames:
        X_brain_area = X_array[Y_brain_array == brain_area]

        X_brain_area = X_brain_area.T
        n_neurons = X_brain_area.shape[1]
        Y_brain_this_area = [brain_area] * n_neurons

        population_data[window_name][brain_area] = {
            'X': X_brain_area,  # (trials × neurons)
            'Y_freq': Y_freq_array,  # (trials,)
            'Y_brain': Y_brain_this_area  # (neurons,)
        }

        print(f"  {brain_area}: {X_brain_area.shape}")


print("\nSaving session-level data...")
save_speech_data()
print("Done!")


celldb = celldatabase.load_hdf(params.fullPath)
simpleSiteNames = celldb['recordingSiteName'].str.split(',').apply(lambda x: x[0])
simpleSiteNames.name = 'simpleSiteName'
celldb = pd.concat([celldb, simpleSiteNames], axis=1)
celldb_subset = celldb[celldb['simpleSiteName'].isin(params.targetSiteNames)].reset_index()


print("Calculating firing rates for natural sounds")
fr_arrays = calculate_fr_arrays(celldb_subset, 'naturalSound',
                                params.naturalStimVar, params.NS_timeRange, params.natSounds_allPeriods)
save_non_speech_arrays('naturalSound', fr_arrays)


print("Calculating firing rates for AM")
fr_arrays = calculate_fr_arrays(celldb_subset, 'AM',
                                params.simpleStimVar, params.AM_timeRange, params.natSounds_allPeriods)
save_non_speech_arrays('AM', fr_arrays)

print("Calculating firing rates for pure tone sounds")
fr_arrays = calculate_fr_arrays(celldb_subset, 'pureTones',
                                params.simpleStimVar, params.PT_timeRange, params.natSounds_allPeriods)
save_non_speech_arrays('pureTones', fr_arrays)
