# TODO: make file that defines DataClass for loading the data, FiringRateClass for calculating firing rate,
from .. import funcs, studyparams as params
import numpy as np
import os
from copy import deepcopy

data = {}
X_all = []
Y_brain_all = []
Y_frequency_all = []
previous_freq = params.previous_frequency_speech

def calculate_firing_rate(sound_label, window, ensemble, ephysData, bdata, targetSiteName):
    """Calculate firing rate for a specific sound type and time window"""
    if sound_label == "speech":
        eventOnsetTimes = ephysData['events']['stimOn']
        _, _, _ = ensemble.eventlocked_spiketimes(eventOnsetTimes, window)
        spikeCounts = ensemble.spiketimes_to_spikecounts(params.binEdges)
        sumEvokedFR = spikeCounts.sum(axis=2)
        spikesPerSecEvoked = sumEvokedFR / (window[1] - window[0])

        FTParamsEachTrial = bdata['targetFTpercent']
        VOTParamsEachTrial = bdata['targetVOTpercent']
        nTrials = len(bdata['targetFTpercent'])
        Y_frequency = np.array([(FTParamsEachTrial[i], VOTParamsEachTrial[i]) for i in range(nTrials)])
        return targetSiteName, spikesPerSecEvoked, Y_frequency

    else:
        nTrials = len(bdata['currentFreq'])
        Y_frequency = np.array(bdata['currentFreq'])
        eventOnsetTimes = ephysData['events']['stimOn'][:nTrials]
        _, _, _ = ensemble.eventlocked_spiketimes(eventOnsetTimes, window)
        if sound_label == "PT":
            spikeCounts = ensemble.spiketimes_to_spikecounts(params.binEdgesPT)
            sumEvokedFR = spikeCounts.sum(axis=2)
            spikesPerSecEvoked = sumEvokedFR / (window[1] - window[0])
            return targetSiteName, spikesPerSecEvoked, Y_frequency

        elif sound_label == "AM":
            spikeCounts = ensemble.spiketimes_to_spikecounts(params.binEdges)
            sumEvokedFR = spikeCounts.sum(axis=2)
            spikesPerSecEvoked = sumEvokedFR / (window[1] - window[0])
            return targetSiteName, spikesPerSecEvoked, Y_frequency


def normalize_firing_rate(targetSiteName, spikesPerSecEvoked, Y_frequency):
    # Normalize by trial means
    trialMeans = spikesPerSecEvoked.mean(axis=1)
    spikesPerSecEvokedNormalized = spikesPerSecEvoked.T - trialMeans
    spikesPerSecEvokedNormalized = spikesPerSecEvokedNormalized.T

    # Subsample if needed
    if spikesPerSecEvokedNormalized.shape[1] > params.leastCellsArea:
        subsetIndex = np.random.choice(spikesPerSecEvokedNormalized.shape[1], params.leastCellsArea, replace=False)
        spikeRateNormalized = spikesPerSecEvokedNormalized[:, subsetIndex]
    else:
        spikeRateNormalized = spikesPerSecEvokedNormalized

    Y_brain_area_array = [targetSiteName] * spikeRateNormalized.shape[0]

    return spikeRateNormalized, Y_brain_area_array, Y_frequency

# clean data before normalizing
def clean_data(subject, date, targetSiteName, sound_type, X, y_freq, previous_freq):
    if sound_type == "speech":
        # Initialize valid indices to keep only the trials matching the minimum occurrences
        valid_indices = []
        freq_kept_counts = params.freq_kept_counts

        # Filter the trials for each frequency based on min_speech_freq_dict
        for i, freq in enumerate(y_freq[0]):
            freq_tuple = tuple(freq)
            # Check if the count for this frequency hasn't exceeded the minimum allowed count
            if freq_kept_counts[freq_tuple] < params.min_speech_freq_dict[freq_tuple]:
                valid_indices.append(i)
                 freq_kept_counts[freq_tuple] += 1

        # Filter X_speech and Y arrays based on valid indices
        if len(valid_indices) < params.max_trials['speech']:
            print(f'Not enough speech trials for subject {subject}, on {date} in brain area {targetSiteName}')
            pass
        else:
            X_speech = np.array(X).T
            X_speech_filtered = X_speech[valid_indices].T
            Y_frequency_speech_filtered = y_freq[0][valid_indices]

            if len(X_speech_filtered) != 0:
                # Sort Y_frequency_speech_adjusted
                if isinstance(Y_frequency_speech_filtered, list):
                    Y_frequency_speech_filtered = np.array(Y_frequency_speech_filtered[0])

                # Use np.lexsort to sort by the second element of the tuple first, and then by the first element
                indices_speech = np.lexsort(
                    (Y_frequency_speech_filtered[:, 1], Y_frequency_speech_filtered[:, 0]))

                # Use these sorted indices to rearrange the array
                Y_frequency_speech_sorted = Y_frequency_speech_filtered[indices_speech]
                # Check if frequency lists are all the same
                if previous_freq is not None:
                    assert np.array_equal(Y_frequency_speech_sorted, previous_freq), (
                        f"Frequency mismatch for subject: {subject}, date: {date}, target site: {targetSiteName}"
                        f"Previous: {previous_freq} and sorted: {Y_frequency_speech_sorted}")

                previous_freq = deepcopy(Y_frequency_speech_sorted)

                return X_speech_filtered, Y_frequency_speech_sorted, previous_freq, indices_speech
    else:
        # Apply adjustments
        X, Y_freq, Y_brain = funcs.adjust_array_and_labels(X, y_freq, y_brain, params.max_trials['AM'], subject, date,
                                                           targetSiteName)
        if len(X) != 0:
            # Sort Y_frequency_AM_adjusted
            sorted_indices = np.argsort(np.array(Y_freq))
            sorted_Y_freq = Y_freq[0][sorted_indices]
            Y_freq_sorted = sorted_Y_freq
            indices = np.argsort(np.array(Y_freq_sorted[0]))  # Sort by frequency values

            # Check if frequency lists are all the same
            if previous_freq is not None:
                assert np.array_equal(Y_freq, previous_freq), (
                    f"Frequency mismatch for subject: {subject}, date: {date}, target site: {targetSiteName}")
            previous_freq = deepcopy(Y_freq)

            return X, Y_freq, previous_freq, indices


def save_data():
    """Save population data for each sound type and time window"""
    # Save each sound type separately
    for sound_label in ['speech', 'AM', 'PT']:
        # Collect data for all windows
        onset_X = []
        sustained_X = []
        offset_X = []
        brain_regions = []
        stim_array = []
        mouseIDs = []
        sessionIDs = []

        for window_name in ['onset', 'sustained', 'offset']:
            if len(data[sound_label][window_name]['X']) == 0:
                print(f"No data for {sound_label} - {window_name}, skipping...")
                continue

            # Concatenate all sessions for this window
            X_array = np.concatenate(data[sound_label][window_name]['X'], axis=0)

            if window_name == 'onset':
                onset_X = X_array
                # Get metadata from onset (same for all windows)
                brain_regions = np.array(data[sound_label][window_name]['Y_brain'])
                stim_array = np.concatenate(data[sound_label][window_name]['Y_freq'], axis=0) if len(
                    data[sound_label][window_name]['Y_freq']) > 0 else np.array([])
                mouseIDs = np.array(data[sound_label][window_name]['mouseID'])
                sessionIDs = np.array(data[sound_label][window_name]['sessionID'])
            elif window_name == 'sustained':
                sustained_X = X_array
            elif window_name == 'offset':
                offset_X = X_array

        # Check if we have data
        if len(onset_X) == 0 and len(sustained_X) == 0 and len(offset_X) == 0:
            print(f"No data for {sound_label}, skipping...")
            continue

        # Save to file
        fr_arrays_filename = os.path.join(params.figdataPath, f'fr_arrays_{sound_label}.npz')
        print(f"Saving {sound_label} data to {fr_arrays_filename}")
        print(f"  Onset shape: {onset_X.shape if len(onset_X) > 0 else 'N/A'}")
        print(f"  Sustained shape: {sustained_X.shape if len(sustained_X) > 0 else 'N/A'}")
        print(f"  Offset shape: {offset_X.shape if len(offset_X) > 0 else 'N/A'}")
        print(f"  Brain regions: {brain_regions.shape}")
        print(f"  Stim: {stim_array.shape}")
        print(f"  Unique mice: {np.unique(mouseIDs)}")
        print(f"  Unique sessions: {len(np.unique(sessionIDs))}")

        np.savez(fr_arrays_filename,
                 onsetfr=onset_X,
                 sustainedfr=sustained_X,
                 offsetfr=offset_X,
                 brainRegionArray=brain_regions,
                 stimArray=stim_array,
                 mouseIDArray=mouseIDs,
                 sessionIDArray=sessionIDs)
        print(f"Saved {sound_label}!")


# load all sessions
print("Loading all sessions...")
for subject in params.subject_list:
    for date, targetSiteName in params.recordingDate_list[subject], params.targetSiteNames:
        for sound_type, sound_type_load in params.sound_type_load:
            print(f"Processing: {subject}, {date}, {targetSiteName}, {sound_type}}")
            ensemble, ephys, bdata = funcs.load_data(subject, date, targetSiteName, sound_type_load)
            if not ensemble:
                print(f"No data for {subject}, {date}, {targetSiteName}, {sound_type}")
                continue
            for window_name, window in params.spike_windows:
                targetSiteName, spikesPerSecEvoked, Y_frequency = calculate_firing_rate(sound_type, window, ensemble,
                                                                                        ephys, bdata, targetSiteName)
                X, y_brain, y_freq = normalize_firing_rate(targetSiteName, spikesPerSecEvoked, Y_frequency)
                X, Y_freq, previous_freq, indices = clean_data(subject, date, targetSiteName, sound_type, X, y_freq, previous_freq)

                # Append to lists
                X_all.extend([X])
                Y_brain_all.extend(y_brain)

    # Apply sorting to the X arrays
    X_sorted = funcs.sort_x_arrays(X_all, indices, sound_type)
    # Concatenate the sorted arrays
    X_array = np.concatenate(X_sorted, axis=0)

    result = {}

    # Add data to the dictionary for each brain area and sound type
    for brain_area in params.targetSiteNames:
        for sound_type, X_array, Y_brain_area_all, Y_frequency_sorted in zip(
                ['speech', 'AM', 'PT'],
                [X_speech_array, X_AM_array, X_PT_array],
                [Y_brain_area_speech_all, Y_brain_area_AM_all, Y_brain_area_PT_all],
                [Y_frequency_speech_sorted, Y_frequency_AM_sorted, Y_frequency_pureTones_sorted]):
            brain_area_array = np.array(Y_brain_area_all)
            X_array_adjusted = X_array[brain_area_array == brain_area]
            X_array_adjusted = X_array_adjusted.T
            X_array_adjusted = funcs.select_neurons(X_array_adjusted.T, brain_area, params.min_neuron_dict)
            result[(brain_area, sound_type)] = {'X': X_array_adjusted, 'Y': Y_frequency_sorted}

                result = {'X': X, 'Y_brain': y_brain, 'Y_freq': y_freq}
                # Get number of neurons in this session
                n_neurons = result['X'].shape[0] if result['X'].ndim > 1 else 1

                # Store data
                data[sound_type][window_name]['X'].append(result['X'])
                data[sound_type][window_name]['Y_brain'].extend(result['Y_brain'])
                data[sound_type][window_name]['Y_freq'].append(result['Y_freq'])

                # Create mouseID and sessionID arrays for each neuron
                data[sound_type][window_name]['mouseID'].extend([subject] * n_neurons)
                data[sound_type][window_name]['sessionID'].extend(
                    [f"{subject}_{date}_{targetSiteName}"] * n_neurons)