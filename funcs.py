# Functions file for AudPopAnalysis Repo

import numpy as np
import matplotlib.pyplot as plt
import os
from jaratoolbox import celldatabase, settings, ephyscore
from copy import deepcopy
import pandas as pd
from sklearn.decomposition import PCA

# %% Constants
subject_list = ['feat004', 'feat005', 'feat006', 'feat007', 'feat008', 'feat009', 'feat010']
recordingDate_list = {
    'feat004': ['2022-01-11', '2022-01-19', '2022-01-21'],
    'feat005': ['2022-02-07', '2022-02-08', '2022-02-11', '2022-02-14', '2022-02-15', '2022-02-16'],
    'feat006': ['2022-02-21', '2022-02-22', '2022-02-24', '2022-02-25', '2022-02-26', '2022-02-28', '2022-03-01', '2022-03-02'],
    'feat007': ['2022-03-10', '2022-03-11', '2022-03-15', '2022-03-16', '2022-03-18', '2022-03-21'],
    'feat008': ['2022-03-23', '2022-03-24', '2022-03-25'],
    'feat009': ['2022-06-04', '2022-06-05', '2022-06-06', '2022-06-07', '2022-06-09', '2022-06-10'],
    'feat010': ['2022-06-21', '2022-06-22', '2022-06-27', '2022-06-28', '2022-06-30']
}
targetSiteNames = ["Primary auditory area", "Dorsal auditory area", "Ventral auditory area"]
leastCellsArea = 10000
evoked_start = 0.015
evoked_end = 0.3
pt_evoked_end = 0.1
binWidth = 0.01
binEdges = np.arange(evoked_start, evoked_end, binWidth)
binEdgesPT = np.arange(evoked_start, pt_evoked_end, binWidth)
periodsNameSpeech = ['base200', 'respOnset', 'respSustained']
max_trials = {'PT': 640, 'AM': 220, 'speech': 381}
unique_labels = [(0, 0), (0, 33), (0, 67), (0, 100), (33, 100), (67, 100), (100, 100), (100, 67), (100, 33),
                 (100, 0), (67, 0), (33, 0)]
min_speech_freq_dict = {(0, 0): 31, (0, 33): 29, (0, 67): 32, (0, 100): 24, (33, 100): 34, (67, 100): 35,
                        (100, 100): 33, (100, 67): 29, (100, 33): 35, (100, 0): 35, (67, 0): 31, (33, 0): 33}
frequency_counts_dict = {tuple(freq): [] for freq in unique_labels}
data_dict = {}

# %% Load dataframe
databaseDir = os.path.join(settings.DATABASE_PATH, '2024popanalysis')
fullDbPath = 'celldb_2024popanalysis.h5'
fullPath = os.path.join(databaseDir, fullDbPath)
fullDb = celldatabase.load_hdf(fullPath)
simpleSiteNames = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])
simpleSiteNames = simpleSiteNames.replace("Posterior auditory area", "Dorsal auditory area")
fullDb["recordingSiteName"] = simpleSiteNames


# %% Initialize plot and subset dataframe
def load_data(subject, date, targetSiteName, sound_type_load):
    celldb = fullDb[(fullDb.subject == subject)]
    celldbSubset = celldb[(celldb.date == date)]
    celldbSubset = celldbSubset[(celldbSubset.recordingSiteName == targetSiteName)]

    if celldbSubset.empty:
        print(f"No data in {targetSiteName} on {date} for Speech, AM, and PT.")
        return None, None, None

    ensemble = ephyscore.CellEnsemble(celldbSubset)
    try:
        ephysData, bdata = ensemble.load(sound_type_load)
    except IndexError:
        print(f"No FTVOTBorder data for {targetSiteName} on {date} for {subject}")
        return None, None, None

    return ensemble, ephysData, bdata


# Calculate Spike Rate
def spike_rate(sound_type, ensemble, ephysData, bdata, targetSiteName):
    '''Calculate firing rate as spikes per second evoked
            sound_type: str sound type label. ex. "speech"
            ensemple: ephyscore.CellEnsemble(celldbSubset)
            ephysData: ephyscore.CellEnsemble(celldbSubset)
            bdata: ephyscore.CellEnsemble(celldbSubset)
            targetSiteName: str brain area ex. "Primary auditory area"

        Returns X array (spikeRateNormalized) of firing rates for specified sound type and brain area (trials, neurons)
                Y brain area meta data (Y_brain_area_array) brain area meta data for each neuron
                Y sound frequency (Hrz) (Y_frequency) meta data for each trial
    '''
    X_array = []

    if sound_type == "speech":
        eventOnsetTimes = ephysData['events']['stimOn']
        _, _, _ = ensemble.eventlocked_spiketimes(eventOnsetTimes, [evoked_start, evoked_end])
        spikeCounts = ensemble.spiketimes_to_spikecounts(binEdges)
        sumEvokedFR = spikeCounts.sum(axis=2)
        spikesPerSecEvoked = sumEvokedFR / (evoked_end - evoked_start)

        FTParamsEachTrial = bdata['targetFTpercent']
        VOTParamsEachTrial = bdata['targetVOTpercent']
        nTrials = len(bdata['targetFTpercent'])

        # Create and sort Y_frequency for speech
        Y_frequency = np.array([(FTParamsEachTrial[i], VOTParamsEachTrial[i]) for i in range(nTrials)])

    if sound_type == "AM":
        nTrials = len(bdata['currentFreq'])
        eventOnsetTimes = ephysData['events']['stimOn'][:nTrials]
        _, _, _ = ensemble.eventlocked_spiketimes(eventOnsetTimes, [evoked_start, evoked_end])
        spikeCounts = ensemble.spiketimes_to_spikecounts(binEdges)
        sumEvokedFR = spikeCounts.sum(axis=2)
        spikesPerSecEvoked = sumEvokedFR / (evoked_end - evoked_start)

        # Create and sort Y_frequency for AM/PT
        Y_frequency = np.array(bdata['currentFreq'])

    if sound_type == "PT":
        nTrials = len(bdata['currentFreq'])
        eventOnsetTimes = ephysData['events']['stimOn'][:nTrials]
        _, _, _ = ensemble.eventlocked_spiketimes(eventOnsetTimes, [evoked_start, pt_evoked_end])
        spikeCounts = ensemble.spiketimes_to_spikecounts(binEdgesPT)
        sumEvokedFR = spikeCounts.sum(axis=2)
        spikesPerSecEvoked = sumEvokedFR / (evoked_end - pt_evoked_end)

        # Create and sort Y_frequency for AM/PT
        Y_frequency = np.array(bdata['currentFreq'])

    trialMeans = spikesPerSecEvoked.mean(axis=1)
    spikesPerSecEvokedNormalized = spikesPerSecEvoked.T - trialMeans  # why negative
    spikesPerSecEvokedNormalized = spikesPerSecEvokedNormalized.T

    if spikesPerSecEvokedNormalized.shape[1] > leastCellsArea:
        subsetIndex = np.random.choice(spikesPerSecEvokedNormalized.shape[1], leastCellsArea, replace=False)
        spikeRateNormalized = spikesPerSecEvokedNormalized[:, subsetIndex]
    else:
        spikeRateNormalized = spikesPerSecEvokedNormalized

    Y_brain_area_array = [targetSiteName] * spikeRateNormalized.shape[0]

    return spikeRateNormalized, Y_brain_area_array, Y_frequency


def adjust_array_and_labels(x_array, y_array, Y_brain_area_PT, max_length, subject, date, targetSiteName):
    # Check if any row in x_array has fewer than max_length trials
    if x_array.shape[1] < max_length:
        print(f"Not enough PT trials recorded for subject {subject}, on {date} in brain area {targetSiteName}.")
        return None, None, None
    else:
        adjusted_x = x_array[:, :max_length]  # Truncate across all neurons

    adjusted_y = y_array[:max_length]
    adjusted_yba = Y_brain_area_PT[:max_length]

    return adjusted_x, adjusted_y, adjusted_yba


def sort_x_arrays(X_list, indices, sound_type):
    sorted_x_list = []
    for x in X_list:
        if sound_type == "am" or sound_type == "pt":
            sorted_x = x[indices]
            sorted_x_list.append(np.array(sorted_x))
        if sound_type == "speech":
            sorted_x = x[indices]
            sorted_x_list.append(np.array(sorted_x))
    return sorted_x_list


# Function to randomly select neurons based on the min_neuron_dict
def select_neurons(data, brain_area, min_neuron_dict):
    # Check if brain area in dictionary and apply subsampling if necessary
    if brain_area in min_neuron_dict:
        n_neurons_to_select = min_neuron_dict[brain_area]
        total_neurons = data.shape[1]

        # Ensure that we don't select more neurons than are available
        if total_neurons > n_neurons_to_select:
            selected_indices = np.random.choice(total_neurons, n_neurons_to_select, replace=False)
            data = data[:, selected_indices]
        else:
            print(f"Not enough neurons in {brain_area}, using all {total_neurons} neurons.")
    return data


def create_figure_grid(rows, cols, title, figsize=(18, 10)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    return fig, axes


def min_speech_dict(Y_frequency_speech, X_speech, subject, date, targetSiteName, previous_frequency_speech):
    valid_indices = []
    freq_kept_counts = {tuple(freq): 0 for freq in unique_labels}

    # Filter the trials for each frequency based on min_speech_freq_dict
    for i, freq in enumerate(Y_frequency_speech[0]):
        freq_tuple = tuple(freq)
        # Check if the count for this frequency hasn't exceeded the minimum allowed count
        if freq_kept_counts[freq_tuple] < min_speech_freq_dict[freq_tuple]:
            valid_indices.append(i)
            freq_kept_counts[freq_tuple] += 1

    # Filter X_speech and Y arrays based on valid indices
    if len(valid_indices) < max_trials['speech']:
        print(
            f'Not enough speech trials for subject {subject}, on {date} in brain area {targetSiteName}')
        pass
    else:
        X_speech = np.array(X_speech)
        X_speech = X_speech.T
        X_speech_filtered = X_speech[valid_indices]
        X_speech_filtered = X_speech_filtered.T
        Y_frequency_speech_filtered = Y_frequency_speech[0][valid_indices]
    return X_speech, X_speech_filtered, Y_frequency_speech_filtered


def adjust_speech_length(subject, date, brain_area, X_speech, Y_frequency_speech, Y_brain_area_speech, previous_frequency_speech):
    valid_indices = []
    freq_kept_counts = {tuple(freq): 0 for freq in unique_labels}

    # Filter valid trials based on frequency count
    for i, freq in enumerate(Y_frequency_speech):
        freq_tuple = tuple(freq)
        if freq_kept_counts[freq_tuple] < min_speech_freq_dict[freq_tuple]:
            valid_indices.append(i)
            freq_kept_counts[freq_tuple] += 1

    # Filter X_speech and Y arrays based on valid indices
    if len(valid_indices) < max_trials['speech']:
        print(f'Not enough speech trials for subject {subject}, on {date} in brain area {brain_area}')
        return None, None, None, None
    else:
        X_speech = np.array(X_speech)
        X_speech = X_speech.T
        X_speech_filtered = X_speech[valid_indices]
        X_speech_filtered = X_speech_filtered.T
        Y_frequency_speech_filtered = Y_frequency_speech[valid_indices]

        if len(X_speech_filtered) != 0:
            # Sort Y_frequency_speech_adjusted
            if isinstance(Y_frequency_speech_filtered, list):
                Y_frequency_speech_filtered = np.array(Y_frequency_speech_filtered)

            # Use np.lexsort to sort by the second element of the tuple first, and then by the first element
            indices_speech = np.lexsort(
                (Y_frequency_speech_filtered[:, 1], Y_frequency_speech_filtered[:, 0]))

            # Use these sorted indices to rearrange the array
            Y_frequency_speech_sorted = Y_frequency_speech_filtered[indices_speech]

            # Check if frequency lists are all the same
            if previous_frequency_speech is not None:
                assert np.array_equal(Y_frequency_speech_sorted, previous_frequency_speech), (
                    f"Frequency mismatch for subject: {subject}, date: {date}, target site: {brain_area}"
                    f"Previous: {previous_frequency_speech} and sorted: {Y_frequency_speech_sorted}")

            previous_frequency_speech = deepcopy(Y_frequency_speech_sorted)

        return X_speech_filtered, Y_frequency_speech_sorted, previous_frequency_speech, indices_speech


def sort_sound_array(subject, date, brain_area, X_adjusted, Y_brain_area_all, Y_frequency_adjusted, previous_frequency):
    X_all = None
    Y_brain_area_all_combined = None

    if X_adjusted is not None:
        if len(X_adjusted) != 0:
            # Sort Y_frequency_AM_adjusted
            Y_frequency = np.array(Y_frequency_adjusted)
            sorted_indices = np.argsort(Y_frequency)
            sorted_Y_freq = Y_frequency[sorted_indices]
            Y_frequency_sorted = sorted_Y_freq

            Y_frequency_sorted = np.array(Y_frequency_sorted)
            indices = np.argsort(Y_frequency_sorted)  # Sort by frequency values

            # Check if frequency lists are all the same
            if previous_frequency is not None:
                assert np.array_equal(Y_frequency_sorted, previous_frequency), (
                    f"Frequency mismatch for subject: {subject}, date: {date}, target site: {brain_area}")
            previous_frequency = deepcopy(Y_frequency_sorted)

            # Concatenate data instead of extending lists
            if X_all is None:
                X_all = X_adjusted
            else:
                X_all = np.concatenate((X_all, X_adjusted), axis=0)

            if Y_brain_area_all_combined is None:
                Y_brain_area_all_combined = Y_brain_area_all
            else:
                Y_brain_area_all_combined = np.concatenate((Y_brain_area_all_combined, Y_brain_area_all), axis=0)
    else:
        return None, None, None, previous_frequency, None

    return X_all, Y_frequency_sorted, Y_brain_area_all_combined, previous_frequency, indices


def sort_speech_frequencies(X_speech_filtered, Y_frequency_speech_filtered, subject, date, targetSiteName,
                            previous_frequency_speech):
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
        if previous_frequency_speech is not None:
            assert np.array_equal(Y_frequency_speech_sorted, previous_frequency_speech), (
                f"Frequency mismatch for subject: {subject}, date: {date}, target site: {targetSiteName}")
        previous_frequency_speech = deepcopy(Y_frequency_speech_sorted)
    return Y_frequency_speech_sorted, indices_speech, previous_frequency_speech


def clean_and_concatenate(subject, recordingDate_list, targetSiteName, previous_frequency_AM, previous_frequency_PT, previous_frequency_speech):
    global X_speech_array, X_AM_array, X_PT_array, Y_frequency_speech_sorted, Y_frequency_AM_sorted, Y_frequency_pureTones_sorted
    y_max = 0.17
    X_speech_all = []
    Y_brain_area_speech_all = []
    X_AM_all = []
    Y_brain_area_AM_all = []
    X_pureTones_all = []
    Y_brain_area_PT_all = []
    indices_AM = None
    indices_PT = None
    indices_speech = None

    for date in recordingDate_list[subject]:
        # Load and process data for Speech
        speechEnsemble, speechEphys, speechBdata = load_data(subject, date, targetSiteName,
                                                             "FTVOTBorders")
        if speechEnsemble:
            X_speech, Y_brain_area_speech, Y_frequency_speech = spike_rate(
                "speech", speechEnsemble, speechEphys, speechBdata, targetSiteName)

            # Initialize valid indices to keep only the trials matching the minimum occurrences
            valid_indices = []
            freq_kept_counts = {tuple(freq): 0 for freq in unique_labels}

            # Filter the trials for each frequency based on min_speech_freq_dict
            for i, freq in enumerate(Y_frequency_speech[0]):
                freq_tuple = tuple(freq)
                # Check if the count for this frequency hasn't exceeded the minimum allowed count
                if freq_kept_counts[freq_tuple] < min_speech_freq_dict[freq_tuple]:
                    valid_indices.append(i)
                    freq_kept_counts[freq_tuple] += 1

            # Filter X_speech and Y arrays based on valid indices
            if len(valid_indices) < max_trials['speech']:
                print(
                    f'Not enough speech trials for subject {subject}, on {date} in brain area {targetSiteName}')
                pass
            else:
                X_speech = np.array(X_speech)
                X_speech = X_speech.T
                X_speech_filtered = X_speech[valid_indices]
                X_speech_filtered = X_speech_filtered.T
                Y_frequency_speech_filtered = Y_frequency_speech[0][valid_indices]

                if len(X_speech_filtered) != 0:
                    # Sort Y_frequency_speech_adjusted
                    if isinstance(Y_frequency_speech_filtered, list):
                        Y_frequency_speech_filtered = np.array(Y_frequency_speech_filtered[0])

                    # Use np.lexsort to sort by the second element of the tuple first, and then by the first
                    # element
                    indices_speech = np.lexsort(
                        (Y_frequency_speech_filtered[:, 1], Y_frequency_speech_filtered[:, 0]))

                    # Use these sorted indices to rearrange the array
                    Y_frequency_speech_sorted = Y_frequency_speech_filtered[indices_speech]

                    # Check if frequency lists are all the same
                    if previous_frequency_speech is not None:
                        assert np.array_equal(Y_frequency_speech_sorted, previous_frequency_speech), (
                            f"Frequency mismatch for subject: {subject}, date: {date}, target site: {targetSiteName}")
                    previous_frequency_speech = deepcopy(Y_frequency_speech_sorted)

                # Append to lists
                X_speech_all.extend([X_speech_filtered])
                Y_brain_area_speech_all.extend(Y_brain_area_speech)
        else:
            continue

        # Load and process data for AM
        amEnsemble, amEphys, amBdata = load_data(subject, date, targetSiteName, "AM")
        if amEnsemble:
            X_AM, Y_brain_area_AM, Y_frequency_AM = spike_rate(
                "AM", amEnsemble, amEphys, amBdata, targetSiteName)

            # Apply adjustments
            X_AM_adjusted, Y_frequency_AM_adjusted, Yba_AM_adj, ignored_x_AM, ignored_y_AM, ignored_yba_AM = (
                adjust_array_and_labels(X_AM, Y_frequency_AM, Y_brain_area_AM, max_trials['AM'], subject,
                                        date,
                                        targetSiteName))

            if len(X_AM_adjusted) != 0:
                # Sort Y_frequency_AM_adjusted
                Y_frequency_AM = np.array(Y_frequency_AM_adjusted)
                sorted_indices = np.argsort(Y_frequency_AM)
                sorted_Y_freq = Y_frequency_AM[0][sorted_indices]
                Y_frequency_AM_sorted = sorted_Y_freq

                Y_frequency_AM_sorted = np.array(Y_frequency_AM_sorted[0])
                indices_AM = np.argsort(Y_frequency_AM_sorted)  # Sort by frequency values

                # Check if frequency lists are all the same
                if previous_frequency_AM is not None:
                    assert np.array_equal(Y_frequency_AM_sorted, previous_frequency_AM), (
                        f"Frequency mismatch for subject: {subject}, date: {date}, target site: {targetSiteName}")
                previous_frequency_AM = deepcopy(Y_frequency_AM_sorted)

            # Append to lists
            X_AM_all.extend(X_AM_adjusted)
            Y_brain_area_AM_all.extend(Yba_AM_adj)
        else:
            continue

        # Load and process data for Pure Tones
        ptEnsemble, ptEphys, ptBdata = load_data(subject, date, targetSiteName, "pureTones")
        if ptEnsemble:
            X_pureTones, Y_brain_area_PT, Y_frequency_pureTones = spike_rate(
                "PT", ptEnsemble, ptEphys, ptBdata, targetSiteName)

            # Apply adjustments
            X_PT_adjusted, Y_frequency_PT_adjusted, Yba_PT_adj, ignored_x_PT, ignored_y_PT, ignored_yba_PT = (
                adjust_array_and_labels(X_pureTones, Y_frequency_pureTones, Y_brain_area_PT,
                                        max_trials['PT'],
                                        subject, date, targetSiteName))

            if len(X_PT_adjusted) != 0:
                # Convert Y_frequency_pureTones_adjusted
                Y_frequency_pureTones = np.array(Y_frequency_PT_adjusted)
                sorted_indices = np.argsort(Y_frequency_pureTones)
                sorted_Y_freq = Y_frequency_pureTones[0][sorted_indices]
                Y_frequency_pureTones_sorted = sorted_Y_freq

                Y_frequency_pureTones_sorted = np.array(Y_frequency_pureTones_sorted[0])
                indices_PT = np.argsort(Y_frequency_pureTones_sorted)  # Sort by frequency values

                # Check if frequency lists are all the same
                if previous_frequency_PT is not None:
                    assert np.array_equal(Y_frequency_pureTones_sorted, previous_frequency_PT), (
                        f"Frequency mismatch for subject: {subject}, date: {date}, target site: {targetSiteName}")
                previous_frequency_PT = deepcopy(Y_frequency_pureTones_sorted)

            # Append to the lists
            X_pureTones_all.extend(X_PT_adjusted)
            Y_brain_area_PT_all.extend(Yba_PT_adj)
        else:
            continue
    if X_AM_all:
        X_AM_sorted = sort_x_arrays(X_AM_all, indices_AM, "am")
        X_AM_array = np.concatenate(X_AM_sorted, axis=0)
    if X_pureTones_all:
        X_PT_sorted = sort_x_arrays(X_pureTones_all, indices_PT, "pt")
        X_PT_array = np.concatenate(X_PT_sorted, axis=0)
    if X_speech_all:
        X_speech_sorted = sort_x_arrays(X_speech_all, indices_speech, "speech")
        X_speech_array = np.concatenate(X_speech_sorted, axis=0)

    return (X_speech_array, X_AM_array, X_PT_array, Y_brain_area_speech_all, Y_brain_area_AM_all, Y_brain_area_PT_all,
            Y_frequency_speech_sorted, Y_frequency_AM_sorted, Y_frequency_pureTones_sorted)


def calculate_participation_ratio(explained_variance_ratio):
    return ((np.sum(explained_variance_ratio)) ** 2) / np.sum(explained_variance_ratio ** 2)


def plot_scree_plot(ax, data, title, y_max, particRatio, color='black'):
    # Perform PCA
    pca = PCA()
    pca.fit(data)
    explained_variance_ratio = pca.explained_variance_ratio_

    # Plot settings
    x_max = min(len(explained_variance_ratio), 11)
    ax.bar(range(x_max), explained_variance_ratio[:x_max], color=color)

    # Axis formatting
    ax.set_xticks(range(x_max))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, y_max)

    # Participation ratio annotation
    ax.text(0.5, 0.9,
            f"Participation Ratio = {particRatio:.3f}",
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=28,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.6))


def plot_2d_pca(ax, data, labels, title, cmap):
    """
    Plots 2D PCA of the input data on the given axis.

    Parameters:
    - ax: Matplotlib axis object to plot on.
    - data: Dictionary containing data['X'] (features) and optionally other keys.
    - labels: Array-like, labels for coloring the data points.
    - title: Title of the plot.
    - cmap: Colormap for the scatter plot (default: 'viridis').

    Returns:
    - scatter: The scatter plot object for further customization if needed.
    """
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(data['X'])

    explained_variance = pca.explained_variance_ratio_

    scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap=cmap, s=32)

    ax.set_xlabel(f'PCA 1 ({explained_variance[0] * 100:.2f}% variance)')
    ax.set_ylabel(f'PCA 2 ({explained_variance[1] * 100:.2f}% variance)')

    return scatter

def euclidean_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)


def trial_distances(trials, mean_vec):
    distances = []
    for trial in trials:
        dist = sum((x - m) ** 2 for x, m in zip(trial, mean_vec)) ** 0.5
        distances.append(dist)
    return distances