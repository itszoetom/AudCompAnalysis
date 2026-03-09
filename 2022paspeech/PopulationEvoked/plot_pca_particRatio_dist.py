"""
Purpose
- Generate final box-and-whisker plots summarizing PCA participation ratios
  across brain areas and sound types.

Inputs
- Spike-rate matrices for Speech, AM, and Pure Tones
- Trial labels (brain area, frequency, subject, date)
- Analysis parameters from studyparams (trial limits, neuron thresholds)
- Preprocessed HDF5 database of recording sites

Processing
- Load and clean neural data across subjects, dates, and sites
- Equalize trial counts across frequencies and sound types
- Sort trials to ensure consistent frequency ordering
- Concatenate neurons across sessions within each brain area
- Standardize neural activity and perform PCA
- Compute participation ratio per session / condition
- Aggregate participation ratios by brain area × sound type

Outputs
- Lists of participation ratios for:
  - Primary auditory area (Speech, AM, PT)
  - Ventral auditory area (Speech, AM, PT)
- Box-and-whisker plots comparing participation ratios across sound types
- Saved figures for final visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from .. import studyparams as params, funcs
from jaratoolbox import celldatabase, settings

# Create arrays to hold participation ratios for each brain area and sound type combo
primary_speech = []
primary_am = []
primary_pt = []
ventral_speech = []
ventral_am = []
ventral_pt = []

# %% Loop through each mouse-date combo
for subject in params.subject_list:
    # Create a 3x3 grid for 2D PCA subplots
    y_max = 0.17

    X_speech_all = []
    Y_brain_area_speech_all = []
    Y_frequency_speech_all = []
    X_AM_all = []
    Y_brain_area_AM_all = []
    Y_frequency_AM_all = []
    X_pureTones_all = []
    Y_brain_area_PT_all = []
    Y_frequency_pureTones_all = []

    for date in params.recordingDate_list[subject]:
        for targetSiteName in params.targetSiteNames:
            # Load and process data for Speech
            speechEnsemble, speechEphys, speechBdata = funcs.load_data(subject, date, targetSiteName, "FTVOTBorders")
            if speechEnsemble:
                X_speech, Y_brain_area_speech, Y_frequency_speech = funcs.spike_rate(
                    "speech", speechEnsemble, speechEphys, speechBdata, targetSiteName)

                # Initialize valid indices to keep only the trials matching the minimum occurrences
                valid_indices = []
                freq_kept_counts = {tuple(freq): 0 for freq in params.unique_labels}

                # Filter the trials for each frequency based on min_speech_freq_dict
                for i, freq in enumerate(Y_frequency_speech[0]):
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
                    X_speech = np.array(X_speech)
                    X_speech = X_speech.T
                    X_speech_filtered = X_speech[valid_indices]
                    X_speech_filtered = X_speech_filtered.T
                    Y_frequency_speech_filtered = Y_frequency_speech[0][valid_indices]

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

                    # Append to lists
                    X_speech_all.extend([X_speech_filtered])
                    Y_brain_area_speech_all.extend(Y_brain_area_speech)

            # Load and process data for AM
            amEnsemble, amEphys, amBdata = funcs.load_data(subject, date, targetSiteName, "AM")
            if amEnsemble:
                X_AM, Y_brain_area_AM, Y_frequency_AM = funcs.spike_rate(
                    "AM", amEnsemble, amEphys, amBdata, targetSiteName)

                # Apply adjustments
                X_AM_adjusted, Y_frequency_AM_adjusted, Yba_AM_adj, ignored_x_AM, ignored_y_AM, ignored_yba_AM = (
                    funcs.adjust_array_and_labels(X_AM, Y_frequency_AM, Y_brain_area_AM, params.max_trials['AM'], subject, date,
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

            # Load and process data for Pure Tones
            ptEnsemble, ptEphys, ptBdata = funcs.load_data(subject, date, targetSiteName, "pureTones")
            if ptEnsemble:
                X_pureTones, Y_brain_area_PT, Y_frequency_pureTones = funcs.spike_rate(
                    "PT", ptEnsemble, ptEphys, ptBdata, targetSiteName)

                # Apply adjustments
                X_PT_adjusted, Y_frequency_PT_adjusted, Yba_PT_adj, ignored_x_PT, ignored_y_PT, ignored_yba_PT = (
                    funcs.adjust_array_and_labels(X_pureTones, Y_frequency_pureTones, Y_brain_area_PT, params.max_trials['PT'],
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

    # Apply sorting to the X arrays
    X_AM_sorted = funcs.sort_x_arrays(X_AM_all, indices_AM, "am")
    X_PT_sorted = funcs.sort_x_arrays(X_pureTones_all, indices_PT, "pt")
    X_speech_sorted = funcs.sort_x_arrays(X_speech_all, indices_speech, "speech")

    # Concatenate the sorted arrays
    X_speech_array = np.concatenate(X_speech_sorted, axis=0)
    X_AM_array = np.concatenate(X_AM_sorted, axis=0)
    X_PT_array = np.concatenate(X_PT_sorted, axis=0)

    data_dict = {}

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
            data_dict[(brain_area, sound_type)] = {'X': X_array_adjusted, 'Y': Y_frequency_sorted}

    for i, brain_area in enumerate(params.targetSiteNames):
        for j, sound_type in enumerate(['speech', 'AM', 'PT']):
            data = data_dict[(brain_area, sound_type)]

            # Perform PCA and calculate participation ratio
            scaler = StandardScaler()
            data_standardized = scaler.fit_transform(data['X'])
            pca = PCA()
            pca.fit(data_standardized)
            explained_variance_ratio = pca.explained_variance_ratio_
            particRatio = funcs.calculate_participation_ratio_(explained_variance_ratio)
            n_neurons = data['X'].shape[1]  # Number of neurons
            # particRatioPercent = calculate_participation_ratio_percent(explained_variance_ratio, n_neurons)

            if i == 0 and j == 0:
                primary_speech.append(particRatio)
            if i == 0 and j == 1:
                primary_am.append(particRatio)
            if i == 0 and j == 2:
                primary_pt.append(particRatio)
            if i == 1 and j == 0:
                ventral_speech.append(particRatio)
            if i == 1 and j == 1:
                ventral_am.append(particRatio)
            if i == 1 and j == 2:
                ventral_pt.append(particRatio)


def box_and_whisker_plot(speech_array, am_array, pt_array, brain_area, n_neurons):
    # Define sound types
    sound_types = ['Speech', 'AM', 'PT']

    # Combine all arrays into a list for boxplot
    data = [speech_array, am_array, pt_array]

    # Create the boxplot
    fig, ax = plt.subplots(figsize=(22, 16))
    fig.suptitle(f'{brain_area} Box Plot for Sounds', fontsize=16)

    # Boxplot for speech, AM, and PT data
    ax.boxplot(data, patch_artist=True, widths=0.6)

    # Set x-axis labels and titles
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(sound_types)
    ax.set_title('Participation Ratio Across Sound Types')
    ax.set_xlabel('Sound Type')
    ax.set_ylabel(f'Participation Ratio - n = {n_neurons}')

    plt.tight_layout()
    plt.show()
    plt.savefig(params.figSavePath + f"{brain_area} Box Plot for {n_neurons}")

box_and_whisker_plot(primary_speech, primary_am, primary_pt, "Primary auditory area", n_neurons)
box_and_whisker_plot(ventral_speech, ventral_am, ventral_pt, "Ventral auditory area", n_neurons)
