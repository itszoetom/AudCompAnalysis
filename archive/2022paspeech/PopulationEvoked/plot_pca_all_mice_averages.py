"""
Purpose
- Generate 2D PCA and scree plots for neural population activity, averaging trials for each frequency
- Creates a 3x3 grid of plots for each brain area × sound type combination
- Focuses on population-level average responses rather than single-trial data

Inputs
- Spike-rate data for all subjects and sessions (neurons × trials)
- Trial labels (brain area, frequency, sound type)
- Predefined spike windows, trial equalization criteria, and neuron subsampling thresholds

Processing
- Load and filter data for Speech, AM, and Pure Tone stimuli
- Equalize trial counts and randomly subsample neurons if needed
- Average spike rates across trials for each unique frequency
- Standardize data and perform PCA for each brain area × sound type
- Calculate explained variance and participation ratio
- Map frequency labels to numeric values for coloring in 2D PCA plots

Outputs
- Scree plots (variance explained per PC) for brain area × sound type averages
- 2D PCA scatter plots (average trials per frequency) for each combination
- Figures saved as PNGs for documentation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from .. import funcs, studyparams as params

# %% Initialize Data Arrays
X_speech_all = []
Y_brain_area_speech_all = []
Y_frequency_speech_all = []

X_AM_all = []
Y_brain_area_AM_all = []
Y_frequency_AM_all = []

X_pureTones_all = []
Y_brain_area_PT_all = []
Y_frequency_pureTones_all = []

for subject in params.subject_list:
    for date in params.recordingDate_list[subject]:
        for targetSiteName in params.targetSiteNames:
            # Load and process data for Speech
            speechEnsemble, speechEphys, speechBdata = funcs.load_data(subject, date, targetSiteName, "FTVOTBorders")
            if speechEnsemble:
                X_speech, Y_brain_area_speech, Y_frequency_speech = funcs.spike_rate(
                    "speech", speechEnsemble, speechEphys, speechBdata, targetSiteName)

                # Apply adjustments
                X_speech_adjusted, Y_frequency_speech_adjusted, Yba_sp_adj, ignored_x_sp, ignored_y_sp, ignored_yba_sp = (
                    funcs.adjust_array_and_labels(X_speech, Y_frequency_speech, Y_brain_area_speech, params.max_trials['speech'],
                                            subject, date, targetSiteName))

                if len(X_speech_adjusted) != 0:
                    # Sort Y_frequency_speech_adjusted
                    if isinstance(Y_frequency_speech_adjusted, list):
                        Y_frequency_speech_adjusted = np.array(Y_frequency_speech_adjusted[0])
                    sorted_array = Y_frequency_speech_adjusted[  # Sort by first tuple and then second
                        np.lexsort((Y_frequency_speech_adjusted[:, 1], Y_frequency_speech_adjusted[:, 0]))]

                    Y_frequency_speech_sorted = [sorted_array]
                    Y_frequency_speech_sorted = np.array(Y_frequency_speech_sorted[0])
                    indices_speech = np.argsort(Y_frequency_speech_sorted, axis=0)  # Sort by frequency combinations

                    # Check if frequency lists are all the same
                    #if previous_frequency_speech is not None:
                        #assert deepcopy(Y_frequency_speech_sorted) == previous_frequency_speech, (
                            #f"Frequency mismatch for subject: {subject}, date: {date}, target site: {targetSiteName}")
                    #previous_frequency_speech = deepcopy(Y_frequency_speech_sorted)

                # Append to lists
                X_speech_all.extend(X_speech_adjusted)
                Y_brain_area_speech_all.extend(Yba_sp_adj)

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
X_AM_sorted = funcs.sort_x_arrays(X_AM_all, indices_AM)
X_PT_sorted = funcs.sort_x_arrays(X_pureTones_all, indices_PT)
X_speech_sorted = funcs.sort_x_arrays(X_speech_all, indices_speech)

# Concatenate the sorted arrays
X_speech_array = np.concatenate(X_speech_sorted, axis=0)
X_speech_array = X_speech_array[:, :, 0]  # flatten to change shape from (neurons, 500, 2) to (neurons, 500)

X_AM_array = np.concatenate(X_AM_sorted, axis=0)
X_PT_array = np.concatenate(X_PT_sorted, axis=0)

# Initialize dictionary to hold average spike rates per frequency
data_dict = {}
y_max = 0.8
avg_spike_rate_dict = {}
X_mean = []

# Add data to the dictionary for each brain area and sound type
for brain_area in params.targetSiteNames:
    # For speech
    brain_area_array_speech = np.array(Y_brain_area_speech_all)
    X_speech_array_adjusted = X_speech_array[brain_area_array_speech == brain_area]
    X_speech_array_adjusted = X_speech_array_adjusted.T
    Y_frequency_speech_array_adjusted = Y_frequency_speech_sorted
    data_dict[(brain_area, 'speech')] = {'X': X_speech_array_adjusted, 'Y': Y_frequency_speech_array_adjusted}

    unique_speech_freqs = np.unique(Y_frequency_speech_array_adjusted, axis=0)
    X_means = []

    for freq in unique_speech_freqs:
        # Create a mask by comparing each element of Y_frequency_speech_array_adjusted to the tuple freq
        freq_mask = np.all(Y_frequency_speech_array_adjusted == freq, axis=1)
        # Apply the mask to X_speech_array_adjusted
        freq_values = X_speech_array_adjusted[freq_mask]
        # Calculate the mean spike rate for this frequency
        if freq_values.size > 0:
            col_means = np.mean(freq_values, axis=0)
            X_means.append(col_means)

    X_means = np.vstack(X_means)

    avg_spike_rate_dict[(brain_area, 'speech')] = {'X': np.array(X_means), 'Y': unique_speech_freqs}

    # For AM
    brain_area_array_AM = np.array(Y_brain_area_AM_all)
    X_AM_array_adjusted = X_AM_array[brain_area_array_AM == brain_area]
    X_AM_array_adjusted = X_AM_array_adjusted.T
    Y_frequency_AM_array_adjusted = Y_frequency_AM_sorted
    data_dict[(brain_area, 'AM')] = {'X': X_AM_array_adjusted, 'Y': Y_frequency_AM_array_adjusted}

    unique_am_freqs = np.unique(Y_frequency_AM_array_adjusted)
    X_means = []

    for freq in unique_am_freqs:
        # Get the indices of all trials with the current frequency
        freq_mask = (Y_frequency_AM_array_adjusted == freq)
        # Apply mask to the trials and calculate the mean spike rate for this frequency
        freq_values = X_AM_array_adjusted[freq_mask]
        col_means = np.mean(freq_values, axis=0)
        col_means = col_means.T
        X_means.append(col_means)

    X_means = np.vstack(X_means)

    avg_spike_rate_dict[(brain_area, 'AM')] = {'X': np.array(X_means), 'Y': unique_am_freqs}

    # For pure tones
    brain_area_array_PT = np.array(Y_brain_area_PT_all)
    X_PT_array_adjusted = X_PT_array[brain_area_array_PT == brain_area]
    X_PT_array_adjusted = X_PT_array_adjusted.T
    Y_frequency_PT_array_adjusted = Y_frequency_pureTones_sorted
    data_dict[(brain_area, 'PT')] = {'X': X_PT_array_adjusted, 'Y': Y_frequency_PT_array_adjusted}

    unique_pt_freqs = np.unique(Y_frequency_PT_array_adjusted)
    X_means = []

    for freq in unique_pt_freqs:
        # Get the indices of all trials with the current frequency
        freq_mask = (Y_frequency_PT_array_adjusted == freq)
        # Apply mask to the trials and calculate the mean spike rate for this frequency
        freq_values = X_PT_array_adjusted[freq_mask, :]
        col_means = np.mean(freq_values, axis=0)
        col_means = col_means.T
        X_means.append(col_means)

    X_means = np.vstack(X_means)

    avg_spike_rate_dict[(brain_area, 'PT')] = {'X': X_means, 'Y': unique_pt_freqs}

# Create a 3x3 grid for subplots
fig_scree, axes_scree = plt.subplots(3, 3, figsize=(22, 16))
fig_scree.suptitle('Scree Plots for Different Brain Areas and Sound Types', fontsize=16)
fig_scree.subplots_adjust(hspace=0.4, wspace=0.4)


def plot_scree_plot(ax, data, title, y_max, particRatio):
    pca = PCA()
    pca.fit(data['X'])
    explained_variance_ratio = pca.explained_variance_ratio_

    n_components = len(explained_variance_ratio)
    x_max = min(n_components, 13)  # Limit x-axis to 13 components
    x_min = 0

    ax.bar(range(x_max), explained_variance_ratio[:x_max], color='black')
    ax.set_xlabel('PCA features', fontsize=12)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(x_max))
    ax.set_xlim(x_min, 13)  # Set x-axis limits from 0 to 13
    ax.set_ylim(0, y_max)  # Set y-axis limits to be consistent
    ax.text(0.6, 0.85, f"Participation Ratio = {particRatio:.3f}",
            horizontalalignment='center', verticalalignment='center',
            fontsize=14, transform=ax.transAxes)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


# Plot Scree plots for each combination
for i, brain_area in enumerate(params.targetSiteNames):
    for j, sound_type in enumerate(['speech', 'AM', 'PT']):
        data = avg_spike_rate_dict.get((brain_area, sound_type), None)
        data_full = data_dict.get((brain_area, sound_type))
        if data is None:
            continue  # Skip if no data available for this combination

        title = f'{brain_area} - {sound_type} n = {data_full["X"].shape[1]}'

        # Perform PCA and calculate participation ratio
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data['X'])

        pca = PCA()
        pca.fit(data['X'])
        explained_variance_ratio = pca.explained_variance_ratio_
        particRatio = funcs.calculate_participation_ratio(explained_variance_ratio)

        # Plot the scree plot
        plot_scree_plot(axes_scree[i, j], data, title, y_max, particRatio)

# Save Scree plots figure
fig_scree.show()
fig_scree.savefig("/Users/zoetomlinson/Desktop/GitHub/neuronalDataResearch/Figures/Population Plots/PopScreeAveragePlots.png")

# Create a 3x3 grid for 2D PCA subplots
fig_pca, axes_pca = plt.subplots(3, 3, figsize=(22, 16))
fig_pca.suptitle('2D PCA Plots for Different Brain Areas and Sound Types', fontsize=16)
fig_pca.subplots_adjust(hspace=0.4, wspace=0.4)


# Plot 2D PCA plots
for i, brain_area in enumerate(params.targetSiteNames):
    for j, sound_type in enumerate(['speech', 'AM', 'PT']):
        data = avg_spike_rate_dict.get((brain_area, sound_type), None)
        data_full = data_dict.get((brain_area, sound_type))
        if data is None:
            continue  # Skip if no data is available

        title = f'{brain_area} - {sound_type}, n = {data_full["X"].shape[1]}'

        # For 'speech' sound type, create a mapping of frequencies to numbers
        if sound_type == 'speech':
            Y_labels = [tuple(row) for row in data["Y"]]
            label_to_number = {label: idx for idx, label in enumerate(params.unique_labels)}
            color_values = np.array([label_to_number[label] for label in Y_labels])
            scatter = funcs.plot_2d_pca(axes_pca[i, j], data, color_values, title)

        # For 'AM' sound type, directly use the 'Y' values
        elif sound_type == 'AM':
            Y_labels = np.array(data_dict[(brain_area, sound_type)]['Y'])
            scatter = funcs.plot_2d_pca(axes_pca[i, j], data, unique_am_freqs, title)

        # For 'PT' sound type, apply log10 transformation to 'Y'
        elif sound_type == 'PT':
            Y_labels = np.array(data_dict[(brain_area, sound_type)]['Y'])
            scatter = funcs.plot_2d_pca(axes_pca[i, j], data, unique_pt_freqs, title)

# Save as pngs
fig_pca.savefig("/Users/zoetomlinson/Desktop/GitHub/neuronalDataResearch/Figures/Population Plots/2D_PCA_Average_Plots.png")
plt.show()