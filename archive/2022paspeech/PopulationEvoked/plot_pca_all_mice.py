"""
Purpose
- Visualize neural population activity using PCA across brain areas and sound types

Inputs
- Spike-rate data for each session (neurons × trials)
- Trial labels (mouse, brain area, sound type)
- Predefined spike windows, trial equalization criteria, and neuron counts

Processing
- Filter trials to meet minimum frequency occurrences
- Concatenate neuron activity across sessions and subjects
- Standardize data and perform PCA
- Calculate explained variance and participation ratio
- Randomly subset neurons for additional plots (111 neurons)

Outputs
- Scree plots (variance explained per PC) for each brain area × sound type
- 2D PCA scatter plots for each brain area × sound type
- Scree and PCA plots for a 111-neuron subset
- Figures saved as PNGs for documentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from .. import studyparams, funcs

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

# %% Loop through each mouse-date combo
for subject in studyparams.subject_list:
    for date in studyparams.recordingDate_list[subject]:
        for targetSiteName in studyparams.targetSiteNames:
            # Load and process data for Speech
            speechEnsemble, speechEphys, speechBdata = funcs.load_data(subject, date, targetSiteName, "FTVOTBorders")
            if speechEnsemble:
                X_speech, Y_brain_area_speech, Y_frequency_speech = funcs.spike_rate(
                    "speech", speechEnsemble, speechEphys, speechBdata, targetSiteName)

                # Increment the neuron count for this subject
                studyparams.neuron_counts[subject] += X_speech[0].shape[0]  # Count number of neurons (rows)

                # Initialize valid indices to keep only the trials matching the minimum occurrences
                valid_indices = []
                freq_kept_counts = {tuple(freq): 0 for freq in studyparams.unique_labels}

                # Filter the trials for each frequency based on min_speech_freq_dict
                for i, freq in enumerate(Y_frequency_speech[0]):
                    freq_tuple = tuple(freq)
                    # Check if the count for this frequency hasn't exceeded the minimum allowed count
                    if freq_kept_counts[freq_tuple] < studyparams.min_speech_freq_dict[freq_tuple]:
                        valid_indices.append(i)
                        freq_kept_counts[freq_tuple] += 1

                # Filter X_speech and Y arrays based on valid indices
                if len(valid_indices) < studyparams.max_trials['speech']:
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
                    funcs.adjust_array_and_labels(X_AM, Y_frequency_AM, Y_brain_area_AM, studyparams.max_trials['AM'], subject, date,
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
                    funcs.adjust_array_and_labels(X_pureTones, Y_frequency_pureTones, Y_brain_area_PT, studyparams.max_trials['PT'],
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

# Create a pandas DataFrame from the neuron counts dictionary
neuron_counts_df = pd.DataFrame(list(studyparams.neuron_counts.items()), columns=['Subject', 'Neuron_Count'])

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
for brain_area in studyparams.targetSiteNames:
    # For speech
    brain_area_array_speech = np.array(Y_brain_area_speech_all)
    X_speech_array_adjusted = X_speech_array[brain_area_array_speech == brain_area]
    X_speech_array_adjusted = X_speech_array_adjusted.T
    Y_frequency_speech_array_adjusted = Y_frequency_speech_sorted
    data_dict[(brain_area, 'speech')] = {'X': X_speech_array_adjusted, 'Y': Y_frequency_speech_array_adjusted}

    # For AM
    brain_area_array_AM = np.array(Y_brain_area_AM_all)
    X_AM_array_adjusted = X_AM_array[brain_area_array_AM == brain_area]
    X_AM_array_adjusted = X_AM_array_adjusted.T
    Y_frequency_AM_array_adjusted = Y_frequency_AM_sorted
    data_dict[(brain_area, 'AM')] = {'X': X_AM_array_adjusted, 'Y': Y_frequency_AM_array_adjusted}

    # For pure tones
    brain_area_array_PT = np.array(Y_brain_area_PT_all)
    X_PT_array_adjusted = X_PT_array[brain_area_array_PT == brain_area]
    X_PT_array_adjusted = X_PT_array_adjusted.T
    Y_frequency_PT_array_adjusted = Y_frequency_pureTones_sorted
    data_dict[(brain_area, 'PT')] = {'X': X_PT_array_adjusted, 'Y': Y_frequency_PT_array_adjusted}

# Figure subplots
y_max = 0.17
fig_scree, axes_scree = plt.subplots(3, 3, figsize=(22, 16))
fig_scree.suptitle('Scree Plots for Different Brain Areas and Sound Types', fontsize=16)
fig_scree.subplots_adjust(hspace=0.4, wspace=0.4)

def plot_scree_plot(ax, data, title, y_max, particRatio):
    pca = PCA()
    pca.fit(data)
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
for i, brain_area in enumerate(studyparams.targetSiteNames):
    for j, sound_type in enumerate(['speech', 'AM', 'PT']):
        data = data_dict[(brain_area, sound_type)]
        title = f'{brain_area} - {sound_type}, n = {data["X"].shape[1]}'

        # Perform PCA and calculate participation ratio
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data['X'])

        pca = PCA()
        pca.fit(data_standardized)
        explained_variance_ratio = pca.explained_variance_ratio_

        particRatio = funcs.calculate_participation_ratio(explained_variance_ratio)

        # Plot the scree plot
        plot_scree_plot(axes_scree[i, j], data_standardized, title, y_max, particRatio)

# Save Scree plots figure
fig_scree.show()
fig_scree.savefig("/Users/zoetomlinson/Desktop/GitHub/neuronalDataResearch/Figures/Population Plots/PopScreePlots.png")

# Create a 3x3 grid for 2D PCA subplots
fig_pca, axes_pca = plt.subplots(3, 3, figsize=(22, 16))
fig_pca.suptitle('2D PCA Plots for Different Brain Areas and Sound Types', fontsize=16)
fig_pca.subplots_adjust(hspace=0.4, wspace=0.4)


def plot_2d_pca(ax, data, labels, title, cmap='viridis'):
    # Perform PCA and calculate participation ratio
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data['X'])

    pca = PCA()
    transformed_data = pca.fit_transform(data_standardized)

    explained_variance_ratio = pca.explained_variance_ratio_

    scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap=cmap, s=32)
    ax.set_title(title)
    ax.set_xlabel(f'PCA 1 ({explained_variance_ratio[0] * 100:.2f}% variance)')
    ax.set_ylabel(f'PCA 2 ({explained_variance_ratio[1] * 100:.2f}% variance)')
    plt.colorbar(scatter, ax=ax, orientation='vertical')


for i, brain_area in enumerate(studyparams.targetSiteNames):
    for j, sound_type in enumerate(['speech', 'AM', 'PT']):
        data = data_dict.get((brain_area, sound_type), None)
        title = f'{brain_area} - {sound_type}, n = {data["X"].shape[1]}'

        # For 'speech' sound type, create a mapping of frequencies to numbers
        if sound_type == 'speech':
            Y_labels = [tuple(row) for row in data["Y"]]
            label_to_number = {label: idx for idx, label in enumerate(studyparams.unique_labels)}
            color_values = np.array([label_to_number[label] for label in Y_labels])
            plot_2d_pca(axes_pca[i, j], data, color_values, title)

        # For 'AM' sound type, directly use the 'Y' values
        elif sound_type == 'AM':
            plot_2d_pca(axes_pca[i, j], data, data["Y"], title)

        # For 'PT' sound type, apply log10 transformation to 'Y'
        elif sound_type == 'PT':
            plot_2d_pca(axes_pca[i, j], data, np.log10(data["Y"]), title)

# Save as pngs
fig_pca.savefig(studyparams.figSavePath + "2D_PCA_Plots.png")
fig_pca.show()

# Create a 3x3 grid for 2D PCA subplots
y_max = 0.12
fig_pca_subset111, axes_pca = plt.subplots(3, 3, figsize=(22, 16))
fig_pca_subset111.suptitle('2D PCA Plots for Different Brain Areas and Sound Types - Subset to 111 neurons', fontsize=16)
fig_pca_subset111.subplots_adjust(hspace=0.4, wspace=0.4)

for i, brain_area in enumerate(["Primary auditory area", "Dorsal auditory area", "Ventral auditory area"]):
    for j, sound_type in enumerate(['speech', 'AM', 'PT']):
        data = data_dict.get((brain_area, sound_type))

        if data is not None:
            # Randomly select 111 neurons (columns)
            selected_indices = np.random.choice(data['X'].shape[1], 111, replace=False)

            # select columns
            data_111_X = data['X'][:, selected_indices]

            data_neurons = {"X": data_111_X, "Y": data['Y']}

            # Update the title to reflect the number of neurons being plotted
            title = f'{brain_area} - {sound_type}, n = {data_neurons["X"].shape[1]}'

            # For 'speech' sound type, create a mapping of frequencies to numbers
            if sound_type == 'speech':
                Y_labels = [tuple(row) for row in data_neurons["Y"]]
                label_to_number = {label: idx for idx, label in enumerate(studyparams.unique_labels)}
                color_values = np.array([label_to_number.get(label, -1) for label in Y_labels])
                plot_2d_pca(axes_pca[i, j], data_neurons, color_values, title)

            # For 'AM' sound type, directly use the 'Y' values
            elif sound_type == 'AM':
                plot_2d_pca(axes_pca[i, j], data_neurons, data_neurons["Y"], title)

            # For 'PT' sound type, apply log10 transformation to 'Y'
            elif sound_type == 'PT':
                plot_2d_pca(axes_pca[i, j], data_neurons, np.log10(data_neurons["Y"]), title)

# Save as PNG
fig_pca_subset111.savefig(studyparams.figSavePath + "2D_PCA_Subset111Neurons_Plots.png")
plt.show()

# Create a 3x3 grid for subplots
fig_scree_subset111, axes_scree = plt.subplots(3, 3, figsize=(22, 16))
fig_scree_subset111.suptitle('Scree Plots for Different Brain Areas and Sound Types - Subset to 111 neurons', fontsize=16)
fig_scree_subset111.subplots_adjust(hspace=0.4, wspace=0.4)

# Plot Scree plots for each combination
for i, brain_area in enumerate(["Primary auditory area", "Dorsal auditory area", "Ventral auditory area"]):
    for j, sound_type in enumerate(['speech', 'AM', 'PT']):
        data = data_dict.get((brain_area, sound_type))
        if data is not None:
            title = f'{brain_area} - {sound_type}, n = {min(data["X"].shape[1], 111)}'

            # Randomly select 111 neurons
            selected_indices = np.random.choice(data['X'].shape[1], 111, replace=False)
            X_first_111 = data['X'][:, selected_indices]

            # Perform PCA and calculate participation ratio
            scaler = StandardScaler()
            data_standardized = scaler.fit_transform(X_first_111)

            pca = PCA()
            pca.fit(data_standardized)
            explained_variance_ratio = pca.explained_variance_ratio_
            particRatio = funcs.calculate_participation_ratio(explained_variance_ratio)

            # Plot the scree plot
            plot_scree_plot(axes_scree[i, j], data_standardized, title, y_max, particRatio)

# Save Scree plots figure
plt.show()
fig_scree_subset111.savefig(studyparams.figSavePath + "PopScreePlots_Subset111Neurons.png")
