"""
Purpose
- Analyze neural population activity using PCA across brain areas and sound types

Inputs
- Spike-rate data (sessions × neurons × trials)
- Trial labels (mouse, brain area, sound type)
- Predefined spike windows and trial equalization parameters

Processing
- Equalize trial counts across conditions
- Concatenate neurons across sessions
- Perform PCA on population activity
- Compute explained variance and participation ratio

Outputs
- Scree plots (variance explained per PC)
- Participation ratio per brain area × sound type
- Statistical comparisons across conditions
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from .. import studyparams, funcs
from scipy import stats

# %% Constants
subject_list = ['feat005', 'feat006', 'feat007', 'feat008', 'feat009']  # , feat004 , feat010

# Create arrays to hold participation ratios for each brain area and sound type combo
primary_speech = []
primary_am = []
primary_pt = []
ventral_speech = []
ventral_am = []
ventral_pt = []


def plot_scree_plot(ax, data, title, particRatioPercent):
    pca = PCA()
    pca.fit(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    y_max = np.max(explained_variance_ratio) * 1.1  # 10% padding

    n_components = len(explained_variance_ratio)
    x_max = min(n_components, 13)  # Limit x-axis to 13 components
    x_min = 0

    ax.bar(range(x_max), explained_variance_ratio[:x_max], color='black')
    ax.set_xlabel('PCA features', fontsize=12)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(x_max))
    ax.set_xlim(x_min, 13)  # Set x-axis limits from 0 to 13
    ax.set_ylim(0, y_max)
    ax.text(0.6, 0.85, f"Participation Ratio Percent = {particRatioPercent:.3f}",
            horizontalalignment='center', verticalalignment='center',
            fontsize=14, transform=ax.transAxes)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    return explained_variance_ratio


# Loop through each mouse-date combo
for subject in subject_list:
    # Create a 3x3 grid for 2D PCA subplots
    y_max = 0.17
    fig_scree, axes_scree = funcs.create_figure_grid(2, 3, f'Scree Plots for Mouse {subject}')
    fig_pca, axes_pca = funcs.create_figure_grid(2, 3, f'2D PCA Plots for Mouse {subject}')

    X_speech_all = []
    Y_brain_area_speech_all = []
    Y_frequency_speech_all = []
    X_AM_all = []
    Y_brain_area_AM_all = []
    Y_frequency_AM_all = []
    X_pureTones_all = []
    Y_brain_area_PT_all = []
    Y_frequency_pureTones_all = []

    for date in studyparams.recordingDate_list[subject]:
        for targetSiteName in studyparams.targetSiteNames:
            # Load and process data for Speech
            speechEnsemble, speechEphys, speechBdata = funcs.load_data(subject, date, targetSiteName, "FTVOTBorders")
            if speechEnsemble:
                X_speech, Y_brain_area_speech, Y_frequency_speech = funcs.spike_rate(
                    "speech", speechEnsemble, speechEphys, speechBdata, targetSiteName)

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
    for brain_area in ["Primary auditory area", "Ventral auditory area"]:
        for sound_type, X_array, Y_brain_area_all, Y_frequency_sorted in zip(
                ['speech', 'AM', 'PT'],
                [X_speech_array, X_AM_array, X_PT_array],
                [Y_brain_area_speech_all, Y_brain_area_AM_all, Y_brain_area_PT_all],
                [Y_frequency_speech_sorted, Y_frequency_AM_sorted, Y_frequency_pureTones_sorted]):
            brain_area_array = np.array(Y_brain_area_all)
            X_array_adjusted = X_array[brain_area_array == brain_area]
            X_array_adjusted = X_array_adjusted.T
            # X_array_adjusted = select_neurons(X_array_adjusted.T, brain_area, min_neuron_dict)
            data_dict[(brain_area, sound_type)] = {'X': X_array_adjusted, 'Y': Y_frequency_sorted}

    # Plot Scree plots for each combination
    for i, brain_area in enumerate(["Primary auditory area", "Ventral auditory area"]):
        for j, sound_type in enumerate(['speech', 'AM', 'PT']):
            data = data_dict[(brain_area, sound_type)]
            title = f'{brain_area} - {sound_type}, n = {data["X"].shape[1]}'

            # Perform PCA and calculate participation ratio
            scaler = StandardScaler()
            if data['X'].shape[1] != 0:
                data_standardized = scaler.fit_transform(data['X'])

                pca = PCA()
                pca.fit(data_standardized)
                explained_variance_ratio = pca.explained_variance_ratio_
                # particRatio = calculate_participation_ratio_(explained_variance_ratio)
                n_neurons = data['X'].shape[0]  # Number of neurons
                particRatioPercent = funcs.calculate_participation_ratio_percent(explained_variance_ratio, n_neurons)

                # Plot the scree plot
                plot_scree_plot(axes_scree[i, j], data_standardized, title, particRatioPercent)

    # Save Scree plots figure
    # fig_scree.show()
    fig_scree.savefig(studyparams.figSavePath + f"{subject} Scree Plot.png")

    for i, brain_area in enumerate(["Primary auditory area", "Ventral auditory area"]):
        for j, sound_type in enumerate(['speech', 'AM', 'PT']):
            data = data_dict[(brain_area, sound_type)]

            # Perform PCA and calculate participation ratio
            scaler = StandardScaler()
            data_standardized = scaler.fit_transform(data['X'])
            pca = PCA()
            pca.fit(data_standardized)
            explained_variance_ratio = pca.explained_variance_ratio_
            # particRatio = calculate_participation_ratio_(explained_variance_ratio)
            n_neurons = data['X'].shape[0]  # Number of neurons
            particRatioPercent = funcs.calculate_participation_ratio_percent(explained_variance_ratio, n_neurons)

            if i == 0 and j == 0:
                primary_speech.append(particRatioPercent)
            if i == 0 and j == 1:
                primary_am.append(particRatioPercent)
            if i == 0 and j == 2:
                primary_pt.append(particRatioPercent)
            if i == 1 and j == 0:
                ventral_speech.append(particRatioPercent)
            if i == 1 and j == 1:
                ventral_am.append(particRatioPercent)
            if i == 1 and j == 2:
                ventral_pt.append(particRatioPercent)


# Statistical Comparison using Mann-Whitney-U test
results = []
arrays = {
    'Primary auditory area - speech': primary_speech,
    'Primary auditory area - AM': primary_am,
    'Primary auditory area - PT': primary_pt,
    'Ventral auditory area - speech': ventral_speech,
    'Ventral auditory area - AM': ventral_am,
    'Ventral auditory area - PT': ventral_pt}
combinations = list(arrays.keys())

for i, combo1 in enumerate(combinations):
    for j, combo2 in enumerate(combinations):
        if j <= i:
            continue  # Avoid duplicate comparisons

        # Determine if they share brain area or sound type
        combo1_parts = combo1.split(' - ')
        combo2_parts = combo2.split(' - ')
        same_brain_area = combo1_parts[0] == combo2_parts[0]
        same_sound_type = combo1_parts[1] == combo2_parts[1]

        # Only perform the Mann-Whitney U test if they share at least one category
        if not (same_brain_area or same_sound_type):
            continue

        # Perform Mann-Whitney U test
        stat, p_value = stats.mannwhitneyu(arrays[combo1], arrays[combo2], alternative='two-sided')

        # Determine the type of comparison
        comparison_type = 'brain_area' if not same_sound_type else 'sound_type'

        # Check significance
        significant = funcs.check_significance(p_value, comparison_type)

        # Append results
        results.append({
            'Combo 1': combo1,
            'Combo 2': combo2,
            'P-value': p_value,
            'Statistically Significant': significant
        })

# Create DataFrame from results
results_df = pd.DataFrame(results)
print(results_df)