"""
Purpose
- Visualize speech-evoked neural population structure using dimensionality reduction.
- Compare Frequency (FT) vs Voice Onset Time (VOT) representations across brain areas.

Inputs
- Spike-rate matrices for speech stimuli (FTVOTBorders condition)
- Trial labels: brain area, frequency (FT), VOT
- Subject and session metadata from studyparams
- Preprocessed HDF5 recording database

Processing
- Load speech data across subjects, dates, and brain areas
- Equalize trial counts across FT/VOT frequency combinations
- Sort trials to ensure consistent frequency ordering
- Split speech trials into FT and VOT label sets
- Perform 2D PCA projections for visualization
- Perform 2D UMAP projections for visualization

Outputs
- 2D PCA plots for FT and VOT, by brain area
- 2D UMAP plots for FT and VOT, by brain area
- Saved figures for exploratory and presentation use
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from .. import funcs, studyparams as params
from jaratoolbox import celldatabase
#TODO: dont just select the first 111, randomly select

# %% Load dataframe
fullDb = celldatabase.load_hdf(params.fullPath)
simpleSiteNames = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])
simpleSiteNames = simpleSiteNames.replace("Posterior auditory area", "Dorsal auditory area")
fullDb['recordingSiteName'] = simpleSiteNames
counts = fullDb.groupby('recordingSiteName').size()
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
for subject in params.subject_list:
    for date in params.recordingDate_list[subject]:
        for targetSiteName in params.targetSiteNames:
            # Load and process data for Speech
            speechEnsemble, speechEphys, speechBdata = funcs.load_data(subject, date, targetSiteName, "FTVOTBorders")
            if speechEnsemble:
                X_speech, Y_brain_area_speech, Y_frequency_speech = funcs.spike_rate(
                    "speech", speechEnsemble, speechEphys, speechBdata, targetSiteName)

                # Increment the neuron count for this subject
                params.neuron_counts[subject] += X_speech[0].shape[0]  # Count number of neurons (rows)

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

# Create a pandas DataFrame from the neuron counts dictionary
neuron_counts_df = pd.DataFrame(list(params.neuron_counts.items()), columns=['Subject', 'Neuron_Count'])

# Apply sorting to the X arrays
X_AM_sorted = funcs.sort_x_arrays(X_AM_all, indices_AM, "am")
X_PT_sorted = funcs.sort_x_arrays(X_pureTones_all, indices_PT, "pt")
X_speech_sorted = funcs.sort_x_arrays(X_speech_all, indices_speech, "speech")

# Concatenate the sorted arrays
X_speech_array = np.concatenate(X_speech_sorted, axis=0)
X_AM_array = np.concatenate(X_AM_sorted, axis=0)
X_PT_array = np.concatenate(X_PT_sorted, axis=0)

data_dict = {}

# %% Add data to the dictionary for each brain area and sound type
for brain_area in params.targetSiteNames:
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


def calculate_participation_ratio_60_neurons(data):
    X = data['X']  # NumPy array (trials, neurons)
    num_neurons = X.shape[1]  # Number of neurons

    selected_indices = np.random.choice(num_neurons, 60, replace=False)  # Select 60 neurons
    X_n_60 = X[:, selected_indices]  # Slice correctly

    # Standardize data
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(X_n_60)  # No transpose needed

    # Perform PCA
    pca = PCA()
    pca.fit(data_standardized)
    explained_variance_ratio = pca.explained_variance_ratio_

    # Compute participation ratio
    particRatio = funcs.calculate_participation_ratio(explained_variance_ratio)

    # Store updated data
    updated_data = data.copy()
    updated_data['X'] = X_n_60  # Update X to reflect the 60 selected neurons
    updated_data['Participation_Ratio'] = particRatio

    return particRatio, updated_data


# %% Statistical test
num_iterations = 1000
speech_prim_org_prs, speech_dor_org_prs, speech_ven_org_prs = [], [], []
am_prim_org_prs, am_dor_org_prs, am_ven_org_prs = [], [], []
pt_prim_org_prs, pt_dor_org_prs, pt_ven_org_prs = [], [], []
speech_prim_shuf_prs, am_prim_shuf_prs, pt_prim_shuf_prs = [], [], []
speech_dor_shuf_prs, am_dor_shuf_prs, pt_dor_shuf_prs = [], [], []
speech_ven_shuf_prs, am_ven_shuf_prs, pt_ven_shuf_prs = [], [], []

for i, sound_type in enumerate(['speech', 'AM', 'PT']):
    # Select 60 neurons for one brain area and compute PR over 1000 iterations
    for iteration in range(num_iterations):

        all_X, all_Y, brain_labels, sound_PRs = [], [], [], []

        for brain_area in ["Primary auditory area", "Dorsal auditory area", "Ventral auditory area"]:
            data = data_dict[(brain_area, sound_type)]
            particRatio, updated_data = calculate_participation_ratio_60_neurons(data)  # Calculate PR for 60 neurons

            # Append PR for all 3 brain areas to list for sound type
            if sound_type == 'speech':
                if brain_area == 'Primary auditory area':
                    speech_prim_org_prs.append(particRatio)
                elif brain_area == 'Dorsal auditory area':
                    speech_dor_org_prs.append(particRatio)
                else:
                    speech_ven_org_prs.append(particRatio)
            elif sound_type == 'AM':
                if brain_area == 'Primary auditory area':
                    am_prim_org_prs.append(particRatio)
                elif brain_area == 'Dorsal auditory area':
                    am_dor_org_prs.append(particRatio)
                else:
                    am_ven_org_prs.append(particRatio)
            else:
                if brain_area == 'Primary auditory area':
                    pt_prim_org_prs.append(particRatio)
                elif brain_area == 'Dorsal auditory area':
                    pt_dor_org_prs.append(particRatio)
                else:
                    pt_ven_org_prs.append(particRatio)

            all_X.append(updated_data['X'])  # Store X (trials, 60 neurons) --> (trials, 180) for one sound
            all_Y.append(updated_data['Y'])  # Store Y (trials,)
            brain_labels.extend([brain_area] * 60)  # Track brain areas (60, ) --> (180, ) for one sound

            # Combine all selected neurons into one array of shape (trials, 180 neurons)
            combined_X = np.hstack(all_X)  # Concatenate along neuron axis
            combined_Y = np.hstack(all_Y)  # All sound types share the same trial order

            # Shuffle brain area labels and recalculate PR
            shuffled_labels = np.random.permutation(brain_labels)  # Create a shuffled copy

            # Distribute neurons based on shuffled brain labels
            shuffled_X_dict = {"Primary auditory area": [],
                               "Dorsal auditory area": [],
                               "Ventral auditory area": []}

            # TODO: use the shuffled brain labels to resort them into brains and calculate the participation ratios agaun for statistical test

            # Reassign neurons into shuffled brain areas
            for neuron_idx, label in enumerate(shuffled_labels):
                shuffled_X_dict[label].append(combined_X[:, neuron_idx])  # Assign neurons to the correct area

        for key in shuffled_X_dict:
            shuffled_X_dict[key] = np.column_stack(shuffled_X_dict[key])  # Stack along neuron axis

        # Calculate PR for each shuffled brain area and store results
        for brain_area in ["Primary auditory area", "Dorsal auditory area", "Ventral auditory area"]:
            shuffled_particRatio, _ = calculate_participation_ratio_60_neurons({"X": shuffled_X_dict[brain_area]})

            if sound_type == 'speech':
                if brain_area == 'Primary auditory area':
                    speech_prim_shuf_prs.append(shuffled_particRatio)
                elif brain_area == 'Dorsal auditory area':
                    speech_dor_shuf_prs.append(shuffled_particRatio)
                else:
                    speech_ven_shuf_prs.append(shuffled_particRatio)
            elif sound_type == 'AM':
                if brain_area == 'Primary auditory area':
                    am_prim_shuf_prs.append(shuffled_particRatio)
                elif brain_area == 'Dorsal auditory area':
                    am_dor_shuf_prs.append(shuffled_particRatio)
                else:
                    am_ven_shuf_prs.append(shuffled_particRatio)
            else:
                if brain_area == 'Primary auditory area':
                    pt_prim_shuf_prs.append(shuffled_particRatio)
                elif brain_area == 'Dorsal auditory area':
                    pt_dor_shuf_prs.append(shuffled_particRatio)
                else:
                    pt_ven_shuf_prs.append(shuffled_particRatio)

# Perform Mann-Whitney U test
mwu_speech_prim = mannwhitneyu(speech_prim_org_prs, speech_prim_shuf_prs, alternative='two-sided')
mwu_speech_dor = mannwhitneyu(speech_dor_org_prs, speech_dor_shuf_prs, alternative='two-sided')
mwu_speech_ven = mannwhitneyu(speech_ven_org_prs, speech_ven_shuf_prs, alternative= 'two-sided')
mwu_am_prim = mannwhitneyu(am_prim_org_prs, am_prim_shuf_prs, alternative='two-sided')
mwu_am_dor = mannwhitneyu(am_dor_org_prs, am_dor_shuf_prs, alternative='two-sided')
mwu_am_ven = mannwhitneyu(am_ven_org_prs, am_ven_shuf_prs, alternative='two-sided')
mwu_pt_prim = mannwhitneyu(pt_prim_org_prs, pt_prim_shuf_prs, alternative='two-sided')
mwu_pt_dor = mannwhitneyu(pt_dor_org_prs, pt_dor_shuf_prs, alternative='two-sided')
mwu_pt_ven = mannwhitneyu(pt_ven_org_prs, pt_ven_shuf_prs, alternative='two-sided')

# Print results
print(f"Speech-Pimary MWU Test: U={mwu_speech_prim.statistic}, p={mwu_speech_prim.pvalue}")
print(f"Speech-Dorsal MWU Test: U={mwu_speech_dor.statistic}, p={mwu_speech_dor.pvalue}")
print(f"Speech-Ventral MWU Test: U={mwu_speech_ven.statistic}, p={mwu_speech_ven.pvalue}")
print(f"AM-Pimary MWU Test: U={mwu_am_prim.statistic}, p={mwu_am_prim.pvalue}")
print(f"AM-Dorsal MWU Test: U={mwu_am_dor.statistic}, p={mwu_am_dor.pvalue}")
print(f"AM-Ventral MWU Test: U={mwu_am_ven.statistic}, p={mwu_am_ven.pvalue}")
print(f"PT-Pimary MWU Test: U={mwu_pt_prim.statistic}, p={mwu_pt_prim.pvalue}")
print(f"PT-Dorsal MWU Test: U={mwu_pt_dor.statistic}, p={mwu_pt_dor.pvalue}")
print(f"PT-Ventral MWU Test: U={mwu_pt_ven.statistic}, p={mwu_pt_ven.pvalue}")

# Organize data for plotting
org_data = [speech_prim_org_prs, speech_dor_org_prs, speech_ven_org_prs,
            am_prim_org_prs, am_dor_org_prs, am_ven_org_prs,
            pt_prim_org_prs, pt_dor_org_prs, pt_ven_org_prs]

shuf_data = [speech_prim_shuf_prs, speech_dor_shuf_prs, speech_ven_shuf_prs,
             am_prim_shuf_prs, am_dor_shuf_prs, am_ven_shuf_prs,
             pt_prim_shuf_prs, pt_dor_shuf_prs, pt_ven_shuf_prs]

# Create figure and axes
fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey=True)
fig.suptitle('Participation Ratios: Original vs. Shuffled')
sound_types = ['speech', 'AM', 'PT']
brain_areas = ['Primary auditory area', 'Dorsal auditory area', 'Ventral auditory area']

# Loop through the grid and plot
for i, ax in enumerate(axes.flat):
    sns.histplot(org_data[i], color='blue', alpha=0.6, label='Original', ax=ax, kde=True)
    sns.histplot(shuf_data[i], color='red', alpha=0.6, label='Shuffled', ax=ax, kde=True)
    ax.set_title(f'{sound_types[i // 3]} - {brain_areas[i % 3]}')
    ax.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig("/Users/zoetomlinson/Desktop/GitHub/neuronalDataResearch/Figures/Population Plots/PR_MWU_Test_plt.png")
plt.show()
