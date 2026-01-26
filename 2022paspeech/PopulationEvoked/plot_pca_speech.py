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
import matplotlib.pyplot as plt
import os
from jaratoolbox import celldatabase, settings
import umap
from .. import funcs, studyparams
# %% Add data to the dictionary for each brain area and sound type
data_dict = {}
for subject in studyparams.subject_list:
    for brain_area in studyparams.targetSiteNames:
        for date in studyparams.recordingDate_list[subject]:

            # Load and process data for Speech
            speechEnsemble, speechEphys, speechBdata = funcs.load_data(subject, date, brain_area, "FTVOTBorders")
            if speechEnsemble:
                X_speech, Y_brain_area_speech, Y_frequency_speech = funcs.spike_rate(
                    "speech", speechEnsemble, speechEphys, speechBdata, brain_area)

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
                    print(f'Not enough speech trials for subject {subject}, in brain area {brain_area}')
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

            Y_frequency_FT = Y_frequency_speech_sorted[:,0]
            Y_frequency_VOT = Y_frequency_speech_sorted[:, 1]

            for sound_type, X_array, Y_brain_area_all, Y_frequency_sorted in zip(
                    ['FT', 'VOT'],
                    [X_speech, X_speech],
                    [Y_brain_area_speech, Y_brain_area_speech],
                    [Y_frequency_FT, Y_frequency_VOT]):

                brain_area_array = np.array(Y_brain_area_all)
                X_array_adjusted = X_array[brain_area_array == brain_area]
                X_array_adjusted = X_array_adjusted.T
                data_dict[(subject, brain_area, sound_type)] = {'X': X_array_adjusted, 'Y': Y_frequency_sorted}

# %% PCA Plotting
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
fig.suptitle('2D PCA Plots for Different Brain Areas and Sound Types', fontsize=16)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, brain_area in enumerate(studyparams.targetSiteNames):
    for j, sound_type in enumerate(['FT', 'VOT']):
        data = data_dict.get((subject, brain_area, sound_type))

        title = f'{brain_area} - {sound_type}, n = {data["X"].shape[0]}'
        funcs.plot_2d_pca(axes[i, j], data, data["Y"], title)

# Save and show
output_path = studyparams.figSavePath + "Population/PCA/speech_dataset/pca_speech_ft_vot.png"
fig.savefig(output_path)
plt.tight_layout()
plt.show()

# %% UMAP Plotting
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
fig.suptitle('2D UMAP Projections for Different Brain Areas and Sound Types', fontsize=16)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, brain_area in enumerate(studyparams.targetSiteNames):
    for j, sound_type in enumerate(['FT', 'VOT']):
        data = data_dict.get((subject, brain_area, sound_type))

        if data:
            reducer = umap.UMAP(n_components=2, random_state=42)
            transformed_data = reducer.fit_transform(data["X"])
            scatter = axes[i, j].scatter(transformed_data[:, 0], transformed_data[:, 1],
                                         c=data["Y"], cmap='viridis', s=32)
            title = f'{brain_area} - {sound_type}, n = {data["X"].shape[0]}'
            axes[i, j].set_title(title)
            axes[i, j].set_xlabel('UMAP 1')
            axes[i, j].set_ylabel('UMAP 2')
            plt.colorbar(scatter, ax=axes[i, j], orientation='vertical')

# Save and show
output_path = studyparams.figSavePath + "Population/PCA/speech_dataset/Speech_Separated_UMAP.png"
fig.savefig(output_path)
plt.tight_layout()
plt.show()