import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from jaratoolbox import celldatabase, settings
from sklearn.decomposition import PCA
import umap
import funcs
import seaborn as sns
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
targetSiteNames = ["Primary auditory area", "Ventral auditory area"] # "Dorsal auditory area",
leastCellsArea = 10000
evoked_start = 0.015
evoked_end = 0.3
pt_evoked_end = 0.1
binWidth = 0.01
binEdges = np.arange(evoked_start, evoked_end, binWidth)
binEdgesPT = np.arange(evoked_start, pt_evoked_end, binWidth)
max_trials = {'PT': 640, 'AM': 220, 'speech': 381}
data_dict = {}
indices_AM = None
previous_frequency_speech = None
previous_frequency_AM = None
previous_frequency_PT = None

# %% Load dataframe
databaseDir = os.path.join(settings.DATABASE_PATH, '2022paspeech')
fullDbPath = 'fulldb_speech_tuning.h5'
fullPath = os.path.join(databaseDir, fullDbPath)
fullDb = celldatabase.load_hdf(fullPath)
simpleSiteNames = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])
fullDb["recordingSiteName"] = simpleSiteNames

# %% Add data to the dictionary for each brain area and sound type
for subject in subject_list:
    for brain_area in targetSiteNames:

        X_speech_array, X_AM_array, X_PT_array, \
            Y_brain_area_speech_all, Y_brain_area_AM_all, Y_brain_area_PT_all, \
            Y_frequency_speech_sorted, Y_frequency_AM_sorted, Y_frequency_pureTones_sorted \
            = funcs.clean_and_concatenate(subject, recordingDate_list, brain_area,
                                          previous_frequency_AM, previous_frequency_PT,
                                          previous_frequency_speech)

        Y_frequency_FT = Y_frequency_speech_sorted[:,0]
        Y_frequency_VOT = Y_frequency_speech_sorted[:, 1]

        for sound_type, X_array, Y_brain_area_all, Y_frequency_sorted in zip(
                ['FT', 'VOT'],
                [X_speech_array, X_speech_array],
                [Y_brain_area_speech_all, Y_brain_area_speech_all],
                [Y_frequency_FT, Y_frequency_VOT]):

            brain_area_array = np.array(Y_brain_area_all)
            X_array_adjusted = X_array[brain_area_array == brain_area]
            X_array_adjusted = X_array_adjusted.T
            data_dict[(subject, brain_area, sound_type)] = {'X': X_array_adjusted, 'Y': Y_frequency_sorted}

# %% PCA Plotting
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
fig.suptitle('2D PCA Plots for Different Brain Areas and Sound Types', fontsize=16)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, brain_area in enumerate(targetSiteNames):
    for j, sound_type in enumerate(['FT', 'VOT']):
        data = data_dict.get((subject, brain_area, sound_type))

        title = f'{brain_area} - {sound_type}, n = {data["X"].shape[0]}'
        funcs.plot_2d_pca(axes[i, j], data, data["Y"], title)

# Save and show
output_path = "/Users/zoetomlinson/Desktop/neuronalDataResearch/Figures/Population Plots/Speech_Separated_PCA.png"
fig.savefig(output_path)
plt.tight_layout()
plt.show()


# %% UMAP Plotting
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
fig.suptitle('2D UMAP Projections for Different Brain Areas and Sound Types', fontsize=16)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, brain_area in enumerate(targetSiteNames):
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
output_path = "/Users/zoetomlinson/Desktop/neuronalDataResearch/Figures/Population Plots/Speech_Separated_UMAP.png"
fig.savefig(output_path)
plt.tight_layout()
plt.show()