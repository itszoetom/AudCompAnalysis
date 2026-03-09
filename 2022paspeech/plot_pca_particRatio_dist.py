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
import studyparams as params, funcs
from jaratoolbox import celldatabase, settings

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
