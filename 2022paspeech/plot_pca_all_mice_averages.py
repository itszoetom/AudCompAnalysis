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
import funcs, studyparams as params

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


# Create a 3x3 grid for subplots
fig_scree, axes_scree = plt.subplots(3, 3, figsize=(22, 16))
fig_scree.suptitle('Scree Plots for Different Brain Areas and Sound Types', fontsize=16)
fig_scree.subplots_adjust(hspace=0.4, wspace=0.4)

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
        funcs.plot_scree_plot(axes_scree[i, j], data, title, particRatio)

# Save Scree plots figure
fig_scree.show()
fig_scree.savefig(params.figSavePath + "  ")

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
fig_pca.savefig(params.figSavePath + "2D_PCA_Average_Plots.png")
plt.show()