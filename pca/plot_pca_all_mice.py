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
import studyparams as params, funcs

data_dict = {}
figSavePath = params.figSavePath + "/PCA"

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

# Figure subplots
fig_scree, axes_scree = plt.subplots(3, 3, figsize=(22, 16))
fig_scree.suptitle('Scree Plots for Different Brain Areas and Sound Types', fontsize=16)
fig_scree.subplots_adjust(hspace=0.4, wspace=0.4)

# Plot Scree plots for each combination
for i, brain_area in enumerate(params.targetSiteNames):
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
        funcs.plot_scree_plot(axes_scree[i, j], data_standardized, title, particRatio)

# Save Scree plots figure
fig_scree.show()
fig_scree.savefig(params.saveFig)

# Create a 3x3 grid for 2D PCA subplots
fig_pca, axes_pca = plt.subplots(3, 3, figsize=(22, 16))
fig_pca.suptitle('2D PCA Plots for Different Brain Areas and Sound Types', fontsize=16)
fig_pca.subplots_adjust(hspace=0.4, wspace=0.4)

for i, brain_area in enumerate(params.targetSiteNames):
    for j, sound_type in enumerate(['speech', 'AM', 'PT']):
        data = data_dict.get((brain_area, sound_type), None)
        title = f'{brain_area} - {sound_type}, n = {data["X"].shape[1]}'

        scaler = StandardScaler()
        data = scaler.fit_transform(data['X'])

        # For 'speech' sound type, create a mapping of frequencies to numbers
        if sound_type == 'speech':
            Y_labels = [tuple(row) for row in data["Y"]]
            label_to_number = {label: idx for idx, label in enumerate(params.unique_labels)}
            color_values = np.array([label_to_number[label] for label in Y_labels])
            funcs.plot_2d_pca(axes_pca[i, j], data, color_values, title)

        # For 'AM' sound type, directly use the 'Y' values
        elif sound_type == 'AM':
            funcs.plot_2d_pca(axes_pca[i, j], data, data["Y"], title)

        # For 'PT' sound type, apply log10 transformation to 'Y'
        elif sound_type == 'PT':
            funcs.plot_2d_pca(axes_pca[i, j], data, np.log10(data["Y"]), title)

# Save as pngs
fig_pca.savefig(params.figSavePath + "2D_PCA_Plots.png")
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
                label_to_number = {label: idx for idx, label in enumerate(params.unique_labels)}
                color_values = np.array([label_to_number.get(label, -1) for label in Y_labels])
                funcs.plot_2d_pca(axes_pca[i, j], data_neurons, color_values, title)

            # For 'AM' sound type, directly use the 'Y' values
            elif sound_type == 'AM':
                funcs.plot_2d_pca(axes_pca[i, j], data_neurons, data_neurons["Y"], title)

            # For 'PT' sound type, apply log10 transformation to 'Y'
            elif sound_type == 'PT':
                funcs.plot_2d_pca(axes_pca[i, j], data_neurons, np.log10(data_neurons["Y"]), title)

# Save as PNG
fig_pca_subset111.savefig(params.figSavePath + "2D_PCA_Subset111Neurons_Plots.png")
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
            funcs.plot_scree_plot(axes_scree[i, j], data_standardized, title, y_max, particRatio)

# Save Scree plots figure
plt.show()
fig_scree_subset111.savefig(params.figSavePath + "PopScreePlots_Subset111Neurons.png")
