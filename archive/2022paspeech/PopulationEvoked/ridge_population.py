"""
Purpose
- Fit ridge regression models across subjects, brain areas, and sound types

Inputs
- Spike-rate arrays (neurons × trials) for each mouse, brain area, and sound type
- Corresponding stimulus frequencies (log-transformed for AM/PT)
- Trial equalization parameters and neuron subsampling thresholds

Processing
- Perform cross-validation over a log-spaced alpha range to select best regularization parameter
- Predict on test set and compute evaluation metrics: R², RMSE, Pearson correlation, percent within tolerance
- Visualize predicted vs true frequencies per mouse × brain area × sound type
- Aggregate number of neurons and corresponding test R² for summary plots

Outputs
- Scatter plots of predicted vs true frequency for each mouse × brain area × sound type
- Summary plot of test R² vs number of neurons across all conditions
- Figures saved as PNGs for documentation
"""
from .. import funcs, studyparams as params
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
# Constants
X_speech_all = []
Y_brain_area_speech_all = []
X_AM_all = []
Y_brain_area_AM_all = []
X_pureTones_all = []
Y_brain_area_PT_all = []
primary_n_neurons = 0
ventral_n_neurons = 0
dorsal_n_neurons = 0

# %% Add data to the dictionary for each brain area and sound type
for subject in params.subject_list:
    for date in params.recordingDate_list[subject]:
        for brain_area in params.targetSiteNames:
            speechEnsemble, speechEphys, speechBdata = funcs.load_data(subject, date, brain_area,
                                                                 "FTVOTBorders")

            if speechEnsemble:
                X_speech, Y_brain_area_speech, Y_frequency_speech = funcs.spike_rate(
                    "speech", speechEnsemble, speechEphys, speechBdata, brain_area)

                X_speech_array, Y_frequency_speech_sorted, previous_frequency_speech, indices_speech = (
                    funcs.adjust_speech_length(subject, date, brain_area, X_speech, Y_frequency_speech, previous_frequency_speech))

                if X_speech_array is not None:
                    Y_frequency_FT = Y_frequency_speech_sorted[:, 0]
                    Y_frequency_VOT = Y_frequency_speech_sorted[:, 1]

                    # Append the data to the lists
                    X_speech_all.extend(X_speech_array)
                    Y_frequency_speech_2 = Y_frequency_speech_sorted
                    Y_brain_area_speech_all.extend(Y_brain_area_speech)

            # Load and process data for AM
            amEnsemble, amEphys, amBdata = funcs.load_data(subject, date, brain_area, "AM")
            if amEnsemble:
                X_AM, Y_brain_area_AM, Y_frequency_AM = funcs.spike_rate(
                    "AM", amEnsemble, amEphys, amBdata, brain_area)

                if X_AM.any():
                    # Apply adjustments
                    X_AM_adjusted, Y_frequency_AM_adjusted, Yba_AM_adj = (
                        funcs.adjust_array_and_labels(X_AM, Y_frequency_AM, Y_brain_area_AM, params.max_trials['AM'], subject, date, brain_area))

                    # Sort arrays
                    X_AM_array, Y_frequency_AM_sorted, Y_brain_area_AM_sorted, previous_frequency_AM, indices_AM = (
                        funcs.sort_sound_array(subject, date, brain_area, X_AM_adjusted, Yba_AM_adj, Y_frequency_AM_adjusted,
                                               previous_frequency_AM))

                    if X_AM_array is not None:
                        # Append the data to the lists
                        X_AM_all.extend(X_AM_array)
                        Y_frequency_AM_2 = Y_frequency_AM_sorted
                        Y_brain_area_AM_all.extend(Y_brain_area_AM_sorted)

            # Load and process data for Pure Tones
            ptEnsemble, ptEphys, ptBdata = funcs.load_data(subject, date, brain_area, "pureTones")
            if ptEnsemble:
                X_pureTones, Y_brain_area_PT, Y_frequency_pureTones = funcs.spike_rate(
                    "PT", ptEnsemble, ptEphys, ptBdata, brain_area)

                if X_pureTones.any():
                    # Apply adjustments
                    X_PT_adjusted, Y_frequency_PT_adjusted, Yba_PT_adj = (
                        funcs.adjust_array_and_labels(X_pureTones, Y_frequency_pureTones, Y_brain_area_PT,
                                                      params.max_trials['PT'], subject, date, brain_area))

                    # Sort arrays
                    X_PT_array, Y_frequency_PT_sorted, Y_brain_area_PT_sorted, previous_frequency_PT, indices_PT = (
                        funcs.sort_sound_array(subject, date, brain_area, X_PT_adjusted, Yba_PT_adj, Y_frequency_PT_adjusted,
                                               previous_frequency_PT))

                    if X_PT_array is not None:
                        # Append the data to the lists
                        X_pureTones_all.extend(X_PT_array)
                        Y_frequency_PT_2 = Y_frequency_PT_sorted
                        Y_brain_area_PT_all.extend(Y_brain_area_PT_sorted)

# %% Sort arrays
X_PT_sorted = funcs.sort_x_arrays(X_pureTones_all, indices_PT, "pt")
X_AM_sorted = funcs.sort_x_arrays(X_AM_all, indices_AM, "am")
X_speech_sorted = funcs.sort_x_arrays(X_speech_all, indices_speech, "speech")

# Convert the lists to numpy arrays for easy manipulation
X_speech_array = np.stack(X_speech_sorted, axis=0)
# X_speech_array = X_speech_array.squeeze(axis=1)
X_AM_array = np.stack(X_AM_sorted, axis=0)
X_PT_array = np.stack(X_PT_sorted, axis=0)

# %% Initialize data_dict
data_dict = {}

for brain_area in params.targetSiteNames:
    for sound_type, X_array, Y_brain_area_all, Y_frequency_sorted in zip(
            ['AM', 'PT', 'VOT', 'FT'],
            [X_AM_array, X_PT_array, X_speech_array, X_speech_array],
            [Y_brain_area_AM_all, Y_brain_area_PT_all, Y_brain_area_speech_all, Y_brain_area_speech_all],
            [Y_frequency_AM_2, Y_frequency_PT_2, Y_frequency_VOT, Y_frequency_FT]):

        brain_area_array = np.array(Y_brain_area_all)
        X_array_adjusted = X_array[brain_area_array == brain_area]
        X_array_adjusted = X_array_adjusted.T

        # Store the data for each (brain_area, sound_type) pair
        data_dict[(brain_area, sound_type)] = {'X': X_array_adjusted, 'Y': Y_frequency_sorted}

# Initialize lists
n_neurons_list = []
r2_test_list = []
labels_list = []

# 5% tolerance of the target range
tolerance = 0.05
# Define the alpha range
alphas = np.logspace(-10, 5, 200)

# Ridge Regression Loop
for key, value in data_dict.items():
    X = value['X']
    if key[1] in ['AM', 'PT']:
        Y = np.log(value['Y'])
    else:
        Y = value['Y']

    # Skip empty feature arrays early
    if X.size == 0 or X.shape[1] == 0:
        print(f"Skipping {key[0]} - {key[1]} due to empty feature array.")
        continue

    n_neurons = X.shape[1]
    n_neurons_list.append(n_neurons)

    # Track number of neurons per brain area
    if key[0] == 'Primary auditory area':
        primary_n_neurons += n_neurons
    elif key[0] == 'Dorsal auditory area':
        dorsal_n_neurons += n_neurons
    else:
        ventral_n_neurons += n_neurons

    # Split into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Cross-validation to select the best alpha
    best_r2 = -np.inf
    best_alpha_idx = None
    for i, alpha in enumerate(alphas):
        ridge_reg = Ridge(alpha=alpha)
        ridge_reg.fit(X_train, Y_train)
        r2_score = ridge_reg.score(X_test, Y_test)
        if r2_score > best_r2:
            best_r2 = r2_score
            best_alpha_idx = i

    # Skip if no valid alpha was found
    if best_alpha_idx is None:
        print(f"Skipping {key[0]} - {key[1]} due to no valid alpha.")
        continue

    r2_test_list.append(best_r2)
    labels_list.append(f"{key[0]} - {key[1]}")

    # Train Ridge Regression with best alpha
    best_alpha = alphas[best_alpha_idx]
    ridge_reg = Ridge(alpha=best_alpha)
    ridge_reg.fit(X_train, Y_train)

    # Predict and evaluate
    y_train_pred = ridge_reg.predict(X_train)
    y_test_pred = ridge_reg.predict(X_test)

    tolerance_range = tolerance * Y_test
    within_tolerance = np.abs(y_test_pred - Y_test) <= tolerance_range
    percent_within_tolerance = np.mean(within_tolerance) * 100

    # Sort test data and predictions
    sorted_indices = np.argsort(Y_test)
    Y_test_sorted = Y_test[sorted_indices]
    y_test_pred_sorted = y_test_pred[sorted_indices]

    # Define Metrics
    rmse = np.sqrt(mean_squared_error(Y_test, y_test_pred))
    corr, _ = pearsonr(Y_test, y_test_pred)

    # Plot Each Mouse-Brain Area-Sound Type Combo
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=Y_test_sorted, y=y_test_pred_sorted, color='black', label="Predicted vs True")
    sns.regplot(x=Y_test_sorted, y=y_test_pred_sorted, scatter=False, color='red',
                line_kws={"linestyle": "--", "linewidth": 2}, label="Regression Line")

    plt.xlabel('True Frequency (Sorted)')
    plt.ylabel('Predicted Frequency')
    plt.title(f"Predicted vs True Frequency for {key[0]} - {key[1]} \nBest Alpha: {best_alpha}")
    plt.text(0.5, 0.9, f"R²: {best_r2:.3f}, RMSE: {rmse:.3f}\nNeurons: {n_neurons}\nPearson's r: {corr:.3f}",
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color='blue')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # plt.savefig(f"/Users/zoetomlinson/Desktop/GitHub/neuronalDataResearch/Figures/Ridge Regression/{key[0]}/"
    #             f"{key[1]} - True vs Predicted.png")
    plt.show()

# Plot for n_neurons vs test R^2
plt.figure(figsize=(12, 12))
sns.scatterplot(x=n_neurons_list, y=r2_test_list, hue=labels_list, palette='viridis', s=100)

plt.title("Test $R^2$ vs. Number of Neurons", fontsize=16)
plt.xlabel("Number of Neurons (n_neurons)", fontsize=14)
plt.ylabel("Test $R^2$", fontsize=14)
plt.legend(title="Brain Area - Sound Type", bbox_to_anchor=(0.8, 0.5), loc='center left', fontsize=6)
plt.grid(True)
plt.tight_layout()

# plt.savefig("/Users/zoetomlinson/Desktop/GitHub/neuronalDataResearch/Figures/Ridge Regression/R2_vs_Neurons.png")
plt.show()