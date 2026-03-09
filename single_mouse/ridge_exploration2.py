"""
Initial ridge regression exploration with evaluation metrics:
- R², Mean Squared Error (MSE), percent correct within 5% tolerance
- Identify best alpha based on highest test R²
- Visualize R², MSE, and percent correct vs alpha
- Visualize predicted vs test points with connecting lines for each mouse × brain area × sound type

Outputs
- Line plots of R², MSE, and percent correct vs alpha
- Scatter plots showing test points vs predicted points with connecting lines
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from jaratoolbox import celldatabase, settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .. import funcs, studyparams as params

# %% Constants
subject_list = ['feat005', 'feat006', 'feat007', 'feat008', 'feat009'] # 'feat004', 'feat010'
data_dict = {}

# %% Load dataframe
databaseDir = os.path.join(settings.DATABASE_PATH, '2022paspeech')
fullDbPath = 'fulldb_speech_tuning.h5'
fullPath = os.path.join(databaseDir, fullDbPath)
fullDb = celldatabase.load_hdf(fullPath)
simpleSiteNames = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])
fullDb["recordingSiteName"] = simpleSiteNames

X_speech_array, X_AM_array, X_PT_array,\
    Y_brain_area_speech_all, Y_brain_area_AM_all, Y_brain_area_PT_all, \
    Y_frequency_speech_sorted, Y_frequency_AM_sorted, Y_frequency_pureTones_sorted \
    = funcs.clean_and_concatenate(subject_list, params.recordingDate_list, params.targetSiteNames, None, None, None)

# %% Add data to the dictionary for each brain area and sound type
for subject in subject_list:
    for brain_area in ["Primary auditory area", "Ventral auditory area"]:  # Removed speech sounds for now
        for sound_type, X_array, Y_brain_area_all, Y_frequency_sorted in zip(
                ['AM', 'PT'],
                [X_AM_array, X_PT_array],
                [Y_brain_area_AM_all, Y_brain_area_PT_all],
                [Y_frequency_AM_sorted, Y_frequency_pureTones_sorted]):
            brain_area_array = np.array(Y_brain_area_all)
            X_array_adjusted = X_array[brain_area_array == brain_area]
            X_array_adjusted = X_array_adjusted.T
            data_dict[(subject, brain_area, sound_type)] = {'X': X_array_adjusted, 'Y': Y_frequency_sorted}

# Define the alpha range with a wider spectrum for potentially more variation
alphas = np.logspace(-2, 3, 100)  # Extend lambda values

# Define tolerance level for "percent correct" evaluation
tolerance = 0.05  # 5% tolerance of the target range
# %% Main loop
for key, value in data_dict.items():
    X = value['X']
    Y = value['Y']

    # Standardize X to improve Ridge performance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Store R-squared, MSE, and percent correct for each alpha
    train_acc = np.empty_like(alphas)
    test_acc = np.empty_like(alphas)
    train_mse = np.empty_like(alphas)
    test_mse = np.empty_like(alphas)
    train_percent_correct = np.empty_like(alphas)
    test_percent_correct = np.empty_like(alphas)

    for i, alpha in enumerate(alphas):
        ridge_reg = Ridge(alpha=alpha)
        ridge_reg.fit(X_train, Y_train)

        # R-squared values
        train_acc[i] = ridge_reg.score(X_train, Y_train)
        test_acc[i] = ridge_reg.score(X_test, Y_test)

        # Mean Squared Error (MSE) values
        train_mse[i] = mean_squared_error(Y_train, ridge_reg.predict(X_train))
        test_mse[i] = mean_squared_error(Y_test, ridge_reg.predict(X_test))

        # Percent correct
        y_train_pred = ridge_reg.predict(X_train)
        y_test_pred = ridge_reg.predict(X_test)
        train_range = Y_train.max() - Y_train.min()
        test_range = Y_test.max() - Y_test.min()

        train_percent_correct[i] = np.mean(np.abs(Y_train - y_train_pred) <= tolerance * train_range) * 100
        test_percent_correct[i] = np.mean(np.abs(Y_test - y_test_pred) <= tolerance * test_range) * 100

    # Identify the best alpha with the highest R-squared on the test set
    best_alpha_idx = np.argmax(test_acc)
    best_alpha = alphas[best_alpha_idx]
    best_train_mse = train_mse[best_alpha_idx]
    best_test_mse = test_mse[best_alpha_idx]
    best_train_r2 = train_acc[best_alpha_idx]
    best_test_r2 = test_acc[best_alpha_idx]
    best_train_pc = train_percent_correct[best_alpha_idx]
    best_test_pc = test_percent_correct[best_alpha_idx]

    # Print the results
    print(f"Model for {key[0]} (brain area: {key[1]}, sound type: {key[2]}) - Best Lambda: {best_alpha}")
    print(f"Train MSE: {best_train_mse}, Test MSE: {best_test_mse}")
    print(f"Train R-squared: {best_train_r2}, Test R-squared: {best_test_r2}")
    print(f"Train Percent Correct: {best_train_pc}%, Test Percent Correct: {best_test_pc}%")
    print()  # For better readability between results

    # Plotting R-squared, MSE, and Percent Correct values
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].plot(alphas, train_acc, label="Train R-squared", color='blue')
    axs[0].plot(alphas, test_acc, label="Test R-squared", color='red')
    axs[0].axhline(1 / len(np.unique(Y)), color='gray', linestyle='--', label='Baseline')
    axs[0].set_xscale('log')
    axs[0].set_xlabel("Lambda (α)")
    axs[0].set_ylabel("R-squared")
    axs[0].legend()
    axs[0].set_title(f"R-squared vs Lambda for {key}")

    axs[1].plot(alphas, train_mse, label="Train MSE", color='blue')
    axs[1].plot(alphas, test_mse, label="Test MSE", color='red')
    axs[1].set_xscale('log')
    axs[1].set_xlabel("Lambda (α)")
    axs[1].set_ylabel("Mean Squared Error (MSE)")
    axs[1].legend()
    axs[1].set_title(f"MSE vs Lambda for {key}")

    axs[2].plot(alphas, train_percent_correct, label="Train Percent Correct", color='blue')
    axs[2].plot(alphas, test_percent_correct, label="Test Percent Correct", color='red')
    axs[2].set_xscale('log')
    axs[2].set_xlabel("Lambda (α)")
    axs[2].set_ylabel("Percent Correct (%)")
    axs[2].legend()
    axs[2].set_title(f"Percent Correct vs Lambda for {key}")

    plt.tight_layout()
    plt.show()

    # Plot Train and Test Points with Connecting Line
    # Add a jitter for better visualization of overlapping points
    jitter = 0.02 * np.random.randn(len(Y_test))

    # Create the new plot
    plt.figure(figsize=(10, 6))
    for i in range(len(Y_test)):
        plt.plot([0, 1], [Y_test[i], y_test_pred[i]], color='gray', alpha=0.5)  # Line connecting test to pred

    # Scatter the test points
    sns.scatterplot(x=np.zeros(len(Y_test)) + jitter, y=Y_test, color='blue', label="Test Points", alpha=0.8)
    # Scatter the predicted points
    sns.scatterplot(x=np.ones(len(y_test_pred)) + jitter, y=y_test_pred, color='orange', label="Predicted Points", alpha=0.8)

    # Customize the plot
    plt.xticks([0, 1], ['Test', 'Pred'])
    plt.xlabel('Data Points')
    plt.ylabel('Frequency')
    plt.title(f"Test vs Prediction Frequencies - {key[1]} ({sound_type}) - {key[0]}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    # Save the figure to the specified directory
    plt.savefig(f"/Users/zoetomlinson/Desktop/NeuroAI/Figures/Model Plots/"
                f"{key[2]} ({key[1]}) Test vs Pred Connection Plot for {key[0]}")
    plt.show()
