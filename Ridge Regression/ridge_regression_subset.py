import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from jaratoolbox import celldatabase, settings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
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
targetSiteNames = ["Primary auditory area", "Dorsal auditory area", "Ventral auditory area"]
leastCellsArea = 10000
evoked_start = 0.015
evoked_end = 0.3
pt_evoked_end = 0.1
binWidth = 0.01
binEdges = np.arange(evoked_start, evoked_end, binWidth)
binEdgesPT = np.arange(evoked_start, pt_evoked_end, binWidth)
iterations = 30  # Number of random sampling iterations
random_seed = 42
alphas = np.logspace(-5, 10, 200)
np.random.seed(random_seed)  # Ensure reproducibility
periodsNameSpeech = ['base200', 'respOnset', 'respSustained']
max_trials = {'PT': 640, 'AM': 220, 'speech': 381}
unique_labels = [(0, 0), (0, 33), (0, 67), (0, 100), (33, 100), (67, 100), (100, 100), (100, 67), (100, 33),
                 (100, 0), (67, 0), (33, 0)]
min_speech_freq_dict = {(0, 0): 31, (0, 33): 29, (0, 67): 32, (0, 100): 24, (33, 100): 34, (67, 100): 35,
                        (100, 100): 33, (100, 67): 29, (100, 33): 35, (100, 0): 35, (67, 0): 31, (33, 0): 33}
# Initialize a dictionary to store counts for each frequency across mouse-date combos
frequency_counts_dict = {tuple(freq): [] for freq in unique_labels}
data_dict = {}
previous_frequency_speech = None
previous_frequency_AM = None
previous_frequency_PT = None
n_neurons_list = []
r2_test_list = []
labels_list = []
primary_n_neurons = 0
ventral_n_neurons = 0
indices_AM = None
results = []
smallest_neuron_count = 9

# %% Load Data
databaseDir = os.path.join(settings.DATABASE_PATH, '2022paspeech')
fullDbPath = 'fulldb_speech_tuning.h5'
fullPath = os.path.join(databaseDir, fullDbPath)
fullDb = celldatabase.load_hdf(fullPath)
fullDb["recordingSiteName"] = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])

# %% Add data to the dictionary for each brain area and sound type
for subject in subject_list:
    for brain_area in ["Primary auditory area", "Ventral auditory area"]:
        # Clean and load the data for each subject and brain area
        X_speech_array, X_AM_array, X_PT_array, \
            Y_brain_area_speech_all, Y_brain_area_AM_all, Y_brain_area_PT_all, \
            Y_frequency_speech_sorted, Y_frequency_AM_sorted, Y_frequency_pureTones_sorted \
            = funcs.clean_and_concatenate(subject, recordingDate_list, brain_area,
                                          previous_frequency_AM, previous_frequency_PT,
                                          previous_frequency_speech)

        Y_frequency_FT = Y_frequency_speech_sorted[:, 0]
        Y_frequency_VOT = Y_frequency_speech_sorted[:, 1]

        for sound_type, X_array, Y_brain_area_all, Y_frequency_sorted in zip(
                ['AM', 'PT', 'FT', 'VOT'],
                [X_AM_array, X_PT_array, X_speech_array, X_speech_array],
                [Y_brain_area_AM_all, Y_brain_area_PT_all, Y_brain_area_speech_all, Y_brain_area_speech_all],
                [Y_frequency_AM_sorted, Y_frequency_pureTones_sorted, Y_frequency_FT, Y_frequency_VOT]):
            brain_area_array = np.array(Y_brain_area_all)
            X_array_adjusted = X_array[brain_area_array == brain_area]
            X_array_adjusted = X_array_adjusted.T
            data_dict[(subject, brain_area, sound_type)] = {'X': X_array_adjusted, 'Y': Y_frequency_sorted}

# %% Ridge Regression
for key, value in data_dict.items():
    X = value['X']
    n_neurons = X.shape[1]
    smallest_neuron_count = min(smallest_neuron_count, n_neurons)

    if key == 'AM' or key == 'PT':
        Y = np.log(value['Y'])
    else:
        Y = value['Y']

    # Sampling iterations
    for iteration in range(iterations):
        sampled_indices = np.random.choice(X.shape[1], smallest_neuron_count, replace=False)
        X_sampled = X[:, sampled_indices]
        n_neurons = X_sampled.shape[1]

        # Train-test split
        X_train, X_test, Y_train, Y_test = train_test_split(X_sampled, Y, test_size=0.2,
                                                            random_state=random_seed)

        # Find best alpha using Ridge Regression
        best_r2, best_alpha = -np.inf, None
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train, Y_train)
            r2 = ridge.score(X_test, Y_test)
            if r2 > best_r2:
                best_r2 = r2
                best_alpha = alpha

        # Append results
        results.append({
            'Subject': key[0],
            'Brain Area': key[1],
            'Sound Type': key[2],
            'R2 Test': best_r2
        })

    if key[1] == "Primary auditory area":
        primary_n_neurons += n_neurons
    else:
        ventral_n_neurons += n_neurons


# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Compute the average R² for each Subject, Brain Area, and Sound Type
average_r2_df = results_df.groupby(['Subject', 'Brain Area', 'Sound Type'], as_index=False)['R2 Test'].mean()

# Create the figure
plt.figure(figsize=(14, 8))

# Boxplot for each Brain Area-Sound Type combination
sns.boxplot(x=f'Brain Area', y='R2 Test', hue='Sound Type', data=average_r2_df,
    palette='Set2', dodge=True, width=0.6, showcaps=True, boxprops={'edgecolor':'black'},
    whiskerprops={'color':'black'}, medianprops={'color':'black', 'linewidth':2})

# Add individual points for the 5 mice/observations
sns.stripplot(x='Brain Area', y='R2 Test', hue='Sound Type', data=average_r2_df,
      dodge=True, palette='dark:.3', jitter=True, alpha=0.7, linewidth=0.5)

# Customize the plot
plt.title("Distribution of R² Values by Brain Area and Sound Type", fontsize=16)
plt.xlabel(f"Brain Area (Primary Total Neurons: {primary_n_neurons} - Ventral Total Neurons: {ventral_n_neurons}", fontsize=14)
plt.ylabel("R² Value", fontsize=14)
plt.legend(title="Sound Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and show plot
output_path_avg = "/Users/zoetomlinson/Desktop/neuronalDataResearch/Figures/Ridge Regression/Average_R2_BrainArea_SoundType.png"
plt.savefig(output_path_avg)
plt.show()