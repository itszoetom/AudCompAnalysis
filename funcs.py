# Functions file for AudPopAnalysis/2022paspeech Project
import numpy as np
import matplotlib.pyplot as plt
from jaratoolbox import celldatabase, ephyscore, spikesanalysis
from copy import deepcopy
from sklearn.decomposition import PCA
import params
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm


# %% Initialize plot and subset dataframe
def load_data(subject, date, targetSiteName, sound_type_load):
    fullDb = celldatabase.load_hdf(params.fullPath)
    simpleSiteNames = fullDb["recordingSiteName"].str.split(',').apply(lambda x: x[0])
    # simpleSiteNames = simpleSiteNames.replace("Posterior auditory area", "Dorsal auditory area")

    fullDb["recordingSiteName"] = simpleSiteNames
    celldb = fullDb[(fullDb.subject == subject)]
    celldbSubset = celldb[(celldb.date == date)]
    celldbSubset = celldbSubset[(celldbSubset.recordingSiteName == targetSiteName)]

    if celldbSubset.empty:
        print(f"No data in {targetSiteName} on {date} for Speech, AM, and PT.")
        return None, None, None

    ensemble = ephyscore.CellEnsemble(celldbSubset)
    try:
        ephysData, bdata = ensemble.load(sound_type_load)
    except IndexError:
        print(f"No sound data for {targetSiteName} on {date} for {subject}")
        return None, None, None
    return ensemble, ephysData, bdata


# %% Calculate Spike Rate
def spike_rate(sound_type, ensemble, ephysData, bdata, targetSiteName):
    """
    Calculate firing rate as spikes per second evoked
            sound_type: str sound type label. ex. "speech"
            ensemple: ephyscore.CellEnsemble(celldbSubset)
            ephysData: ephyscore.CellEnsemble(celldbSubset)
            bdata: ephyscore.CellEnsemble(celldbSubset)
            targetSiteName: str brain area ex. "Primary auditory area"

        Returns X array (spikeRateNormalized) of firing rates for specified sound type and brain area (trials, neurons)
                Y brain area metadata (Y_brain_area_array) brain area metadata for each neuron
                Y sound frequency (Hrz) (Y_frequency) metadata for each trial
    """
    if sound_type == "speech":
        eventOnsetTimes = ephysData['events']['stimOn']
        _, _, _ = ensemble.eventlocked_spiketimes(eventOnsetTimes, [params.evoked_start, params.evoked_end])
        spikeCounts = ensemble.spiketimes_to_spikecounts(params.binEdges)
        sumEvokedFR = spikeCounts.sum(axis=2)
        spikesPerSecEvoked = sumEvokedFR / (params.evoked_end - params.evoked_start)

        FTParamsEachTrial = bdata['targetFTpercent']
        VOTParamsEachTrial = bdata['targetVOTpercent']
        nTrials = len(bdata['targetFTpercent'])

        # Create and sort Y_frequency for speech
        Y_frequency = np.array([(FTParamsEachTrial[i], VOTParamsEachTrial[i]) for i in range(nTrials)])

    if sound_type == "AM" or sound_type == "PT":
        nTrials = len(bdata['currentFreq'])
        eventOnsetTimes = ephysData['events']['stimOn'][:nTrials]
        _, _, _ = ensemble.eventlocked_spiketimes(eventOnsetTimes, [params.evoked_start, params.evoked_end])
        spikeCounts = ensemble.spiketimes_to_spikecounts(params.binEdges)
        sumEvokedFR = spikeCounts.sum(axis=2)
        spikesPerSecEvoked = sumEvokedFR / (params.evoked_end - params.evoked_start)

        # Create and sort Y_frequency for AM/PT
        Y_frequency = np.array(bdata['currentFreq'])

    trialMeans = spikesPerSecEvoked.mean(axis=1)
    spikesPerSecEvokedNormalized = spikesPerSecEvoked.T - trialMeans  # why negative
    spikesPerSecEvokedNormalized = spikesPerSecEvokedNormalized.T

    if spikesPerSecEvokedNormalized.shape[1] > params.leastCellsArea:
        subsetIndex = np.random.choice(spikesPerSecEvokedNormalized.shape[1], params.leastCellsArea, replace=False)
        spikeRateNormalized = spikesPerSecEvokedNormalized[:, subsetIndex]
    else:
        spikeRateNormalized = spikesPerSecEvokedNormalized

    Y_brain_area_array = [targetSiteName] * spikeRateNormalized.shape[0]

    return spikeRateNormalized, Y_brain_area_array, Y_frequency


def adjust_array_and_labels(x_list, y_list, brain_area, max_length, subject, date, targetSiteName):
    adjusted_x_list = []
    adjusted_y_list = []
    adjusted_ba_list = []
    ignored_x_lists = []
    ignored_y_lists = []
    ignored_ba_lists = []

    for i, x in enumerate(x_list):
        if any(arr.shape[0] < max_length for arr in x):
            ignored_x_lists.append(f"X list {i} (lengths: {[arr.shape[0] for arr in x]})")
            ignored_ba_lists.extend(f"X list {i} (lengths: {[arr.shape[0] for arr in x]}")
            print(f"Not enough PT trials recorded for subject {subject}, on {date} in brain area {targetSiteName}.")
            continue

        # Truncate each array in the list to max_length
        adjusted_x_list.append([arr[:max_length] for arr in x])
        adjusted_ba_list.extend(brain_area)

    for i, y in enumerate(y_list):
        if len(y) < max_length:
            ignored_y_lists.append(f"Y list {i} (length: {len(y)})")
            ignored_ba_lists.append(f"X list {i} (lengths: {[arr.shape[0] for arr in x]}")
            continue

        adjusted_y_list.append(y[:max_length])

    return adjusted_x_list, adjusted_y_list, adjusted_ba_list

def sort_x_arrays(X_list, indices, sound_type):
    sorted_x_list = []
    for x in X_list:
        if sound_type == "am" or sound_type == "pt":
            sorted_x = [arr[indices] for arr in x]
            sorted_x_list.append(np.array(sorted_x))
        if sound_type == "speech":
            for z in x:
                sorted_x = [arr[indices] for arr in z]
                sorted_x_list.append(np.array(sorted_x))
    return sorted_x_list


def spike_rate_for_windows(sound_type: str, ensemble, ephysData, bdata, targetSiteName):
    '''
    Calculate firing rates for multiple evoked spike windows.

    Returns a dictionary of spikeRateNormalized arrays keyed by window name.
    Each value is (trials, neurons).

    sound_type = ['speech', 'pt', 'am']
    '''
    X_dict = {}
    Y_brain_area_array = []

    if sound_type == "speech":
        eventOnsetTimes = ephysData['events']['stimOn']
        FTParamsEachTrial = bdata['targetFTpercent']
        VOTParamsEachTrial = bdata['targetVOTpercent']
        Y_frequency = np.array([(FTParamsEachTrial[i], VOTParamsEachTrial[i]) for i in range(len(FTParamsEachTrial))])

    else:  # AM or PT
        eventOnsetTimes = ephysData['events']['stimOn'][:len(bdata['currentFreq'])]
        Y_frequency = np.array(bdata['currentFreq'])

    # Loop over relevant windows for this sound_type
    for label, (start, end) in params.spike_windows.items():
        if sound_type in label:
            _, _, _ = ensemble.eventlocked_spiketimes(eventOnsetTimes, [start, end])
            spikeCounts = ensemble.spiketimes_to_spikecounts(np.arange(start, end, 0.01))  # Bin width of 10 ms
            sumEvokedFR = spikeCounts.sum(axis=2)
            spikesPerSecEvoked = sumEvokedFR / (end - start)

            trialMeans = spikesPerSecEvoked.mean(axis=1)
            spikesPerSecEvokedNormalized = spikesPerSecEvoked.T - trialMeans
            spikesPerSecEvokedNormalized = spikesPerSecEvokedNormalized.T

            if spikesPerSecEvokedNormalized.shape[1] > params.leastCellsArea:
                subsetIndex = np.random.choice(spikesPerSecEvokedNormalized.shape[1], params.leastCellsArea,
                                               replace=False)
                spikeRateNormalized = spikesPerSecEvokedNormalized[:, subsetIndex]
            else:
                spikeRateNormalized = spikesPerSecEvokedNormalized

            X_dict[label] = spikeRateNormalized
            Y_brain_area_array = [targetSiteName] * spikeRateNormalized.shape[0]

    return X_dict, Y_brain_area_array, Y_frequency


def adjust_speech_length(subject, date, brain_area, X_speech, Y_frequency_speech, previous_frequency_speech):
    valid_indices = []
    freq_kept_counts = {tuple(freq): 0 for freq in params.unique_labels}

    # Filter valid trials based on frequency count
    for i, freq in enumerate(Y_frequency_speech):
        freq_tuple = tuple(freq)
        if freq_kept_counts[freq_tuple] < params.min_speech_freq_dict[freq_tuple]:
            valid_indices.append(i)
            freq_kept_counts[freq_tuple] += 1

    # Filter X_speech and Y arrays based on valid indices
    if len(valid_indices) < params.max_trials['speech']:
        print(f'Not enough speech trials for subject {subject}, on {date} in brain area {brain_area}')
        return None, None, None, None
    else:
        X_speech = np.array(X_speech)
        X_speech = X_speech.T
        X_speech_filtered = X_speech[valid_indices]
        X_speech_filtered = X_speech_filtered.T
        Y_frequency_speech_filtered = Y_frequency_speech[valid_indices]

        if len(X_speech_filtered) != 0:
            # Sort Y_frequency_speech_adjusted
            if isinstance(Y_frequency_speech_filtered, list):
                Y_frequency_speech_filtered = np.array(Y_frequency_speech_filtered)

            # Use np.lexsort to sort by the second element of the tuple first, and then by the first element
            indices_speech = np.lexsort(
                (Y_frequency_speech_filtered[:, 1], Y_frequency_speech_filtered[:, 0]))

            # Use these sorted indices to rearrange the array
            Y_frequency_speech_sorted = Y_frequency_speech_filtered[indices_speech]

            # Check if frequency lists are all the same
            if previous_frequency_speech is not None:
                assert np.array_equal(Y_frequency_speech_sorted, previous_frequency_speech), (
                    f"Frequency mismatch for subject: {subject}, date: {date}, target site: {brain_area}"
                    f"Previous: {previous_frequency_speech} and sorted: {Y_frequency_speech_sorted}")

            previous_frequency_speech = deepcopy(Y_frequency_speech_sorted)

        return X_speech_filtered, Y_frequency_speech_sorted, previous_frequency_speech, indices_speech


# Function to randomly select neurons based on the min_neuron_dict
def select_neurons(data, brain_area, min_neuron_dict):
    n_neurons_to_select = min_neuron_dict[brain_area]
    total_neurons = data.shape[1]

    # Ensure that we don't select more neurons than are available
    if total_neurons > n_neurons_to_select:
        selected_indices = np.random.choice(total_neurons, n_neurons_to_select, replace=False)
        data = data[:, selected_indices]
    else:
        print(f"Not enough neurons in {brain_area}, using all {total_neurons} neurons.")
    return data


def sort_sound_array(subject, date, brain_area, X_adjusted, Y_brain_area_all, Y_frequency_adjusted, previous_frequency):
    X_all = None
    Y_brain_area_all_combined = None

    if X_adjusted is not None:
        if len(X_adjusted) != 0:
            # Sort Y_frequency_AM_adjusted
            Y_frequency = np.array(Y_frequency_adjusted)
            sorted_indices = np.argsort(Y_frequency)
            sorted_Y_freq = Y_frequency[sorted_indices]
            Y_frequency_sorted = sorted_Y_freq

            Y_frequency_sorted = np.array(Y_frequency_sorted)
            indices = np.argsort(Y_frequency_sorted)  # Sort by frequency values

            # Check if frequency lists are all the same
            if previous_frequency is not None:
                assert np.array_equal(Y_frequency_sorted, previous_frequency), (
                    f"Frequency mismatch for subject: {subject}, date: {date}, target site: {brain_area}")
            previous_frequency = deepcopy(Y_frequency_sorted)

            # Concatenate data instead of extending lists
            if X_all is None:
                X_all = X_adjusted
            else:
                X_all = np.concatenate((X_all, X_adjusted), axis=0)

            if Y_brain_area_all_combined is None:
                Y_brain_area_all_combined = Y_brain_area_all
            else:
                Y_brain_area_all_combined = np.concatenate((Y_brain_area_all_combined, Y_brain_area_all), axis=0)
    else:
        return None, None, None, previous_frequency, None

    return X_all, Y_frequency_sorted, Y_brain_area_all_combined, previous_frequency, indices


def calculate_participation_ratio(explained_variance_ratio):
    return ((np.sum(explained_variance_ratio)) ** 2) / np.sum(explained_variance_ratio ** 2)


def calculate_participation_ratio_percent(explained_variance_ratio, n_neurons):
    participation_ratio = ((np.sum(explained_variance_ratio)) ** 2) / np.sum(explained_variance_ratio ** 2)
    participation_ratio_percent = participation_ratio / n_neurons
    return participation_ratio_percent


def create_figure_grid(rows, cols, title, figsize=(18, 10)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    return fig, axes


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


def plot_2d_pca(ax, data, labels, title, cmap):
    """
    Plots 2D PCA of the input data on the given axis.

    Parameters:
    - ax: Matplotlib axis object to plot on.
    - data: Dictionary containing data['X'] (features) and optionally other keys.
    - labels: Array-like, labels for coloring the data points.
    - title: Title of the plot.
    - cmap: Colormap for the scatter plot (default: 'viridis').

    Returns:
    - scatter: The scatter plot object for further customization if needed.
    """
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(data['X'])

    explained_variance = pca.explained_variance_ratio_

    scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap=cmap, s=32, alpha=0.5)

    ax.set_xlabel(f'PCA 1 ({explained_variance[0] * 100:.2f}% variance)')
    ax.set_ylabel(f'PCA 2 ({explained_variance[1] * 100:.2f}% variance)')
    plt.colorbar(scatter, ax=ax, orientation='vertical')

    return scatter


# Function to check statistical significance based on brain area or sound type
def check_significance(p_value, comparison_type):
    if comparison_type == 'brain_area':
        return p_value < 0.05
    elif comparison_type == 'sound_type':
        return p_value < 0.01667



def euclidean_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)


def trial_distances(trials, mean_vec):
    distances = []
    for trial in trials:
        dist = sum((x - m) ** 2 for x, m in zip(trial, mean_vec)) ** 0.5
        distances.append(dist)
    return distances


def plot_5fold_cv(X, Y, title_str, brain_area, window_name, condition_name):
    n_neurons = X.shape[1]
    alphas = np.logspace(-10, 5, 200)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    fig.suptitle(f"{title_str} 5-Fold CV True vs Predicted", fontsize=16)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train_raw = X[train_idx, :]
        X_test_raw = X[test_idx, :]
        Y_train = Y[train_idx]
        Y_test = Y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        best_r2 = -np.inf
        best_alpha = None
        best_model = None

        for alpha in alphas:
            model = Ridge(alpha=alpha, solver='lsqr')
            model.fit(X_train, Y_train)
            r2 = model.score(X_test, Y_test)
            if r2 > best_r2:
                best_r2 = r2
                best_alpha = alpha
                best_model = model

        y_test_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, y_test_pred))
        corr, _ = pearsonr(Y_test, y_test_pred)

        # Sort for plotting
        sorted_idx = np.argsort(Y_test)
        ax = axes[fold_idx]
        sns.scatterplot(x=Y_test[sorted_idx], y=y_test_pred[sorted_idx], ax=ax, color='black', s=20)
        sns.regplot(x=Y_test[sorted_idx], y=y_test_pred[sorted_idx], scatter=False, ax=ax, color='red',
                    line_kws={'linestyle': '--', 'linewidth': 2})
        ax.set_title(f"Fold {fold_idx}\nAlpha={best_alpha:.1e}\nR²={best_r2:.3f}\nRMSE={rmse:.3f}\nr={corr:.3f}")
        ax.set_xlabel("True")
        if fold_idx == 0:
            ax.set_ylabel("Predicted")
        ax.grid(True)

        fold_results.append({
            'brain_area': brain_area,
            'window': window_name,
            'fold': fold_idx,
            'r2': best_r2,
            'condition': condition_name,
            'n_neurons': n_neurons
        })

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return fold_results

def hex_to_rgba(hex_color, alpha=1):
    """
    Converts a hexadecimal color string (e.g., "#RRGGBB" or "#RRGGBBAA")
    to an RGBA tuple (R, G, B, A), where A is between 0 and 1.
    """
    hex_color = hex_color.lstrip('#')  # Remove '#' if present

    if len(hex_color) == 6:
        # Assume full opacity if no alpha component is provided
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = alpha
    elif len(hex_color) == 8:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = alpha
    else:
        raise ValueError("Invalid hex color format. Expected 6 or 8 characters.")

    return (r, g, b, a)


def plot_rasters(celldb:pd.DataFrame, sortingInds: np.ndarray, trialIndexForEachSpikeAll: list,
                 spikeTimesFromEventOnsetAll: list, rows: int=3, cols: int=3,
                 random: bool=True, specificInds: list=None, title: str=None, plot: bool=True,
                 subplot_titles: bool=True) -> plotly.graph_objs._figure.Figure:

    n_cells = rows*cols
    if random:
        someCells = celldb.sample(n=n_cells).index.tolist()
    elif random == False and specificInds is not None:
        someCells = specificInds
    elif random == False and specificInds is None:
        someCells = np.arange(n_cells)

    # Initialize subplots with shared X and Y axes
    fig = make_subplots(
        rows=rows, cols=cols,
        shared_xaxes=False, shared_yaxes=True,
        subplot_titles=celldb.loc[someCells]['simpleSiteName'].to_list() if subplot_titles else None,
    )

    for count, indcell in enumerate(someCells):
        row = count // 3 + 1
        col = count % 3 + 1

        sortedIndexForEachSpike = sortingInds[trialIndexForEachSpikeAll[indcell]]
        fig.add_trace(
            go.Scatter(
                x=spikeTimesFromEventOnsetAll[indcell],
                y=sortedIndexForEachSpike,
                mode='markers',
                marker=dict(size=2, color='black'),
                name=f'Cell {indcell}'
            ),
            row=row, col=col
        )
        # Add axis labels
        fig.update_xaxes(title_text="Time (s)", row=row, col=col)
        fig.update_yaxes(title_text=f"[{indcell}] Sorted trials", row=row, col=col)

    fig.add_vline(
        x=0,
        line=dict(color='red', width=2),  # Customize color and width
    )

    if title is not None:
        fig.update_layout(
            title=title,
            showlegend=False,
            height=400 * rows,
            width=1200,
            title_font=dict(size=16, family='Arial', color='black')
        )
    else:
        fig.update_layout(
            showlegend=False,
            height=400 * rows,
            width=1200,
        )

    if plot:
        fig.show();

    return fig


def calculate_fr_arrays(celldb:pd.DataFrame, stimType:str, stimVar:str, timeRange:list, allPeriods:list) -> list:
    """
    Calculates the firing rate arrays for given cell data, stimulus type, and time range.

    This function processes a given set of cell data from a DataFrame alongside the
    corresponding stimulus type and a specified time range. It calculates the firing
    rate arrays for different response periods, such as baseline, onset response, and
    sustained response periods. The calculations take into account the number of cells,
    categories, and specified time ranges provided.

    Args:
        celldb (pd.DataFrame): A DataFrame containing cell information for tracking relevant ephys data.
        stimType (str): The type of stimulus used in the experiment. Valid values are 'Sine', 'naturalSound', and
        'AM'.
        stimVar (str): The name of the variable in the behavior data representing the stimulus.
        timeRange (list): A list specifying the time range to analyze, in the form
            [start_time, end_time].
        allPeriods (list): A list of tuples/lists specifying the periods to analyze. Assumes order of
        [baseline, onset, sustained, offset].

    Returns:
        list: A list containing the calculated firing rate arrays for all periods and an array of trial stims
        [baseline, onset, sustained, offset, stimArray]
    """
    nCells = len(celldb)
    periodDuration = [x[1] - x[0] for x in allPeriods]

    if stimType == 'AM':
        nTrials = 220
        nCategories = 11
    elif stimType == 'naturalSound':
        nTrials = 200
        nCategories = len(params.SOUND_CATEGORIES)
    elif stimType == 'pureTones':
        nTrials = 320
        nCategories = 16
    else:
        raise ValueError(f"Unrecognized stimulus type: {stimType}. Should be in ['AM', 'naturalSound', 'pureTones']")

    basefr = np.full((nCells, nTrials), np.nan)
    onsetfr = np.full((nCells, nTrials), np.nan)
    sustainedfr = np.full((nCells, nTrials), np.nan)
    offsetfr = np.full((nCells, nTrials), np.nan)
    stimArray = np.full((nCells, nTrials), np.nan)
    brainRegion = np.empty(nCells, object)
    mouseID = np.empty(nCells, object)
    sessionID = np.empty(nCells, object)

    num_iterations = len(celldb)
    indCell = -1
    for indRow, dbRow in tqdm(celldb.iterrows(), total=num_iterations, desc=f"Calculating firing rates for {stimType}"):
        indCell += 1
        oneCell = ephyscore.Cell(dbRow)
        ephysData, bdata = oneCell.load(stimType)

        spikeTimes = ephysData['spikeTimes']
        eventOnsetTimes = ephysData['events']['stimOn'][:nTrials]
        currentStim = bdata[stimVar][:nTrials]

        # -- Test if trials from behavior don't match ephys -- Shouldn't matter since I am manually subsetting both above
        if (len(currentStim) > len(eventOnsetTimes)) or \
                (len(currentStim) < len(eventOnsetTimes) - 1):
            print(f'[{indRow}] Warning! BevahTrials ({len(currentStim)}) and ' +
                  f'EphysTrials ({len(eventOnsetTimes)})')
            continue
        if len(currentStim) == len(eventOnsetTimes) - 1:
            eventOnsetTimes = eventOnsetTimes[:len(currentStim)]

        (spikeTimesFromEventOnset, trialIndexForEachSpike, indexLimitsEachTrial) = \
            spikesanalysis.eventlocked_spiketimes(spikeTimes, eventOnsetTimes, timeRange)

        spikesEachTrialEachPeriod = []
        for indPeriod, period in enumerate(allPeriods):
            spikeCountMat = spikesanalysis.spiketimes_to_spikecounts(spikeTimesFromEventOnset,
                                                                     indexLimitsEachTrial, period)
            spikesEachTrial = spikeCountMat[:, 0]
            spikesEachTrialEachPeriod.append(spikesEachTrial)

        # for indcond in range(nCategories):
            # trialsThisCond = trialsEachCateg[:, indcond]
        firingRateBase = spikesEachTrialEachPeriod[0] / periodDuration[0]
        firingRateOnset = spikesEachTrialEachPeriod[1] / periodDuration[1]
        firingRateSustain = spikesEachTrialEachPeriod[2] / periodDuration[2]
        firingRateOffset = spikesEachTrialEachPeriod[3] / periodDuration[3]

        sorted_stim_ind = np.argsort(currentStim)
        sorted_stim_array = currentStim[sorted_stim_ind]
        sorted_fr_base = firingRateBase[sorted_stim_ind]
        sorted_fr_onset = firingRateOnset[sorted_stim_ind]
        sorted_fr_sustained = firingRateSustain[sorted_stim_ind]
        sorted_fr_offset = firingRateOffset[sorted_stim_ind]


        basefr[indCell, :] = sorted_fr_base
        onsetfr[indCell, :] = sorted_fr_onset
        sustainedfr[indCell, :] = sorted_fr_sustained
        offsetfr[indCell, :] = sorted_fr_offset
        stimArray[indCell, :] = sorted_stim_array
        brainRegion[indCell] = dbRow['simpleSiteName']
        mouseID[indCell] = dbRow['subject']
        sessionID[indCell] = dbRow['date']

    return [basefr, onsetfr, sustainedfr, offsetfr, stimArray, brainRegion, mouseID, sessionID]