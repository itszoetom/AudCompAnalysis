import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut
from jaratoolbox import settings
import os
from tqdm import tqdm
import studyparams
# %% SETTINGS
file_path = settings.FIGURES_DATA_PATH
save_dir = settings.SAVE_PATH
response_ranges = ["onset", "sustained", "offset"]
stim_types = ["naturalSound", "AM", "pureTones"]
colors = {
    'Dorsal auditory area': '#1f77b4',
    'Posterior auditory area': '#ff7f0e',
    'Primary auditory area': '#2ca02c',
    'Ventral auditory area': '#d62728'}

hyperparameters = np.logspace(-5, 1, 20)


def load_stim_data(file_path, stim):
    """Load stimulus arrays from file."""
    return np.load(f"{file_path}/fr_arrays_{stim}.npz", allow_pickle=True)


def extract_frequency_mappings(stim_arrays, stim_type):
    """
    Extract frequency mappings from the loaded data's y array.
    Returns a dictionary mapping stimulus indices to frequency values.
    """
    freq_map = {}

    if 'y' in stim_arrays:
        y_array = stim_arrays['y']
        stimArray = stim_arrays["stimArray"][0, :]
        uniqStims = np.unique(stimArray)

        # Map each unique stimulus index to its corresponding frequency
        for stim_idx in uniqStims:
            # Find first occurrence of this stimulus
            first_occurrence = np.where(stimArray == stim_idx)[0][0]
            freq_value = y_array[first_occurrence]

            if stim_type in ['AM', 'pureTones']:
                # Format as Hz
                freq_map[stim_idx] = f"{freq_value}Hz"
            else:
                freq_map[stim_idx] = str(freq_value)

    return freq_map


def get_stim_config(stim, stim_arrays=None):
    """Get stimulus configuration including trials, categories, and labels."""
    if stim == 'AM':
        nTrials = 220
        nCategories = 11
        stimVals = np.arange(nCategories)
    elif stim == 'naturalSound':
        nTrials = 200
        soundCats = studyparams.SOUND_CATEGORIES
        nCategories = len(soundCats)
        nInstances = 4
        stimVals = np.array([f"{soundCats[i]}_{j + 1}" for i in range(nCategories) for j in range(nInstances)])
    elif stim == 'pureTones':
        nTrials = 320
        nCategories = 16
        stimVals = np.arange(nCategories)

    # Extract frequency mapping from data if available
    freq_map = None
    if stim_arrays is not None:
        freq_map = extract_frequency_mappings(stim_arrays, stim)

    # Generate labels
    if stim == 'naturalSound':
        labels = stimVals
    elif freq_map is not None:
        # Use actual frequencies from data
        labels = [freq_map.get(i, str(i)) for i in range(nCategories)]
    else:
        # Fallback to indices
        labels = [str(i) for i in range(nCategories)]

    return {
        'nTrials': nTrials,
        'nCategories': nCategories,
        'stimVals': stimVals,
        'labels': labels,
        'freq_map': freq_map
    }


def compute_pairwise_svm(brain_resp_array, stimArray, uniqStims, c_value):
    """
    Compute pairwise SVM accuracies for all stimulus pairs.

    Returns:
        svm_stim_vals: Matrix of pairwise accuracies
    """
    svm_stim_vals = np.full((len(uniqStims), len(uniqStims)), np.nan)

    for i1, stim1 in enumerate(uniqStims):
        mask1 = stimArray == stim1
        resp1 = brain_resp_array[:, mask1]  # neurons, trials

        for i2, stim2 in enumerate(uniqStims):
            if i1 == i2:
                continue
            mask2 = stimArray == stim2
            resp2 = brain_resp_array[:, mask2]

            X_pair = np.hstack([resp1, resp2]).T
            y_pair = np.array([0] * resp1.shape[1] + [1] * resp2.shape[1])

            # Shuffle the dataset so trials are randomized
            shuffle_idx = np.random.permutation(len(y_pair))
            X_pair = X_pair[shuffle_idx]
            y_pair = y_pair[shuffle_idx]

            # Leave-one-out cross-validation
            loo = LeaveOneOut()
            acc_list = []
            for train_idx, test_idx in loo.split(X_pair, y_pair):
                svm = LinearSVC(max_iter=10000, dual='auto', C=c_value)
                svm.fit(X_pair[train_idx], y_pair[train_idx])
                acc_list.append(svm.score(X_pair[test_idx], y_pair[test_idx]))

            accuracy = np.mean(acc_list)
            svm_stim_vals[i1, i2] = accuracy

    return svm_stim_vals


def save_dataset_info(stim_types, file_path, save_dir):
    """Save out the number of neurons and trials for each sound type and brain area combo."""
    dataset_info = []
    all_freq_mappings = []

    for stim in stim_types:
        stim_arrays = load_stim_data(file_path, stim)
        brainRegionArray = stim_arrays["brainRegionArray"]
        uniqRegions = np.unique(brainRegionArray)

        config = get_stim_config(stim, stim_arrays)

        for respRange in response_ranges:
            respArray = stim_arrays[f"{respRange}fr"]

            for brainRegion in uniqRegions:
                brain_mask = brainRegionArray == brainRegion
                brain_resp_array = respArray[brain_mask, :]

                n_neurons = brain_resp_array.shape[0]
                n_trials = brain_resp_array.shape[1]

                dataset_info.append({
                    'stim_type': stim,
                    'brain_region': brainRegion,
                    'response_window': respRange,
                    'n_neurons': n_neurons,
                    'n_trials': n_trials,
                    'n_categories': config['nCategories']
                })

        # Collect frequency mappings from data
        if config['freq_map'] is not None:
            for idx, freq in config['freq_map'].items():
                all_freq_mappings.append({
                    'stim_type': stim,
                    'index': idx,
                    'frequency': freq
                })

    # Save dataset info
    info_df = pd.DataFrame(dataset_info)
    info_path = os.path.join(save_dir, "dataset_info.csv")
    info_df.to_csv(info_path, index=False)
    print(f"Saved dataset information to {info_path}")

    # Save frequency mappings extracted from data
    if all_freq_mappings:
        freq_df = pd.DataFrame(all_freq_mappings)
        freq_path = os.path.join(save_dir, "frequency_mappings.csv")
        freq_df.to_csv(freq_path, index=False)
        print(f"Saved frequency mappings to {freq_path}")

    return info_df


def create_classifier_visualization(brain_resp_array, stimArray, stim1_idx, stim2_idx,
                                    c_value, stim_type, save_dir):
    """
    Create visualization of classifier performance for two similar stimuli.
    Shows data points colored by stimulus and the decision boundary.
    """
    mask1 = stimArray == stim1_idx
    mask2 = stimArray == stim2_idx

    resp1 = brain_resp_array[:, mask1].T
    resp2 = brain_resp_array[:, mask2].T

    X = np.vstack([resp1, resp2])
    y = np.array([0] * resp1.shape[0] + [1] * resp2.shape[0])

    # Train SVM on all data for visualization
    svm = LinearSVC(max_iter=10000, dual='auto', C=c_value)
    svm.fit(X, y)

    # Use first 2 principal components for visualization if more than 2 neurons
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(X)

    # Transform decision boundary
    # Create mesh grid
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Transform mesh back to original space and predict
    mesh_original = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    Z = svm.decision_function(mesh_original)
    Z = Z.reshape(xx.shape)

    xlabel, ylabel = 'PC1', 'PC2'

    # Create plot
    fig = go.Figure()

    # Decision boundary (tolerance margins)
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z,
        colorscale='RdBu',
        opacity=0.3,
        showscale=False,
        contours=dict(start=-1 / c_value, end=1 / c_value, size=0.5 / c_value)
    ))

    # Decision boundary line
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z,
        showscale=False,
        contours=dict(
            start=0, end=0,
            coloring='lines',
            showlabels=False
        ),
        line=dict(color='black', width=2)
    ))

    # Data points
    fig.add_trace(go.Scatter(
        x=X_vis[y == 0, 0], y=X_vis[y == 0, 1],
        mode='markers',
        name=f'Stim {stim1_idx}',
        marker=dict(size=8, color='blue', opacity=0.6)
    ))

    fig.add_trace(go.Scatter(
        x=X_vis[y == 1, 0], y=X_vis[y == 1, 1],
        mode='markers',
        name=f'Stim {stim2_idx}',
        marker=dict(size=8, color='red', opacity=0.6)
    ))

    fig.update_layout(
        title=f'SVM Classifier: {stim_type} Stim {stim1_idx} vs {stim2_idx} (C={c_value:.3f})',
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=700,
        height=700
    )

    return fig


# %% Main Analysis

# Save dataset information
print("=== Saving dataset information ===")
dataset_info_df = save_dataset_info(stim_types, file_path, save_dir)
print(dataset_info_df)

# Initialize storage
boxplot_data = {}
all_results = []
hyperparameter_results = {}

# HYPERPARAMETER TUNING
print("\n=== HYPERPARAMETER TUNING ===")

for stim in stim_types:
    print(f"\n=== Processing stim type: {stim} ===")

    stim_arrays = load_stim_data(file_path, stim)
    config = get_stim_config(stim, stim_arrays)
    brainRegionArray = stim_arrays["brainRegionArray"]
    stimArray = stim_arrays["stimArray"][0, :]
    uniqStims = np.unique(stimArray)
    uniqRegions = np.unique(brainRegionArray)

    for respRange in response_ranges:
        respArray = stim_arrays[f"{respRange}fr"]

        for brainRegion in uniqRegions:
            print(f"   -> {respRange} - {brainRegion}")
            brain_mask = brainRegionArray == brainRegion
            brain_resp_array = respArray[brain_mask, :]

            c_accuracies = []
            total_pairs = len(uniqStims) * (len(uniqStims) - 1)

            with tqdm(total=total_pairs * len(hyperparameters),
                      desc=f"{stim} | {respRange} | {brainRegion}", leave=True) as pbar:

                for c_value in hyperparameters:
                    svm_stim_vals = compute_pairwise_svm(
                        brain_resp_array, stimArray, uniqStims, c_value
                    )
                    pbar.update(total_pairs)

                    # Average upper triangle
                    upper_tri = np.triu_indices(len(uniqStims), k=1)
                    avg_accuracy = np.nanmean(svm_stim_vals[upper_tri])
                    c_accuracies.append(avg_accuracy)

            # Store results
            key = f"{stim}_{brainRegion}_{respRange}"
            hyperparameter_results[key] = {
                'c_values': hyperparameters,
                'accuracies': c_accuracies
            }

# CREATE HYPERPARAMETER TUNING PLOTS
print("\n=== Creating hyperparameter tuning plots ===")

for stim in stim_types:
    stim_arrays = load_stim_data(file_path, stim)
    brainRegionArray = stim_arrays["brainRegionArray"]
    uniqRegions = np.unique(brainRegionArray)

    fig = make_subplots(
        rows=len(uniqRegions), cols=len(response_ranges),
        subplot_titles=[f"{reg} - {rr}" for reg in uniqRegions for rr in response_ranges],
        x_title='C (Regularization Parameter)',
        y_title='Average Upper Triangle Accuracy'
    )

    for row_idx, brainRegion in enumerate(uniqRegions, start=1):
        for col_idx, respRange in enumerate(response_ranges, start=1):
            key = f"{stim}_{brainRegion}_{respRange}"

            if key in hyperparameter_results:
                c_values = hyperparameter_results[key]['c_values']
                accuracies = hyperparameter_results[key]['accuracies']

                best_idx = np.argmax(accuracies)
                best_c = c_values[best_idx]
                best_acc = accuracies[best_idx]

                fig.add_trace(
                    go.Scatter(
                        x=c_values, y=accuracies,
                        mode='lines+markers',
                        name=f"{brainRegion[:4]}-{respRange[:3]}",
                        showlegend=False,
                        line=dict(color=colors.get(brainRegion, '#000000')),
                        hovertemplate='C: %{x:.3f}<br>Accuracy: %{y:.3f}<extra></extra>'
                    ),
                    row=row_idx, col=col_idx
                )

                fig.add_trace(
                    go.Scatter(
                        x=[best_c], y=[best_acc],
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='star'),
                        showlegend=False,
                        hovertemplate=f'Best C: {best_c:.3f}<br>Accuracy: {best_acc:.3f}<extra></extra>'
                    ),
                    row=row_idx, col=col_idx
                )

                print(f"{key}: Best C = {best_c:.3f}, Accuracy = {best_acc:.3f}")

    fig.update_xaxes(type='log', title_text='C')
    fig.update_yaxes(title_text='Accuracy')
    fig.update_layout(
        title=f"Hyperparameter Tuning for {stim}",
        height=1400, width=1400
    )

    fig.write_html(f"{save_dir}hyperparameter_tuning_{stim}.html")
    print(f"Saved hyperparameter tuning plot for {stim}")

# Save hyperparameter results
hp_results_list = []
for key, data in hyperparameter_results.items():
    stim, region, window = key.split('_', 2)
    for c_val, acc in zip(data['c_values'], data['accuracies']):
        hp_results_list.append({
            'stim': stim,
            'region': region,
            'window': window,
            'C': c_val,
            'avg_accuracy': acc
        })

hp_df = pd.DataFrame(hp_results_list)
hp_save_path = os.path.join(save_dir, "hyperparameter_tuning_results.csv")
hp_df.to_csv(hp_save_path, index=False)
print(f"\nSaved hyperparameter tuning results to {hp_save_path}")

# FINAL ANALYSIS WITH OPTIMAL C
print("\n=== Running final analysis with optimal C values ===")

for stim in stim_types:
    print(f"\n=== Processing stim type: {stim} (with optimal C) ===")

    stim_arrays = load_stim_data(file_path, stim)
    config = get_stim_config(stim, stim_arrays)
    brainRegionArray = stim_arrays["brainRegionArray"]
    stimArray = stim_arrays["stimArray"][0, :]
    uniqStims = np.unique(stimArray)
    uniqRegions = np.unique(brainRegionArray)

    fig_sub = make_subplots(
        rows=len(uniqRegions), cols=len(response_ranges),
        subplot_titles=[f"{reg} - {rr}" for reg in uniqRegions for rr in response_ranges]
    )

    for col_idx, respRange in enumerate(response_ranges, start=1):
        respArray = stim_arrays[f"{respRange}fr"]

        for row_idx, brainRegion in enumerate(uniqRegions, start=1):
            print(f"   -> {respRange} - {brainRegion}")
            brain_mask = brainRegionArray == brainRegion
            brain_resp_array = respArray[brain_mask, :]

            # Get best C
            key = f"{stim}_{brainRegion}_{respRange}"
            if key in hyperparameter_results:
                c_values = hyperparameter_results[key]['c_values']
                accuracies = hyperparameter_results[key]['accuracies']
                best_c = c_values[np.argmax(accuracies)]
            else:
                best_c = 1.0

            svm_stim_vals = np.full((len(uniqStims), len(uniqStims)), np.nan)
            total_pairs = len(uniqStims) * (len(uniqStims) - 1)

            with tqdm(total=total_pairs, desc=f"{stim} | {respRange} | {brainRegion} (C={best_c:.3f})",
                      leave=True) as pbar:
                for i1, stim1 in enumerate(uniqStims):
                    mask1 = stimArray == stim1
                    resp1 = brain_resp_array[:, mask1]

                    for i2, stim2 in enumerate(uniqStims):
                        if i1 == i2:
                            pbar.update(1)
                            continue
                        mask2 = stimArray == stim2
                        resp2 = brain_resp_array[:, mask2]

                        X_pair = np.hstack([resp1, resp2]).T
                        y_pair = np.array([0] * resp1.shape[1] + [1] * resp2.shape[1])

                        if len(np.unique(y_pair)) < 2:
                            pbar.update(1)
                            continue

                        shuffle_idx = np.random.permutation(len(y_pair))
                        X_pair = X_pair[shuffle_idx]
                        y_pair = y_pair[shuffle_idx]

                        loo = LeaveOneOut()
                        acc_list = []
                        for train_idx, test_idx in loo.split(X_pair, y_pair):
                            svm = LinearSVC(max_iter=10000, dual='auto', C=best_c)
                            svm.fit(X_pair[train_idx], y_pair[train_idx])
                            acc_list.append(svm.score(X_pair[test_idx], y_pair[test_idx]))

                        accuracy = np.mean(acc_list)
                        svm_stim_vals[i1, i2] = accuracy
                        all_results.append({
                            "stim": stim,
                            "region": brainRegion,
                            "window": respRange,
                            "stim1": stim1,
                            "stim2": stim2,
                            "accuracy": accuracy,
                            "C": best_c
                        })
                        pbar.update(1)

                # Create classifier visualization for adjacent stimuli (e.g., 5 vs 6)
                if stim in ['AM', 'pureTones'] and len(uniqStims) > 1:
                    # Pick middle adjacent pair for visualization
                    mid_idx = len(uniqStims) // 2
                    if mid_idx < len(uniqStims) - 1:
                        vis_fig = create_classifier_visualization(
                            brain_resp_array, stimArray,
                            uniqStims[mid_idx], uniqStims[mid_idx + 1],
                            best_c, stim, save_dir
                        )
                        vis_path = os.path.join(save_dir,
                                                f"classifier_vis_{stim}_{brainRegion}_{respRange}.html")
                        vis_fig.write_html(vis_path)

                #