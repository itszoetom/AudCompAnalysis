import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from jaratoolbox import settings
import os
from tqdm import tqdm
import studyparams

# SETTINGS
file_path = settings.FIGURES_DATA_PATH
save_dir = settings.SAVE_PATH
response_ranges = ["onset", "sustained", "offset"]
stim_types = ["naturalSound", "AM", "pureTones"]
colors = {
    'Dorsal auditory area': '#1f77b4',
    'Posterior auditory area': '#ff7f0e',
    'Primary auditory area': '#2ca02c',
    'Ventral auditory area': '#d62728'}

hyperparameters = np.logspace(-8, 8, 20)
boxplot_data = {}
all_results = []

# Dictionary to store hyperparameter tuning results
hyperparameter_results = {}

# Dictionary to store hard pairs for visualization
hard_pairs = {}

for stim in stim_types:
    print(f"\n=== Processing stim type: {stim} ===")

    # Stimulus setup
    if stim == 'AM':
        nTrials = 220
        nCategories = 11
        stimVals = np.arange(nCategories)
        labels = [str(i) for i in range(nCategories)]
    elif stim == 'naturalSound':
        nTrials = 200
        soundCats = studyparams.SOUND_CATEGORIES
        nCategories = len(soundCats)
        nInstances = 4
        stimVals = np.array([f"{soundCats[i]}_{j + 1}" for i in range(nCategories) for j in range(nInstances)])
        labels = stimVals
    elif stim == 'pureTones':
        nTrials = 320
        nCategories = 16
        stimVals = np.arange(nCategories)
        labels = [str(i) for i in range(nCategories)]

    stim_arrays = np.load(f"{file_path}/fr_arrays_{stim}.npz", allow_pickle=True)
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

            # Store accuracies for each C value
            c_accuracies = []

            total_pairs = len(uniqStims) * (len(uniqStims) - 1)

            with tqdm(total=total_pairs * len(hyperparameters),
                      desc=f"{stim} | {respRange} | {brainRegion}", leave=True) as pbar:

                # Loop through each C hyperparameter
                for c_value in hyperparameters:
                    svm_stim_vals = np.full((len(uniqStims), len(uniqStims)), np.nan)

                    for i1, stim1 in enumerate(uniqStims):
                        mask1 = stimArray == stim1
                        resp1 = brain_resp_array[:, mask1]  # neurons, trials

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

                            # Shuffle the dataset so trials are randomized (not all 0s then all 1s)
                            shuffle_idx = np.random.permutation(len(y_pair))
                            X_pair = X_pair[shuffle_idx]
                            y_pair = y_pair[shuffle_idx]

                            loo = LeaveOneOut()
                            acc_list = []
                            for train_idx, test_idx in loo.split(X_pair, y_pair):
                                svm = LinearSVC(max_iter=10000, dual='auto', C=c_value)
                                svm.fit(X_pair[train_idx], y_pair[train_idx])
                                acc_list.append(svm.score(X_pair[test_idx], y_pair[test_idx]))
                            accuracy = np.mean(acc_list)
                            svm_stim_vals[i1, i2] = accuracy
                            pbar.update(1)

                    # Average the accuracy values across all stimulus pairs in the upper triangle
                    upper_tri = np.triu_indices(len(uniqStims), k=1)
                    upper_tri_accuracies = svm_stim_vals[upper_tri]
                    avg_accuracy = np.nanmean(upper_tri_accuracies)
                    c_accuracies.append(avg_accuracy)

            # Store hyperparameter tuning results
            key = f"{stim}_{brainRegion}_{respRange}"
            hyperparameter_results[key] = {
                'c_values': hyperparameters,
                'accuracies': c_accuracies
            }

# Create hyperparameter tuning plots
print("\n=== Creating hyperparameter tuning plots ===")

for stim in stim_types:
    stim_arrays = np.load(f"{file_path}/fr_arrays_{stim}.npz", allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    uniqRegions = np.unique(brainRegionArray)

    # Create subplot for this stimulus type
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

                # Find best C
                best_idx = np.argmax(accuracies)
                best_c = c_values[best_idx]
                best_acc = accuracies[best_idx]

                # Add line trace
                fig.add_trace(
                    go.Scatter(
                        x=c_values,
                        y=accuracies,
                        mode='lines+markers',
                        name=f"{brainRegion[:4]}-{respRange[:3]}",
                        showlegend=False,
                        line=dict(color=colors.get(brainRegion, '#000000')),
                        hovertemplate='C: %{x:.3f}<br>Accuracy: %{y:.3f}<extra></extra>'
                    ),
                    row=row_idx, col=col_idx
                )

                # Add marker for best C
                fig.add_trace(
                    go.Scatter(
                        x=[best_c],
                        y=[best_acc],
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='star'),
                        showlegend=False,
                        hovertemplate=f'Best C: {best_c:.3f}<br>Accuracy: {best_acc:.3f}<extra></extra>'
                    ),
                    row=row_idx, col=col_idx
                )

                print(f"{key}: Best C = {best_c:.3f}, Accuracy = {best_acc:.3f}")

    # Update layout
    fig.update_xaxes(type='log', title_text='C')
    fig.update_yaxes(title_text='Accuracy')
    fig.update_layout(
        title=f"Hyperparameter Tuning for {stim}",
        height=1400,
        width=1400
    )

    # Save figure
    fig.write_html(f"{save_dir}hyperparameter_tuning_{stim}.html")
    print(f"Saved hyperparameter tuning plot for {stim}")

# Save hyperparameter results to CSV
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

# Now run final analysis with best C for each condition
print("\n=== Running final analysis with optimal C values ===")

for stim in stim_types:
    print(f"\n=== Processing stim type: {stim} (with optimal C) ===")

    # Stimulus setup
    if stim == 'AM':
        nTrials = 220
        nCategories = 11
        stimVals = np.arange(nCategories)
        labels = [str(i) for i in range(nCategories)]
    elif stim == 'naturalSound':
        nTrials = 200
        soundCats = studyparams.SOUND_CATEGORIES
        nCategories = len(soundCats)
        nInstances = 4
        stimVals = np.array([f"{soundCats[i]}_{j + 1}" for i in range(nCategories) for j in range(nInstances)])
        labels = stimVals
    elif stim == 'pureTones':
        nTrials = 320
        nCategories = 16
        stimVals = np.arange(nCategories)
        labels = [str(i) for i in range(nCategories)]

    stim_arrays = np.load(f"{file_path}/fr_arrays_{stim}.npz", allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    stimArray = stim_arrays["stimArray"][0, :]
    uniqStims = np.unique(stimArray)
    uniqRegions = np.unique(brainRegionArray)

    # Subplot figure
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

            # Get best C for this condition
            key = f"{stim}_{brainRegion}_{respRange}"
            if key in hyperparameter_results:
                c_values = hyperparameter_results[key]['c_values']
                accuracies = hyperparameter_results[key]['accuracies']
                best_c = c_values[np.argmax(accuracies)]
            else:
                best_c = 1.0  # Default

            svm_stim_vals = np.full((len(uniqStims), len(uniqStims)), np.nan)
            pair_data = {}  # Store data for each pair
            total_pairs = len(uniqStims) * (len(uniqStims) - 1)

            with tqdm(total=total_pairs, desc=f"{stim} | {respRange} | {brainRegion} (C={best_c:.3f})",
                      leave=True) as pbar:
                for i1, stim1 in enumerate(uniqStims):
                    mask1 = stimArray == stim1
                    resp1 = brain_resp_array[:, mask1]  # neurons, trials

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

                        # Shuffle the dataset so trials are randomized (not all 0s then all 1s)
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

                        # Store pair data including X, y for visualization (only upper triangle)
                        if i1 < i2:  # Only store upper triangle pairs
                            pair_data[(i1, i2)] = {
                                'X': X_pair,
                                'y': y_pair,
                                'accuracy': accuracy,
                                'stim1': stim1,
                                'stim2': stim2
                            }

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

            # Find hardest pair to classify (lowest accuracy in upper triangle)
            upper_tri = np.triu_indices(len(uniqStims), k=1)
            upper_tri_accuracies = svm_stim_vals[upper_tri]
            if len(upper_tri_accuracies) > 0 and not np.all(np.isnan(upper_tri_accuracies)):
                hardest_idx = np.nanargmin(upper_tri_accuracies)
                hardest_pair = (upper_tri[0][hardest_idx], upper_tri[1][hardest_idx])
                if hardest_pair in pair_data:  # Make sure we have the data
                    hard_pairs[key] = {
                        'pair_indices': hardest_pair,
                        'data': pair_data[hardest_pair],
                        'C': best_c
                    }

            # Store upper-triangle for boxplots
            boxplot_data[f"{brainRegion}_{stim}_{respRange}"] = svm_stim_vals[upper_tri].flatten()

            # Heatmap
            show_cb = (row_idx == 1 and col_idx == len(response_ranges))
            heatmap = go.Heatmap(
                z=svm_stim_vals,
                zmin=0, zmax=1,
                colorscale='Viridis',
                colorbar=dict(title="SVM Accuracy") if show_cb else None,
                hovertemplate='Stim1: %{x}<br>Stim2: %{y}<br>Accuracy: %{z}<extra></extra>'
            )
            fig_sub.add_trace(heatmap, row=row_idx, col=col_idx)
            fig_sub.update_xaxes(tickvals=list(range(len(uniqStims))), ticktext=labels, row=row_idx, col=col_idx)
            fig_sub.update_yaxes(tickvals=list(range(len(uniqStims))), autorange='reversed', ticktext=labels,
                                 row=row_idx, col=col_idx)

    # Save subplot
    os.makedirs(save_dir, exist_ok=True)
    fig_sub.update_layout(title=f"SVM Accuracy Heatmaps for {stim} (Optimized C)", height=1400, width=1400)
    fig_sub.write_html(f"{save_dir}SVM_heatmaps_{stim}_optimized.html")
    print(f"Saved {stim} heatmaps with optimized C")

# Save all results for later replotting
results_df = pd.DataFrame(all_results)
results_save_path = os.path.join(save_dir, "svm_pairwise_results.csv")
results_df.to_csv(results_save_path, index=False)
print(f"\nSaved all pairwise SVM results to {results_save_path}")

# Create decision boundary visualizations for hard pairs
print("\n=== Creating decision boundary visualizations ===")

for stim in stim_types:
    stim_arrays = np.load(f"{file_path}/fr_arrays_{stim}.npz", allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    uniqRegions = np.unique(brainRegionArray)

    # Create subplot for decision boundaries
    n_regions = len(uniqRegions)
    n_windows = len(response_ranges)

    fig_boundaries = make_subplots(
        rows=n_regions, cols=n_windows,
        subplot_titles=[f"{reg} - {rr}" for reg in uniqRegions for rr in response_ranges]
    )

    for row_idx, brainRegion in enumerate(uniqRegions, start=1):
        for col_idx, respRange in enumerate(response_ranges, start=1):
            key = f"{stim}_{brainRegion}_{respRange}"

            if key not in hard_pairs:
                continue

            pair_info = hard_pairs[key]
            X = pair_info['data']['X']
            y = pair_info['data']['y']
            best_c = pair_info['C']
            stim1 = pair_info['data']['stim1']
            stim2 = pair_info['data']['stim2']
            accuracy = pair_info['data']['accuracy']

            # Select two neurons with highest variance for visualization
            # This gives us the most informative 2D slice of the data
            neuron_vars = np.var(X, axis=0)
            top_neurons = np.argsort(neuron_vars)[-2:]  # Indices of top 2 neurons
            X_2d = X[:, top_neurons]

            # Train SVM on just these 2 neurons for visualization
            svm_2d = LinearSVC(max_iter=10000, dual='auto', C=best_c)
            svm_2d.fit(X_2d, y)

            # Create mesh for decision boundary
            x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
            y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))
            Z = svm_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Add contour for decision boundary
            fig_boundaries.add_trace(
                go.Contour(
                    x=np.linspace(x_min, x_max, 100),
                    y=np.linspace(y_min, y_max, 100),
                    z=Z,
                    showscale=False,
                    contours=dict(
                        start=0,
                        end=0,
                        size=1,
                        coloring='lines'
                    ),
                    line=dict(width=3, color='black'),
                    name='Decision Boundary',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row_idx, col=col_idx
            )

            # Add points for stimulus 1
            mask_stim1 = y == 0
            fig_boundaries.add_trace(
                go.Scatter(
                    x=X_2d[mask_stim1, 0],
                    y=X_2d[mask_stim1, 1],
                    mode='markers',
                    marker=dict(size=8, color='blue', symbol='circle'),
                    name=f'Stim {stim1}',
                    showlegend=(row_idx == 1 and col_idx == 1),
                    hovertemplate=f'Stim {stim1}<br>Neuron {top_neurons[0]}: %{{x:.2f}} Hz<br>Neuron {top_neurons[1]}: %{{y:.2f}} Hz<extra></extra>'
                ),
                row=row_idx, col=col_idx
            )

            # Add points for stimulus 2
            mask_stim2 = y == 1
            fig_boundaries.add_trace(
                go.Scatter(
                    x=X_2d[mask_stim2, 0],
                    y=X_2d[mask_stim2, 1],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='diamond'),
                    name=f'Stim {stim2}',
                    showlegend=(row_idx == 1 and col_idx == 1),
                    hovertemplate=f'Stim {stim2}<br>Neuron {top_neurons[0]}: %{{x:.2f}} Hz<br>Neuron {top_neurons[1]}: %{{y:.2f}} Hz<extra></extra>'
                ),
                row=row_idx, col=col_idx
            )

            # Update axes with neuron indices
            fig_boundaries.update_xaxes(title_text=f"Neuron {top_neurons[0]} (Hz)", row=row_idx, col=col_idx)
            fig_boundaries.update_yaxes(title_text=f"Neuron {top_neurons[1]} (Hz)", row=row_idx, col=col_idx)

            print(
                f"{key}: Hardest pair = {stim1} vs {stim2}, Accuracy = {accuracy:.3f}, C = {best_c:.3f}, Neurons = {top_neurons}")

    # Update layout
    fig_boundaries.update_layout(
        title=f"SVM Decision Boundaries for Hardest Pairs - {stim}<br><sub>Points colored by stimulus; black line = decision boundary; axes show firing rates of two most variable neurons</sub>",
        height=400 * n_regions,
        width=400 * n_windows,
        showlegend=True
    )

    # Save figure
    fig_boundaries.write_html(f"{save_dir}decision_boundaries_{stim}.html")
    print(f"Saved decision boundary visualization for {stim}")

print("\n=== Analysis complete ===")