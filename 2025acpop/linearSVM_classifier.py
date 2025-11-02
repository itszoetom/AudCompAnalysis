import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut
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

hyperparameters = np.logspace(-3, 2, 30)
boxplot_data = {}
all_results = []

# Dictionary to store hyperparameter tuning results
hyperparameter_results = {}

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
        stimVals = np.array([f"{soundCats[i]}_{j+1}" for i in range(nCategories) for j in range(nInstances)])
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

                # Store upper-triangle for boxplots
                upper_tri = np.triu_indices(len(uniqStims), k=1)
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