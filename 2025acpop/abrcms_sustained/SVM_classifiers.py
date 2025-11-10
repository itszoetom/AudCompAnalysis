import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from jaratoolbox import settings
import os
from tqdm import tqdm
import studyparams

# SETTINGS
file_path = settings.FIGURES_DATA_PATH
save_dir = settings.SAVE_PATH
response_window = "sustained"  # Only sustained window
stim_types = ["naturalSound", "AM", "pureTones"]
colors = {
    'Dorsal auditory area': '#1f77b4',
    'Posterior auditory area': '#ff7f0e',
    'Primary auditory area': '#2ca02c',
    'Ventral auditory area': '#d62728'}

hyperparameters = np.logspace(-4, 5, 20)
boxplot_data = {}
all_results = []
max_neurons = 265  # Maximum neurons per brain region

# Dictionary to store hyperparameter tuning results
hyperparameter_results = {}

# HYPERPARAMETER TUNING PHASE
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
    respArray = stim_arrays[f"{response_window}fr"]
    uniqStims = np.unique(stimArray)
    uniqRegions = np.unique(brainRegionArray)

    for brainRegion in uniqRegions:
        print(f"   -> {response_window} - {brainRegion}")
        brain_mask = brainRegionArray == brainRegion
        brain_resp_array = respArray[brain_mask, :]

        # Randomly select up to max_neurons neurons
        n_neurons = brain_resp_array.shape[0]
        if n_neurons > max_neurons:
            np.random.seed(42)  # For reproducibility
            selected_neurons = np.random.choice(n_neurons, max_neurons, replace=False)
            brain_resp_array = brain_resp_array[selected_neurons, :]
            print(f"      Subsampled from {n_neurons} to {max_neurons} neurons")
        else:
            print(f"      Using all {n_neurons} neurons")

        # Store accuracies for each C value
        c_accuracies = []
        total_pairs = len(uniqStims) * (len(uniqStims) - 1)

        with tqdm(total=total_pairs * len(hyperparameters),
                  desc=f"{stim} | {response_window} | {brainRegion}", leave=True) as pbar:

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

                        # Shuffle the dataset so trials are randomized
                        shuffle_idx = np.random.permutation(len(y_pair))
                        X_pair = X_pair[shuffle_idx]
                        y_pair = y_pair[shuffle_idx]

                        # Standardize inputs before training
                        scaler = StandardScaler()
                        X_pair = scaler.fit_transform(X_pair)

                        svm = LinearSVC(max_iter=10000, dual='auto', C=c_value)
                        loo = LeaveOneOut()
                        acc_list = []
                        for tr, te in loo.split(X_pair):
                            X_train = X_pair[tr]
                            X_test = X_pair[te]
                            acc = svm.fit(X_train, y_pair[tr]).score(X_test, y_pair[te])
                            acc_list.append(acc)

                        accuracy = np.mean(acc_list)
                        svm_stim_vals[i1, i2] = accuracy
                        pbar.update(1)

                # Average the accuracy values across all stimulus pairs in the upper triangle
                upper_tri = np.triu_indices(len(uniqStims), k=1)
                upper_tri_accuracies = svm_stim_vals[upper_tri]
                avg_accuracy = np.nanmean(upper_tri_accuracies)
                c_accuracies.append(avg_accuracy)

        # Store hyperparameter tuning results
        key = f"{stim}_{brainRegion}_{response_window}"
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
    n_regions = len(uniqRegions)
    fig = make_subplots(
        rows=n_regions, cols=1,
        subplot_titles=[f"{reg} - {response_window}" for reg in uniqRegions],
        x_title='C (Regularization Parameter)',
        y_title='Average Upper Triangle Accuracy'
    )

    for row_idx, brainRegion in enumerate(uniqRegions, start=1):
        key = f"{stim}_{brainRegion}_{response_window}"

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
                    name=f"{brainRegion[:4]}",
                    showlegend=False,
                    line=dict(color=colors.get(brainRegion, '#000000')),
                    hovertemplate='C: %{x:.3f}<br>Accuracy: %{y:.3f}<extra></extra>'
                ),
                row=row_idx, col=1
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
                row=row_idx, col=1
            )

            print(f"{key}: Best C = {best_c:.3f}, Accuracy = {best_acc:.3f}")

    # Update layout
    fig.update_xaxes(type='log', title_text='C')
    fig.update_yaxes(title_text='Accuracy')
    fig.update_layout(
        title=f"Hyperparameter Tuning for {stim}",
        height=400 * n_regions,
        width=1000
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

# FINAL ANALYSIS WITH OPTIMAL C VALUES
print("\n=== Running final analysis with optimal C values ===")

for stim in stim_types:
    print(f"\n=== Processing stim type: {stim} (with optimal C) ===")

    stim_arrays = np.load(f"{file_path}/fr_arrays_{stim}.npz", allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    stimArray = stim_arrays["stimArray"][0, :]
    respArray = stim_arrays[f"{response_window}fr"]
    uniqStims = np.unique(stimArray)
    uniqRegions = np.unique(brainRegionArray)

    # Create figure with one column per brain region
    fig_sub = make_subplots(
        rows=1, cols=len(uniqRegions),
        subplot_titles=[f"{reg} - {response_window}" for reg in uniqRegions]
    )

    for col_idx, brainRegion in enumerate(uniqRegions, start=1):
        print(f"   -> {response_window} - {brainRegion}")
        brain_mask = brainRegionArray == brainRegion
        brain_resp_array = respArray[brain_mask, :]

        # Randomly select up to max_neurons neurons (same seed for consistency)
        n_neurons = brain_resp_array.shape[0]
        if n_neurons > max_neurons:
            np.random.seed(42)
            selected_neurons = np.random.choice(n_neurons, max_neurons, replace=False)
            brain_resp_array = brain_resp_array[selected_neurons, :]

        # Get best C for this condition
        key = f"{stim}_{brainRegion}_{response_window}"
        if key in hyperparameter_results:
            c_values = hyperparameter_results[key]['c_values']
            accuracies = hyperparameter_results[key]['accuracies']
            best_c = c_values[np.argmax(accuracies)]
        else:
            best_c = 1.0  # Default

        svm_stim_vals = np.full((len(uniqStims), len(uniqStims)), np.nan)
        total_pairs = len(uniqStims) * (len(uniqStims) - 1)

        # Track best and worst pairs for visualization
        pair_accuracies = []

        with tqdm(total=total_pairs, desc=f"{stim} | {response_window} | {brainRegion} (C={best_c:.3f})",
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

                    if len(np.unique(y_pair)) < 2 or X_pair.shape[0] < 10:
                        pbar.update(1)
                        continue

                    # Shuffle the dataset
                    shuffle_idx = np.random.permutation(len(y_pair))
                    X_pair = X_pair[shuffle_idx]
                    y_pair = y_pair[shuffle_idx]

                    # Standardize inputs
                    svm = LinearSVC(max_iter=10000, dual='auto', C=best_c)
                    loo = LeaveOneOut()
                    acc_list = []
                    for tr, te in loo.split(X_pair):
                        scaler_fold = StandardScaler()
                        X_train = scaler_fold.fit_transform(X_pair[tr])
                        X_test = scaler_fold.transform(X_pair[te])
                        acc = svm.fit(X_train, y_pair[tr]).score(X_test, y_pair[te])
                        acc_list.append(acc)

                    accuracy = np.mean(acc_list)
                    svm_stim_vals[i1, i2] = accuracy

                    # Store for finding best/worst pairs
                    pair_accuracies.append({
                        'stim1': stim1, 'stim2': stim2, 'accuracy': accuracy,
                        'X_pair': X_pair, 'y_pair': y_pair
                    })

                    all_results.append({
                        "stim": stim,
                        "region": brainRegion,
                        "window": response_window,
                        "stim1": stim1,
                        "stim2": stim2,
                        "accuracy": accuracy,
                        "C": best_c
                    })
                    pbar.update(1)

        # Create visualization for best and worst classifier pairs
        if len(pair_accuracies) >= 2 and brain_resp_array.shape[0] >= 2:
            pair_accuracies.sort(key=lambda x: x['accuracy'])
            worst_pair = pair_accuracies[0]
            best_pair = pair_accuracies[-1]

            # Select two neurons with highest variance
            neuron_vars = np.var(brain_resp_array, axis=1)
            neuron_idx = np.argsort(neuron_vars)[-2:]

            fig_viz = make_subplots(rows=1, cols=2,
                                    subplot_titles=[
                                        f"Worst: {worst_pair['stim1']} vs {worst_pair['stim2']} (acc={worst_pair['accuracy']:.2f})",
                                        f"Best: {best_pair['stim1']} vs {best_pair['stim2']} (acc={best_pair['accuracy']:.2f})"])

            for viz_col, pair in enumerate([worst_pair, best_pair], start=1):
                X_2d = pair['X_pair'][:, neuron_idx]
                y = pair['y_pair']

                # Standardize for visualization
                scaler_viz = StandardScaler()
                X_2d_scaled = scaler_viz.fit_transform(X_2d)

                # Train final SVM on full data
                svm_final = LinearSVC(max_iter=10000, dual='auto', C=best_c)
                svm_final.fit(X_2d_scaled, y)

                # Decision boundary
                w = svm_final.coef_[0]
                b = svm_final.intercept_[0]
                margin = 1 / np.linalg.norm(w)

                x_min, x_max = X_2d_scaled[:, 0].min() - 1, X_2d_scaled[:, 0].max() + 1
                xx = np.linspace(x_min, x_max, 100)
                yy_boundary = -(w[0] * xx + b) / w[1]
                yy_upper = yy_boundary + margin * np.sqrt(1 + (w[0] / w[1]) ** 2)
                yy_lower = yy_boundary - margin * np.sqrt(1 + (w[0] / w[1]) ** 2)

                # Plot data points
                fig_viz.add_trace(go.Scatter(x=X_2d_scaled[y == 0, 0], y=X_2d_scaled[y == 0, 1], mode='markers',
                                             marker=dict(color='#d62728', size=8), name=f'Stim {pair["stim1"]}',
                                             showlegend=(viz_col == 1)), row=1, col=viz_col)
                fig_viz.add_trace(go.Scatter(x=X_2d_scaled[y == 1, 0], y=X_2d_scaled[y == 1, 1], mode='markers',
                                             marker=dict(color='#1f77b4', size=8), name=f'Stim {pair["stim2"]}',
                                             showlegend=(viz_col == 1)), row=1, col=viz_col)

                # Plot boundaries
                fig_viz.add_trace(go.Scatter(x=xx, y=yy_boundary, mode='lines',
                                             line=dict(color='black', width=2), name='Decision',
                                             showlegend=(viz_col == 1)), row=1, col=viz_col)

                fig_viz.update_xaxes(title_text=f'Neuron {neuron_idx[0]} (scaled)', row=1, col=viz_col)
                fig_viz.update_yaxes(title_text=f'Neuron {neuron_idx[1]} (scaled)', row=1, col=viz_col)

            fig_viz.update_layout(height=500, width=1000,
                                  title=f"Classifier Examples: {stim} - {brainRegion} - {response_window} (C={best_c:.3f})")
            fig_viz.write_html(f"{save_dir}classifier_examples_{stim}_{brainRegion}_{response_window}.html")

        # Heatmap
        show_cb = (col_idx == len(uniqRegions))
        heatmap = go.Heatmap(
            z=svm_stim_vals,
            colorscale='Viridis',
            colorbar=dict(title="SVM Accuracy") if show_cb else None,
            hovertemplate='Stim1: %{x}<br>Stim2: %{y}<br>Accuracy: %{z}<extra></extra>'
        )
        fig_sub.add_trace(heatmap, row=1, col=col_idx)
        fig_sub.update_xaxes(tickvals=list(range(len(uniqStims))), ticktext=uniqStims, row=1, col=col_idx)
        fig_sub.update_yaxes(tickvals=list(range(len(uniqStims))), autorange='reversed', ticktext=uniqStims,
                             row=1, col=col_idx)

    # Save subplot
    os.makedirs(save_dir, exist_ok=True)
    fig_sub.update_layout(title=f"SVM Accuracy Heatmaps for {stim} (Optimized C)", height=600,
                          width=400 * len(uniqRegions))
    fig_sub.write_html(f"{save_dir}SVM_heatmaps_{stim}_optimized.html")
    print(f"Saved {stim} heatmaps with optimized C")

# Save all results for later replotting
results_df = pd.DataFrame(all_results)
results_save_path = os.path.join(save_dir, "svm_pairwise_results.csv")
results_df.to_csv(results_save_path, index=False)
print(f"\nSaved all pairwise SVM results to {results_save_path}")