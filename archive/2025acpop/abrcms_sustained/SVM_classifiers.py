import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from jaratoolbox import settings
from tqdm import tqdm
import studyparams

# SETTINGS
file_path = settings.FIGURES_DATA_PATH
save_dir = settings.SAVE_PATH
os.makedirs(save_dir, exist_ok=True)

response_window = "sustained"
stim_types = ["naturalSound", "AM", "pureTones"]
colors = {
    'Dorsal auditory area': '#1f77b4',
    'Posterior auditory area': '#ff7f0e',
    'Primary auditory area': '#2ca02c',
    'Ventral auditory area': '#d62728'
}

hyperparameters = np.logspace(-2, 4, 20)
all_results = []
max_neurons = 265
hyperparameter_results = {}


def get_stim_config(stim):
    """Get stimulus-specific configuration."""
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

    return nTrials, nCategories, stimVals, labels


def subsample_neurons(brain_resp_array, max_neurons, seed=42):
    """Randomly subsample neurons if there are too many."""
    n_neurons = brain_resp_array.shape[0]
    if n_neurons > max_neurons:
        np.random.seed(seed)
        selected_neurons = np.random.choice(n_neurons, max_neurons, replace=False)
        brain_resp_array = brain_resp_array[selected_neurons, :]
        print(f"      Subsampled from {n_neurons} to {max_neurons} neurons")
    else:
        print(f"      Using all {n_neurons} neurons")

    return brain_resp_array


def run_svm_pairwise(resp1, resp2, c_value):
    """
    Run SVM classification for a pair of stimuli with leave-one-out cross-validation.

    IMPORTANT CHANGE:
    resp1 and resp2 are assumed to ALREADY be standardized in a common space.
    We do NOT fit a StandardScaler here anymore.
    """
    # shape: neurons x trials
    X_pair = np.hstack([resp1, resp2]).T  # trials x features
    y_pair = np.array([0] * resp1.shape[1] + [1] * resp2.shape[1])

    # Shuffle
    shuffle_idx = np.random.permutation(len(y_pair))
    X_pair = X_pair[shuffle_idx]
    y_pair = y_pair[shuffle_idx]

    svm = LinearSVC(max_iter=10000, C=c_value)
    loo = LeaveOneOut()
    acc_list = []
    for tr, te in loo.split(X_pair):
        X_train = X_pair[tr]
        X_test = X_pair[te]
        acc = svm.fit(X_train, y_pair[tr]).score(X_test, y_pair[te])
        acc_list.append(acc)

    accuracy = np.mean(acc_list)
    return accuracy, X_pair, y_pair


def compute_pairwise_accuracies(brain_resp_array_std, stimArray, uniqStims, c_value, desc=""):
    """
    Compute pairwise SVM accuracies for all stimulus pairs.

    brain_resp_array_std: ALREADY standardized (neurons x trials) for this region+stim set.
    """
    svm_stim_vals = np.full((len(uniqStims), len(uniqStims)), np.nan)
    total_pairs = len(uniqStims) * (len(uniqStims) - 1)
    pair_accuracies = []

    with tqdm(total=total_pairs, desc=desc, leave=True) as pbar:
        for i1, stim1 in enumerate(uniqStims):
            mask1 = stimArray == stim1
            resp1 = brain_resp_array_std[:, mask1]

            for i2, stim2 in enumerate(uniqStims):
                if i1 == i2:
                    pbar.update(1)
                    continue
                mask2 = stimArray == stim2
                resp2 = brain_resp_array_std[:, mask2]

                accuracy, X_pair, y_pair = run_svm_pairwise(resp1, resp2, c_value)
                svm_stim_vals[i1, i2] = accuracy

                pair_accuracies.append({
                    'stim1': stim1, 'stim2': stim2, 'accuracy': accuracy,
                    'X_pair': X_pair, 'y_pair': y_pair
                })
                pbar.update(1)

    return svm_stim_vals, pair_accuracies


def create_classifier_visualization(brain_resp_array_std, pair_accuracies, stim, brainRegion,
                                    response_window, best_c, save_dir):
    """Create visualization of best and worst classifier examples."""
    pair_accuracies.sort(key=lambda x: x['accuracy'])
    worst_pair = pair_accuracies[0]
    best_pair = pair_accuracies[-1]

    # variance from standardized data
    neuron_vars = np.var(brain_resp_array_std, axis=1)
    neuron_idx = np.argsort(neuron_vars)[-2:]  # top 2 var neurons

    fig_viz = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Worst: {worst_pair['stim1']} vs {worst_pair['stim2']} (acc={worst_pair['accuracy']:.2f})",
            f"Best: {best_pair['stim1']} vs {best_pair['stim2']} (acc={best_pair['accuracy']:.2f})"
        ]
    )

    for viz_col, pair in enumerate([worst_pair, best_pair], start=1):
        # pair['X_pair'] is trials x features (already in standardized space for those trials)
        X_2d = pair['X_pair'][:, neuron_idx]
        y = pair['y_pair']

        # we can re-standardize just for plotting 2D
        scaler_viz = StandardScaler()
        X_2d_scaled = scaler_viz.fit_transform(X_2d)

        svm_final = LinearSVC(max_iter=10000, C=best_c)
        svm_final.fit(X_2d_scaled, y)

        w = svm_final.coef_[0]
        b = svm_final.intercept_[0]
        norm_w = np.linalg.norm(w)
        margin = 1.0 / norm_w if norm_w > 0 else 0.0

        x_min, x_max = X_2d_scaled[:, 0].min() - 1, X_2d_scaled[:, 0].max() + 1
        xx = np.linspace(x_min, x_max, 200)

        if abs(w[1]) < 1e-6:
            yy_boundary = np.full_like(xx, -b / (w[1] + 1e-6))
            slope_factor = 1.0
        else:
            yy_boundary = -(w[0] * xx + b) / w[1]
            slope_factor = np.sqrt(1 + (w[0] / w[1]) ** 2)

        yy_upper = yy_boundary + margin * slope_factor
        yy_lower = yy_boundary - margin * slope_factor

        fig_viz.add_trace(
            go.Scatter(
                x=X_2d_scaled[y == 0, 0],
                y=X_2d_scaled[y == 0, 1],
                mode='markers',
                marker=dict(color='#31688e', size=8),
                name='Stim 1',
                showlegend=(viz_col == 1)
            ),
            row=1, col=viz_col
        )
        fig_viz.add_trace(
            go.Scatter(
                x=X_2d_scaled[y == 1, 0],
                y=X_2d_scaled[y == 1, 1],
                mode='markers',
                marker=dict(color='#35b779', size=8),
                name='Stim 2',
                showlegend=(viz_col == 1)
            ),
            row=1, col=viz_col
        )

        # Decision boundary + margins
        fig_viz.add_trace(
            go.Scatter(
                x=xx, y=yy_boundary,
                mode='lines',
                line=dict(color='black', width=2),
                name='Decision boundary',
                showlegend=(viz_col == 1)
            ),
            row=1, col=viz_col
        )
        fig_viz.add_trace(
            go.Scatter(
                x=xx, y=yy_upper,
                mode='lines',
                line=dict(color='black', width=1, dash='dot'),
                name='Margin',
                showlegend=False
            ),
            row=1, col=viz_col
        )
        fig_viz.add_trace(
            go.Scatter(
                x=xx, y=yy_lower,
                mode='lines',
                line=dict(color='black', width=1, dash='dot'),
                name='Margin',
                showlegend=False
            ),
            row=1, col=viz_col
        )

        fig_viz.update_xaxes(title_text=f'Neuron {neuron_idx[0]} (scaled)', row=1, col=viz_col)
        fig_viz.update_yaxes(title_text=f'Neuron {neuron_idx[1]} (scaled)', row=1, col=viz_col)

    fig_viz.update_layout(
        height=500,
        width=1000,
        title=f"Classifier Examples: {stim} - {brainRegion} (C={best_c:.3f})"
    )
    fig_viz.write_html(os.path.join(
        save_dir,
        f"classifier_examples_{stim}_{brainRegion}_{response_window}.html"
    ))


# ================== HYPERPARAMETER TUNING PHASE ==================
for stim in stim_types:
    print(f"\n=== Processing stim type: {stim} ===")

    nTrials, nCategories, stimVals, labels = get_stim_config(stim)

    stim_arrays = np.load(os.path.join(file_path, f"fr_arrays_{stim}.npz"), allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    stimArray = stim_arrays["stimArray"][0, :]
    respArray = stim_arrays[f"{response_window}fr"]
    uniqStims = np.unique(stimArray)
    uniqRegions = np.unique(brainRegionArray)

    for brainRegion in uniqRegions:
        print(f"   -> {response_window} - {brainRegion}")
        brain_mask = brainRegionArray == brainRegion
        brain_resp_array = respArray[brain_mask, :]
        brain_resp_array = subsample_neurons(brain_resp_array, max_neurons)

        # GLOBAL STANDARDIZATION FOR THIS REGION + STIM SET
        # brain_resp_array: neurons x trials
        scaler = StandardScaler()
        brain_resp_array_std = scaler.fit_transform(brain_resp_array.T).T  # standardize over trials, keep neurons in rows

        c_accuracies = []
        total_pairs = len(uniqStims) * (len(uniqStims) - 1)

        with tqdm(total=total_pairs * len(hyperparameters),
                  desc=f"{stim} | {response_window} | {brainRegion}", leave=True) as pbar:

            for c_value in hyperparameters:
                svm_stim_vals = np.full((len(uniqStims), len(uniqStims)), np.nan)

                for i1, stim1 in enumerate(uniqStims):
                    mask1 = stimArray == stim1
                    resp1 = brain_resp_array_std[:, mask1]

                    for i2, stim2 in enumerate(uniqStims):
                        if i1 == i2:
                            pbar.update(1)
                            continue
                        mask2 = stimArray == stim2
                        resp2 = brain_resp_array_std[:, mask2]

                        accuracy, _, _ = run_svm_pairwise(resp1, resp2, c_value)
                        svm_stim_vals[i1, i2] = accuracy
                        pbar.update(1)

                upper_tri = np.triu_indices(len(uniqStims), k=1)
                upper_tri_accuracies = svm_stim_vals[upper_tri]
                avg_accuracy = np.nanmean(upper_tri_accuracies)
                c_accuracies.append(avg_accuracy)

        key = f"{stim}_{brainRegion}_{response_window}"
        hyperparameter_results[key] = {
            'c_values': hyperparameters,
            'accuracies': c_accuracies,
            # save scaler params if you want to reuse later
            # but we can just recompute in final phase the same way
        }

# ================== CREATE HYPERPARAMETER TUNING PLOTS ==================
print("\n=== Creating hyperparameter tuning plots ===")

for stim in stim_types:
    stim_arrays = np.load(os.path.join(file_path, f"fr_arrays_{stim}.npz"), allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    uniqRegions = np.unique(brainRegionArray)

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

            best_idx = np.argmax(accuracies)
            best_c = c_values[best_idx]
            best_acc = accuracies[best_idx]

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

    fig.update_xaxes(type='log', title_text='C')
    fig.update_yaxes(title_text='Accuracy')
    fig.update_layout(
        title=f"Hyperparameter Tuning for {stim}",
        height=400 * n_regions,
        width=1000
    )

    fig.write_html(os.path.join(save_dir, f"hyperparameter_tuning_{stim}.html"))
    print(f"Saved hyperparameter tuning plot for {stim}")

# ================== SAVE HYPERPARAMETER RESULTS ==================
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

# ================== FINAL ANALYSIS WITH OPTIMAL C VALUES ==================
print("\n=== Running final analysis with optimal C values ===")

for stim in stim_types:
    print(f"\n=== Processing stim type: {stim} (with optimal C) ===")

    stim_arrays = np.load(os.path.join(file_path, f"fr_arrays_{stim}.npz"), allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    stimArray = stim_arrays["stimArray"][0, :]
    respArray = stim_arrays[f"{response_window}fr"]
    uniqStims = np.unique(stimArray)
    uniqRegions = np.unique(brainRegionArray)

    fig_sub = make_subplots(
        rows=1, cols=len(uniqRegions),
        subplot_titles=[f"{reg} - {response_window}" for reg in uniqRegions]
    )

    for col_idx, brainRegion in enumerate(uniqRegions, start=1):
        print(f"   -> {response_window} - {brainRegion}")
        brain_mask = brainRegionArray == brainRegion
        brain_resp_array = respArray[brain_mask, :]
        brain_resp_array = subsample_neurons(brain_resp_array, max_neurons)

        # standardize globally for this region+stim set (same as tuning phase)
        scaler = StandardScaler()
        brain_resp_array_std = scaler.fit_transform(brain_resp_array.T).T

        # Get best C
        key = f"{stim}_{brainRegion}_{response_window}"
        if key in hyperparameter_results:
            c_values = hyperparameter_results[key]['c_values']
            accuracies = hyperparameter_results[key]['accuracies']
            best_c = c_values[np.argmax(accuracies)]
        else:
            best_c = 1.0

        svm_stim_vals, pair_accuracies = compute_pairwise_accuracies(
            brain_resp_array_std, stimArray, uniqStims, best_c,
            desc=f"{stim} | {response_window} | {brainRegion} (C={best_c:.3f})"
        )

        # Store results
        for pair in pair_accuracies:
            all_results.append({
                "stim": stim,
                "region": brainRegion,
                "window": response_window,
                "stim1": pair['stim1'],
                "stim2": pair['stim2'],
                "accuracy": pair['accuracy'],
                "C": best_c
            })

        # Create classifier visualization
        create_classifier_visualization(
            brain_resp_array_std, pair_accuracies, stim, brainRegion,
            response_window, best_c, save_dir
        )

        # Create heatmap
        show_cb = (col_idx == len(uniqRegions))
        heatmap = go.Heatmap(
            z=svm_stim_vals,
            colorscale='Viridis',
            colorbar=dict(title="SVM Accuracy") if show_cb else None,
            hovertemplate='Stim1: %{x}<br>Stim2: %{y}<br>Accuracy: %{z}<extra></extra>'
        )
        fig_sub.add_trace(heatmap, row=1, col=col_idx)
        fig_sub.update_xaxes(
            tickvals=list(range(len(uniqStims))),
            ticktext=uniqStims.astype(str),
            row=1, col=col_idx
        )
        fig_sub.update_yaxes(
            tickvals=list(range(len(uniqStims))),
            ticktext=uniqStims.astype(str),
            autorange='reversed',
            row=1, col=col_idx
        )

    fig_sub.update_layout(
        title=f"SVM Accuracy Heatmaps for {stim} (Optimized C)",
        height=600,
        width=400 * len(uniqRegions)
    )
    fig_sub.write_html(os.path.join(save_dir, f"SVM_heatmaps_{stim}_optimized.html"))
    print(f"Saved {stim} heatmaps with optimized C")

# ================== SAVE ALL RESULTS ==================
results_df = pd.DataFrame(all_results)
results_save_path = os.path.join(save_dir, "svm_pairwise_results.csv")
results_df.to_csv(results_save_path, index=False)
print(f"\nSaved all pairwise SVM results to {results_save_path}")
