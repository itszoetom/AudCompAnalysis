import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from jaratoolbox import settings
import os
from tqdm import tqdm
import studyparams
# studyparams = __import__('2025acpop.studyparams').studyparams

# SETTINGS
file_path = settings.FIGURES_DATA_PATH
save_dir = settings.SAVE_PATH + "LDA"
response_ranges = ["onset", "sustained", "offset"]
stim_types = ["naturalSound", "AM", "pureTones"]
colors = {
    'Dorsal auditory area': '#1f77b4',
    'Posterior auditory area': '#ff7f0e',
    'Primary auditory area': '#2ca02c',
    'Ventral auditory area': '#d62728'}

lda = LinearDiscriminantAnalysis()
boxplot_data = {}
all_results = []

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

    # Subplot figure
    fig_sub = make_subplots(
        rows=len(uniqRegions), cols=len(response_ranges),
        subplot_titles=[f"{reg} - {rr}" for reg in uniqRegions for rr in response_ranges]
    )

    for col_idx, respRange in enumerate(response_ranges, start=1):
        respArray = stim_arrays[f"{respRange}fr"]

        for row_idx, brainRegion in enumerate(uniqRegions, start=1):
            # TODO: subset neuron count for each brain region down to 265 (randomly select 265 neurons)
            print(f"   -> {respRange} - {brainRegion}")
            brain_mask = brainRegionArray == brainRegion
            brain_resp_array = respArray[brain_mask, :]

            lda_stim_vals = np.full((len(uniqStims), len(uniqStims)), np.nan)
            total_pairs = len(uniqStims) * (len(uniqStims) - 1)

            with tqdm(total=total_pairs, desc=f"{stim} | {respRange} | {brainRegion}", leave=True) as pbar:
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

                        # TODO: standardize the inputs before training

                        # Shuffle the dataset so trials are randomized (not all 0s then all 1s)
                        shuffle_idx = np.random.permutation(len(y_pair))
                        X_pair = X_pair[shuffle_idx]
                        y_pair = y_pair[shuffle_idx]

                        loo = LeaveOneOut()
                        acc_list = []
                        for train_idx, test_idx in loo.split(X_pair, y_pair):
                            lda.fit(X_pair[train_idx], y_pair[train_idx])
                            acc_list.append(lda.score(X_pair[test_idx], y_pair[test_idx]))
                        accuracy = np.mean(acc_list)
                        lda_stim_vals[i1, i2] = accuracy
                        all_results.append({
                            "stim": stim,
                            "region": brainRegion,
                            "window": respRange,
                            "stim1": stim1,
                            "stim2": stim2,
                            "accuracy": accuracy
                        })
                        pbar.update(1)

            # Store upper-triangle for boxplots
            upper_tri = np.triu_indices(len(uniqStims), k=1)
            boxplot_data[f"{brainRegion}_{stim}_{respRange}"] = lda_stim_vals[upper_tri].flatten()

            # Heatmap
            show_cb = (row_idx == 1 and col_idx == len(response_ranges))
            heatmap = go.Heatmap(
                z=lda_stim_vals,
                zmin=0, zmax=1,
                colorscale='Viridis',
                colorbar=dict(title="LDA Accuracy") if show_cb else None,
                hovertemplate='Stim1: %{x}<br>Stim2: %{y}<br>Accuracy: %{z}<extra></extra>'
            )
            fig_sub.add_trace(heatmap, row=row_idx, col=col_idx)

            fig_sub.update_xaxes(tickvals=list(range(len(uniqStims))), ticktext=labels, row=row_idx, col=col_idx)
            fig_sub.update_yaxes(tickvals=list(range(len(uniqStims))), autorange='reversed', ticktext=labels, row=row_idx, col=col_idx)

    # Save subplot
    os.makedirs(save_dir, exist_ok=True)
    fig_sub.update_layout(title=f"LDA Accuracy Heatmaps for {stim}", height=1400, width=1400)
    fig_sub.write_html(f"{save_dir}LDA_heatmaps_{stim}.html")
    print(f"Saved {stim} heatmaps")

# Save all results for later replotting
results_df = pd.DataFrame(all_results)
results_save_path = os.path.join(save_dir, "lda_pairwise_results.csv")
results_df.to_csv(results_save_path, index=False)
print(f"Saved all pairwise LDA results to {results_save_path}")