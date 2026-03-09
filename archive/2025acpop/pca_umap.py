import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from jaratoolbox import settings
import os

studyparams = __import__('2025acpop.studyparams').studyparams

# SETTINGS
file_path = settings.FIGURES_DATA_PATH
response_ranges = ["onset", "sustained", "offset"]
stim_types = ["naturalSound", "AM", "pureTones"]
colors = {
    'Dorsal auditory area': '#1f77b4',
    'Posterior auditory area': '#ff7f0e',
    'Primary auditory area': '#2ca02c',
    'Ventral auditory area': '#d62728'}

# Create output directory
output_dir = "/Users/zoetomlinson/Desktop/MurrayLab/neuronalDataResearch/Figures/PCA/"
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
MAX_NEURONS = 265


def encode_natural_sounds(stim_array):
    """Encode natural sound strings as numbers 1-20"""
    unique_sounds = np.unique(stim_array)
    sound_to_num = {sound: i + 1 for i, sound in enumerate(unique_sounds)}
    return np.array([sound_to_num[s] for s in stim_array])


for stim in stim_types:
    print(f"\n=== Processing stim type: {stim} ===")

    # Load data
    stim_arrays = np.load(f"{file_path}fr_arrays_{stim}.npz", allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    stimArray = stim_arrays["stimArray"][0, :]
    uniqRegions = np.unique(brainRegionArray)

    # Encode frequencies for coloring
    if stim == 'naturalSound':
        freq_encoded = encode_natural_sounds(stimArray)
    else:
        freq_encoded = np.log(stimArray.astype(float))

    # Create figure for PCA (4 rows x 3 cols)
    fig_pca, axes_pca = plt.subplots(4, 3, figsize=(15, 16))
    fig_pca.suptitle(f'PCA - {stim}', fontsize=16, fontweight='bold', y=0.995)

    # Create figure for UMAP (4 rows x 3 cols)
    fig_umap, axes_umap = plt.subplots(4, 3, figsize=(15, 16))
    fig_umap.suptitle(f'UMAP - {stim}', fontsize=16, fontweight='bold', y=0.995)

    for row_idx, brainRegion in enumerate(uniqRegions):
        brain_mask = brainRegionArray == brainRegion

        for col_idx, respRange in enumerate(response_ranges):
            print(f"   -> {brainRegion} - {respRange}")

            respArray = stim_arrays[f"{respRange}fr"]
            brain_resp_array = respArray[brain_mask, :]  # neurons × trials

            # Randomly subsample neurons if more than MAX_NEURONS
            n_neurons = brain_resp_array.shape[0]
            if n_neurons > MAX_NEURONS:
                neuron_indices = np.random.choice(n_neurons, MAX_NEURONS, replace=False)
                brain_resp_array = brain_resp_array[neuron_indices, :]
                n_neurons_used = MAX_NEURONS
            else:
                n_neurons_used = n_neurons

            # Transpose so trials are rows and neurons are columns
            X = brain_resp_array.T  # trials × neurons

            ax_pca = axes_pca[row_idx, col_idx]
            ax_umap = axes_umap[row_idx, col_idx]

            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # PCA
            n_components_pca = min(10, X_scaled.shape[1])
            pca = PCA(n_components=n_components_pca)
            X_pca = pca.fit_transform(X_scaled)

            # UMAP
            n_neighbors = min(15, X_scaled.shape[0] - 1)
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1,
                                random_state=42, n_jobs=1)
            X_umap = reducer.fit_transform(X_scaled)

            # PCA plot
            scatter_pca = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1],
                                         c=freq_encoded, cmap='rainbow',
                                         alpha=0.6, s=15, vmin=freq_encoded.min(),
                                         vmax=freq_encoded.max())
            ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=8)
            ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=8)
            ax_pca.set_title(f'{brainRegion}\n{respRange}\nn={n_neurons_used} neurons',
                             fontsize=9)
            ax_pca.tick_params(labelsize=7)

            # UMAP plot
            scatter_umap = ax_umap.scatter(X_umap[:, 0], X_umap[:, 1],
                                           c=freq_encoded, cmap='rainbow',
                                           alpha=0.6, s=15, vmin=freq_encoded.min(),
                                           vmax=freq_encoded.max())
            ax_umap.set_xlabel('UMAP 1', fontsize=8)
            ax_umap.set_ylabel('UMAP 2', fontsize=8)
            ax_umap.set_title(f'{brainRegion}\n{respRange}\nn={n_neurons_used} neurons',
                              fontsize=9)
            ax_umap.tick_params(labelsize=7)

    # Add colorbars
    fig_pca.subplots_adjust(right=0.92, hspace=0.35, wspace=0.3)
    cbar_ax_pca = fig_pca.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar_pca = fig_pca.colorbar(scatter_pca, cax=cbar_ax_pca)
    if stim == 'naturalSound':
        cbar_pca.set_label('Sound ID (1-20)', fontsize=10)
    else:
        cbar_pca.set_label('Frequency', fontsize=10)
    cbar_pca.ax.tick_params(labelsize=8)

    fig_umap.subplots_adjust(right=0.92, hspace=0.35, wspace=0.3)
    cbar_ax_umap = fig_umap.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar_umap = fig_umap.colorbar(scatter_umap, cax=cbar_ax_umap)
    if stim == 'naturalSound':
        cbar_umap.set_label('Sound ID (1-20)', fontsize=10)
    else:
        cbar_umap.set_label('Frequency', fontsize=10)
    cbar_umap.ax.tick_params(labelsize=8)

    # Save figures
    pca_filename = f"{stim}_PCA_all_regions_windows.png"
    umap_filename = f"{stim}_UMAP_all_regions_windows.png"

    fig_pca.savefig(os.path.join(output_dir, pca_filename), dpi=200, bbox_inches='tight')
    fig_umap.savefig(os.path.join(output_dir, umap_filename), dpi=200, bbox_inches='tight')

    plt.close(fig_pca)
    plt.close(fig_umap)

    print(f"   Saved: {pca_filename}")
    print(f"   Saved: {umap_filename}")

print(f"\n=== All figures saved to: {output_dir} ===")