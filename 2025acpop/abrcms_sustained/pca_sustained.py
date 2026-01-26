import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from jaratoolbox import settings
import os

studyparams = __import__('2025acpop.studyparams').studyparams

# SETTINGS
file_path = settings.FIGURES_DATA_PATH
stim_types = ["naturalSound", "AM", "pureTones"]
output_dir = "/Users/zoetomlinson/Desktop/MurrayLab/ABRCMS_poster_figs/PCA/"
os.makedirs(output_dir, exist_ok=True)

# Formal names dictionary
formal_names = {
    'pureTones': 'Pure Tones',
    'AM': 'AM White Noise',
    'naturalSound': 'Natural Sounds'
}

short_names = {
    "Dorsal auditory area": "Dorsal",
    "Primary auditory area": "Primary",
    "Ventral auditory area": "Ventral",
    "Posterior auditory area": "Posterior"
}

np.random.seed(42)
MAX_NEURONS = 265

for stim in stim_types:
    print(f"\n=== Processing stim type: {stim} ===")

    # Load data
    stim_arrays = np.load(f"{file_path}fr_arrays_{stim}.npz", allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    stimArray = stim_arrays["stimArray"][0, :]  # trials-long
    uniqRegions = np.unique(brainRegionArray)

    # COLOR VALUES & LABELS
    if stim == "naturalSound":
        soundCats = studyparams.SOUND_CATEGORIES
        nCategories = len(soundCats)
        nInstances = 4
        stimVals = np.array([f"{soundCats[i]}_{j + 1}" for i in range(nCategories) for j in range(nInstances)])
        # stimArray contains numeric indices into this list (floats -> cast to int)
        stim_indices = np.array(stimArray, dtype=int)
        color_values = stim_indices + 1
        labels = stimVals
        colorbar_label = "Natural sound"
        is_natural = True
    else:
        # AM / PT: actual numbers
        color_values = stimArray.astype(float)
        if stim == "AM":
            colorbar_label = "AM White Noise rate (Hz)"
        else:
            colorbar_label = "Pure tone (Hz)"
        is_natural = False

    # FIGURE SETUP - increased height and adjusted spacing
    fig, axes = plt.subplots(1, 4, figsize=(45, 11), squeeze=False)
    fig.suptitle(f"Principal Component Analysis PC1 vs PC2 - {formal_names.get(stim, stim)}", fontsize=55, fontweight="bold", y=1.1)

    last_scatter = None

    for col_idx, brainRegion in enumerate(uniqRegions):
        ax = axes[0, col_idx]

        brain_mask = (brainRegionArray == brainRegion)
        respArray = stim_arrays["sustainedfr"]
        brain_resp_array = respArray[brain_mask, :]

        n_neurons = brain_resp_array.shape[0]
        if n_neurons > MAX_NEURONS:
            neuron_indices = np.random.choice(n_neurons, MAX_NEURONS, replace=False)
            brain_resp_array = brain_resp_array[neuron_indices, :]
            n_neurons_used = MAX_NEURONS
        else:
            n_neurons_used = n_neurons

        # trials × neurons
        X = brain_resp_array.T

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_components_pca = min(10, X_scaled.shape[1])
        pca = PCA(n_components=n_components_pca)
        X_pca = pca.fit_transform(X_scaled)

        last_scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=color_values,
            cmap="rainbow",
            s=120,
            alpha=0.75,
            vmin=color_values.min(),
            vmax=color_values.max(),
        )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=45, labelpad=15)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=45, labelpad=15)

        # Use short names for brain regions
        region_short = short_names.get(brainRegion, brainRegion)
        ax.set_title(f"{region_short}\nn={n_neurons_used} neurons", fontsize=40, pad=25)
        ax.tick_params(labelsize=40, pad=8)

    # Adjusted spacing for less cramping
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.18, wspace=0.38)

    # COLORBAR
    cbar_ax = fig.add_axes([0.06, 0, 0.9, 0.04])  # [left, bottom, width, height]
    cbar = fig.colorbar(last_scatter, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(colorbar_label, fontsize=40, labelpad=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=45, pad=10)

    if is_natural:
        all_vals = np.arange(1, len(labels) + 1)
        tick_labels = [labels[int(v - 1)] for v in all_vals]
        tick_labels = [label.replace("_", " ") for label in labels]
        cbar.set_ticks(all_vals)
        cbar.set_ticklabels(tick_labels, fontsize=45)
        for lbl in cbar.ax.get_xticklabels():
            lbl.set_rotation(25)
            lbl.set_ha("right")
    else:
        uniq_vals = np.unique(color_values)
        cbar.set_ticks(uniq_vals)
        cbar.set_ticklabels([f"{v:.0f}" for v in uniq_vals], fontsize=35)
        for lbl in cbar.ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha("right")

    # SAVE
    outname = f"{stim}_PCA_all_regions_sustained_poster.png"
    fig.savefig(os.path.join(output_dir, outname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {outname}")

print(f"\n=== All figures saved to: {output_dir} ===") 