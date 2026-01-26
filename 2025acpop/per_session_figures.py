import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from jaratoolbox import settings, celldatabase

studyparams = __import__('2025acpop.studyparams').studyparams

# Constants
stim_types = ["naturalSound", "AM", "pureTones"]

figdataPath = os.path.join(settings.FIGURES_DATA_PATH)
results_dir = os.path.join(settings.SAVE_PATH)
np.random.seed(42)

# LOAD CELLDB
dbCoordsFilename = os.path.join(figdataPath, f'celldb_{studyparams.STUDY_NAME}_responsive_all_stims_index_new.h5')
celldb = celldatabase.load_hdf(dbCoordsFilename)
celldb['simpleSiteName'] = celldb['recordingSiteName'].str.split(',').apply(lambda x: x[0])

areas_of_interest = [
    "Dorsal auditory area",
    "Primary auditory area",
    "Ventral auditory area",
    "Posterior auditory area"
]
short_names = {
    "Dorsal auditory area": "Dorsal",
    "Primary auditory area": "Primary",
    "Ventral auditory area": "Ventral",
    "Posterior auditory area": "Posterior"
}
formal_names = {'pureTones': 'Pure Tones',
                'AM': 'AM White Noise',
                'naturalSound': 'Natural Sounds'}

aud_db = celldb[celldb['simpleSiteName'].isin(areas_of_interest)].reset_index()

# Store data for summary statistics
summary_stats = []

# Create separate figure for each sound type
for stim in stim_types:
    stim_arrays = np.load(os.path.join(figdataPath, f"fr_arrays_{stim}.npz"), allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    sessionArray = stim_arrays["sessionIDArray"]

    # Create figure for this sound type
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Count neurons per session per area
    session_counts = {}

    for area_idx, area in enumerate(areas_of_interest):
        area_mask = brainRegionArray == area
        area_sessions = sessionArray[area_mask]
        unique_sessions, counts = np.unique(area_sessions, return_counts=True)
        session_counts[area] = counts

        # Calculate statistics
        summary_stats.append({
            'Sound Type': formal_names[stim],
            'Brain Area': short_names[area],
            'Mean': np.mean(counts),
            'Median': np.median(counts),
            'Std': np.std(counts),
            'Min': np.min(counts),
            'Max': np.max(counts),
            'N Sessions': len(counts)
        })

        # Create histogram for this area
        ax = axes[area_idx]

        # Histogram with KDE overlay
        ax.hist(counts, bins=15, alpha=0.6, color='steelblue', edgecolor='black')

        # Add KDE curve
        from scipy import stats

        if len(counts) > 1 and np.std(counts) > 0:
            kde = stats.gaussian_kde(counts)
            x_range = np.linspace(counts.min(), counts.max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

        # Add vertical lines for mean and median
        mean_val = np.mean(counts)
        median_val = np.median(counts)
        ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_val:.1f}')

        # Formatting
        ax.set_xlabel('Neurons per Session', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{short_names[area]}', fontsize=14, fontweight='bold')

        # Add stats text box
        stats_text = f'n={len(counts)} sessions\nσ={np.std(counts):.1f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3, linestyle='--')

    plt.suptitle(f'{formal_names[stim]}: Distribution of Neurons per Session',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure with sound type in filename
    filename = f'neuron_counts_per_session_{stim}.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved: {filename}")

# Create summary statistics table
summary_df = pd.DataFrame(summary_stats)
print("\n" + "=" * 80)
print("SUMMARY STATISTICS: Neurons per Session")
print("=" * 80)
print(summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("Analysis complete! Figures saved to:", results_dir)
print("=" * 80)