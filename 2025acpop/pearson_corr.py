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

# Parameters
n_neurons_per_session = 30
n_subsamplings = 100

# LOAD CELLDB
dbCoordsFilename = os.path.join(figdataPath, f'celldb_{studyparams.STUDY_NAME}_responsive_all_stims_index_new.h5')
celldb = celldatabase.load_hdf(dbCoordsFilename)
celldb['simpleSiteName'] = celldb['recordingSiteName'].str.split(',').apply(lambda x: x[0])

# Exclude Posterior
areas_of_interest = [
    "Dorsal auditory area",
    "Primary auditory area",
    "Ventral auditory area"
]
short_names = {
    "Dorsal auditory area": "Dorsal",
    "Primary auditory area": "Primary",
    "Ventral auditory area": "Ventral"
}
formal_names = {
    'pureTones': 'Pure Tones',
    'AM': 'AM White Noise',
    'naturalSound': 'Natural Sounds'
}

# Load all stim data
stim_data = {}
for stim in stim_types:
    stim_arrays = np.load(os.path.join(figdataPath, f"fr_arrays_{stim}.npz"), allow_pickle=True)

    # Standardize firing rates
    respArray_raw = stim_arrays["sustainedfr"]
    scaler = StandardScaler()
    respArray = scaler.fit_transform(respArray_raw.T).T

    stim_data[stim] = {
        'brainRegionArray': stim_arrays["brainRegionArray"],
        'sessionArray': stim_arrays["sessionIDArray"],
        'stimArray': stim_arrays["stimArray"][0, :],  # 1D trial vector
        'respArray': respArray  # neurons × trials
    }

# Find sessions with >= n_neurons_per_session for each area
valid_sessions_per_area = {}
for area in areas_of_interest:
    valid_sessions = None

    for stim in stim_types:
        area_mask = stim_data[stim]['brainRegionArray'] == area
        area_sessions = stim_data[stim]['sessionArray'][area_mask]
        unique_sessions, counts = np.unique(area_sessions, return_counts=True)
        sessions_with_enough = set(unique_sessions[counts >= n_neurons_per_session])

        if valid_sessions is None:
            valid_sessions = sessions_with_enough
        else:
            valid_sessions = valid_sessions.intersection(sessions_with_enough)

    valid_sessions_per_area[area] = sorted(list(valid_sessions))
    print(f"{short_names[area]}: {len(valid_sessions_per_area[area])} sessions with >= {n_neurons_per_session} neurons")

# Find minimum number of sessions across areas
min_sessions = min(len(sessions) for sessions in valid_sessions_per_area.values())
print(f"\nUsing {min_sessions} sessions per area (minimum across areas)")

# Equalize number of sessions across areas
for area in areas_of_interest:
    valid_sessions_per_area[area] = valid_sessions_per_area[area][:min_sessions]

# Compute discriminability matrices for each stimulus type and area
results = {}

for area in areas_of_interest:
    print(f"\nProcessing {short_names[area]}...")
    sessions = valid_sessions_per_area[area]
    results[area] = {}

    for stim in stim_types:
        print(f"  Stimulus: {formal_names[stim]}")

        # Get unique stimulus values
        stim_array = stim_data[stim]['stimArray']
        unique_stims = np.unique(stim_array)
        n_stims = len(unique_stims)

        # Store C and D matrices for each session and subsampling
        C_matrices = []

        for session_idx, session_id in enumerate(sessions):
            for k in range(n_subsamplings):
                # Get neurons from this session
                session_mask = stim_data[stim]['sessionArray'] == session_id
                area_mask = stim_data[stim]['brainRegionArray'] == area
                mask = session_mask & area_mask

                session_resp = stim_data[stim]['respArray'][mask, :]  # neurons × trials
                session_stims = stim_array  # All trials

                # Randomly subsample neurons
                n_available = session_resp.shape[0]
                if n_available > n_neurons_per_session:
                    selected_idx = np.random.choice(n_available, n_neurons_per_session, replace=False)
                    session_resp = session_resp[selected_idx, :]

                # Compute trial average for each unique stimulus
                stim_vectors = []
                for stim_val in unique_stims:
                    trials_mask = session_stims == stim_val
                    if np.sum(trials_mask) > 0:
                        # Average across trials for this stimulus: (n_neurons,)
                        avg_resp = np.mean(session_resp[:, trials_mask], axis=1)
                        stim_vectors.append(avg_resp)

                # Stack: (n_unique_stims, n_neurons)
                stim_vectors = np.array(stim_vectors)

                # Compute correlation matrix between stimuli
                C = np.corrcoef(stim_vectors)  # (n_stims, n_stims)
                C_matrices.append(C)

            if (session_idx + 1) % 5 == 0:
                print(f"    Processed {session_idx + 1}/{len(sessions)} sessions")

        # Average across all sessions and subsamplings
        C_matrices = np.array(C_matrices)
        C_avg = np.mean(C_matrices, axis=0)  # (n_stims, n_stims)
        D_avg = 1 - C_avg

        results[area][stim] = {
            'C_avg': C_avg,
            'D_avg': D_avg,
            'unique_stims': unique_stims,
            'n_sessions': len(sessions),
            'n_subsamplings': n_subsamplings
        }

# Create visualizations - separate figure for each stimulus type
for stim in stim_types:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for area_idx, area in enumerate(areas_of_interest):
        C_avg = results[area][stim]['C_avg']
        D_avg = results[area][stim]['D_avg']
        unique_stims = results[area][stim]['unique_stims']
        n_sess = results[area][stim]['n_sessions']
        n_sub = results[area][stim]['n_subsamplings']

        # Create labels based on stimulus type
        if stim == 'pureTones':
            labels = [f'{int(s)} Hz' if s >= 1000 else f'{int(s * 1000)} Hz'
                      for s in unique_stims]
        elif stim == 'AM':
            labels = [f'{int(s)} Hz' for s in unique_stims]
        else:  # naturalSound
            # Use SOUND_CATEGORIES from studyparams
            SOUND_CATEGORIES = studyparams.SOUND_CATEGORIES
            sounds_per_category = len(unique_stims) // len(SOUND_CATEGORIES)
            labels = []
            for i, stim_val in enumerate(unique_stims):
                category_idx = int(i // sounds_per_category)
                within_category_idx = int(i % sounds_per_category) + 1
                if category_idx < len(SOUND_CATEGORIES):
                    labels.append(f'{SOUND_CATEGORIES[category_idx]}_{within_category_idx}')
                else:
                    labels.append(f'Sound_{int(stim_val)}')

        # Plot C (correlation) - top row
        ax_c = axes[0, area_idx]
        im_c = ax_c.imshow(C_avg, cmap='viridis', aspect='auto', vmin=-0.5, vmax=1)
        ax_c.set_xticks(range(len(labels)))
        ax_c.set_yticks(range(len(labels)))
        ax_c.set_xticklabels(labels, rotation=90, fontsize=8)
        ax_c.set_yticklabels(labels, fontsize=8)
        ax_c.set_title(f'{short_names[area]}\nCorrelation (C)', fontsize=12, fontweight='bold')
        plt.colorbar(im_c, ax=ax_c, label='Correlation')

        # Add grid
        ax_c.set_xticks(np.arange(len(labels)) - 0.5, minor=True)
        ax_c.set_yticks(np.arange(len(labels)) - 0.5, minor=True)
        ax_c.grid(which='minor', color='white', linewidth=0.5)

        # Plot D (dissimilarity) - bottom row
        ax_d = axes[1, area_idx]
        im_d = ax_d.imshow(D_avg, cmap='viridis', aspect='auto', vmin=0, vmax=2)
        ax_d.set_xticks(range(len(labels)))
        ax_d.set_yticks(range(len(labels)))
        ax_d.set_xticklabels(labels, rotation=90, fontsize=8)
        ax_d.set_yticklabels(labels, fontsize=8)
        ax_d.set_title(f'{short_names[area]}\nDissimilarity (D = 1 - C)', fontsize=12, fontweight='bold')
        plt.colorbar(im_d, ax=ax_d, label='Dissimilarity')

        # Add grid
        ax_d.set_xticks(np.arange(len(labels)) - 0.5, minor=True)
        ax_d.set_yticks(np.arange(len(labels)) - 0.5, minor=True)
        ax_d.grid(which='minor', color='white', linewidth=0.5)

        # Add info text on rightmost column
        if area_idx == 2:
            info_text = f'n={n_sess} sessions\n{n_sub} subsamplings\n{n_neurons_per_session} neurons/session'
            ax_c.text(1.3, 0.5, info_text, transform=ax_c.transAxes,
                      fontsize=10, verticalalignment='center',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.suptitle(f'{formal_names[stim]}: Within-Stimulus Discriminability Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = f'discriminability_{stim}_n{n_neurons_per_session}.png'
    plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

print("\n" + "=" * 80)
print(f"Analysis complete! Figures saved to: {results_dir}")
print("=" * 80)