from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np
from jaratoolbox import celldatabase, settings
studyparams = __import__('2025acpop.studyparams').studyparams
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cv_results = []
stimTypes = ["AM", "pureTones", "naturalSound"]
response_ranges = ["onsetfr", "sustainedfr", "offsetfr"]
figdataPath = os.path.join(settings.FIGURES_DATA_PATH)

for stimType in stimTypes:
    # Set trial info
    if stimType == "AM":
        nTrials = 220
        nCategories = 11
    elif stimType == "pureTones":
        nTrials = 320
        nCategories = 16
    elif stimType == "naturalSound":
        nTrials = 200
        nCategories = len(studyparams.SOUND_CATEGORIES)
        soundCats = studyparams.SOUND_CATEGORIES
        nInstances = 4
        stimVals = np.empty(nInstances*nCategories, dtype=object)
        for i in range(nCategories):
            for j in range(nInstances):
                stimVals[i*nInstances+j] = soundCats[i] + f"_{j+1}"

    # Load arrays
    fr_arrays_filename = os.path.join(figdataPath, f"fr_arrays_{stimType}.npz")
    data = np.load(fr_arrays_filename, allow_pickle=True)

    y = data["stimArray"]
    brainRegionArray = data["brainRegionArray"]
    uniqRegions = np.unique(brainRegionArray)

    for respRange in response_ranges:
        respArray = data[respRange]  # neurons x trials

        for brainRegion in uniqRegions:
            mask = brainRegionArray == brainRegion
            X = respArray[mask, :].T  # trials x neurons
            Y = y[mask, :].T          # trials x neurons

            print(f"Shape of X array for {respRange}-{brainRegion}-{stimType}: {X.shape}")
            print(f"Shape of Y array for {respRange}-{brainRegion}-{stimType}: {Y.shape}")

            # log-transform for AM or PT
            if stimType in ["AM", "pureTones"]:
                Y = np.log(Y + 1e-8)

            # 5-fold CV Ridge
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]

                ridge = RidgeCV(alphas=np.logspace(-3, 3, 7), cv=None)
                ridge.fit(X_train, Y_train)
                Y_pred = ridge.predict(X_test)
                r2_val = r2_score(Y_test, Y_pred, multioutput='raw_values').mean()

                cv_results.append({
                    "stim_type": stimType,
                    "brain_area": brainRegion,
                    "window": respRange,
                    "r2": r2_val,
                    "condition": f"{stimType} {respRange} {brainRegion}"
                })

# Convert results to DataFrame
df_results = pd.DataFrame(cv_results)

# For each sound type, plot a single figure
for stimType in stimTypes:
    plt.figure(figsize=(12, 6))

    # Subset results for this sound type
    df_subset = df_results[df_results['stim_type'] == stimType].copy()

    # Sort brain areas for consistent order
    order = sorted(df_subset['brain_area'].unique())

    sns.boxplot(
        x='brain_area',
        y='r2',
        hue='window',
        data=df_subset,
        order=order,
        palette='Set2'
    )

    plt.xlabel('Brain Area')
    plt.ylabel('R²')
    plt.title(f'R² Distribution Across Brain Areas and Spike Windows for {stimType}')
    plt.legend(title='Spike Window')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = "/Users/zoetomlinson/Desktop/GitHub/neuronalDataResearch/Figures/Population Plots"
    filename = f"{stimType}_ridge_reg_boxplots.png"
    plt.savefig(os.path.join(save_path, filename))
    plt.show()