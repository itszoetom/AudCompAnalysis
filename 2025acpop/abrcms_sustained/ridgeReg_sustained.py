import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import wilcoxon
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from jaratoolbox import settings, celldatabase

studyparams = __import__('2025acpop.studyparams').studyparams

# ===================== GLOBAL POSTER STYLE =====================
plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 32,
    "axes.labelsize": 28,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 22
})

# ===================== SETTINGS =====================
neuron_threshold = 20
stim_types = ["naturalSound", "AM", "pureTones"]

# ONLY KEEP SUSTAINED WINDOW
response_ranges = ["sustained"]

alphas = np.logspace(-3, 3, 20)
n_splits = 5  # k-fold CV

figdataPath = os.path.join(settings.FIGURES_DATA_PATH)

results_dir = os.path.join(settings.SAVE_PATH, "Ridge Regression")
os.makedirs(results_dir, exist_ok=True)

# Load session info from celldb
dbCoordsFilename = os.path.join(figdataPath, f'celldb_{studyparams.STUDY_NAME}_responsive_all_stims_index_new.h5')
celldb = celldatabase.load_hdf(dbCoordsFilename)
celldb['simpleSiteName'] = celldb['recordingSiteName'].str.split(',').apply(lambda x: x[0])

areas_of_interest = [
    "Dorsal auditory area",
    "Primary auditory area",
    "Ventral auditory area",
    "Posterior auditory area"
]

aud_db = celldb[celldb['simpleSiteName'].isin(areas_of_interest)].reset_index()

# Group sessions with enough neurons
grouped_data = aud_db.groupby(['simpleSiteName', 'date']).size()
session_list = grouped_data[grouped_data >= neuron_threshold]

print(f"Number of sessions with ≥{neuron_threshold} neurons: {len(session_list)}")

results = []

# ===================== MAIN LOOP =====================
np.random.seed(42)

for stim in stim_types:
    stim_arrays = np.load(os.path.join(figdataPath, f"fr_arrays_{stim}.npz"), allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    sessionArray = stim_arrays["sessionIDArray"]
    stimArray = stim_arrays["stimArray"][0, :]  # 1D trial vector

    for respRange in response_ranges:  # really just "sustained"
        respArray = stim_arrays[f"{respRange}fr"]  # neurons × trials

        for area, date in session_list.index:
            area_idx = np.where(brainRegionArray == area)[0]
            session_idx = np.where(sessionArray == date)[0]
            neuron_idx = np.intersect1d(area_idx, session_idx)

            # need at least 30 to make them comparable
            if neuron_idx.size < neuron_threshold:
                continue

            # choose exactly 30 neurons at random
            chosen_neurons = np.random.choice(neuron_idx, size=neuron_threshold, replace=False)

            X = respArray[chosen_neurons, :].T  # trials × neurons
            y = stimArray.copy()

            # transform y for AM and tones
            if stim in ["AM", "pureTones"]:
                y = np.log(y + 1e-8)

            # k-fold cross-validation
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            r2_scores = []
            # we'll store the last fold's predictions for plotting
            last_y_test = None
            last_y_pred = None

            for fold_i, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                ridge = RidgeCV(alphas=alphas, store_cv_values=False)
                ridge.fit(X_train, y_train)
                y_pred = ridge.predict(X_test)
                r2_scores.append(r2_score(y_test, y_pred))

                # keep for plotting (last fold is fine for quick QC)
                last_y_test = y_test
                last_y_pred = y_pred

            mean_r2 = np.mean(r2_scores)

            results.append({
                "stim_type": stim,
                "brain_area": area,
                "window": respRange,
                "date": date,
                "r2": mean_r2
            })

            # one figure per (stim, area, date)
            if last_y_test is not None:
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.scatter(last_y_test, last_y_pred, alpha=0.6, s=40)
                min_val = min(last_y_test.min(), last_y_pred.min())
                max_val = max(last_y_test.max(), last_y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3)

                ax.set_title(
                    f"{stim} — {area} — {date}\nPredicted vs Actual ({respRange}, 30 neurons)",
                    fontsize=26,
                    pad=20
                )
                ax.set_xlabel("Actual", fontsize=24, labelpad=10)
                ax.set_ylabel("Predicted", fontsize=24, labelpad=10)
                ax.tick_params(axis='both', labelsize=20)
                ax.set_xlim(min_val, max_val)
                ax.set_ylim(min_val, max_val)
                ax.grid(True, linestyle='--', alpha=0.4)

                pvadir = os.path.join(results_dir, "pred_vs_actual")
                os.makedirs(pvadir, exist_ok=True)
                fname = f"{stim}_{area.replace(' ', '_')}_{date}_{respRange}_pred_vs_actual.png"
                fig.savefig(os.path.join(pvadir, fname), dpi=300, bbox_inches='tight')
                plt.close(fig)

# ===================== SAVE RESULTS TABLE =====================
df = pd.DataFrame(results)
results_save_path = os.path.join(results_dir, "ridge_results_sustained_only.csv")
df.to_csv(results_save_path, index=False)
print(f"Saved results to {results_save_path}")

# ===================== STATS & PLOTTING (SUSTAINED ONLY) =====================
# For EVERY brain area pair combo, run Wilcoxon test
apply_bonferroni = True
region_order_all = [
    "Dorsal auditory area",
    "Posterior auditory area",
    "Primary auditory area",
    "Ventral auditory area"
]

wilcoxon_rows = []
for stim in stim_types:
    sdf = df[df["stim_type"] == stim]
    # pivot by date to do paired tests across areas
    wdf = sdf[["date", "brain_area", "r2"]].copy()
    if wdf.empty:
        continue
    wide = wdf.pivot_table(index="date", columns="brain_area", values="r2", aggfunc="mean")
    areas_present = [a for a in region_order_all if a in wide.columns]

    # Test EVERY pair of brain areas
    for a1, a2 in combinations(areas_present, 2):
        paired = wide[[a1, a2]].dropna()
        if paired.shape[0] == 0:
            continue
        try:
            stat, p = wilcoxon(paired[a1].values, paired[a2].values, alternative="two-sided")
        except ValueError:
            continue
        wilcoxon_rows.append([stim, a1, a2, stat, p])

wilcox_df = pd.DataFrame(wilcoxon_rows, columns=["stim", "a1", "a2", "stat", "p"])
if apply_bonferroni and not wilcox_df.empty:
    wilcox_df["p_corrected"] = multipletests(wilcox_df["p"], method="bonferroni")[1]
else:
    wilcox_df["p_corrected"] = wilcox_df["p"]

wilcox_df.to_csv(os.path.join(results_dir, "wilcoxon_results_sustained_only.csv"), index=False)


# ===================== POSTER-SIZED PLOTS =====================
def add_bracket(ax, x1, x2, base_y, h, text):
    ax.plot([x1, x1, x2, x2], [base_y, base_y + h, base_y + h, base_y],
            lw=3, c='k', clip_on=False)
    ax.text((x1 + x2) / 2, base_y + h, text,
            ha='center', va='bottom', color='k', fontsize=24, weight='bold')


for stim in stim_types:
    df_sub = df[df["stim_type"] == stim].copy()
    if df_sub.empty:
        continue

    # FIRST: Replace the text in df_sub
    df_sub["brain_area"] = df_sub["brain_area"].str.replace(" auditory area", "")

    # THEN: Create region_order with the UPDATED values
    region_order_short = [r.replace(" auditory area", "") for r in region_order_all]
    region_order = [r for r in region_order_short if r in df_sub["brain_area"].unique()]

    # Count sessions per brain area
    session_counts = df_sub.groupby("brain_area")["date"].nunique()

    # Create figure with reduced width and skinnier boxes
    plt.figure(figsize=(14, 12))
    ax = sns.boxplot(
        x="brain_area", y="r2", data=df_sub,
        order=region_order,
        hue="brain_area",
        palette="Set2",
        showfliers=False,
        width=0.5,
        legend=False
    )
    sns.stripplot(
        x="brain_area", y="r2", data=df_sub,
        order=region_order,
        dodge=False, alpha=0.6, size=10, color='black'  # Black points
    )

    ax.set_xlabel("Brain Area", fontsize=32, labelpad=15, weight='bold')
    ax.set_ylabel("R² (5-fold CV per session)", fontsize=32, labelpad=15, weight='bold')
    ax.set_title(f"Ridge Regression Performance — {stim}",
                 fontsize=36, pad=25, weight='bold')

    # Horizontal x-axis labels with session counts
    xticklabels = [f"{area}\n(n={session_counts[area]} sessions)" for area in region_order]
    ax.set_xticklabels(xticklabels, rotation=0, ha='center')
    ax.tick_params(axis='both', labelsize=24)

    # Add significance brackets for ALL pairs
    sub_wx = wilcox_df[wilcox_df["stim"] == stim].copy()
    if not sub_wx.empty:
        # SHORTEN THE AREA NAMES IN sub_wx to match region_order
        sub_wx['a1'] = sub_wx['a1'].str.replace(" auditory area", "")
        sub_wx['a2'] = sub_wx['a2'].str.replace(" auditory area", "")

        ymax = df_sub["r2"].max()
        ymin = df_sub["r2"].min()
        y_range = ymax - ymin if ymax > ymin else 1.0
        step = 0.08 * y_range
        base_top = ymax + 0.05 * y_range

        # Sort by distance between areas for better bracket placement
        sub_wx['x1_pos'] = sub_wx['a1'].apply(lambda x: region_order.index(x))
        sub_wx['x2_pos'] = sub_wx['a2'].apply(lambda x: region_order.index(x))
        sub_wx['distance'] = abs(sub_wx['x2_pos'] - sub_wx['x1_pos'])
        sub_wx = sub_wx.sort_values('distance')

        bracket_idx = 0
        for row in sub_wx.itertuples(index=False):
            a1 = row.a1
            a2 = row.a2
            p = row.p_corrected
            if p < 0.05:
                x1 = region_order.index(a1)
                x2 = region_order.index(a2)
                star_text = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                add_bracket(ax, x1, x2, base_top + bracket_idx * step, step * 0.7, star_text)
                bracket_idx += 1

    ax.grid(True, linestyle="--", alpha=0.3, axis='y')
    plt.tight_layout()
    out_path = os.path.join(results_dir, f"{stim}_brain_areas_sustained_boxplot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved poster plot to {out_path}")