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
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from jaratoolbox import settings, celldatabase
studyparams = __import__('2025acpop.studyparams').studyparams

# SETTINGS
neuron_threshold = 65
stim_types = ["naturalSound", "AM", "pureTones"]
response_ranges = ["onset", "sustained", "offset"]
alphas = np.logspace(-3, 3, 20)
n_splits = 5  # k-fold CV

figdataPath = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME)

# Load session info from celldb
dbPath = os.path.join(settings.DATABASE_PATH, studyparams.STUDY_NAME)
dbCoordsFilename = os.path.join(dbPath, f'celldb_{studyparams.STUDY_NAME}_responsive_all_stims_index_new.h5')
celldb = celldatabase.load_hdf(dbCoordsFilename)
celldb['simpleSiteName'] = celldb['recordingSiteName'].str.split(',').apply(lambda x: x[0])
areas_of_interest = ["Dorsal auditory area", "Primary auditory area", "Ventral auditory area", "Posterior auditory area"]
aud_db = celldb[celldb['simpleSiteName'].isin(areas_of_interest)].reset_index()

# Group sessions with enough neurons
grouped_data = aud_db.groupby(['simpleSiteName', 'date']).size()
session_list = grouped_data[grouped_data > neuron_threshold]

print(f"Number of sessions with >{neuron_threshold} neurons: {len(session_list)}")

# Ridge Regression per session with 5-fold CV
results = []

for stim in stim_types:
    stim_arrays = np.load(os.path.join(figdataPath, f"fr_arrays_{stim}.npz"), allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    sessionArray = stim_arrays["sessionIDArray"]
    stimArray = stim_arrays["stimArray"][0, :]  # 1D trial vector

    for respRange in response_ranges:
        respArray = stim_arrays[f"{respRange}fr"]  # neurons × trials

        for area, date in session_list.index:
            area_idx = np.where(brainRegionArray == area)[0]
            session_idx = np.where(sessionArray == date)[0]
            neuron_idx = np.intersect1d(area_idx, session_idx)

            if neuron_idx.size < 2:
                continue

            X = respArray[neuron_idx, :].T  # trials × neurons
            y = stimArray.copy()

            if stim in ["AM", "pureTones"]:
                y = np.log(y + 1e-8)

            # k-fold cross-validation
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            r2_scores = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                ridge = RidgeCV(alphas=alphas, store_cv_values=False)
                ridge.fit(X_train, y_train)
                y_pred = ridge.predict(X_test)
                r2_scores.append(r2_score(y_test, y_pred))

            mean_r2 = np.mean(r2_scores)

            results.append({
                "stim_type": stim,
                "brain_area": area,
                "window": respRange,
                "date": date,
                "r2": mean_r2
            })

# Save results
df = pd.DataFrame(results)
results_save_path = os.path.join(figdataPath, "ridge_results.csv")
df.to_csv(results_save_path, index=False)
print(f"Saved results to {results_save_path}")
print(df.head())

# Load saved results (or keep df from the main loop)
results_save_path = os.path.join(figdataPath, "ridge_results.csv")
df = pd.read_csv(results_save_path)

# Config
apply_bonferroni = True  # set True when want correction back on
window_order = ["onset", "sustained", "offset"]
region_order_all = ["Dorsal auditory area", "Posterior auditory area",
                    "Primary auditory area", "Ventral auditory area"]


# Helper functions
def significance_stars(p):
    # show only the strongest level (no stacked *,** at the same spot)
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"


def add_bracket(ax, x1, x2, base_y, h, p):
    """Draw a bracket from x1 to x2 at height base_y+h with a centered star label."""
    if significance_stars(p):
        ax.plot([x1, x1, x2, x2], [base_y, base_y+h, base_y+h, base_y], lw=1.5, c='k', clip_on=False)
    ax.text((x1+x2)/2, base_y+h, significance_stars(p), ha='center', va='bottom', color='k')


def hue_centers_in_group(group_x, hue_levels, group_width=0.8):
    """
    Compute the x centers for each hue box within a grouped seaborn boxplot category.
    Assumes seaborn default dodge spacing ~ equal division of group_width across hue levels.
    """
    n = len(hue_levels)
    # equally-spaced centers across group_width
    offsets = np.linspace(-group_width/2, group_width/2, n, endpoint=True)
    return {lvl: group_x + off for lvl, off in zip(hue_levels, offsets)}


# Plot 1: Window comparisons within each brain area (MWU)
mwu_rows = []
for stim in stim_types:
    sdf = df[df["stim_type"] == stim]
    for area in sdf["brain_area"].dropna().unique():
        adf = sdf[sdf["brain_area"] == area]
        # test EVERY pair of windows
        for w1, w2 in combinations(window_order, 2):
            vals1 = adf.loc[adf["window"] == w1, "r2"].values
            vals2 = adf.loc[adf["window"] == w2, "r2"].values
            if len(vals1) == 0 or len(vals2) == 0:
                continue
            stat, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
            mwu_rows.append([stim, area, w1, w2, stat, p])

mwu_df = pd.DataFrame(mwu_rows, columns=["stim", "area", "w1", "w2", "stat", "p"])
if apply_bonferroni and not mwu_df.empty:
    mwu_df["p"] = multipletests(mwu_df["p"], method="bonferroni")[1]  # overwrite with corrected p

# Plot 2: Brain-area comparisons within each window (Wilcoxon, paired by session date)
wilcoxon_rows = []
for stim in stim_types:
    sdf = df[df["stim_type"] == stim]
    for win in window_order:
        wdf = sdf[sdf["window"] == win][["date", "brain_area", "r2"]].copy()
        if wdf.empty:
            continue
        # pivot to align by date for paired test
        wide = wdf.pivot_table(index="date", columns="brain_area", values="r2", aggfunc="mean")
        areas_present = [a for a in region_order_all if a in wide.columns]
        for a1, a2 in combinations(areas_present, 2):
            paired = wide[[a1, a2]].dropna()
            if paired.shape[0] == 0:
                continue
            try:
                stat, p = wilcoxon(paired[a1].values, paired[a2].values, alternative="two-sided")
            except ValueError:
                # all differences zero, Wilcoxon undefined
                continue
            wilcoxon_rows.append([stim, win, a1, a2, stat, p])

wilcox_df = pd.DataFrame(wilcoxon_rows, columns=["stim", "window", "a1", "a2", "stat", "p"])
if apply_bonferroni and not wilcox_df.empty:
    wilcox_df["p"] = multipletests(wilcox_df["p"], method="bonferroni")[1]  # overwrite with corrected p

# Save stats for record
mwu_df.to_csv(os.path.join(figdataPath, "mwu_results.csv"), index=False)
wilcox_df.to_csv(os.path.join(figdataPath, "wilcoxon_results.csv"), index=False)

# Visualization (two plots per sound)
save_path = os.path.join("/Users/zoetomlinson/Desktop/GitHub/neuronalDataResearch/Figures/Population Plots")
os.makedirs(save_path, exist_ok=True)

for stim in stim_types:
    df_sub = df[df["stim_type"] == stim].copy()
    if df_sub.empty:
        continue

    # Plot 1: X = Brain Area, Hue = Window (annotate MWU for ALL window pairs within each area)
    plt.figure(figsize=(13, 6))
    region_order = [r for r in region_order_all if r in df_sub["brain_area"].unique()]
    ax = sns.boxplot(
        x="brain_area", y="r2", hue="window", data=df_sub,
        order=region_order, hue_order=window_order,
        palette="Set2", showfliers=False
    )
    sns.stripplot(
        x="brain_area", y="r2", hue="window", data=df_sub,
        order=region_order, hue_order=window_order,
        dodge=True, alpha=0.6, size=4, palette="Set2"
    )

    # Remove duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Spike Window", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Annotations: every pair of windows per area
    sub_mwu = mwu_df[mwu_df["stim"] == stim]
    if not sub_mwu.empty:
        ymax = df_sub["r2"].max()
        ymin = df_sub["r2"].min()
        step = 0.04 * (ymax - ymin if ymax > ymin else 1.0)
        base_top = ymax + step  # starting height above data
        for area in region_order:
            area_rows = sub_mwu[sub_mwu["area"] == area]
            if area_rows.empty:
                continue
            # center x for each hue within this area
            group_x = region_order.index(area)
            centers = hue_centers_in_group(group_x, window_order, group_width=0.8)
            y_cur = base_top
            # draw every pair
            for (w1, w2) in combinations(window_order, 2):
                row = area_rows[((area_rows["w1"] == w1) & (area_rows["w2"] == w2)) |
                                ((area_rows["w1"] == w2) & (area_rows["w2"] == w1))]
                if row.empty:
                    continue
                p = float(row["p"].iloc[0])
                x1, x2 = centers[w1], centers[w2]
                add_bracket(ax, x1, x2, y_cur, step*0.6, p)
                y_cur += step  # stack within the same area

    ax.set_xlabel("Brain Area")
    ax.set_ylabel("R² (5-fold CV per session)")
    ax.set_title(f"R² Across Brain Areas and Spike Windows — {stim} with MWU test and Bonferroni correction")
    ax.grid(True, linestyle="--", alpha=0.4, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{stim}_regions_vs_windows.png"), dpi=300)
    plt.show()

    # Plot 2: X = Window, Hue = Brain Area (annotate Wilcoxon for ALL region pairs within each window)
    plt.figure(figsize=(13, 6))
    ax = sns.boxplot(
        x="window", y="r2", hue="brain_area", data=df_sub,
        order=window_order, hue_order=region_order,
        palette="Set2", showfliers=False
    )
    sns.stripplot(
        x="window", y="r2", hue="brain_area", data=df_sub,
        order=window_order, hue_order=region_order,
        dodge=True, alpha=0.6, size=4, palette="Set2"
    )

    # legend dedupe
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Brain Area", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Annotations: every pair of regions per window
    sub_wx = wilcox_df[wilcox_df["stim"] == stim]
    if not sub_wx.empty:
        ymax = df_sub["r2"].max()
        ymin = df_sub["r2"].min()
        step = 0.04 * (ymax - ymin if ymax > ymin else 1.0)
        base_top = ymax + step
        for win in window_order:
            win_rows = sub_wx[sub_wx["window"] == win]
            if win_rows.empty:
                continue
            group_x = window_order.index(win)
            centers = hue_centers_in_group(group_x, region_order, group_width=0.8)
            y_cur = base_top
            for (a1, a2) in combinations(region_order, 2):
                row = win_rows[((win_rows["a1"] == a1) & (win_rows["a2"] == a2)) |
                               ((win_rows["a1"] == a2) & (win_rows["a2"] == a1))]
                if row.empty:
                    continue
                p = float(row["p"].iloc[0])
                x1, x2 = centers[a1], centers[a2]
                add_bracket(ax, x1, x2, y_cur, step*0.6, p)
                y_cur += step

    ax.set_xlabel("Spike Window")
    ax.set_ylabel("R² (5-fold CV per session)")
    ax.set_title(f"R² Across Spike Windows and Brain Areas — {stim} with Wilcoxon test and Bonferroni correction")
    ax.grid(True, linestyle="--", alpha=0.4, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{stim}_windows_vs_regions.png"), dpi=300)
    plt.show()