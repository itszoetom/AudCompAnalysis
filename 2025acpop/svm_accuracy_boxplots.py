import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_rel, mannwhitneyu, wilcoxon, spearmanr
from statsmodels.stats.multitest import multipletests
from jaratoolbox import settings

# %% Load data
studyparams = __import__('2025acpop.studyparams').studyparams
file_path = settings.SAVE_PATH + "SVM/svm_pairwise_results.csv"
df = pd.read_csv(file_path)

response_ranges = ["onset", "sustained", "offset"]
stim_types = ["naturalSound", "AM", "pureTones"]
region_order_all = df["region"].unique().tolist()
window_order = response_ranges.copy()  # explicitly define

save_path = "/Users/zoetomlinson/Desktop/MurrayLab/neuronalDataResearch/Figures/SVM"
os.makedirs(save_path, exist_ok=True)

apply_bonferroni = True

cutoffs = {"AM": 5, "naturalSound": 10, "pureTones": 7}


# %% Helper functions
def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def add_bracket(ax, x1, x2, base_y, h, p):
    star = significance_stars(p)
    if star:
        ax.plot([x1, x1, x2, x2], [base_y, base_y + h, base_y + h, base_y], lw=1.5, c='k', clip_on=False)
        ax.text((x1 + x2) / 2, base_y + h, star, ha='center', va='bottom', color='k')


def hue_centers_in_group(group_x, hue_levels, group_width=0.8):
    n = len(hue_levels)
    offsets = np.linspace(-group_width / 2, group_width / 2, n, endpoint=True)
    return {lvl: group_x + off for lvl, off in zip(hue_levels, offsets)}


def stim_to_category(stim):
    try:
        return int(float(stim)) // 4
    except:
        return np.nan


def try_int(x):
    try:
        return int(float(x))
    except:
        return np.nan


# %% Heatmaps with paired t-test annotation
for stim in stim_types:
    if stim == 'AM':
        nCategories = 11
        labels = [str(i) for i in range(nCategories)]
    elif stim == 'naturalSound':
        soundCats = studyparams.SOUND_CATEGORIES
        nCategories = len(soundCats)
        nInstances = 4
        labels = [f"{soundCats[i]}_{j + 1}" for i in range(nCategories) for j in range(nInstances)]
    elif stim == 'pureTones':
        nCategories = 16
        labels = [str(i) for i in range(nCategories)]

    cutoff = cutoffs[stim]

    n_rows = len(region_order_all)
    n_cols = len(response_ranges)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), constrained_layout=True)
    if n_rows == 1: axes = np.expand_dims(axes, 0)
    if n_cols == 1: axes = np.expand_dims(axes, 1)

    for row_idx, reg in enumerate(region_order_all):
        for col_idx, win in enumerate(response_ranges):
            ax = axes[row_idx, col_idx]
            sub = df[(df["region"] == reg) & (df["window"] == win) & (df['stim'] == stim)]

            uniq_stims = sorted(set(sub["stim1"]).union(sub["stim2"]))
            stim_to_idx = {s: i for i, s in enumerate(uniq_stims)}

            svm_stim_vals = np.full((len(uniq_stims), len(uniq_stims)), np.nan)
            for _, r in sub.iterrows():
                i, j = stim_to_idx[r["stim1"]], stim_to_idx[r["stim2"]]
                svm_stim_vals[i, j] = r["accuracy"]

            # Paired t-test for heatmap t-test
            lower_vals, upper_vals = [], []
            for i in range(len(uniq_stims)):
                for j in range(i + 1, len(uniq_stims)):
                    stim_diff = abs(i - j)
                    if stim_diff <= cutoff:
                        lower_vals.append(svm_stim_vals[i, j])
                    else:
                        upper_vals.append(svm_stim_vals[i, j])

            sig_star = ''
            min_len = min(len(lower_vals), len(upper_vals))
            lower_vals = np.array(lower_vals[:min_len])
            upper_vals = np.array(upper_vals[:min_len])
            stat, p_val = ttest_rel(lower_vals, upper_vals)
            sig_star = significance_stars(p_val)

            sns.heatmap(
                svm_stim_vals,
                ax=ax,
                square=True,
                cmap="viridis",
                vmin=0,
                vmax=1,
                cbar=(row_idx == 0 and col_idx == n_cols - 1),
                cbar_kws={"label": "SVM Accuracy"},
                xticklabels=labels,
                yticklabels=labels
            )
            ax.set_title(f"{reg} - {win} {sig_star}", fontsize=12)
            ax.set_xlabel("Stim2")
            ax.set_ylabel("Stim1")

    # Create output directory if needed
    heatmap_dir = os.path.join(save_path, "svm_accuracy_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    out_file = os.path.join(heatmap_dir, f"SVM_heatmaps_{stim}.png")
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved heatmaps for {stim} → {out_file}")

# %% Compute MWU and Wilcoxon statistics
mwu_rows, wilcox_rows = [], []

for stim in stim_types:
    sdf = df[df["stim"] == stim]

    # MWU: window comparisons within each brain area
    for area in sdf["region"].unique():
        adf = sdf[sdf["region"] == area]
        for w1, w2 in combinations(window_order, 2):
            vals1 = adf[adf["window"] == w1]["accuracy"]
            vals2 = adf[adf["window"] == w2]["accuracy"]
            if len(vals1) > 0 and len(vals2) > 0:
                stat, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
                mwu_rows.append([stim, area, w1, w2, stat, p])

    # Wilcoxon: region comparisons within each window
    for w in window_order:
        wdf = sdf[sdf["window"] == w][["region", "accuracy"]].copy()
        regions_present = [r for r in region_order_all if r in wdf["region"].unique()]
        for a1, a2 in combinations(regions_present, 2):
            vals1 = wdf[wdf["region"] == a1]["accuracy"]
            vals2 = wdf[wdf["region"] == a2]["accuracy"]
            if len(vals1) > 0 and len(vals2) > 0:
                stat, p = wilcoxon(vals1, vals2)
                wilcox_rows.append([stim, w, a1, a2, stat, p])

mwu_df = pd.DataFrame(mwu_rows, columns=["stim", "area", "w1", "w2", "stat", "p"])
wilcox_df = pd.DataFrame(wilcox_rows, columns=["stim", "window", "a1", "a2", "stat", "p"])

if apply_bonferroni:
    if not mwu_df.empty: mwu_df["p"] = multipletests(mwu_df["p"], method="bonferroni")[1]
    if not wilcox_df.empty: wilcox_df["p"] = multipletests(wilcox_df["p"], method="bonferroni")[1]

mwu_df.to_csv(os.path.join(save_path, "svm_mwu_results.csv"), index=False)
wilcox_df.to_csv(os.path.join(save_path, "svm_wilcox_results.csv"), index=False)

# %% Statistics Boxplots
boxplot_dir = os.path.join(save_path, "svm_boxplot_summaries")
os.makedirs(boxplot_dir, exist_ok=True)

for stim in stim_types:
    df_sub = df[df["stim"] == stim].copy()

    # Plot 1: X=Brain Area, Hue=Window
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        x="region", y="accuracy", hue="window", data=df_sub,
        order=region_order_all, hue_order=window_order,
        palette="Set2", showfliers=False
    )
    sns.stripplot(
        x="region", y="accuracy", hue="window", data=df_sub,
        order=region_order_all, hue_order=window_order,
        dodge=True, alpha=0.6, size=4, palette="Set2"
    )
    # Legend dedupe
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Spike Window", bbox_to_anchor=(1.02, 1), loc='upper left')

    # MWU annotations
    sub_mwu = mwu_df[mwu_df["stim"] == stim]
    if not sub_mwu.empty:
        ymax, ymin = df_sub["accuracy"].max(), df_sub["accuracy"].min()
        step = 0.04 * (ymax - ymin if ymax > ymin else 1)
        base_top = ymax + step
        for area in region_order_all:
            area_rows = sub_mwu[sub_mwu["area"] == area]
            if area_rows.empty: continue
            group_x = region_order_all.index(area)
            centers = hue_centers_in_group(group_x, window_order, group_width=0.8)
            y_cur = base_top
            for w1, w2 in combinations(window_order, 2):
                row = area_rows[((area_rows["w1"] == w1) & (area_rows["w2"] == w2)) |
                                ((area_rows["w1"] == w2) & (area_rows["w2"] == w1))]
                if row.empty: continue
                add_bracket(ax, centers[w1], centers[w2], y_cur, step * 0.6, float(row["p"].iloc[0]))
                y_cur += step

    ax.set_xlabel("Brain Area")
    ax.set_ylabel("SVM Accuracy")
    ax.set_title(f"SVM Accuracy by Brain Area & Spike Window - {stim} - MWU with Bonferroni")
    ax.grid(True, linestyle="--", alpha=0.4, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(boxplot_dir, f"{stim}_regions_vs_windows.png"), dpi=300)
    plt.show()

    # Plot 2: X=Window, Hue=Brain Area
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        x="window", y="accuracy", hue="region", data=df_sub,
        order=window_order, hue_order=region_order_all,
        palette="Set2", showfliers=False
    )
    sns.stripplot(
        x="window", y="accuracy", hue="region", data=df_sub,
        order=window_order, hue_order=region_order_all,
        dodge=True, alpha=0.6, size=4, palette="Set2"
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Brain Area", bbox_to_anchor=(1.02, 1), loc='upper left')

    # Wilcoxon annotations
    sub_wx = wilcox_df[wilcox_df["stim"] == stim]
    if not sub_wx.empty:
        ymax, ymin = df_sub["accuracy"].max(), df_sub["accuracy"].min()
        step = 0.04 * (ymax - ymin if ymax > ymin else 1)
        base_top = ymax + step
        for win in window_order:
            win_rows = sub_wx[sub_wx["window"] == win]
            if win_rows.empty: continue
            group_x = window_order.index(win)
            centers = hue_centers_in_group(group_x, region_order_all, group_width=0.8)
            y_cur = base_top
            for a1, a2 in combinations(region_order_all, 2):
                row = win_rows[((win_rows["a1"] == a1) & (win_rows["a2"] == a2)) |
                               ((win_rows["a1"] == a2) & (win_rows["a2"] == a1))]
                if row.empty: continue
                add_bracket(ax, centers[a1], centers[a2], y_cur, step * 0.6, float(row["p"].iloc[0]))
                y_cur += step

    ax.set_xlabel("Spike Window")
    ax.set_ylabel("SVM Accuracy")
    ax.set_title(f"SVM Accuracy by Spike Window & Brain Area - {stim} - Wilcoxon with Bonferroni")
    ax.grid(True, linestyle="--", alpha=0.4, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(boxplot_dir, f"{stim}_windows_vs_regions.png"), dpi=300)
    plt.show()

print("\n=== Analysis complete ===")
print(f"Heatmaps saved to: {os.path.join(save_path, 'svm_accuracy_heatmaps')}")
print(f"Boxplots saved to: {boxplot_dir}")
print(f"Statistics saved to: {save_path}")