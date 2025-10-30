# %% Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
from jaratoolbox import settings

# %% Load data
studyparams = __import__('2025acpop.studyparams').studyparams
data_path = settings.SAVE_PATH + "SVM/svm_pairwise_results.csv"
results_save_path = os.path.join(data_path)
df = pd.read_csv(results_save_path)

response_ranges = ["onset", "sustained", "offset"]
stim_types = ["naturalSound", "AM", "pureTones"]
region_order_all = df["region"].unique().tolist()

save_path = "/Users/zoetomlinson/Desktop/MurrayLab/neuronalDataResearch/Figures/SVM"
os.makedirs(save_path, exist_ok=True)


# %% Helper functions
def significance_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return ""

def add_bracket(ax, x1, x2, base_y, h, p):
    star = significance_stars(p)
    if star:
        ax.plot([x1, x1, x2, x2], [base_y, base_y+h, base_y+h, base_y], lw=1.5, c='k', clip_on=False)
        ax.text((x1+x2)/2, base_y+h, star, ha='center', va='bottom', color='k')

def try_int(x):
    try:
        return int(float(x))
    except:
        return np.nan

def stim_to_category(stim):
    try:
        return int(float(stim)) // 4
    except:
        return np.nan

def get_boxplot_x_positions(ax, region_order, hue_levels):
    n_hue = len(hue_levels)
    offsets = np.linspace(-0.4, 0.4, n_hue, endpoint=True)
    mapping = {}
    for i, reg in enumerate(region_order):
        for j, hl in enumerate(hue_levels):
            mapping[(reg, hl)] = i + offsets[j]
    return mapping

# %% Discriminability vs Stimulus Distance for AM / pureTones
q_list = []
df_dist = df[df["stim"].isin(["AM", "pureTones"])].copy()
df_dist["stim1_int"] = df_dist["stim1"].apply(try_int)
df_dist["stim2_int"] = df_dist["stim2"].apply(try_int)
df_dist = df_dist.dropna(subset=["stim1_int", "stim2_int"]).copy()
df_dist = df_dist[df_dist["stim1_int"] > df_dist["stim2_int"]].copy()

for stim_name in df_dist["stim"].unique():
    sub = df_dist[df_dist["stim"] == stim_name].copy()
    freq_groups = 2  # number of groups to divide frequencies into
    sub["q1"] = pd.cut(sub["stim1_int"], freq_groups, labels=False) # TODO: what number should this be?
    sub["q2"] = pd.cut(sub["stim2_int"], freq_groups, labels=False)
    sub["group_pair"] = np.where(sub["q1"] == sub["q2"], "within-group", "between-group")
    q_list.append(sub)
df_dist = pd.concat(q_list, ignore_index=True)

# %% Natural Sounds
df_nat = df[df["stim"] == "naturalSound"].copy()
df_nat["cat1"] = df_nat["stim1"].apply(stim_to_category)
df_nat["cat2"] = df_nat["stim2"].apply(stim_to_category)
df_nat["pair_type"] = np.where(df_nat["cat1"] == df_nat["cat2"], "within-category", "between-category")

region_order = sorted(df["region"].unique().tolist())

# %% Plotting + Stats
for stim in stim_types:
    if stim == "naturalSound":
        data_source = df_nat
        hue_col = "pair_type"
        hue_levels = ["within-category", "between-category"]
    else:
        data_source = df_dist[df_dist["stim"] == stim].copy()
        hue_col = "group_pair"
        hue_levels = ["within-group", "between-group"]

    stats_within_between = []
    stats_region_comp = []

    fig, axes = plt.subplots(len(response_ranges), 1, figsize=(15, 7*len(response_ranges)), sharey=True)

    for ax_idx, window in enumerate(response_ranges):
        ax = axes[ax_idx]
        sub_win = data_source[data_source["window"] == window].copy()

        sns.boxplot(
            x="region", y="accuracy", hue=hue_col, data=sub_win,
            order=region_order, ax=ax, showfliers=False,
            palette={hue_levels[0]: "skyblue", hue_levels[1]: "salmon"}
        )
        sns.stripplot(
            x="region", y="accuracy", hue=hue_col, data=sub_win,
            order=region_order, dodge=True, alpha=0.5, size=3,
            palette={hue_levels[0]: "skyblue", hue_levels[1]: "salmon"}, ax=ax, linewidth=0
        )
        ax.set_title(f"{stim}: {window}")
        ax.set_xlabel("Brain Area")
        ax.set_ylabel("SVM Accuracy" if ax_idx == 0 else "")

        # Within vs Between (Mann-Whitney + Bonferroni)
        wb_pvals = []
        wb_tmp = []
        for region in region_order:
            sub = sub_win[sub_win["region"] == region]
            vals_w = sub[sub[hue_col] == hue_levels[0]]["accuracy"]
            vals_b = sub[sub[hue_col] == hue_levels[1]]["accuracy"]

            statval, p_val = mannwhitneyu(vals_w, vals_b, alternative="two-sided")

            wb_tmp.append((region, statval, p_val))
            wb_pvals.append(p_val)
            corrected = multipletests(wb_pvals, method="bonferroni")[1]

            for (region, statval, p_val), p_corr in zip(wb_tmp, corrected):
                stats_within_between.append({"stim": stim, "window": window, "region": region,
                                             "stat": statval, "p": p_val, "p_corrected": p_corr})
                # annotate
                if p_corr < 0.05:
                    xmap = get_boxplot_x_positions(ax, region_order, hue_levels)
                    x1, x2 = xmap[(region, hue_levels[0])], xmap[(region, hue_levels[1])]
                    ymax = sub_win[sub_win["region"] == region]["accuracy"].max()
                    base_y = (ymax if not pd.isna(ymax) else ax.get_ylim()[1]*0.5) + 0.01
                    add_bracket(ax, x1, x2, base_y, 0.03, p_corr)

    plt.tight_layout()
    fig_fn = os.path.join(save_path, f"{stim}_within_between_by_window_1x3.png")
    plt.savefig(fig_fn, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {fig_fn}")

    pd.DataFrame(stats_within_between).to_csv(os.path.join(save_path, f"{stim}_within_between_stats.csv"), index=False)
    pd.DataFrame(stats_region_comp).to_csv(os.path.join(save_path, f"{stim}_region_stats.csv"), index=False)

print("All plotting + stats complete.")
