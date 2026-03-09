import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_rel, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from jaratoolbox import settings

# GLOBAL / LOAD DATA
studyparams = __import__('2025acpop.studyparams').studyparams
SOUND_CATEGORIES = studyparams.SOUND_CATEGORIES  # ['Frogs', 'Crickets', 'Streamside', 'Bubbling', 'Bees']

file_path = os.path.join(settings.SAVE_PATH, "SVM", "svm_pairwise_results.csv")
df = pd.read_csv(file_path)

# Only keep sustained
WINDOW = "sustained"
stim_types = ["naturalSound", "AM", "pureTones"]
region_order_all = df["region"].unique().tolist()

formal_names = {'pureTones': 'Pure Tones',
                'AM': 'AM White Noise',
                'naturalSound': 'Natural Sounds'}

# where to save
save_path = "/Users/zoetomlinson/Desktop/MurrayLab/neuronalDataResearch/Figures/SVM"
os.makedirs(save_path, exist_ok=True)

# multiple-comparison correction
apply_bonferroni = True

# Poster-y fonts (we’ll bump sizes later in plots too)
plt.rcParams.update({
    "font.size": 32,            # base font
    "axes.titlesize": 44,       # plot titles
    "axes.labelsize": 40,       # x / y labels
    "xtick.labelsize": 32,      # tick labels
    "ytick.labelsize": 32,
    "legend.fontsize": 34,
    "legend.title_fontsize": 36,
    "figure.titlesize": 48,     # suptitles
    "figure.labelsize": 40,
})



# helpers
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
        ax.plot([x1, x1, x2, x2],
                [base_y, base_y + h, base_y + h, base_y],
                lw=2, c='k', clip_on=False)
        ax.text((x1 + x2) / 2, base_y + h,
                star, ha='center', va='bottom', color='k',
                fontsize=18, fontweight='bold')


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


def is_upper_triangle_row(row):
    if row["stim"] in ["AM", "pureTones"]:
        s1 = try_int(row["stim1"])
        s2 = try_int(row["stim2"])
        if pd.isna(s1) or pd.isna(s2):
            return False
        return s1 > s2
    else:
        return str(row["stim1"]) > str(row["stim2"])


def make_freq_label(val):
    """For AM / pureTones, convert label to clean int (actual frequency)."""
    try:
        f = float(val)
        return str(int(round(f)))
    except:
        return str(val)


def natural_every_other_label(i):
    """
    i = stimulus index.
    We assume: 4 stims per natural category.
    We only label even indices (0,2,4,6,...).
    """
    if i % 2 != 0:
        return ""
    cat_idx = i // 4
    pos_in_cat = i % 4
    if cat_idx >= len(SOUND_CATEGORIES):
        return ""
    cat_name = SOUND_CATEGORIES[cat_idx]
    if pos_in_cat == 0:
        return f"{cat_name} 1"
    elif pos_in_cat == 2:
        return f"{cat_name} 3"
    else:
        return ""


# ============== HEATMAPS (unchanged) ==============
for stim in stim_types:
    n_regions = len(region_order_all)
    fig, axes = plt.subplots(
        1, n_regions,
        figsize=(7 * n_regions, 7),
        constrained_layout=True
    )

    fig.suptitle(f"{formal_names.get(stim)} Pairwise Stimuli Discrimination\nLinear SVM Classifier Accuracy",
                 fontsize=45, y=1.18, fontweight="bold")

    for col_idx, reg in enumerate(region_order_all):
        ax = axes[col_idx]

        sub = df[
            (df["region"] == reg) &
            (df["window"] == WINDOW) &
            (df["stim"] == stim)
        ]

        uniq_stims = sorted(set(sub["stim1"]).union(sub["stim2"]))
        stim_to_idx = {s: i for i, s in enumerate(uniq_stims)}
        n_stims = len(uniq_stims)
        svm_stim_vals = np.full((n_stims, n_stims), np.nan)
        for _, r in sub.iterrows():
            i, j = stim_to_idx[r["stim1"]], stim_to_idx[r["stim2"]]
            svm_stim_vals[i, j] = r["accuracy"]

        sns.heatmap(
            svm_stim_vals,
            ax=ax,
            square=True,
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar=(col_idx == n_regions - 1),
            cbar_kws={
                "label": "SVM Accuracy",
                "shrink": 0.8,
                "ticks": np.linspace(0, 1, 6)
            }
        )

        ax.set_xticks(range(n_stims))
        ax.set_yticks(range(n_stims))

        xlabels = []
        ylabels = []
        for i, lbl in enumerate(uniq_stims):
            if i % 2 == 0:
                if stim == "naturalSound":
                    lab = natural_every_other_label(i)
                else:
                    lab = make_freq_label(lbl)
                xlabels.append(lab)
                ylabels.append(lab)
            else:
                xlabels.append("")
                ylabels.append("")

        ax.set_xticklabels((label.replace("_", " ") for label in xlabels), rotation=50, ha='right', fontsize=25)
        ax.set_yticklabels((label.replace("_", " ") for label in ylabels), rotation=0, fontsize=30)

        short_name = reg.replace('-', ' ').split()[0]
        ax.set_title(f"{short_name}", pad=10, fontsize=35)

        ax.set_xlabel("Stim 2 (hz)", fontsize=35)
        if col_idx == 0:
            ax.set_ylabel("Stim 1 (hz)", fontsize=35)
        else:
            ax.set_ylabel("")

    heatmap_dir = os.path.join(save_path, "svm_accuracy_heatmaps_sustained")
    os.makedirs(heatmap_dir, exist_ok=True)
    out_file = os.path.join(heatmap_dir, f"SVM_heatmaps_{stim}_sustained.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmaps for {stim} → {out_file}")

# ============== REGION COMPARISON BOXPLOTS (ridge-style) ==============
boxplot_dir = os.path.join(save_path, "svm_boxplot_summaries_sustained")
os.makedirs(boxplot_dir, exist_ok=True)

mwu_rows = []

for stim in stim_types:
    df_sub = df[(df["stim"] == stim) & (df["window"] == WINDOW)].copy()
    # upper triangle only
    df_sub = df_sub[df_sub.apply(is_upper_triangle_row, axis=1)]

    cmap = plt.cm.viridis
    region_colors = [cmap(x) for x in np.linspace(0.15, 0.9, len(region_order_all))]
    short_region_names = [reg.replace('-', ' ').split()[0] for reg in region_order_all]

    fig, ax = plt.subplots(figsize=(12, 12))
    ax = sns.boxplot(
        x="region", y="accuracy",
        data=df_sub,
        order=region_order_all,
        showfliers=False,
        palette=region_colors,
        width=0.5
    )
    ax.set_xticklabels(short_region_names, rotation=0, ha='center', fontsize=40)

    # points on top
    sns.stripplot(
        x="region", y="accuracy",
        data=df_sub,
        order=region_order_all,
        alpha=0.6, size=8, color="black",
        jitter=0.08,
        ax=ax,
        zorder=5
    )

    # stats
    region_pairs = list(combinations(region_order_all, 2))
    pvals = []
    tmp = []
    for a1, a2 in region_pairs:
        vals1 = df_sub[df_sub["region"] == a1]["accuracy"]
        vals2 = df_sub[df_sub["region"] == a2]["accuracy"]
        if len(vals1) > 0 and len(vals2) > 0:
            stat, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
            pvals.append(p)
            tmp.append((a1, a2, stat, p))

    if apply_bonferroni and pvals:
        corr = multipletests(pvals, method="bonferroni")[1]
    else:
        corr = pvals

    ymax = df_sub["accuracy"].max()
    ymin = df_sub["accuracy"].min()
    y_range = ymax - ymin if ymax > 0 else 1.0
    step = 0.07 * y_range
    y_cur = ymax + step * 1.2

    for (a1, a2, stat, p), p_corr in zip(tmp, corr):
        if p_corr < 0.05:
            x1 = region_order_all.index(a1)
            x2 = region_order_all.index(a2)
            add_bracket(ax, x1, x2, y_cur, step * 0.6, p_corr)
            y_cur += step * 1.0
        mwu_rows.append([stim, a1, a2, stat, p, p_corr])

    ax.set_ylim(ymin - 0.01, y_cur + step * 0.5)
    ax.set_ylabel("SVM Accuracy", fontsize=35, labelpad=12, fontweight='bold')
    ax.set_title(f"{formal_names.get(stim)}\nSVM Pairwise Accuracy by Area", pad=16, fontsize=45, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.3, axis='y')

    plt.tight_layout()
    out_fig = os.path.join(boxplot_dir, f"{stim}_regions_sustained.png")
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved boxplot for {stim} → {out_fig}")

mwu_df = pd.DataFrame(
    mwu_rows,
    columns=["stim", "region1", "region2", "stat", "p_raw", "p_bonf"]
)
mwu_df.to_csv(os.path.join(save_path, "svm_region_stats_sustained.csv"), index=False)

# ============== WITHIN vs BETWEEN (styled to match + bars for within-area comparisons) ==============
dist_dir = os.path.join(save_path, "svm_within_between_sustained")
os.makedirs(dist_dir, exist_ok=True)

df_dist = df[
    (df["stim"].isin(["AM", "pureTones"])) &
    (df["window"] == WINDOW)
].copy()
df_dist = df_dist[df_dist.apply(is_upper_triangle_row, axis=1)]
df_dist["stim1_int"] = df_dist["stim1"].apply(try_int)
df_dist["stim2_int"] = df_dist["stim2"].apply(try_int)
df_dist = df_dist.dropna(subset=["stim1_int", "stim2_int"]).copy()

df_nat = df[
    (df["stim"] == "naturalSound") &
    (df["window"] == WINDOW)
].copy()
df_nat = df_nat[df_nat.apply(is_upper_triangle_row, axis=1)]
df_nat["cat1"] = df_nat["stim1"].apply(stim_to_category)
df_nat["cat2"] = df_nat["stim2"].apply(stim_to_category)
df_nat["pair_type"] = np.where(
    df_nat["cat1"] == df_nat["cat2"],
    "within",
    "between"
)

region_order = sorted(df["region"].unique().tolist())
short_region_names = [reg.replace('-', ' ').split()[0] for reg in region_order]

for stim in stim_types:
    if stim == "naturalSound":
        data_source = df_nat
        hue_col = "pair_type"
        hue_levels = ["within", "between"]
    else:
        sub = df_dist[df_dist["stim"] == stim].copy()
        sub["q1"] = pd.cut(sub["stim1_int"], 2, labels=False)
        sub["q2"] = pd.cut(sub["stim2_int"], 2, labels=False)
        sub["group_pair"] = np.where(sub["q1"] == sub["q2"], "within", "between")
        data_source = sub
        hue_col = "group_pair"
        hue_levels = ["within", "between"]

    stats_rows = []
    fig, ax = plt.subplots(figsize=(18,10))

    cmap = plt.cm.viridis
    pal = {hue_levels[i]: cmap(0.3 + 0.4 * i) for i in range(len(hue_levels))}

    sns.boxplot(
        x="region", y="accuracy",
        hue=hue_col, data=data_source,
        order=region_order,
        hue_order=hue_levels,
        showfliers=False,
        ax=ax,
        palette=pal,
        width=0.55
    )
    ax.set_xticklabels(short_region_names, fontsize=40)

    sns.stripplot(
        x="region", y="accuracy",
        hue=hue_col, data=data_source,
        order=region_order,
        hue_order=hue_levels,
        dodge=True, alpha=0.5, size=7,
        ax=ax,
        palette=pal,
        zorder=5
    )

    # de-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title=hue_col,
              bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=30, title_fontsize=30, frameon=True,
              edgecolor='black', fancybox=False, shadow=False,
              handlelength=1.5, handleheight=0.7, labelspacing=0.3)

    # -------- STATS --------
    all_p = []
    tmp_entries = []

    # 1) per-region: within vs between
    for reg in region_order:
        subr = data_source[data_source["region"] == reg]
        vals_within = subr[subr[hue_col] == hue_levels[0]]["accuracy"]
        vals_between = subr[subr[hue_col] == hue_levels[1]]["accuracy"]
        if len(vals_within) > 0 and len(vals_between) > 0:
            stat, p = mannwhitneyu(vals_within, vals_between, alternative="two-sided")
            all_p.append(p)
            tmp_entries.append(("per_region", stim, reg, None, stat, p))

    # 2) pairwise across areas, but ONLY comparing the "within" group
    for regA, regB in combinations(region_order, 2):
        subA = data_source[
            (data_source["region"] == regA) &
            (data_source[hue_col] == hue_levels[0])
        ]["accuracy"]
        subB = data_source[
            (data_source["region"] == regB) &
            (data_source[hue_col] == hue_levels[0])
        ]["accuracy"]
        if len(subA) > 0 and len(subB) > 0:
            stat, p = mannwhitneyu(subA, subB, alternative="two-sided")
            all_p.append(p)
            tmp_entries.append(("pairwise_within", stim, regA, regB, stat, p))

    # Bonferroni across everything in this figure
    if all_p:
        corr_p = multipletests(all_p, method="bonferroni")[1]
    else:
        corr_p = []

    ymax = data_source["accuracy"].max()
    ymin = data_source["accuracy"].min()
    y_range = ymax - ymin if ymax > ymin else 1.0
    step = 0.07 * y_range
    y_start = ymax + step * 0.5

    # we’ll draw per-region first, then pairwise_within above
    per_region_height = y_start
    pairwise_height = y_start

    # first pass: per-region bars
    for (entry, p_corr) in zip(tmp_entries, corr_p):
        kind, stim_name, reg1, reg2, stat, p_raw = entry
        if kind == "per_region":
            stats_rows.append({
                "stim": stim_name,
                "region1": reg1,
                "region2": "",
                "test_type": "within_vs_between",
                "stat": stat,
                "p_raw": p_raw,
                "p_bonf": p_corr
            })
            if p_corr < 0.05:
                group_x = region_order.index(reg1)
                x1 = group_x - 0.18
                x2 = group_x + 0.18
                add_bracket(ax, x1, x2, per_region_height, step * 0.4, p_corr)
                per_region_height += step * 0.55

    # second pass: pairwise within-category area comparisons
    # stack them above the per-region ones
    pairwise_height = per_region_height + step * 0.4
    for (entry, p_corr) in zip(tmp_entries, corr_p):
        kind, stim_name, reg1, reg2, stat, p_raw = entry
        if kind == "pairwise_within":
            stats_rows.append({
                "stim": stim_name,
                "region1": reg1,
                "region2": reg2,
                "test_type": "within_vs_within_pair",
                "stat": stat,
                "p_raw": p_raw,
                "p_bonf": p_corr
            })
            if p_corr < 0.05:
                x1 = region_order.index(reg1)
                x2 = region_order.index(reg2)
                add_bracket(ax, x1, x2, pairwise_height, step * 0.4, p_corr)
                pairwise_height += step * 0.55

    ax.set_ylim(ymin - 0.02, pairwise_height + step * 0.8)
    ax.set_title(f"{formal_names.get(stim)}\nWithin vs Between Sound Category", pad=14,
                 fontsize=45, fontweight='bold')
    ax.set_ylabel("SVM Accuracy", fontsize=35, labelpad=12, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.3, axis='y')

    plt.tight_layout()
    fig_fn = os.path.join(dist_dir, f"{stim}_within_between_sustained.png")
    plt.savefig(fig_fn, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(stats_rows).to_csv(
        os.path.join(dist_dir, f"{stim}_within_between_sustained_stats.csv"),
        index=False
    )
    print(f"Saved within/between for {stim} → {fig_fn}")

print("\n=== Sustained-only analysis complete ===")
print(f"Heatmaps: {os.path.join(save_path, 'svm_accuracy_heatmaps_sustained')}")
print(f"Boxplots: {boxplot_dir}")
print(f"Within/Between: {dist_dir}")
