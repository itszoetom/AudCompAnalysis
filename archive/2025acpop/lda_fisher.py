# %% Discriminability-vs-Distance (AM / pureTones) + Local Fisher Information
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from jaratoolbox import settings
from scipy.stats import norm, spearmanr, linregress

# %% Load data
studyparams = __import__('2025acpop.studyparams').studyparams
file_path = os.path.join(settings.FIGURES_DATA_PATH, studyparams.STUDY_NAME)
results_save_path = os.path.join(file_path, "lda_pairwise_results.csv")
df = pd.read_csv(results_save_path)

response_ranges = ["onset", "sustained", "offset"]
stim_types = ["naturalSound", "AM", "pureTones"]
region_order_all = df["region"].unique().tolist()

save_path = "/Users/zoetomlinson/Desktop/GitHub/neuronalDataResearch/Figures/Population Plots/LDA"
os.makedirs(save_path, exist_ok=True)


# %% Helper functions: clamp to avoid inf ppf
def _clip01(p, eps=1e-4):
    return np.minimum(1.0-eps, np.maximum(eps, p))

# 2AFC mapping: accuracy = Phi(d'/sqrt(2))  =>  d' = sqrt(2) * Phi^{-1}(accuracy)
def accuracy_to_dprime(p):
    p = _clip01(np.asarray(p))
    return np.sqrt(2.0) * norm.ppf(p)


def try_int(x):
    try:
        return int(float(x))
    except:
        return np.nan


# %% Build AM/PT dataframe with integer-coded stimuli and pair distances
df_pair = df[df["stim"].isin(["AM", "pureTones"])].copy()
df_pair["stim1_int"] = df_pair["stim1"].apply(try_int)
df_pair["stim2_int"] = df_pair["stim2"].apply(try_int)
df_pair = df_pair.dropna(subset=["stim1_int", "stim2_int"]).copy()
df_pair["delta"] = (df_pair["stim1_int"] - df_pair["stim2_int"]).abs().astype(int)
df_pair = df_pair[df_pair["delta"] > 0].copy()  # exclude identical stimuli
df_pair["dprime"] = accuracy_to_dprime(df_pair["accuracy"])
df_pair["fisher_local"] = (df_pair["dprime"]**2) / (df_pair["delta"]**2)

# --- Analysis 1: Accuracy vs stimulus separation
accdist_stats = []  # per region/window/stim: Spearman rho, linear slope
for stim in ["AM", "pureTones"]:
    sub_stim = df_pair[df_pair["stim"] == stim]
    # For consistent ordering on plots
    region_order_dist = sorted(sub_stim["region"].unique().tolist())

    for window in response_ranges:
        sub_w = sub_stim[sub_stim["window"] == window].copy()

        # 1a) Per-region scatter + binned means
        fig, axarr = plt.subplots(
            nrows=len(region_order_dist),
            ncols=1,
            figsize=(10, 3.2*len(region_order_dist)),
            sharex=True, sharey=True
        )

        # Determine available deltas (x-axis)
        all_deltas = np.sort(sub_w["delta"].unique())
        for r_idx, region in enumerate(region_order_dist):
            ax = axarr[r_idx] if len(region_order_dist) > 1 else axarr
            swr = sub_w[sub_w["region"] == region].copy()
            # Raw scatter (light jitter on x for visibility)
            jitter = (np.random.rand(len(swr)) - 0.5) * 0.1
            ax.scatter(swr["delta"] + jitter, swr["accuracy"], s=12, alpha=0.35)

            # Binned mean ± SE across pairs
            mean_by_delta = swr.groupby("delta")["accuracy"].agg(['mean', 'count', 'std'])
            mean_by_delta["se"] = mean_by_delta["std"] / np.sqrt(mean_by_delta["count"])
            ax.errorbar(
                mean_by_delta.index.values,
                mean_by_delta["mean"].values,
                yerr=mean_by_delta["se"].values,
                fmt='-o', markersize=4, linewidth=1.5
            )

            ax.set_title(f"{region}")
            ax.set_ylabel("LDA accuracy")
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

            # 1b) Stats: Spearman rank corr & linear slope (accuracy ~ delta)
            if swr["delta"].nunique() > 1:
                rho, rho_p = spearmanr(swr["delta"], swr["accuracy"])
                slope, intercept, r_val, p_val, stderr = linregress(swr["delta"], swr["accuracy"])
            else:
                rho, rho_p, slope, p_val = np.nan, np.nan, np.nan, np.nan
            accdist_stats.append({
                "stim": stim, "window": window, "region": region,
                "spearman_rho": rho, "spearman_p": rho_p,
                "lin_slope_per_delta": slope, "lin_p": p_val,
                "n_pairs": len(swr)
            })

        axarr[-1].set_xlabel("|Δ stimulus|")
        plt.suptitle(f"{stim} — {window}: Discriminability vs distance", y=0.995, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fig_fn = os.path.join(save_path, f"{stim}_{window}_accuracy_vs_distance_by_region.png")
        plt.savefig(fig_fn, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved figure → {fig_fn}")

# Save stats
accdist_stats_df = pd.DataFrame(accdist_stats)
accdist_stats_fn = os.path.join(save_path, "AM_PT_accuracy_vs_distance_stats.csv")
accdist_stats_df.to_csv(accdist_stats_fn, index=False)
print(f"Saved stats → {accdist_stats_fn}")

# --- Analysis 2: Local Fisher information from nearest-neighbor pairs (Δ=1)
#    Interpretation: higher J suggests steeper local discriminability per unit step.
fisher_stats = []
for stim in ["AM", "pureTones"]:
    sub = df_pair[(df_pair["stim"] == stim) & (df_pair["delta"] == 1)].copy()
    if sub.empty:
        continue
    for window in response_ranges:
        sw = sub[sub["window"] == window]
        for region in sorted(sw["region"].unique().tolist()):
            swr = sw[sw["region"] == region]
            # Summary across all Δ=1 pairs
            J_mean = np.nanmean(swr["fisher_local"]) if len(swr) else np.nan
            J_median = np.nanmedian(swr["fisher_local"]) if len(swr) else np.nan
            fisher_stats.append({
                "stim": stim, "window": window, "region": region,
                "J_mean_delta1": J_mean, "J_median_delta1": J_median, "n_pairs": len(swr)
            })

        # Plot bar of J (Δ=1) across regions for this stim/window
        plot_df = pd.DataFrame([row for row in fisher_stats
                                if row["stim"] == stim and row["window"] == window]).copy()
        if not plot_df.empty:
            # Sort by mean J for a nice visual
            plot_df.sort_values("J_mean_delta1", ascending=False, inplace=True)
            plt.figure(figsize=(10, 5))
            sns.barplot(data=plot_df, x="region", y="J_mean_delta1")
            plt.ylabel("Local Fisher info (Δ=1 proxy)")
            plt.xlabel("Brain Area")
            plt.title(f"{stim} — {window}: Local Fisher info (Δ=1)")
            plt.xticks(rotation=40, ha="right")
            plt.tight_layout()
            fig_fn = os.path.join(save_path, f"{stim}_{window}_localFisher_delta1_bar.png")
            plt.savefig(fig_fn, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved figure → {fig_fn}")

# Save Fisher info summaries
fisher_stats_df = pd.DataFrame(fisher_stats)
fisher_stats_fn = os.path.join(save_path, "AM_PT_localFisher_delta1_stats.csv")
fisher_stats_df.to_csv(fisher_stats_fn, index=False)
print(f"Saved stats → {fisher_stats_fn}")

# --- Analysis 3 (Optional): Compare AM vs PT slope/J within each region/window
#     Mann-Whitney on pairwise slopes is awkward; instead compare Δ=1 J per pair.
amJ = fisher_stats_df[fisher_stats_df["stim"] == "AM"]
ptJ = fisher_stats_df[fisher_stats_df["stim"] == "pureTones"]
comp_rows = []
for window in response_ranges:
    for region in sorted(df_pair["region"].unique().tolist()):
        a = amJ[(amJ["window"] == window) & (amJ["region"] == region)]["J_mean_delta1"]
        p = ptJ[(ptJ["window"] == window) & (ptJ["region"] == region)]["J_mean_delta1"]
        if len(a) and len(p) and np.isfinite(a.values[0]) and np.isfinite(p.values[0]):
            comp_rows.append({
                "window": window, "region": region,
                "AM_J_mean_delta1": a.values[0],
                "PT_J_mean_delta1": p.values[0],
                "diff_AM_minus_PT": a.values[0] - p.values[0]
            })
ampt_comp = pd.DataFrame(comp_rows)
ampt_comp_fn = os.path.join(save_path, "AM_vs_PT_localFisher_delta1_comparison.csv")
ampt_comp.to_csv(ampt_comp_fn, index=False)
print(f"Saved comparison → {ampt_comp_fn}")
