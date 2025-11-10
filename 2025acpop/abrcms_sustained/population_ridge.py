import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import wilcoxon
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from jaratoolbox import settings
from tqdm import tqdm
import studyparams

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
file_path = settings.FIGURES_DATA_PATH
save_dir = os.path.join(settings.SAVE_PATH, "Ridge_Regression_Population")
os.makedirs(save_dir, exist_ok=True)

response_window = "sustained"  # Only sustained window
stim_types = ["naturalSound", "AM", "pureTones"]
colors = {
    'Dorsal auditory area': '#1f77b4',
    'Posterior auditory area': '#ff7f0e',
    'Primary auditory area': '#2ca02c',
    'Ventral auditory area': '#d62728'
}

alphas = np.logspace(-3, 3, 20)
max_neurons = 265  # Maximum neurons per brain region to match SVM script

print(f"Ridge Regression Population Analysis (LOO CV)")
print(f"Response window: {response_window}")
print(f"Max neurons per region: {max_neurons}")
print(f"Alpha range: {alphas.min():.3f} to {alphas.max():.3f}\n")

# ===================== MAIN ANALYSIS =====================
all_results = []
np.random.seed(42)

for stim in stim_types:
    print(f"\n=== Processing stim type: {stim} ===")

    # Load stimulus arrays
    stim_arrays = np.load(os.path.join(file_path, f"fr_arrays_{stim}.npz"), allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    stimArray = stim_arrays["stimArray"][0, :]  # 1D trial vector
    respArray = stim_arrays[f"{response_window}fr"]  # neurons × trials
    uniqRegions = np.unique(brainRegionArray)

    for brainRegion in uniqRegions:
        print(f"   -> {response_window} - {brainRegion}")

        # Get neurons for this brain region
        brain_mask = brainRegionArray == brainRegion
        brain_resp_array = respArray[brain_mask, :]

        # Randomly select up to max_neurons neurons
        n_neurons = brain_resp_array.shape[0]
        if n_neurons > max_neurons:
            selected_neurons = np.random.choice(n_neurons, max_neurons, replace=False)
            brain_resp_array = brain_resp_array[selected_neurons, :]
            print(f"      Subsampled from {n_neurons} to {max_neurons} neurons")
        else:
            print(f"      Using all {n_neurons} neurons")

        # Prepare data
        X = brain_resp_array.T  # trials × neurons
        y = stimArray.copy()

        # Transform y for AM and tones (log transform)
        if stim in ["AM", "pureTones"]:
            y = np.log(y + 1e-8)

        # Leave-one-out cross-validation
        loo = LeaveOneOut()
        n_trials = X.shape[0]
        r2_scores_loo = []

        print(f"      Running LOO CV on {n_trials} trials...")
        with tqdm(total=n_trials, desc=f"      LOO progress", leave=False) as pbar:
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Standardize using only training data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Fit Ridge with cross-validation for alpha selection
                ridge = RidgeCV(alphas=alphas)
                ridge.fit(X_train_scaled, y_train)

                # Predict on test trial
                y_pred = ridge.predict(X_test_scaled)

                # Calculate R² for this single prediction
                # For single point R², use squared correlation approach
                r2 = 1 - ((y_test[0] - y_pred[0]) ** 2) / ((y_test[0] - np.mean(y_train)) ** 2)
                r2_scores_loo.append(r2)

                pbar.update(1)

        # Store all LOO R² values
        for r2_val in r2_scores_loo:
            all_results.append({
                "stim_type": stim,
                "brain_area": brainRegion,
                "window": response_window,
                "r2": r2_val,
                "n_neurons": brain_resp_array.shape[0]
            })

        print(f"      Mean R²: {np.mean(r2_scores_loo):.3f} ± {np.std(r2_scores_loo):.3f}")

# ===================== SAVE RESULTS =====================
df = pd.DataFrame(all_results)
results_save_path = os.path.join(save_dir, "ridge_results_population_loo.csv")
df.to_csv(results_save_path, index=False)
print(f"\nSaved results to {results_save_path}")

# ===================== STATISTICS =====================
print("\n=== Running statistical tests ===")
apply_bonferroni = True
region_order_all = [
    "Dorsal auditory area",
    "Posterior auditory area",
    "Primary auditory area",
    "Ventral auditory area"
]

wilcoxon_rows = []
for stim in stim_types:
    sdf = df[df["stim_type"] == stim].copy()
    if sdf.empty:
        continue

    areas_present = [a for a in region_order_all if a in sdf["brain_area"].unique()]

    # Test EVERY pair of brain areas
    for a1, a2 in combinations(areas_present, 2):
        r2_a1 = sdf[sdf["brain_area"] == a1]["r2"].values
        r2_a2 = sdf[sdf["brain_area"] == a2]["r2"].values

        if len(r2_a1) == 0 or len(r2_a2) == 0:
            continue

        try:
            stat, p = wilcoxon(r2_a1, r2_a2, alternative="two-sided")
            wilcoxon_rows.append([stim, a1, a2, stat, p, len(r2_a1), len(r2_a2)])
        except ValueError:
            continue

wilcox_df = pd.DataFrame(wilcoxon_rows, columns=["stim", "a1", "a2", "stat", "p", "n1", "n2"])
if apply_bonferroni and not wilcox_df.empty:
    wilcox_df["p_corrected"] = multipletests(wilcox_df["p"], method="bonferroni")[1]
else:
    wilcox_df["p_corrected"] = wilcox_df["p"]

wilcox_save_path = os.path.join(save_dir, "wilcoxon_results_population_loo.csv")
wilcox_df.to_csv(wilcox_save_path, index=False)
print(f"Saved statistical results to {wilcox_save_path}")

# ===================== PLOTTING =====================
print("\n=== Creating plots ===")


def add_bracket(ax, x1, x2, base_y, h, text):
    """Add significance bracket to plot"""
    ax.plot([x1, x1, x2, x2], [base_y, base_y + h, base_y + h, base_y],
            lw=3, c='k', clip_on=False)
    ax.text((x1 + x2) / 2, base_y + h, text,
            ha='center', va='bottom', color='k', fontsize=24, weight='bold')


for stim in stim_types:
    df_sub = df[df["stim_type"] == stim].copy()
    if df_sub.empty:
        continue

    # Shorten brain area names
    df_sub["brain_area"] = df_sub["brain_area"].str.replace(" auditory area", "")

    # Create region order with shortened names
    region_order_short = [r.replace(" auditory area", "") for r in region_order_all]
    region_order = [r for r in region_order_short if r in df_sub["brain_area"].unique()]

    # Count trials per brain area
    trial_counts = df_sub.groupby("brain_area").size()

    # Create figure
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
        dodge=False, alpha=0.4, size=6, color='black'  # More transparent due to many points
    )

    ax.set_xlabel("Brain Area", fontsize=32, labelpad=15, weight='bold')
    ax.set_ylabel("R² (LOO CV per trial)", fontsize=32, labelpad=15, weight='bold')
    ax.set_title(f"Ridge Regression Performance — {stim}",
                 fontsize=36, pad=25, weight='bold')

    # Horizontal x-axis labels with trial counts
    xticklabels = [f"{area}\n(n={trial_counts[area]} trials)" for area in region_order]
    ax.set_xticklabels(xticklabels, rotation=0, ha='center')
    ax.tick_params(axis='both', labelsize=24)

    # Add significance brackets
    sub_wx = wilcox_df[wilcox_df["stim"] == stim].copy()
    if not sub_wx.empty:
        # Shorten area names in wilcoxon results
        sub_wx['a1'] = sub_wx['a1'].str.replace(" auditory area", "")
        sub_wx['a2'] = sub_wx['a2'].str.replace(" auditory area", "")

        ymax = df_sub["r2"].max()
        ymin = df_sub["r2"].min()
        y_range = ymax - ymin if ymax > ymin else 1.0
        step = 0.08 * y_range
        base_top = ymax + 0.05 * y_range

        # Sort by distance between areas for better bracket placement
        sub_wx['x1_pos'] = sub_wx['a1'].apply(lambda x: region_order.index(x) if x in region_order else -1)
        sub_wx['x2_pos'] = sub_wx['a2'].apply(lambda x: region_order.index(x) if x in region_order else -1)
        sub_wx = sub_wx[(sub_wx['x1_pos'] >= 0) & (sub_wx['x2_pos'] >= 0)]
        sub_wx['distance'] = abs(sub_wx['x2_pos'] - sub_wx['x1_pos'])
        sub_wx = sub_wx.sort_values('distance')

        bracket_idx = 0
        for row in sub_wx.itertuples(index=False):
            if row.p_corrected < 0.05:
                x1 = region_order.index(row.a1)
                x2 = region_order.index(row.a2)
                star_text = "***" if row.p_corrected < 0.001 else "**" if row.p_corrected < 0.01 else "*"
                add_bracket(ax, x1, x2, base_top + bracket_idx * step, step * 0.7, star_text)
                bracket_idx += 1

    ax.grid(True, linestyle="--", alpha=0.3, axis='y')
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{stim}_brain_areas_population_loo_boxplot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {out_path}")

print("\n=== Analysis complete ===")
print(f"Results saved to: {save_dir}")