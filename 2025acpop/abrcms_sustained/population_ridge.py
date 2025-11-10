import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
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
    'Dorsal': '#1f77b4',
    'Posterior': '#ff7f0e',
    'Primary': '#2ca02c',
    'Ventral': '#d62728'
}

alphas = np.logspace(-3, 3, 20)
max_neurons = 265  # Maximum neurons per brain region to match SVM script
n_splits = 5  # 5-fold cross-validation

print(f"Ridge Regression Population Analysis (5-Fold CV)")
print(f"Response window: {response_window}")
print(f"Max neurons per region: {max_neurons}")
print(f"Number of folds: {n_splits}")
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

        # 5-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        r2_scores = []

        print(f"      Running {n_splits}-fold CV...")
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Standardize using only training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit Ridge with cross-validation for alpha selection
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(X_train_scaled, y_train)

            # Predict on test fold
            y_pred = ridge.predict(X_test_scaled)

            # Calculate R² for this fold
            r2 = r2_score(y_test, y_pred)
            r2_scores.append(r2)
            print(f"         Fold {fold_idx + 1}: R² = {r2:.3f}")

        # Store all fold R² values
        for fold_idx, r2_val in enumerate(r2_scores):
            all_results.append({
                "stim_type": stim,
                "brain_area": brainRegion,
                "window": response_window,
                "r2": r2_val,
                "n_neurons": brain_resp_array.shape[0],
                "fold": fold_idx + 1
            })

        print(f"      Mean R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# ===================== SAVE RESULTS =====================
df = pd.DataFrame(all_results)
results_save_path = os.path.join(save_dir, "ridge_results_population_5fold.csv")
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

wilcox_save_path = os.path.join(save_dir, "wilcoxon_results_population_5fold.csv")
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

    # Get neuron counts per brain area
    neuron_counts = df_sub.groupby("brain_area")["n_neurons"].first()

    # Create color palette for regions
    region_colors = {r: colors.get(r, '#808080') for r in region_order}
    palette = [region_colors[r] for r in region_order]

    # Create figure
    plt.figure(figsize=(14, 12))
    ax = sns.boxplot(
        x="brain_area", y="r2", data=df_sub,
        order=region_order,
        palette=palette,
        showfliers=False,
        width=0.5,
        linewidth=2.5
    )

    # Add stripplot with matching colors
    for i, area in enumerate(region_order):
        area_data = df_sub[df_sub["brain_area"] == area]["r2"]
        x_positions = np.random.normal(i, 0.04, size=len(area_data))
        ax.scatter(x_positions, area_data,
                  alpha=0.7, s=80, color=region_colors[area], edgecolors='black', linewidths=1.5)

    ax.set_xlabel("Brain Area", fontsize=32, labelpad=15, weight='bold')
    ax.set_ylabel("R² (5-fold CV)", fontsize=32, labelpad=15, weight='bold')
    ax.set_title(f"Ridge Regression Performance — {stim}",
                 fontsize=36, pad=25, weight='bold')

    # Horizontal x-axis labels with neuron counts
    xticklabels = [f"{area}\n(n={neuron_counts[area]} neurons)" for area in region_order]
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
    out_path = os.path.join(save_dir, f"{stim}_brain_areas_population_5fold_boxplot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {out_path}")

print("\n=== Analysis complete ===")
print(f"Results saved to: {save_dir}")