import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from jaratoolbox import settings, celldatabase

studyparams = __import__('2025acpop.studyparams').studyparams

# SETTINGS
plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 32,
    "axes.labelsize": 28,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 22
})

neuron_threshold = 30
stim_types = ["naturalSound", "AM", "pureTones"]
response_ranges = ["sustained"]   # only sustained
alphas = np.logspace(-3, 3, 20)
n_splits = 5
n_sessions = 3   # <-- max sessions per area

figdataPath = os.path.join(settings.FIGURES_DATA_PATH)

results_dir = os.path.join(settings.SAVE_PATH, "Ridge Regression")
os.makedirs(results_dir, exist_ok=True)

np.random.seed(42)

# LOAD CELLDB
dbCoordsFilename = os.path.join(
    figdataPath,
    f'celldb_{studyparams.STUDY_NAME}_responsive_all_stims_index_new.h5'
)
celldb = celldatabase.load_hdf(dbCoordsFilename)
celldb['simpleSiteName'] = celldb['recordingSiteName'].str.split(',').apply(lambda x: x[0])

areas_of_interest = [
    "Dorsal auditory area",
    "Primary auditory area",
    "Ventral auditory area",
    "Posterior auditory area"
]
short_names = {
    "Dorsal auditory area": "Dorsal",
    "Primary auditory area": "Primary",
    "Ventral auditory area": "Ventral",
    "Posterior auditory area": "Posterior"
}
formal_names = {'pureTones': 'Pure Tones',
                'AM': 'AM White Noise',
                'naturalSound': 'Natural Sounds'}

aud_db = celldb[celldb['simpleSiteName'].isin(areas_of_interest)].reset_index()

# FIND SESSIONS WITH ENOUGH NEURONS
grouped_data = aud_db.groupby(['simpleSiteName', 'date']).size()
eligible_sessions = grouped_data[grouped_data >= neuron_threshold].reset_index(name='n_neurons')

print(f"Total sessions with ≥{neuron_threshold} neurons (all areas): {len(eligible_sessions)}")

# pick up to n_sessions PER AREA
selected_sessions_list = []
for area in areas_of_interest:
    area_sessions = eligible_sessions[eligible_sessions['simpleSiteName'] == area]
    if area_sessions.empty:
        continue
    if len(area_sessions) > n_sessions:
        area_sessions = area_sessions.sample(n=n_sessions, random_state=42)
    selected_sessions_list.append(area_sessions)

selected_sessions = pd.concat(selected_sessions_list, ignore_index=True)

print("Using these sessions (per area):")
for _, row in selected_sessions.iterrows():
    print(f"  {row['simpleSiteName']} — {row['date']}")

# MAIN LOOP
fold_results = []
session_results = []

for stim in stim_types:
    stim_arrays = np.load(os.path.join(figdataPath, f"fr_arrays_{stim}.npz"), allow_pickle=True)
    brainRegionArray = stim_arrays["brainRegionArray"]
    sessionArray = stim_arrays["sessionIDArray"]
    stimArray = stim_arrays["stimArray"][0, :]  # 1D trial vector

    for respRange in response_ranges:
        # neurons × trials (raw)
        respArray_raw = stim_arrays[f"{respRange}fr"]

        # === STANDARDIZE ONCE PER STIM × WINDOW ===
        # standardize over trials, keep neurons as rows
        scaler = StandardScaler()
        respArray = scaler.fit_transform(respArray_raw.T).T

        # loop over the pre-selected sessions (max 3 per area)
        for _, ses in selected_sessions.iterrows():
            area = ses['simpleSiteName']
            date = ses['date']

            area_idx = np.where(brainRegionArray == area)[0]
            session_idx = np.where(sessionArray == date)[0]
            neuron_idx = np.intersect1d(area_idx, session_idx)

            # need at least 30 to make them comparable
            if neuron_idx.size < neuron_threshold:
                continue

            # choose exactly 30 neurons at random
            chosen_neurons = np.random.choice(neuron_idx, size=neuron_threshold, replace=False)

            # X: trials × neurons (already standardized globally above)
            X = respArray[chosen_neurons, :].T
            y = stimArray.copy()

            # transform y for AM and tones
            if stim in ["AM", "pureTones"]:
                y = np.log(y + 1e-8)

            # k-fold cross-validation
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            r2_scores = []
            last_y_test = None
            last_y_pred = None

            for fold_i, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                ridge = RidgeCV(alphas=alphas)
                ridge.fit(X_train, y_train)
                y_pred = ridge.predict(X_test)
                fold_r2 = r2_score(y_test, y_pred)
                r2_scores.append(fold_r2)

                # save fold-level result
                fold_results.append({
                    "stim_type": stim,
                    "brain_area": area,
                    "window": respRange,
                    "date": date,
                    "fold": fold_i,
                    "r2": fold_r2
                })

                last_y_test = y_test
                last_y_pred = y_pred

            # session-level mean (for stats)
            mean_r2 = np.mean(r2_scores)
            session_results.append({
                "stim_type": stim,
                "brain_area": area,
                "window": respRange,
                "date": date,
                "r2": mean_r2
            })

            # QC plot per (stim, area, date)
            if last_y_test is not None:
                # Reverse log transform for AM and pureTones
                if stim in ["AM", "pureTones"]:
                    plot_y_test = np.exp(last_y_test) - 1e-8
                    plot_y_pred = np.exp(last_y_pred) - 1e-8
                else:
                    plot_y_test = last_y_test
                    plot_y_pred = last_y_pred

                fig, ax = plt.subplots(figsize=(15, 15))  # increased from (10, 10)
                ax.scatter(plot_y_test, plot_y_pred, alpha=0.8, s=120)  # increased from s=60
                min_val = min(plot_y_test.min(), plot_y_pred.min())
                max_val = max(plot_y_test.max(), plot_y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=11)  # increased from 6

                ax.set_title(
                    f"Prediction Performance for {formal_names.get(stim)} in {short_names.get(area)}"
                    f"\nPredicted vs Actual R²={mean_r2:.2f}, n=30",
                    fontsize=40,  # increased from 35
                    pad=25  # increased from 20
                )
                ax.set_xlabel("Actual", fontsize=40, labelpad=15)  # increased from 35, 10
                ax.set_ylabel("Predicted", fontsize=40, labelpad=15)  # increased from 35, 10
                ax.tick_params(axis='both', labelsize=35)  # increased from 30
                ax.set_xlim(min_val, max_val)
                ax.set_ylim(min_val, max_val)
                ax.grid(True, linestyle='--', alpha=0.4)

                pvadir = os.path.join(results_dir, "pred_vs_actual")
                os.makedirs(pvadir, exist_ok=True)
                fname = f"{stim}_{area.replace(' ', '_')}_{date}_pred_vs_actual.png"
                fig.savefig(os.path.join(pvadir, fname), dpi=300, bbox_inches='tight')
                plt.close(fig)

# SAVE RESULTS TABLES
df_folds = pd.DataFrame(fold_results)
df_sessions = pd.DataFrame(session_results)

fold_save_path = os.path.join(results_dir, "ridge_results_sustained_only_folds.csv")
sess_save_path = os.path.join(results_dir, "ridge_results_sustained_only_sessions.csv")
df_folds.to_csv(fold_save_path, index=False)
df_sessions.to_csv(sess_save_path, index=False)
print(f"Saved fold-level results to {fold_save_path}")
print(f"Saved session-level results to {sess_save_path}")

# STATS (fold-level distributions, MWU + Bonferroni)
apply_bonferroni = True
region_order_all = [
    "Dorsal auditory area",
    "Posterior auditory area",
    "Primary auditory area",
    "Ventral auditory area"
]

mwu_rows = []

for stim in stim_types:
    fdf = df_folds[df_folds["stim_type"] == stim].copy()
    if fdf.empty:
        continue

    areas_present = fdf["brain_area"].unique()

    for a1, a2 in combinations(areas_present, 2):
        vals1 = fdf.loc[fdf["brain_area"] == a1, "r2"].dropna().values
        vals2 = fdf.loc[fdf["brain_area"] == a2, "r2"].dropna().values

        if len(vals1) < 2 or len(vals2) < 2:
            continue

        stat, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
        mwu_rows.append({
            "stim": stim,
            "area_1": a1,
            "area_2": a2,
            "n1": len(vals1),
            "n2": len(vals2),
            "U": stat,
            "p": p
        })

mwu_df = pd.DataFrame(mwu_rows)

if not mwu_df.empty and apply_bonferroni:
    corrected = []
    for stim in mwu_df["stim"].unique():
        sub = mwu_df[mwu_df["stim"] == stim].copy()
        sub["p_corrected"] = multipletests(sub["p"], method="bonferroni")[1]
        corrected.append(sub)
    mwu_df = pd.concat(corrected, ignore_index=True)
else:
    mwu_df["p_corrected"] = mwu_df["p"]

mwu_outpath = os.path.join(results_dir, "mwu_results_fold_level.csv")
mwu_df.to_csv(mwu_outpath, index=False)
print(f"Saved MWU tests on fold-level distributions to {mwu_outpath}")


def add_bracket(ax, x1, x2, base_y, h, text):
    ax.plot([x1, x1, x2, x2], [base_y, base_y + h, base_y + h, base_y],
            lw=3, c='k', clip_on=False)
    ax.text((x1 + x2) / 2, base_y + h, text,
            ha='center', va='bottom', color='k', fontsize=24, weight='bold')


# marker styles for sessions
marker_styles = ['o', 's', '^', 'D', 'P', 'X', '*']

for stim in stim_types:
    df_sub = df_folds[df_folds["stim_type"] == stim].copy()
    if df_sub.empty:
        continue

    # shorten area names for x-axis
    df_sub["brain_area_short"] = df_sub["brain_area"].str.replace(" auditory area", "")
    region_order_short = [r.replace(" auditory area", "") for r in region_order_all]
    region_order = [r for r in region_order_short if r in df_sub["brain_area_short"].unique()]

    # session counts for label
    sess_sub = df_sessions[df_sessions["stim_type"] == stim].copy()
    sess_sub["brain_area_short"] = sess_sub["brain_area"].str.replace(" auditory area", "")
    session_counts = sess_sub.groupby("brain_area_short")["date"].nunique()

    plt.figure(figsize=(14, 12))
    ax = sns.boxplot(
        x="brain_area_short", y="r2", data=df_sub,
        order=region_order,
        palette="Set2",
        showfliers=False,
        width=0.5
    )

    # overlay fold points
    for i, area_short in enumerate(region_order):
        area_mask = df_sub["brain_area_short"] == area_short
        area_df = df_sub[area_mask]

        area_dates = area_df["date"].unique()
        for j, date in enumerate(area_dates):
            marker = marker_styles[j % len(marker_styles)]
            ses_df = area_df[area_df["date"] == date]
            x_vals = np.full(len(ses_df), i, dtype=float)
            x_vals = x_vals + (np.linspace(-0.08, 0.08, len(ses_df)))
            ax.scatter(
                x_vals,
                ses_df["r2"].values,
                marker=marker,
                s=160,
                color='black',
                edgecolor='white',
                linewidth=1.2,
                zorder=5,
                label=None if i > 0 or j > 0 else "session folds"
            )

    ax.set_ylabel("R² (5-fold CV, per session)", fontsize=35, labelpad=15, weight='bold')
    ax.set_title(f"{formal_names.get(stim)} Ridge Regression R²", fontsize=45, pad=25, weight='bold')

    xticklabels = []
    for area_short in region_order:
        n_sess = session_counts.get(area_short, 0)
        xticklabels.append(f"{area_short}\n(n={n_sess})")
    ax.set_xticklabels(xticklabels, rotation=0, ha='center')
    ax.tick_params(axis='both', labelsize=35)

    # stats brackets
    sub_mwu = mwu_df[mwu_df["stim"] == stim].copy()
    if not sub_mwu.empty:
        sub_mwu["area_1_short"] = sub_mwu["area_1"].str.replace(" auditory area", "")
        sub_mwu["area_2_short"] = sub_mwu["area_2"].str.replace(" auditory area", "")

        ymax = df_sub["r2"].max()
        ymin = df_sub["r2"].min()
        y_range = ymax - ymin if ymax > ymin else 1.0
        step = 0.08 * y_range
        base_top = ymax + 0.05 * y_range

        sub_mwu = sub_mwu[
            sub_mwu["area_1_short"].isin(region_order)
            & sub_mwu["area_2_short"].isin(region_order)
        ]

        sub_mwu["x1_pos"] = sub_mwu["area_1_short"].apply(lambda x: region_order.index(x))
        sub_mwu["x2_pos"] = sub_mwu["area_2_short"].apply(lambda x: region_order.index(x))
        sub_mwu["distance"] = (sub_mwu["x2_pos"] - sub_mwu["x1_pos"]).abs()
        sub_mwu = sub_mwu.sort_values("distance")

        bracket_idx = 0
        for row in sub_mwu.itertuples(index=False):
            if row.p_corrected < 0.05:
                x1 = row.x1_pos
                x2 = row.x2_pos
                p = row.p_corrected
                star_text = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                add_bracket(
                    ax,
                    x1,
                    x2,
                    base_top + bracket_idx * step,
                    step * 0.7,
                    star_text
                )
                bracket_idx += 1

    ax.grid(True, linestyle="--", alpha=0.3, axis='y')
    plt.tight_layout()
    out_path = os.path.join(results_dir, f"{stim}_brain_areas_sustained_boxplot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved poster plot to {out_path}")
