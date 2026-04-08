# Thesis Methods Summary

Repo-grounded notes for the thesis writeup. This version is scoped to the analyses currently included in the thesis: methods/descriptive figures, PCA, participation ratio, per-session ridge, and linear SVM.

## Scope

- Included analysis families:
  - Methods/descriptive figures
  - Encoding: PCA, participation ratio
  - Decoding/discriminability: per-session ridge, linear SVM
- Main repo locations:
  - Shared settings: [params.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/shared/params.py)
  - Shared utilities/CV helpers: [funcs.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/shared/funcs.py)
  - Statistical annotations: [plot_stats.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/shared/plot_stats.py)
  - Methods figures: [methods/](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/methods)
  - Encoding figures: [pca/](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/pca)
  - Ridge figures: [ridge/](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/ridge)
  - Discriminability figures: [discriminability/](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/discriminability)

## Datasets

- Two datasets:
  - Speech
  - Non-speech
- Brain areas recorded:
  - Primary auditory area
  - Dorsal auditory area
  - Ventral auditory area
  - Posterior auditory area
- Speech population-style analyses exclude dorsal auditory area because of low neuron count in the saved arrays and plotting logic ([funcs.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/shared/funcs.py)).

## Stimuli

- Pure tones:
  - 16 frequencies, logarithmically spaced from 2 to 40 kHz
- AM white noise:
  - 11 modulation rates, logarithmically spaced from 4 to 128 Hz
- Natural sounds:
  - 20 exemplars from 5 categories
  - Categories: Frogs, Crickets, Streamside, Bubbling, Bees
  - 4 exemplars per category
- Speech:
  - 12 FT/VOT stimuli from the boundary/corner positions of the 2D space
  - Use the implemented values including `67`, not `66`
  - Endpoint syllables: `ba`, `da`, `pa`, `ta`

## Trial Structure

- Sounds were passively presented to head-fixed mice.
- No task-performance or reward-contingency variable is used in the analysis code.
- Speech preprocessing:
  - keep the first `20` repeats per FT/VOT token
  - sort trials into a consistent FT/VOT order across sessions
- Non-speech preprocessing caps total trial count at load time:
  - PT: `320` total trials (`20` per stimulus)
  - AM: `220` total trials (`20` per stimulus)
  - natural sounds: `200` total trials (`10` per stimulus)
- If behavior and electrophysiology differ by exactly one trial, the final ephys trial is dropped; larger mismatches are skipped.

## Firing-Rate Arrays

- Raw HDF databases are converted into shared `.npz` firing-rate arrays by [build_firing_rate_arrays.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/preprocessing/build_firing_rate_arrays.py).
- Saved arrays are reused by the downstream analyses in the thesis.
- Saved metadata includes:
  - onset, sustained, and offset firing-rate arrays
  - brain-region labels
  - stimulus labels
  - mouse IDs
  - session IDs
- Speech-only preprocessing:
  - firing rates are mean-centered within neuron across trials before saving shared arrays

## Spike Windows Used in the Thesis Analyses

- Speech:
  - onset: `0.0–0.2 s`
  - sustained: `0.2–0.5 s`
  - offset: `0.5–0.7 s`
- AM:
  - onset: `0.0–0.2 s`
  - sustained: `0.2–0.5 s`
  - offset: `0.5–0.7 s`
- Natural sounds:
  - onset: `0.0–0.5 s`
  - sustained: `1.0–4.0 s`
  - offset: `4.0–4.5 s`
- Pure tones:
  - onset: `0.0–0.05 s`
  - sustained: `0.05–0.1 s`
  - offset: `0.1–0.15 s`

## Encoding

### PCA

- Dataset:
  - equal-neuron population datasets
- Preprocessing:
  - mean-centered within neuron before PCA
- Model:
  - `sklearn.decomposition.PCA()` with default full component computation
- Stored/used outputs:
  - PC scores
  - explained variance ratio
  - participation ratio
  - neuron count
  - trial count

### Participation Ratio

- Computed from the PCA explained-variance spectrum:
  - \((\sum \lambda)^2 / \sum \lambda^2\)
- Used as a scalar measure of effective dimensionality.

## Per-Session Ridge

- Normalization:
  - each session matrix is z-scored before neuron subsampling and CV
- Session balancing and neuron subsampling:
  - keep all eligible sessions in each brain area
  - subsample each retained session to the same neuron count
  - neurons per session:
    - speech: `10`
    - non-speech: `30`
- Speech ridge target:
  - one ordered 12-stimulus tuple target based on [params.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/shared/params.py)`::unique_labels`
- AM target:
  - modulation rate
  - log-transformed before fitting
- PT target:
  - frequency
  - log-transformed before fitting
- Natural-sound target:
  - exemplar identity encoded numerically
- Alpha grid:
  - `np.logspace(-10, 5, 200)`
- Per-session pipeline:
  - identify sessions with enough neurons for the per-session threshold
  - keep all included eligible sessions
  - for each retained session, region, and spike window:
    - z-score predictors across trials
    - randomly subsample neurons `100` times
    - for each subsample, run outer `5`-fold shuffled `KFold`
    - within each outer fold, fit `RidgeCV` on the training split using the same alpha grid and shuffled inner CV
  - final plotted values:
    - one mean test `R^2` value per session
    - fold values are first averaged across the `100` neuron subsamples and then averaged across the `5` outer folds
- Metrics currently stored for per-session ridge:
  - `R^2`
  - `RMSE`
  - mean selected alpha
  - neuron count
  - trial count
- Figure labels/annotations:
  - x-axis labels include `n = number of sessions`
  - panel text shows mean `R^2`, mean `RMSE`, and mean selected alpha
- Appendix-style companion figures:
  - separate per-session alpha summary plots
  - PCA scree plots
  - linear-SVM hyperparameter tuning curves
  - linear-SVM boundary example
  - data-summary figures
- Ridge statistics:
  - Mann-Whitney U

## Discriminability

- Preprocessing:
  - equal-neuron population matrices are z-scored with `StandardScaler.fit_transform()`
- Equal-neuron population counts:
  - speech: `99` neurons per region
  - non-speech: `278` neurons per region
- Region-comparison boxplots:
  - currently use `test_mode="unpaired"`
  - therefore use Mann-Whitney U
- Natural within-vs-between boxplots:
  - also use unpaired Mann-Whitney U in the current figure code
  - compare brain regions within the `Within` and `Between` categories separately

### Linear SVM

- Model:
  - `LinearSVC`
  - `max_iter = 200000`
- Pairwise decoding:
  - binary classification for each stimulus pair
  - leave-one-out CV
- Hyperparameter tuning:
  - tune `C` separately for each sound type, brain area, and spike window
  - grid: `np.logspace(-5, 4, 20)`
- Data handling:
  - pairwise datasets are shuffled with deterministic seeds
  - no additional standardization is applied inside classifier CV because the full population matrix is already z-scored upstream
  - `tol = 1e-3`
- Outputs:
  - pairwise accuracy heatmaps
  - region-comparison boxplots
  - natural-sound within-vs-between category boxplots

### Natural-Sound Within-vs-Between Comparisons

- Implemented for linear SVM in the thesis-facing analyses.
- Pair labels:
  - `Within` if both natural sounds come from the same category
  - `Between` otherwise
- Compared across brain regions within each spike window and pair category.
- Displayed as hue-based boxplots.
- Current plotting uses viridis-based palettes for consistency with the rest of the thesis figures.

## Statistical Analysis

- Multiple-comparison correction:
  - Bonferroni
- Significance annotation style:
  - brackets plus stars
  - only shown for corrected `p < 0.05`
- Star thresholds:
  - `*` for `< 0.05`
  - `**` for `< 0.01`
  - `***` for `< 0.001`

## Thesis Wording Reminders

- Do not describe discriminability as a session-level pipeline.
- Do not say any of the current thesis boxplots use Wilcoxon tests.
- Do not say speech ridge is fit separately to FT and VOT if the thesis is describing the current per-session figure.
- Do not mention population ridge unless you decide to include those figures/results.
- Do not describe UMAP, Pearson, or LDA as part of the current thesis results.
