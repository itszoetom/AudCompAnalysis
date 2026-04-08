# Thesis PDF vs Repo Alignment

This document compares the current text extracted from [UOThesis.pdf](/Users/zoetomlinson/Desktop/College/Honors%20College/Thesis/UOThesis.pdf) against the implemented analysis code in this repository. It focuses on sections that materially affect the Methods, Results framing, and appendix.

## Readability Note
- I extracted the PDF text locally with `pypdf`, so I can compare the draft without needing `pdftotext`.
- I did not modify the PDF itself because only a PDF and `.pages` files are available here, not a plain-text or LaTeX source.

## High-Priority Mismatches

### 2.2.4 Speech Syllables
PDF draft:
- FT and VOT values are listed as `0, 33, 66, 100`.

Repo:
- The code uses the 12 labels in `params.unique_labels`, which include `67`, not `66` ([params.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/shared/params.py)).

Recommended fix:
- Change `66` to `67` everywhere the speech stimulus grid is described.

### 3.4.2 UMAP
PDF draft:
- The section says PCA and UMAP both project trials onto the first two principal components.

Repo:
- PCA uses PC1 and PC2 ([pca/plot_pca_population.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/pca/plot_pca_population.py#L78)).
- UMAP produces `UMAP 1` and `UMAP 2` with `n_neighbors=min(15, n_trials - 1)`, `min_dist=0.1`, `random_state=42`, `n_jobs=1` ([pca/plot_umap_population.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/pca/plot_umap_population.py#L73)).

Recommended fix:
- Rewrite this sentence so PCA is described as projecting trials onto the first two principal components and UMAP is described as embedding trials in a two-dimensional nonlinear manifold.

### 3.5.1 Ridge Regression
PDF draft:
- Population and per-mouse ridge are described as using an `80/20` train-test split to pick alpha.
- Per-session ridge is described as using outer `5-fold` CV with inner `RidgeCV`.
- The metrics list includes a tolerance-band percentage metric.

Repo:
- The shared ridge implementation uses outer `5-fold` CV with shuffled folds plus inner `RidgeCV` alpha selection for the main ridge pipeline ([funcs.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/shared/funcs.py)).
- Population ridge repeats this shared CV pipeline for `30` random seeds and stores `R2 Test`, `RMSE`, `Pearson r`, and `Best Alpha` ([ridge/ridge_analysis.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/ridge/ridge_analysis.py#L139)).
- There is no tolerance-band metric in the implemented pipeline.

Recommended fix:
- Replace the `80/20 train-test split` description with `outer 5-fold cross-validation with inner RidgeCV alpha selection`.
- Remove the tolerance-band metric unless you plan to implement it.

### 3.5.2 Pearson Correlation and Session-Based Discriminability
PDF draft:
- Discriminability is described as a session-level analysis with up to five sessions per region.
- It says neurons are subsampled `100` times per session with `10` neurons for speech and `30` for non-speech.

Repo:
- Discriminability is currently implemented as a population-level equal-neuron analysis built from `build_population_dataset(...)` ([discriminability/discriminability_analysis.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/discriminability/discriminability_analysis.py#L56)).
- Population neuron counts are `99` for speech and `278` for non-speech ([funcs.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/shared/funcs.py)).
- The session-based repeated subsampling logic exists for ridge per-session, not for discriminability ([ridge/plot_ridge_per_session.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/ridge/plot_ridge_per_session.py#L96)).

Recommended fix:
- Rewrite the discriminability section as a population-level equal-neuron analysis.
- Remove the `up to five sessions` description unless you intend to re-implement discriminability that way.

### 3.5.3 Linear SVM
PDF draft:
- Linear SVM is described as `LinearSVC` with fixed `C=1.0`, `dual=auto`, and stratified cross-validation with up to five folds.

Repo:
- The implemented model is `LinearSVC(C=c_value, max_iter=10000)` ([discriminability/discriminability_analysis.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/discriminability/discriminability_analysis.py#L165)).
- `C` is tuned separately for each sound type, brain area, and spike window over `np.logspace(-5, 4, 20)` ([discriminability/discriminability_analysis.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/discriminability/discriminability_analysis.py), [funcs.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/shared/funcs.py)).
- Classification uses leave-one-out CV, not stratified 5-fold CV ([funcs.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/shared/funcs.py)).
- The current code sets `dual="auto"` and `tol=1e-3`.

Recommended fix:
- Describe linear SVM as leave-one-out pairwise decoding with condition-specific `C` tuning over a logarithmic grid.

### 4.4 Statistical Analysis
PDF draft:
- Ridge boxplots are described as using MWU only.
- Discriminability boxplots are described as using Wilcoxon when matched and MWU otherwise.

Repo:
- Ridge boxplots now use MWU in the thesis-facing figures ([ridge/ridge_analysis.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/ridge/ridge_analysis.py), [plot_stats.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/shared/plot_stats.py)).
- Discriminability region and natural within-between boxplots are also called with `test_mode="unpaired"`, so they use MWU ([discriminability/discriminability_analysis.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/discriminability/discriminability_analysis.py)).
- Bonferroni correction and bracket/star annotations are implemented in the shared plotting helper ([plot_stats.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/shared/plot_stats.py)).

Recommended fix:
- Reverse the current wording: ridge uses auto Wilcoxon-or-MWU, while discriminability currently uses MWU in the plotted figures.

### 3.3.2 Neuron Subsampling
PDF draft:
- Session-level analyses are described as subsampling `n=30` neurons per session.

Repo:
- Current per-session ridge settings are `10` neurons for speech and `30` for AM, PT, and natural sounds ([params.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/shared/params.py)).
- Per-session ridge also balances the number of sessions across brain areas before scoring ([ridge/plot_ridge_per_session.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/ridge/plot_ridge_per_session.py#L66)).

Recommended fix:
- Replace the `n=30` statement with the current `10/20` settings and add the session-balancing sentence.

### Appendix A.1 Linear Discriminant Analysis
PDF draft:
- LDA is described as excluded from the main analysis.

Repo:
- LDA is still part of the main discriminability pipeline (`plot_lda.py`, `ANALYSIS_METHODS`) ([discriminability/plot_lda.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/discriminability/plot_lda.py), [discriminability/discriminability_analysis.py](/Users/zoetomlinson/Desktop/MurrayLab/AudCompAnalysis/discriminability/discriminability_analysis.py#L175)).

Recommended fix:
- Either remove the appendix claim or actually remove LDA from the main pipeline if that is the thesis decision.

## Lower-Priority Mismatches

### Results Section File References
PDF draft:
- Results pages list old script names such as `pca/plot_pca_all_mice.py`, `ridge/plot_ridge_boxplot.py`, and `ridge/plot_ridge_per_mouse.py`.

Repo:
- The current entry points are:
- `pca/plot_pca_population.py`
- `pca/plot_pca_population_avgs.py`
- `pca/plot_pca_speech.py`
- `pca/plot_umap_population.py`
- `ridge/plot_ridge_population.py`
- `ridge/plot_ridge_per_session.py`
- `discriminability/plot_pearson.py`
- `discriminability/plot_linear_svm.py`
- `discriminability/plot_lda.py`

Recommended fix:
- Update the Results placeholders so they reference existing scripts only.

## Repo-Aligned Replacement Text

### Replacement for Ridge Methods Paragraph
"Ridge regression was used to decode stimulus features from neural population activity. For speech stimuli, formant transition (FT) and voice onset time (VOT) were treated as separate regression targets and modeled independently. For AM and pure tones, target values were log-transformed before fitting. The ridge pipeline used shuffled outer 5-fold cross-validation with feature standardization performed within each training fold and inner RidgeCV selection of the regularization parameter over 200 values spanning 10^-10 to 10^5. Performance was summarized using held-out R^2, root mean squared error, Pearson correlation between predicted and observed targets, and the mean selected alpha."

### Replacement for Discriminability Methods Paragraph
"Population discriminability analyses were performed on equal-neuron population datasets constructed separately for each sound type, brain region, and spike window. Predictor matrices were z-scored before analysis. Pearson discriminability quantified similarity between mean population response vectors for each stimulus pair as 1 - r. Linear SVM and LDA evaluated pairwise stimulus discrimination using leave-one-out cross-validation. For linear SVM, the regularization parameter C was tuned separately for each sound type, brain region, and spike window over a logarithmically spaced grid from 10^-2 to 10^4."

### Replacement for Statistics Paragraph
"Statistical comparisons were applied to summary distribution figures rather than descriptive example plots. Pairwise p-values were Bonferroni-corrected within panel. In the current implementation, discriminability summary plots use two-sided Mann-Whitney U tests, whereas ridge summary plots use Wilcoxon signed-rank tests when matched observations are available and Mann-Whitney U tests otherwise. Corrected p-values below 0.05 were annotated on figures using bracket-and-star notation."

## Next Editing Constraint
- I can compare and rewrite against the PDF, but I cannot directly edit [UOThesis.pdf](/Users/zoetomlinson/Desktop/College/Honors%20College/Thesis/UOThesis.pdf) as a normal text document.
- If you want me to make the changes directly, I need an editable source:
- exported `.docx`
- `.tex`
- copied/pasted thesis text
- or a `.pages` export that includes editable text in a format I can patch reliably
