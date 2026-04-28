# Decoding: Discriminability Figures

Pairwise stimulus discriminability analyses using linear SVM classifiers, corresponding to Figures 9–11 and Supplemental Figures S9–S13 of the thesis. Pearson dissimilarity and LDA are also computed as supplemental comparisons.

## Run

```bash
python discriminability/run_all.py
```

Runtime is approximately 30–40 minutes for the full pipeline.

## Scripts

### `discriminability_analysis.py`
Shared analysis and plotting helpers. Key functions:
- `run_pairwise_analysis`: runs one discriminability method across all sound × region × window conditions
- `run_method_analysis`: orchestrates loading, computing, and saving results for one method
- `draw_heatmap_grid`: renders pairwise similarity/accuracy matrices per subregion and window
- `plot_region_boxplots`: boxplots of mean pairwise accuracy across subregions and windows
- `plot_natural_within_between_boxplots`: within- vs. between-category breakdown for natural sounds
- `plot_svm_hyperparameter_tuning`: C-parameter tuning curves per subregion and window
- `plot_linear_svm_example`: 2×2 grid of example SVM decision boundaries in PCA-reduced space

### `plot_linear_svm.py` → Figures 9–11, S9–S13
Generates all linear SVM figures: heatmaps, regional boxplots, natural-sound within/between boxplots, C-tuning curves, and decision boundary examples.

### `plot_pearson.py` → supplemental comparison
Pearson dissimilarity heatmaps and boxplots (1 − correlation between trial-averaged responses).

### `plot_lda.py` → supplemental comparison
LDA classification accuracy heatmaps and boxplots.

### `run_all.py`
Runs analyses and all plotting scripts in sequence: Pearson → linear SVM → LDA.

## Linear SVM Pipeline

1. **Population matrix:** Neurons from all sessions and animals within each subregion are concatenated along the neuron axis.
2. **Subsampling:** Neurons are randomly subsampled to a fixed count (278 for non-speech, 99 for speech) using a deterministic seed.
3. **Standardization:** Predictor matrices are z-scored independently before classification.
4. **Classifier:** `LinearSVC` with regularization `C` selected from a grid of 20 logarithmically spaced values from 10⁻⁵ to 10⁴ (`np.logspace(-5, 4, 20)`), tuned per sound type × subregion × window.
5. **Cross-validation:** Pairwise classification accuracy is estimated using **shuffled leave-one-out cross-validation** (one held-out trial per fold).
6. **Result:** Mean LOO accuracy across all stimulus pairs for each subregion × window condition.

## Natural Sound Within/Between Categories

For natural sounds, stimulus pairs are labeled as *Within-category* (both exemplars from the same environmental category) or *Between-category* (exemplars from different categories). Results are split and compared separately to assess categorical organization.

## Statistical Analysis

All pairwise subregion comparisons use **unpaired Mann-Whitney U tests with Bonferroni correction**. Significance is annotated with bracket-and-star notation (∗ p < 0.05, ∗∗ p < 0.01, ∗∗∗ p < 0.001).

## Output

- Heatmaps and boxplots: `figSavePath/decoding/<method_key>/<sound_type>/`
- SVM tuning and example figures: `figSavePath/decoding/discriminability/`
- CSV results: `figSavePath/decoding/<method_key>_results.csv`
