# Decoding: Discriminability

Pairwise stimulus discriminability using linear SVM classifiers (Figures 9–11, Supplemental S9–S13). Pearson dissimilarity and LDA are also computed as supplemental comparisons.

## Run

```bash
python -m scripts.discriminability.run_all
```

Runtime: roughly 30–40 minutes for the full pipeline.

## Scripts

- `plot_linear_svm.py` → **Figures 9–11, S9–S13.** All linear SVM figures: heatmaps, regional boxplots, natural-sound within/between boxplots, C-tuning curves, and decision boundary examples.
- `plot_pearson.py` — supplemental Pearson dissimilarity heatmaps and boxplots.
- `plot_lda.py` — supplemental LDA classification heatmaps and boxplots.
- `discriminability_analysis.py` — shared helpers (`run_method_analysis`, `draw_heatmap_grid`, `plot_region_boxplots`, …).

## Linear SVM Pipeline

1. Population matrices per subregion concatenate all sessions and animals along the neuron axis.
2. Neurons subsampled to a fixed equal count (278 for non-speech, 99 for speech) using a deterministic seed.
3. Predictor firing rates z-scored independently.
4. `LinearSVC` with regularization C tuned over 20 log-spaced values from 10⁻⁵ to 10⁴, fit per sound × region × window.
5. Pairwise accuracy estimated by shuffled leave-one-out cross-validation across all stimulus pairs.

For natural sounds, pairs are also labeled **within-category** vs. **between-category** to assess categorical organization.

Pairwise subregion comparisons use unpaired Mann-Whitney U with Bonferroni correction.

## Output

- Heatmaps and boxplots: `figSavePath/decoding/<method_key>/<sound_type>/`
- SVM tuning and example figures: `figSavePath/decoding/discriminability/`
- CSV results: `figSavePath/decoding/<method_key>_results.csv`
