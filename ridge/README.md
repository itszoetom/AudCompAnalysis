# Decoding: Ridge Regression Figures

Per-session ridge regression decoding analyses for pure tones (PT) and AM white noise, corresponding to Figures 7–8 and Supplemental Figures S6–S8 of the thesis.

Ridge regression is applied to PT and AM only. Natural sounds have no inherent acoustic ordering (arbitrary ranking would be meaningless), and speech stimuli are defined along two independent orthogonal acoustic dimensions (VOT and FT) that cannot be collapsed into a single regression target. Discriminability analysis (pairwise SVM) is used for those sound types instead.

## Run

```bash
python ridge/run_all.py
```

## Scripts

### `plot_ridge_per_session.py` → Figures 7–8
Per-session ridge R² boxplots: one box per subregion, one figure per sound type. Also generates alpha tuning curves (Supplemental Figures S6–S7) and predicted-vs-actual scatter examples (Supplemental Figure S8).

### `plot_ridge_population.py`
Population-level ridge summaries across sound type × subregion × window.

### `ridge_analysis.py`
Shared helpers: `fit_best_ridge`, `run_population_ridge`, `build_target_datasets`, `plot_ridge_summary`.

## Per-Session Ridge Pipeline (Figures 7–8)

1. **Eligibility:** A session is included if it has ≥ 30 neurons in the target subregion.
2. **Subsampling:** Each retained session is subsampled to exactly 30 neurons, repeated 100 times with different random seeds.
3. **Cross-validation:** Each subsample is evaluated with 5-fold cross-validation (folds shuffled; random state fixed for reproducibility).
4. **Alpha selection:** Within each training fold, `RidgeCV` selects the best regularization strength from a grid of 200 values logarithmically spaced from 10⁻⁵ to 10¹⁰ (`np.logspace(-5, 10, 200)`).
5. **Regression target:** Stimulus parameter values are log-transformed before fitting, consistent with the logarithmic spacing of PT frequencies and AM modulation rates.
6. **Score:** Mean R² across the 5 test folds, averaged over the 100 subsamples, is the reported per-session score.
7. **Predictors:** All neuron firing rates are z-scored independently before fitting within each fold.

## Summary Statistics

Pairwise subregion comparisons use **unpaired Mann-Whitney U tests with Bonferroni correction**. Legend entries include session counts per subregion (n = N).

## Output

Figures are written to `figSavePath/decoding/ridge/`.
