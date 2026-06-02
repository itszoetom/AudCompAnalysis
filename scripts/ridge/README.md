# Decoding: Ridge Regression

Per-session ridge regression decoding for pure tones (PT) and AM white noise (Figures 7–8, Supplemental S6–S8).

Ridge regression is applied only to PT and AM. Natural sounds lack inherent acoustic ordering, and speech varies along two orthogonal dimensions (VOT, FT) that cannot be collapsed into a single regression target. Discriminability (pairwise SVM) handles those sound types instead.

## Run

```bash
python -m scripts.ridge.run_all
```

## Scripts

- `plot_ridge_per_session.py` (Figures 7–8): per-session ridge R² boxplots, plus α tuning curves (S6–S7) and predicted-vs-actual scatter examples (S8).
- `plot_ridge_population.py`: population-level ridge summaries across sound, subregion, and window.
- `ridge_analysis.py`: shared helpers (`fit_best_ridge`, `run_population_ridge`, target builders).

## Per-Session Pipeline

1. Session included if it has at least 30 neurons in the target subregion.
2. Subsample to exactly 30 neurons, repeated 100 times with different seeds.
3. Neuron firing rates z-scored independently within each training fold.
4. Shuffled 5-fold cross-validation; `RidgeCV` selects α from 200 log-spaced values between 10⁻⁵ and 10¹⁰.
5. Regression targets log-transformed, matching the log-spaced stimuli.
6. Score is the mean R² across folds, averaged over the 100 subsamples.

Pairwise subregion comparisons use unpaired Mann-Whitney U with Bonferroni correction; legend entries include session counts.

## Output

Figures are written to `figSavePath/decoding/ridge/`.
