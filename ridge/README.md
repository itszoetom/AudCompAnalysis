# Ridge Regression

Population decoding analyses and figures using ridge regression.

## Canonical entry points
- `python ridge/run_all.py`
  runs the full ridge figure set

## Individual figure scripts
- `plot_ridge_boxplot.py`
  population ridge `R^2` boxplots by brain region and spike window, with pairwise MWU plus Bonferroni correction
- `plot_ridge_per_mouse.py`
  per-mouse ridge `R^2` boxplots in the same format
- `plot_ridge_per_session.py`
  per-session ridge `R^2` boxplots in the same format
- `ridge_population.py`
  predicted-vs-actual ridge plots for population datasets

## Shared helpers
- `ridge_analysis.py`
  data loading, repeated ridge fitting, population subsampling, alpha search, and figure styling

## Implementation notes
- ridge uses L2 regularization
- alpha tuning uses a `logspace(1e-10, 1e5)`-style grid implemented as `np.logspace(-10, 5, 200)`
- predictors are z-scored with `StandardScaler`
- population and per-mouse scripts use held-out split tuning
- per-session scripts use outer `5-fold` CV with inner `RidgeCV` tuning

## Outputs
- figures are written to `figSavePath/ridge/`
- summary figures use one panel per spike window and shared axes across brain regions
- long-running scripts include `tqdm` progress bars and stage prints
