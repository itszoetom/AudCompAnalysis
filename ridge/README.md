# Ridge Regression

Population decoding analyses and figures using ridge regression.

## Canonical entry points
- `python ridge/run_all.py`
  runs the full ridge figure set

## Individual figure scripts
- `plot_ridge_per_session.py`
  per-session ridge `R^2` boxplots after `100` random neuron subsamples per session, each scored with `5-fold` CV and averaged
- `plot_ridge_population.py`
  population predicted-vs-actual ridge scatter plots plus population `R^2` boxplot summaries

## Shared helpers
- `ridge_analysis.py`
  ridge-specific target construction and wrappers around the shared CV pipeline in `funcs.py`

## Implementation notes
- ridge uses L2 regularization
- alpha tuning uses a `logspace(1e-10, 1e5)`-style grid implemented as `np.logspace(-10, 5, 200)`
- predictors are z-scored with `StandardScaler`
- all ridge analyses now use the same outer `5-fold` CV with inner `RidgeCV` tuning
- `plot_ridge_per_session.py` randomly subsamples each valid session down to `10` neurons for speech or `30` neurons for non-speech, repeats that `100` times, runs `5-fold` CV on each subsample, and averages the resulting session `R^2`

## Outputs
- figures are written to `figSavePath/ridge/`
- summary figures use one panel per spike window and shared axes across brain regions
- long-running scripts include `tqdm` progress bars and stage prints
