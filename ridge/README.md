# Ridge Regression

Population decoding analyses and figures using ridge regression.

## Canonical entry points
- `python ridge/run_all.py`
  runs the full ridge figure set

## Individual figure scripts
- `plot_ridge_per_session.py`
  per-session ridge `R^2` boxplots after balancing brain areas to the same number of valid sessions, subsampling each session to the same neuron count, then averaging `100` repeated `5-fold` CV ridge fits
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
- `plot_ridge_per_session.py` first downsamples each brain area to the same number of eligible sessions, then randomly subsamples each retained session down to `10` neurons for speech or `20` neurons for non-speech, repeats that `100` times, runs `5-fold` CV on each subsample, and averages the resulting session `R^2`
- speech per-session ridge currently uses one ordered 12-stimulus tuple target based on `params.unique_labels`
- per-session ridge summaries also track mean RMSE and mean selected alpha and show those values on the figure panels
- ridge summary boxplots use Mann-Whitney U tests with Bonferroni correction and bracket/star annotations for corrected `p < 0.05`

## Outputs
- figures are written to `figSavePath/decoding/ridge/`
- summary figures use one panel per spike window and shared axes across brain regions
- long-running scripts include `tqdm` progress bars and stage prints
