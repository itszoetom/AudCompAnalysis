# Ridge Regression

Ridge decoding analyses and figures.

## Run

- `python ridge/run_all.py`

## Main Scripts

- `plot_ridge_per_session.py`
  thesis-facing per-session ridge boxplots
- `plot_ridge_population.py`
  population predicted-vs-actual plots and summaries
- `ridge_analysis.py`
  shared ridge helpers

## Per-Session Ridge

- keep all eligible sessions in each brain area
- subsample each retained session to:
  - speech: `10` neurons
  - non-speech: `30` neurons
- run shuffled outer `5`-fold CV with inner `RidgeCV` tuning
- repeat neuron subsampling `100` times
- average across subsamples within each fold
- plot the resulting `5` fold-level `R^2` values per session
- x-axis labels include `n = number of sessions` for each brain area

## Shared Settings

- L2 regularization
- alpha grid: `np.logspace(-10, 5, 200)`
- predictors z-scored before fitting
- boxplot comparisons use Mann-Whitney U with Bonferroni correction

## Output

- figures are written to `figSavePath/decoding/ridge/`
