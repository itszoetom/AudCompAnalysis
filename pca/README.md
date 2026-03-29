# PCA

Population PCA and UMAP figures for the thesis datasets.

## Canonical entry points
- `python pca/run_all.py`
  runs the full PCA figure set

## Individual figure scripts
- `plot_pca_population.py`
  population PCA scatter plots and scree plots for every sound type
- `plot_pca_population_averages.py`
  trial-averaged PCA scatter plots
- `plot_pca_speech.py`
  speech PCA figures colored separately by FT and VOT
- `plot_umap_population.py`
  population UMAP figures

## Shared helpers
- `pca_analysis.py`
  loading, fixed-neuron subsampling, PCA helpers, averaging helpers, and figure styling

## Outputs
- figures are written to `figSavePath/pca/`
- population figures are organized by sound type, brain region, and spike window

## Notes
- speech figure panels exclude dorsal auditory area because of low neuron count
- long-running scripts include `tqdm` progress bars and stage prints
