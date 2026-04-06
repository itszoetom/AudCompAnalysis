# PCA

Population PCA and UMAP figures for the thesis datasets.

## Run

- `python pca/run_all.py`

## Main Scripts

- `plot_pca_population.py`
  PCA scatter and scree plots
- `plot_pca_population_avgs.py`
  trial-averaged PCA plots
- `plot_pca_speech.py`
  speech PCA plots colored by FT and VOT
- `plot_umap_population.py`
  UMAP plots
- `pca_analysis.py`
  shared PCA/UMAP helpers

## Output

- figures are written to `figSavePath/pca/`
- panels are organized by sound type, brain region, and spike window

## Notes

- speech population-style figures exclude dorsal auditory area
