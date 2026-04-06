# Auditory Cortex Population Analysis

Population-level auditory-cortex analyses for the Murray Lab thesis project.

The repo combines:
- a speech dataset with FT/VOT continua (`speech`)
- a non-speech dataset with pure tones (`PT`), AM white noise (`AM`), and natural sounds (`naturalSound`)

Raw HDF databases are converted into shared `.npz` firing-rate arrays, and those arrays are reused across methods figures, PCA/UMAP, ridge regression, and discriminability analyses.

## Experimental Design

- Speech subjects: `feat004` to `feat010`
- Non-speech subjects: `feat014` to `feat019`
- Brain regions: `Primary`, `Dorsal`, `Ventral`, `Posterior`
- Population neuron counts:
  - speech: `99` per region
  - non-speech: `278` per region

## Main Files

- `build_firing_rate_arrays.py`
  builds the shared `.npz` firing-rate arrays
- `params.py`
  shared paths, metadata, spike windows, and neuron-count settings
- `funcs.py`
  shared data loading, dataset building, subsampling, and CV helpers
- `plot_stats.py`
  Bonferroni-corrected boxplot annotation helpers

Folders:
- `methods/`
  descriptive figures
- `pca/`
  PCA and UMAP figures
- `ridge/`
  ridge decoding figures
- `discriminability/`
  Pearson, linear-SVM, and LDA discriminability figures

## Typical Run Order

1. `python build_firing_rate_arrays.py`
2. `python methods/run_all.py`
3. `python pca/run_all.py`
4. `python ridge/run_all.py`
5. `python discriminability/run_all.py`

## Notes

- Paper-style figures use serif fonts and consistent region naming.
- Boxplot annotations use Mann-Whitney U tests with Bonferroni correction in the current thesis-facing figures.
- Natural-sound discriminability also includes within- vs. between-category summaries.

## Environment

The code expects local access to `jaratoolbox` plus the HDF databases referenced in `params.py`. Saved `.npz` outputs are written to the configured `dbSavePath`, not into this Git repo.
