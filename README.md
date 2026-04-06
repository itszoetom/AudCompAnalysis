# Auditory Cortex Population Analysis

Population-level auditory-cortex analyses for the Murray Lab thesis project. The repo combines:
- a natural-sounds dataset with pure tones (`PT`), AM white noise (`AM`), and natural sounds (`naturalSound`)
- a speech dataset with FT/VOT continua (`speech`)

Raw HDF databases are converted into shared `.npz` firing-rate arrays, and those arrays are reused across PCA/UMAP, ridge-regression, methods figures, and discriminability analyses.

## Datasets

Natural Sounds Dataset:
- Subjects: `feat014` to `feat019`
- Brain regions: `Primary`, `Dorsal`, `Ventral`, `Posterior`
- Stimuli: `PT`, `AM`, `naturalSound`
- Population figure subsampling: `278` neurons per region

Speech Dataset:
- Subjects: `feat004` to `feat010`
- Brain regions in saved arrays: `Primary`, `Dorsal`, `Ventral`, `Posterior`
- Paper-style population figures: exclude `Dorsal` because of low neuron count
- Stimuli: 12 FT/VOT speech identities, with four syllable endpoints (`ba`, `da`, `pa`, `ta`)
- Population figure subsampling: `99` neurons per region

## Main Files

Root:
- `build_firing_rate_arrays.py`
  builds the saved `.npz` response arrays for all sound types
- `funcs.py`
  centralized shared data loading, dataset builders, labeling, neuron sampling, figure setup, and the canonical ridge CV pipeline
- `params.py`
  shared paths, subject lists, stimulus metadata, spike windows, and plotting colors
- `plot_stats.py`
  shared Bonferroni-corrected statistical annotation helpers for boxplot-style figures

`pca/`:
- `pca_analysis.py`
  PCA-specific summaries and plotting helpers built on `funcs.py`
- `README.md`
  folder-level run instructions and outputs
- `run_all.py`
  runs the full PCA figure set
- `plot_pca_population.py`
  population PCA scatter and scree plots
- `plot_pca_population_averages.py`
  trial-averaged PCA plots
- `plot_umap_population.py`
  UMAP population plots

`ridge/`:
- `ridge_analysis.py`
  ridge-specific target builders and wrappers around the shared 5-fold standardized CV pipeline in `funcs.py`
- `README.md`
  folder-level run instructions and outputs
- `run_all.py`
  runs the full ridge figure set
- `plot_ridge_per_session.py`
  per-session ridge `R^2` distributions from repeated within-session neuron subsampling and `5-fold` CV
- `plot_ridge_population.py`
  population predicted-vs-actual ridge plots plus population `R^2` summaries

`methods/`:
- `README.md`
  folder-level run instructions and outputs
- `run_all.py`
  runs the full methods figure set
- `plot_data_info.py`
  combined speech and non-speech neuron-count summaries
- `plot_single_mouse_psth.py`
  raster-plus-PSTH example figures for each thesis sound type

`discriminability/`:
- `discriminability_analysis.py`
  discriminability-specific population analysis, method definitions, pairwise analysis, plotting, and stats helpers
- `README.md`
  folder-level run instructions and outputs
- `plot_pearson.py`
- `plot_linear_svm.py`
- `plot_lda.py`
- `run_all.py`
  runs the full discriminability pipeline

Project documentation:
- `THESIS_METHODS_OUTLINE.md`
  implementation-grounded methods outline for thesis drafting

## Typical Run Order

1. Run `build_firing_rate_arrays.py`.
2. Use the canonical folder-level runners:
   - `python methods/run_all.py`
   - `python pca/run_all.py`
   - `python ridge/run_all.py`
   - `python discriminability/run_all.py`
3. Use the individual plotting scripts when you only want one figure family.

## Figure Conventions

- Paper-style figures use serif fonts and consistent region naming across methods.
- Summary distributions are organized as brain-region comparisons on a shared axis, with one panel per spike window and one figure per sound type.
- Statistical annotations use Bonferroni-corrected pairwise tests.
- Ridge summary boxplots use Wilcoxon when matched samples are available and otherwise MWU, with Bonferroni correction.
- Linear-SVM discriminability also writes per-condition `C` tuning CSV output and hyperparameter-tuning plots.
- Natural-sound discriminability also includes within-category vs between-category summary boxplots.

## Environment Notes

The code expects local access to `jaratoolbox` plus the HDF databases referenced in `params.py`. Saved `.npz` data outputs are written to the `dbSavePath` configured there, not into this Git repo.
