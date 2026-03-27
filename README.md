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
- `params.py`
  shared paths, subject lists, stimulus metadata, spike windows, and plotting colors
- `plot_stats.py`
  shared Bonferroni-corrected statistical annotation helpers for boxplot-style figures

`pca/`:
- `pca_analysis.py`
  shared PCA loading, averaging, labeling, and subsampling helpers
- `README.md`
  folder-level run instructions and outputs
- `run_all.py`
  runs the full PCA figure set
- `plot_pca_all_mice.py`
  population PCA scatter and scree plots
- `plot_pca_all_mice_averages.py`
  trial-averaged PCA plots
- `plot_pca_particRatio_dist.py`
  participation-ratio distributions by brain region and spike window
- `pca_umap.py`
  UMAP population plots

`ridge/`:
- `ridge_analysis.py`
  shared ridge-regression helpers and repeated evaluation functions
- `README.md`
  folder-level run instructions and outputs
- `run_all.py`
  runs the full ridge figure set
- `plot_ridge_boxplot.py`
  population ridge `R^2` distributions by brain region and spike window
- `plot_ridge_per_mouse.py`
  per-mouse ridge `R^2` distributions in the same region-comparison format
- `plot_ridge_per_session.py`
  per-session ridge `R^2` distributions in the same region-comparison format
- `ridge_population.py`
  predicted-vs-actual ridge plots

`methods/`:
- `methods_analysis.py`
  shared array-loading and figure helpers for methods figures
- `README.md`
  folder-level run instructions and outputs
- `run_all.py`
  runs the full methods figure set
- `plot_data_info.py`
  combined speech and non-speech neuron-count summaries
- `plot_single_mouse_psth.py`
  combined raster-plus-PSTH example figures
- `plot_single_mouse_spikerate.py`
  example single-neuron firing-rate figure

`discriminability/`:
- `discriminability_analysis.py`
  shared session-based discriminability pipeline
- `README.md`
  folder-level run instructions and outputs
- `pearson/analysis.py`, `pearson/plot.py`
- `linearSVM/analysis.py`, `linearSVM/plot.py`
- `lda/analysis.py`, `lda/plot.py`
- `run_all_analyses.py`
  runs all discriminability analyses
- `run_all.py`
  runs the full discriminability pipeline

Legacy exploratory discriminability notebooks and scripts are still present in the method subfolders, but the current pipeline is the `analysis.py` and `plot.py` entry points plus the top-level runner scripts.

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

- Paper-style figures use serif fonts and `params.color_palette` for region colors.
- Summary distributions are organized as brain-region comparisons on a shared axis, with one panel per spike window and one figure per sound type.
- Statistical annotations use Bonferroni-corrected pairwise tests.
- Ridge summary boxplots use MWU plus Bonferroni correction.
- Natural-sound discriminability also includes within-category vs between-category summary boxplots.

## Environment Notes

The code expects local access to `jaratoolbox` plus the HDF databases referenced in `params.py`. Saved `.npz` data outputs are written to the `dbSavePath` configured there, not into this Git repo.
