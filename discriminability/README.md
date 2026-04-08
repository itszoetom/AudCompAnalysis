# Discriminability

Population-based pairwise stimulus discriminability analyses.

## Run

- `python discriminability/run_all.py`

## Core Design

- equal-neuron population datasets:
  - speech: `99`
  - non-speech: `278`
- inputs are z-scored before analysis
- linear SVM and LDA use leave-one-out CV
- linear SVM tunes `C` over `np.logspace(-5, 4, 20)`
- boxplot comparisons use Mann-Whitney U with Bonferroni correction

## Main Scripts

- `plot_pearson.py`
  Pearson heatmaps and boxplots
- `plot_linear_svm.py`
  linear-SVM heatmaps, boxplots, tuning plots, and one example boundary figure
- `plot_lda.py`
  LDA figures
- `discriminability_analysis.py`
  shared analysis and plotting helpers

## Output

- thesis-facing Pearson and linear-SVM figures are written into sound-specific folders
- LDA figures are written separately under an omitted folder
- top-level CSVs store pairwise results and SVM tuning results

## Notes

- natural sounds also include within- vs. between-category boxplots
- natural-sound summary boxplots compare brain regions within the `Within` and `Between` categories separately
- thesis-facing summary figures use viridis-based palettes
