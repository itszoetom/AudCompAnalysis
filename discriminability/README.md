# Discriminability

Session-based pairwise stimulus discriminability analyses for the thesis datasets.

## Design
- Cross-validation: `5-fold` within-session stratified CV for `linearSVM` and `lda`
- Session sampling: up to `5` valid sessions per sound and brain region
- Neuron subsampling per session:
  - `speech`: `10`
  - `AM`, `PT`, `naturalSound`: `30`
- Windows: `onset`, `sustained`, `offset`
- Speech stimuli: all `12` FT/VOT identities, with the four syllable endpoints labeled
- Speech regions: `Primary`, `Ventral`, `Posterior` only; `Dorsal` excluded because of low neuron count

## Outputs
Each method folder writes:
- `pairwise_results.csv`:
  one row per sound, brain region, spike window, mouse, session, subsampling iteration, and stimulus pair

Each plotting script creates:
- heatmap grids for every sound type
- region-comparison boxplots for every sound type, one panel per spike window
- natural-sound within-vs-between category boxplots

Statistical tests:
- brain-region comparisons within each spike window: Wilcoxon when matched samples are available, otherwise MWU
- natural within-vs-between comparisons within each brain region: Wilcoxon when matched samples are available, otherwise MWU
- all p-values Bonferroni-corrected

## Structure
- `discriminability_analysis.py`
  shared data loading, session subsampling, plotting, and stats helpers
- `pearson/analysis.py`
- `pearson/plot.py`
- `linearSVM/analysis.py`
- `linearSVM/plot.py`
- `lda/analysis.py`
- `lda/plot.py`
- `run_all_analyses.py`
- `run_all.py`

## Canonical entry points
- `python discriminability/run_all.py`
  runs the full discriminability pipeline
- `python discriminability/run_all_analyses.py`
  runs Pearson, linear SVM, and LDA analyses

## Notes
- long-running analyses include `tqdm` progress bars and stage-level print statements
