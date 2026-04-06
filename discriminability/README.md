# Discriminability

Population-based pairwise stimulus discriminability analyses for the thesis datasets.

## Design
- Cross-validation: leave-one-out population CV for `linearSVM` and `lda`
- Population neuron counts:
  - `speech`: `99`
  - `AM`, `PT`, `naturalSound`: `278`
- Inputs are z-scored once per population dataset before pairwise decoding
- Windows: `onset`, `sustained`, `offset`
- Speech stimuli: all `12` FT/VOT identities, with the four syllable endpoints labeled
- Speech regions: `Primary`, `Ventral`, `Posterior` only; `Dorsal` excluded because of low neuron count
- Linear SVM tunes `C` over an expanded log-spaced grid from `10^-5` to `10^4`, with extra density at the low-`C` end, separately for each sound, brain region, and spike window

## Outputs
Each method writes one top-level CSV in `figSavePath/discriminability/`:
- `<method>_pairwise_results.csv`:
  one row per sound, brain region, spike window, and stimulus pair

Linear SVM also writes:
- `linearSVM_hyperparameter_tuning.csv`:
  one row per sound, brain region, spike window, and tested `C`
- `linearSVM_<sound>_hyperparameter_tuning.png`:
  one tuning figure per sound type, with one panel per brain region and spike window

Each plotting script creates:
- heatmap grids for every sound type
- region-comparison boxplots for every sound type, one panel per spike window
- natural-sound within-vs-between category boxplots
- thesis-facing summary boxplots use viridis-based palettes for consistency across figures

Figure layout:
- thesis-facing Pearson and linear-SVM figures are written into sound-specific folders:
  - `figSavePath/discriminability/speech/`
  - `figSavePath/discriminability/AM/`
  - `figSavePath/discriminability/PT/`
  - `figSavePath/discriminability/naturalSound/`
- omitted LDA figures are written separately under:
  - `figSavePath/discriminability/omitted/lda/<sound>/`

Statistical tests:
- brain-region comparisons within each spike window: MWU
- natural within-vs-between comparisons within each brain region: MWU
- all p-values Bonferroni-corrected

## Structure
- `discriminability_analysis.py`
  shared data loading, population dataset building, pairwise analysis, generic plotting, and stats helpers
- `plot_pearson.py`
- `plot_linear_svm.py`
- `plot_lda.py`
- `run_all.py`

## Canonical entry points
- `python discriminability/run_all.py`
  runs the full discriminability pipeline

## Notes
- long-running analyses include `tqdm` progress bars and stage-level print statements
