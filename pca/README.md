# Encoding: PCA and UMAP Figures

Population encoding analyses using Principal Component Analysis (PCA) and UMAP, corresponding to Figures 3–6 and Supplemental Figures S2–S5 of the thesis.

## Run

```bash
python pca/run_all.py
```

## Scripts

### `plot_pca_population.py` → Figures 3–6
Two-dimensional PCA projections of population firing rates, one figure per sound type. Each figure is a grid of subplots (rows = subregions, columns = spike windows). Each point is one trial, colored by stimulus identity. Participation ratio (PR) is annotated in each panel.

### `plot_pca_population_avgs.py`
Same layout as above but with trials averaged within each stimulus identity before PCA. Used for exploratory visualization of population geometry.

### `plot_pca_speech.py`
Speech-only PCA projections colored separately by FT and VOT dimensions to show how each acoustic dimension is encoded in the population geometry.

### `plot_umap_population.py`
UMAP projections of the same equal-neuron population datasets, using the same grid layout as the PCA figures. UMAP inputs are not additionally scaled beyond the subsampling step.

### `pca_analysis.py`
Shared helpers: `compute_pca_summary`, `calculate_participation_ratio`, `collect_sound_results`, `plot_scree`, `add_stimulus_colorbar`.

## Key Analysis Details

**Mean-centering:** Each neuron's mean firing rate across all trials is subtracted before PCA, so that the principal components capture variance in stimulus-driven responses rather than overall firing rate differences.

**Participation Ratio (PR):** Quantifies effective dimensionality from the PCA variance spectrum:

```
PR = (Σλ)² / Σλ²
```

where λ are the explained-variance eigenvalues. Higher PR indicates more distributed representational variance. PR is bounded between 1 (all variance in one component) and n_components (perfectly uniform spectrum).

**Neuron counts:** Neurons are subsampled to a fixed count per subregion before PCA to avoid confounds from differing population sizes:
- Non-speech (PT, AM, natural sounds): 278 neurons per subregion
- Speech: 99 neurons per subregion (AudD excluded due to insufficient count)

**Statistical note:** PR values cannot be statistically compared across subregions because data were pooled across sessions before PCA — there are no session-level replicates.

## Output

Figures are written to `figSavePath/encoding/`.
