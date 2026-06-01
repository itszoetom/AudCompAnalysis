# Encoding: PCA and UMAP

Population encoding analyses using PCA and UMAP (Figures 3–6, Supplemental S2–S5).

## Run

```bash
python -m scripts.pca.run_all
```

## Scripts

- `plot_pca_population.py` → **Figures 3–6.** 2D PCA projections per sound type, one panel grid per figure (rows = subregions, columns = spike windows). Each point is one trial, colored by stimulus identity; participation ratio (PR) is annotated per panel.
- `plot_pca_population_avgs.py` — same layout but with trials averaged within each stimulus identity before PCA (exploratory).
- `plot_pca_speech.py` — speech-only PCA projections colored separately by FT and VOT.
- `plot_umap_population.py` — UMAP projections in the same grid layout as PCA.
- `pca_analysis.py` — shared helpers (PCA fit, participation ratio, scree, colorbar utilities).

## Pipeline

Neurons from all sessions and animals are concatenated along the neuron axis, then subsampled to a fixed count per subregion (278 for non-speech, 99 for speech). Each neuron is mean-centered across trials before PCA.

Effective dimensionality is reported as the **participation ratio**:

```
PR = (Σλ)² / Σλ²
```

where λ are the explained-variance eigenvalues. Higher PR indicates more distributed representational variance. PR values are not compared statistically across subregions (no session-level replicates after pooling).

## Output

Figures are written to `figSavePath/encoding/`.
