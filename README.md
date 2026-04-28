# Distributed Processing of Sounds Across Mouse Auditory Cortical Subregions

Code for the thesis: *Distributed Processing of Sounds Across Mouse Auditory Cortical Subregions* (Zoe Tomlinson, University of Oregon, 2026). Publicly available at https://github.com/itszoetom/AudCompAnalysis.

Neural population recordings from mouse auditory cortex (AudP, AudD, AudV, AudPo) are analyzed across four stimulus categories — pure tones (PT), amplitude-modulated white noise (AM), environmental natural sounds, and synthetic speech syllables — using encoding (PCA) and decoding (ridge regression, pairwise linear SVM) analyses.

## Datasets

| Dataset | Subjects | Stimuli | Regions |
|---|---|---|---|
| Simple sounds | feat014–feat019 | 16 PT frequencies (2–40 kHz), 11 AM rates (4–128 Hz) | AudP, AudD, AudV, AudPo |
| Complex sounds | feat014–feat019 (natural), feat004–feat010 (speech) | 20 natural sounds (5 categories × 4 exemplars), 12 speech syllables (VOT × FT grid) | AudP, AudD, AudV, AudPo (AudD excluded from speech) |

Raw spike data are stored in HDF databases managed by the Jaramillo Lab. All analysis uses pre-built `.npz` firing-rate arrays (see `shared/build_firing_rate_arrays.py`).

## Repository Structure

```
shared/
  params.py                  — paths, stimulus metadata, spike window definitions, neuron counts
  funcs.py                   — data loading, dataset building, subsampling, CV pipelines
  plot_stats.py              — Mann-Whitney U / Bonferroni boxplot annotation helpers
  build_firing_rate_arrays.py — converts HDF databases to shared .npz arrays

methods/                     — Figure 2: single-cell raster and PSTH examples
pca/                         — Figures 3–6: population PCA scatter plots and scree plots
ridge/                       — Figures 7–8, S6–S8: per-session ridge regression decoding
discriminability/            — Figures 9–11, S9–S13: pairwise linear SVM discriminability

figures/
  make_c_panel.py            — standalone waveform schematic for Figure 1C

settings.py                  — local jaratoolbox path configuration (not analysis code)
```

## Spike Windows

Firing rates are computed over three non-overlapping windows per stimulus:

| Sound | Onset | Sustained | Offset |
|---|---|---|---|
| Pure tones | 0–50 ms | 50–100 ms | 100–150 ms |
| AM / Speech | 0–200 ms | 200–500 ms | 500–700 ms |
| Natural sounds | 0–500 ms | 1000–4000 ms | 4000–4500 ms |

## Population Neuron Counts

Neurons are randomly subsampled to a fixed count before all population analyses to ensure region comparisons are not confounded by sample size:
- Non-speech (PT, AM, natural sounds): **278 neurons** per subregion
- Speech: **99 neurons** per subregion (AudD excluded due to insufficient count)

## Run Order

```bash
python shared/build_firing_rate_arrays.py   # build .npz arrays from HDF databases (run once)
python methods/run_all.py                   # Figure 2
python pca/run_all.py                       # Figures 3–6, S2–S5
python ridge/run_all.py                     # Figures 7–8, S6–S8
python discriminability/run_all.py          # Figures 9–11, S9–S13
```

## Statistical Analysis

All pairwise subregion comparisons within each sound type and spike window use **unpaired Mann-Whitney U tests with Bonferroni correction** for multiple comparisons. Significance is annotated with bracket-and-star notation (∗ p < 0.05, ∗∗ p < 0.01, ∗∗∗ p < 0.001).

## Environment

Requires Python ≥ 3.10, a local `jaratoolbox` copy (bundled under `jaratoolbox/`), and the HDF databases referenced in `shared/params.py`. Figures are written to the path configured in `params.figSavePath`; `.npz` arrays go to `params.dbSavePath`. Neither is committed to this repository.
