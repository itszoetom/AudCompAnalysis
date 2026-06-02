# Functional Specialization Across Mouse Auditory Cortical Subregions

Code and figures accompanying the undergraduate thesis *Functional Specialization Across Mouse Auditory Cortical Subregions* (Zoe Tomlinson, University of Oregon, 2026).

- [`Thesis-Final-Tomlinson.pdf`](Thesis-Final-Tomlinson.pdf): the thesis.
- [`figures/`](figures/): rendered figures.
- [`scripts/`](scripts/): analysis code.

## Overview

Neuropixels recordings from mouse auditory cortex are analyzed across four stimulus categories (pure tones, AM white noise, natural sounds, and speech syllables) to characterize how different auditory subregions specialize for processing sounds of varying complexity. Each session simultaneously targets up to four subregions (primary AudP, dorsal AudD, ventral AudV, posterior AudPo), with placement confirmed histologically against the Allen Mouse Brain Atlas.

Two complementary analyses are applied to every subregion × sound × spike-window combination:

- **Encoding.** PCA and the participation ratio characterize the geometry and effective dimensionality of the population response.
- **Decoding.** Ridge regression (PT, AM) and pairwise linear SVM (all sounds) quantify how reliably stimulus identity can be read out from population activity.

Statistical comparisons between subregions use unpaired Mann-Whitney U tests with Bonferroni correction. Full method details are in the thesis PDF; per-module READMEs in `scripts/` summarize each pipeline.

## Datasets

| Dataset | Subjects | Stimuli |
|---|---|---|
| Simple sounds | feat014–feat019 | 16 pure-tone frequencies (2–40 kHz, log-spaced); 11 AM rates (4–128 Hz, log-spaced) |
| Complex sounds | feat014–feat019 (natural), feat004–feat010 (speech) | 20 natural sounds (5 categories × 4 exemplars); 12 speech syllables on a VOT × FT grid |

Neurons are subsampled to a fixed equal count per subregion before all population analyses (278 for non-speech, 99 for speech; AudD is excluded from the speech dataset for insufficient neuron count). Raw spike data live in HDF databases managed by the Jaramillo Lab; analyses consume pre-built `.npz` firing-rate arrays produced by `scripts/shared/build_firing_rate_arrays.py`.

## Repository Layout

```
Thesis-Final-Tomlinson.pdf   the thesis
scripts/                     analysis code (one subdirectory per analysis; each has its own README)
  shared/                    paths, dataset builders, subsampling, CV helpers, stats utilities
  methods/                   Figure 2, S1
  pca/                       Figures 3–6, S2–S5
  ridge/                     Figures 7–8, S6–S8
  discriminability/          Figures 9–11, S9–S13
figures/                     rendered figures, grouped by analysis stage (see figures/README.md)
```

## Reproducing the figures

```bash
# 1. Build shared firing-rate arrays (run once; needs raw HDF databases + jaratoolbox).
python -m scripts.shared.build_firing_rate_arrays

# 2. Generate figures by analysis stage.
python -m scripts.methods.run_all           # Figure 2, S1
python -m scripts.pca.run_all               # Figures 3–6, S2–S5
python -m scripts.ridge.run_all             # Figures 7–8, S6–S8
python -m scripts.discriminability.run_all  # Figures 9–11, S9–S13   (~30–40 min)
```

## Environment

- Python ≥ 3.10
- A local copy of `jaratoolbox` (Jaramillo Lab toolbox)
- HDF databases at the paths configured in `scripts/shared/params.py`
- Output paths: figures go to `params.figSavePath`, arrays go to `params.dbSavePath` (neither is committed)

Recordings were collected by the Jaramillo Lab (University of Oregon).
