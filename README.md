# Auditory Cortex Population Analysis

Population-level neural data analysis of auditory cortex subregion representations in mice.

## Overview
This repository contains the analysis code for an honors thesis on how auditory cortex subregions differ in their neural population representations of sound. It combines two datasets: a natural-sounds dataset with pure tones, AM white noise, and natural sounds, and a speech dataset with FT/VOT speech continua. The current analysis folders focus on population PCA/UMAP encoding, ridge-regression decoding, and methods figures for example sessions and neuron-count summaries. Preprocessed firing-rate arrays are saved as `.npz` files and then reused by the downstream plotting scripts.

## Datasets
Natural Sounds Dataset:
- Subjects: `feat014`-`feat019`
- Stimuli: pure tones (`PT`), AM white noise (`AM`), natural sounds (`naturalSound`)
- Subregions: A1, AuD, AuV, AuP
- Target subsampling: `278` neurons per subregion

Speech Dataset:
- Subjects: `feat004`-`feat010`
- Stimuli: speech syllables along FT/VOT continua
- Subregions: A1, AuV, AuP, AuP retained in saved arrays, AuD excluded from paper-style figures because of low neuron count
- Target subsampling: `99` neurons per subregion

## Repository Structure
```text
build_firing_rate_arrays.py  - loads .h5 files, preprocesses spike data, saves .npz arrays
params.py                    - shared parameters: subjects, spike windows, colors, paths
database_generation.py       - generates cell database .h5 files
pca/
    pca_analysis.py                - PCA/UMAP helper functions: loading, subsampling, PR
    plot_pca_all_mice.py           - population PCA scatter and scree plots
    plot_pca_all_mice_averages.py  - trial-averaged PCA scatter plots
    plot_pca_particRatio_dist.py   - participation ratio boxplot distributions
    plot_pca_speech.py             - speech FT/VOT separated PCA plots
    pca_umap.py                    - UMAP population scatter plots
ridge/
    ridge_analysis.py              - ridge regression helpers: loading, tuning, evaluation
    plot_ridge_boxplot.py          - population R2 boxplots across regions and windows
    ridge_population.py            - predicted-versus-actual population ridge plots
    plot_ridge_per_mouse.py        - per-mouse R2 boxplots
    plot_ridge_per_session.py      - per-session R2 boxplots
methods/
    methods_analysis.py            - shared utility functions for methods figures
    plot_data_info.py              - neuron-count histograms and session summaries
    plot_single_mouse_psth.py      - example PSTH figures with spike-window markers
    plot_single_mouse_raster.py    - example raster plots with spike-window markers
    plot_single_mouse_spikerate.py - example spike-rate plots across stimuli
discriminability/
    README.md                      - placeholder for future discriminability pass
data/                              - .npz firing-rate arrays (not tracked in git)
archive/                           - legacy scripts (not maintained)
```

## Setup
Use Python 3 with `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `umap-learn`, and `jaratoolbox`. Dataset and figure paths are configured in `params.py`. The expected run order is:

1. Run `build_firing_rate_arrays.py` to build the `.npz` arrays.
2. Run any PCA, ridge, or methods plotting script.
3. Save outputs to the figure path defined in `params.py`.

## Figure Conventions
Paper-style population figures use Times New Roman, viridis for continuous stimulus coloring, and `params.color_palette` for region-level colors. Population PCA, UMAP, and ridge figures are organized as brain-region rows by spike-window columns, with one figure per sound type. Population neuron subsampling is fixed at `278` neurons for the natural-sounds dataset and `99` neurons for speech. Statistical annotations use Mann-Whitney U tests with Bonferroni correction where applicable.
