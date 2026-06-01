# Functional Specialization Across Mouse Auditory Cortical Subregions

Code and figures for the undergraduate thesis *Functional Specialization Across Mouse Auditory Cortical Subregions* (Zoe Tomlinson, University of Oregon, 2026).

📄 **Read the thesis:** [`Thesis-Final-Tomlinson.pdf`](Thesis-Final-Tomlinson.pdf)
🖼️ **Browse the figures:** [`figures/`](figures/)
💻 **Browse the analysis code:** [`scripts/`](scripts/)

---

## Summary

Neuropixels recordings from mouse auditory cortex are analyzed across four stimulus categories (pure tones, AM white noise, natural sounds, speech syllables) to ask how different auditory subregions specialize for processing sounds of varying complexity. Each session simultaneously targets up to four subregions — primary (AudP), dorsal (AudD), ventral (AudV), and posterior (AudPo) — with placement confirmed histologically via the Allen Mouse Brain Atlas.

Two complementary analyses are applied to every subregion × sound × spike-window combination:

- **Encoding (PCA + participation ratio)** — what does the population geometry look like, and how distributed is the representation?
- **Decoding (ridge regression, pairwise linear SVM)** — how well can stimulus identity be read out from population activity?

---

## Datasets

| Dataset | Subjects | Stimuli |
|---|---|---|
| Simple sounds | feat014–feat019 | 16 pure-tone frequencies (2–40 kHz, log-spaced); 11 AM rates (4–128 Hz, log-spaced) |
| Complex sounds | feat014–feat019 (natural), feat004–feat010 (speech) | 20 natural sounds (5 categories × 4 exemplars); 12 speech syllables (VOT × FT grid) |

**Natural sound categories:** frogs, crickets, streamside, bubbling, bees (four distinct exemplars each).

**Speech syllables:** defined along two acoustic dimensions — voice onset time (VOT, 0/33/67/100%) and formant transition (FT, 0/33/67/100%) — forming a 12-point grid with corner syllables /ba/, /da/, /pa/, /ta/. AudD is excluded from speech analyses due to insufficient neuron counts for reliable subsampling.

Raw spike data live in HDF databases managed by the Jaramillo Lab. All analyses consume pre-built `.npz` firing-rate arrays produced by `scripts/shared/build_firing_rate_arrays.py`.

---

## Repository Layout

```
Thesis-Final-Tomlinson.pdf        — the full thesis document

scripts/
  shared/
    params.py                     — paths, stimulus metadata, spike windows, neuron counts
    funcs.py                      — data loading, dataset building, subsampling, CV pipelines
    plot_stats.py                 — Mann-Whitney U / Bonferroni boxplot annotation helpers
    build_firing_rate_arrays.py   — converts HDF databases to shared .npz arrays (run once)
  methods/                        — Figure 2, S1: single-cell raster + PSTH examples, dataset summaries
  pca/                            — Figures 3–6, S2–S5: population PCA, UMAP, participation ratio
  ridge/                          — Figures 7–8, S6–S8: per-session ridge regression decoding
  discriminability/               — Figures 9–11, S9–S13: pairwise linear SVM (+ Pearson, LDA)

figures/                          — rendered thesis figures (PNG / SVG / PDF)
  drafts/                         — schematic panels and exploratory drafts (e.g. Figure 1C waveforms)
  decoding/                       — top-level discriminability summary figures
  figures/                        — figures grouped by analysis stage:
                                    encoding:pca/  decoding/linearSVM/  ridge/  etc.
```

Each subdirectory of `scripts/` has its own README describing the figures it produces and the statistical pipeline behind them.

---

## Methods at a glance

### Firing-rate windows

Spike counts are converted to firing rates (spikes/sec) over three non-overlapping windows per stimulus type:

| Sound | Onset | Sustained | Offset |
|---|---|---|---|
| Pure tones | 0–50 ms | 50–100 ms | 100–150 ms |
| AM white noise | 0–200 ms | 200–500 ms | 500–700 ms |
| Speech syllables | 0–200 ms | 200–500 ms | 500–700 ms |
| Natural sounds | 0–500 ms | 1000–4000 ms | 4000–4500 ms |

All three windows are analyzed for every sound × subregion combination.

### Encoding: PCA + Participation Ratio

Neuron-by-trial matrices from all sessions and animals are concatenated along the neuron axis, then subsampled to a fixed count per subregion (**278 neurons** for non-speech, **99 neurons** for speech) so region comparisons are not confounded by population size. PCA is applied separately for each sound × subregion × window with per-neuron mean centering.

Effective dimensionality is quantified by the **participation ratio** (Recanatesi et al., 2022):

```
PR = (Σλ)² / Σλ²
```

where λ are the PCA explained-variance eigenvalues. Higher PR = variance more distributed across components. PR is annotated in each PCA panel; full scree plots are in Supplemental Figures S2–S5.

PR values are *not* compared statistically across subregions, since pooling across sessions before PCA leaves no session-level replicates.

### Decoding: Ridge regression (PT and AM only)

Natural sounds have no inherent acoustic ordering and speech varies along two orthogonal dimensions (VOT, FT), so ridge regression is applied only to PT and AM. Per session:

1. Session included if ≥ 30 neurons in the target subregion.
2. Subsample to exactly **30 neurons**, repeat **100×** with different seeds.
3. Z-score neuron firing rates independently within each training fold.
4. **5-fold shuffled CV.** Within each training fold, `RidgeCV` chooses α from **200 log-spaced values, 10⁻¹⁰ to 10⁵**.
5. Targets are log-transformed (matching the log-spaced stimuli).
6. Score = mean R² across folds, averaged over 100 subsamples.

### Decoding: Pairwise linear SVM (all sound types)

Population matrices concatenate all sessions and animals; neurons are subsampled to the fixed equal count (278 / 99) with a deterministic seed; firing rates are z-scored. `LinearSVC` regularization **C** is tuned over **20 log-spaced values, 10⁻⁵ to 10⁴** per sound × region × window (Supplemental Figures S9–S12). Pairwise accuracy is estimated by **shuffled leave-one-out cross-validation** (Weese et al., 2025) across all stimulus pairs.

For natural sounds, pairs are additionally split into **within-category** and **between-category** to test for categorical structure.

### Statistics

All pairwise subregion comparisons use **unpaired Mann-Whitney U tests with Bonferroni correction**, annotated with bracket-and-star notation (∗ p < 0.05, ∗∗ p < 0.01, ∗∗∗ p < 0.001).

### Reproducibility

Random subsampling and CV use fixed seeds (`seed = 42` base; deterministically derived from sound type and subregion strings where varied). Shared `.npz` arrays are built once and reused across all analysis modules.

---

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

Each `run_all.py` can also be executed individually; the subdirectory README documents what it produces.

---

## Environment

- Python ≥ 3.10
- A local copy of `jaratoolbox` (Jaramillo Lab toolbox)
- HDF databases at the paths configured in `scripts/shared/params.py`
- Output paths: figures → `params.figSavePath`, arrays → `params.dbSavePath` (neither is committed)

---

## Citation

If you use or reference this work, please cite:

> Tomlinson, Z. (2026). *Functional Specialization Across Mouse Auditory Cortical Subregions* (Undergraduate Thesis). University of Oregon.

Recordings were collected by the Jaramillo Lab (University of Oregon).
