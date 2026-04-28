# Distributed Processing of Sounds Across Mouse Auditory Cortical Subregions

Code for the thesis: *Distributed Processing of Sounds Across Mouse Auditory Cortical Subregions* (Zoe Tomlinson, University of Oregon, 2026). Publicly available at https://github.com/itszoetom/AudCompAnalysis.

Neural population recordings from mouse auditory cortex are analyzed across four stimulus categories using encoding (PCA) and decoding (ridge regression, pairwise linear SVM) analyses to characterize how different auditory cortical subregions specialize for processing sounds of varying complexity.

---

## Datasets

Two independent cohorts of mice were recorded using Neuropixels 1.0 probes during passive listening. Each session targeted up to four auditory subregions simultaneously: primary (AudP), dorsal (AudD), ventral (AudV), and posterior (AudPo). Probe placement was confirmed histologically using the Allen Mouse Brain Atlas.

| Dataset | Subjects | Stimuli |
|---|---|---|
| Simple sounds | feat014–feat019 | 16 PT frequencies (2–40 kHz, log-spaced); 11 AM rates (4–128 Hz, log-spaced) |
| Complex sounds | feat014–feat019 (natural), feat004–feat010 (speech) | 20 natural sounds (5 categories × 4 exemplars); 12 speech syllables (VOT × FT grid) |

**Natural sound categories:** frogs, crickets, streamside, bubbling, bees (four distinct exemplars each).

**Speech syllables:** defined along two acoustic dimensions — voice onset time (VOT, 0/33/67/100%) and formant transition (FT, 0/33/67/100%) — forming a 12-point grid with corner syllables /ba/, /da/, /pa/, /ta/. AudD was excluded from speech analyses due to insufficient neuron counts for reliable subsampling.

Raw spike data are stored in HDF databases managed by the Jaramillo Lab. All analyses consume pre-built `.npz` firing-rate arrays (see `shared/build_firing_rate_arrays.py`).

---

## Repository Structure

```
shared/
  params.py                   — paths, stimulus metadata, spike window definitions, neuron counts
  funcs.py                    — data loading, dataset building, subsampling, CV pipelines
  plot_stats.py               — Mann-Whitney U / Bonferroni boxplot annotation helpers
  build_firing_rate_arrays.py — converts HDF databases to shared .npz arrays (run once)

methods/                      — Figure 2: single-cell raster and PSTH examples
pca/                          — Figures 3–6, S2–S5: population PCA scatter and scree plots
ridge/                        — Figures 7–8, S6–S8: per-session ridge regression decoding
discriminability/             — Figures 9–11, S9–S13: pairwise linear SVM discriminability

figures/
  make_c_panel.py             — standalone waveform schematic for Figure 1C

settings.py                   — local jaratoolbox path configuration (not analysis code)
docs/
  Tomlinson_UOThesis.pdf      — thesis document
```

---

## 3.1 Firing Rate Calculation

Spike counts were divided by window duration to give mean firing rates in spikes/second. Three non-overlapping analysis windows are defined per stimulus type to capture distinct phases of the neural response:

| Sound | Onset | Sustained | Offset |
|---|---|---|---|
| Pure tones | 0–50 ms | 50–100 ms | 100–150 ms |
| AM white noise | 0–200 ms | 200–500 ms | 500–700 ms |
| Speech syllables | 0–200 ms | 200–500 ms | 500–700 ms |
| Natural sounds | 0–500 ms | 1000–4000 ms | 4000–4500 ms |

All three windows are analyzed for every sound type and subregion combination.

---

## 3.2 Encoding: PCA and Participation Ratio

For all encoding analyses, neuron-by-trial matrices from all sessions and animals were concatenated along the neuron axis to form a single population matrix per subregion and stimulus type. Neurons were randomly subsampled to a fixed count before analysis to ensure region comparisons are not confounded by population size:

- **Non-speech (PT, AM, natural sounds):** 278 neurons per subregion
- **Speech syllables:** 99 neurons per subregion

### Principal Component Analysis

PCA was applied separately for each combination of sound type, brain subregion, and spike window. Input matrices were **mean-centered per neuron** (each neuron's across-trial mean subtracted) before decomposition. Figures show all trials projected onto the first two principal components, with points colored by stimulus identity.

### Participation Ratio

Effective dimensionality was quantified using the **participation ratio (PR)**, computed from the PCA explained-variance spectrum (Recanatesi et al., 2022):

```
PR = (Σλ)² / Σλ²
```

where λ are the explained-variance eigenvalues from PCA. PR ranges from 1 (all variance concentrated in one component) to n (perfectly uniform spectrum across all components). Higher PR indicates more distributed representational variance across the population. PR is annotated in each PCA panel; scree plots showing the full variance spectrum are provided in Supplemental Figures S2–S5.

**Statistical note:** PR values cannot be statistically compared across subregions because data were pooled across sessions before PCA — there are no session-level replicates to test against.

---

## 3.3 Decoding

### 3.3.1 Ridge Regression (Figures 7–8, S6–S8)

Ridge regression was applied to **PT and AM only**. Natural sounds have no inherent acoustic ordering (arbitrary ranking would be meaningless), and speech stimuli vary along two independent orthogonal dimensions (VOT and FT) that cannot be collapsed into a single regression target. Pairwise discriminability (Section 3.3.2) was used for those sound types instead.

Ridge regression was applied **at the session level**, independently for each subregion and sound type. L2 regularization shrinks coefficients toward zero without zeroing any out, preserving contributions from the full population and reducing overfitting in high-dimensional correlated data.

**Pipeline:**
1. A session is included if it has ≥ 30 neurons in the target subregion.
2. Each retained session is subsampled to exactly **30 neurons**, repeated **100 times** with different random seeds to average out subsampling variance.
3. All neuron firing rates are **z-scored independently** within each training fold.
4. Each subsample is evaluated with **shuffled 5-fold cross-validation**.
5. Within each training fold, `RidgeCV` selects the best regularization strength from **200 logarithmically spaced α values from 10⁻¹⁰ to 10⁵** (`np.logspace(-10, 5, 200)`).
6. Regression targets are **log-transformed** before fitting, consistent with the logarithmic spacing of PT frequencies and AM modulation rates.
7. Performance is reported as **mean R² across the 5 test folds, averaged over 100 subsamples**.

### 3.3.2 Discriminability: Pairwise Linear SVM (Figures 9–11, S9–S13)

Discriminability quantifies how well population responses distinguish each pair of stimuli. Linear SVM classifiers were applied to population matrices concatenating neurons across all sessions and animals.

**Pipeline:**
1. Population matrices per subregion and window are built by concatenating all sessions and animals along the neuron axis.
2. Neurons are randomly subsampled to a fixed equal count (**278 for non-speech, 99 for speech**) using a deterministic seed.
3. All predictor firing rates are **z-scored** before classification.
4. The regularization hyperparameter **C is tuned over a grid of 20 logarithmically spaced values from 10⁻⁵ to 10⁴** (`np.logspace(-5, 4, 20)`), fit separately for each sound type × subregion × window (Supplemental Figures S9–S12).
5. Pairwise classification accuracy is estimated using **shuffled leave-one-out cross-validation** (Weese et al., 2025).
6. All stimulus pairs are evaluated; mean accuracy across pairs is the reported per-condition score.

For natural sounds, stimulus pairs are additionally labeled as **Within-category** (both from the same environmental category) or **Between-category** (from different categories) to assess categorical organization.

### 3.3.3 Statistical Analysis

All pairwise subregion comparisons within each sound type and spike window use **unpaired Mann-Whitney U tests with Bonferroni correction** for multiple comparisons. Significance is annotated with bracket-and-star notation (∗ p < 0.05, ∗∗ p < 0.01, ∗∗∗ p < 0.001). All neurons were subsampled to the fixed equal count before computing any summary statistic, ensuring that distributional differences reflect neural representation rather than population size.

---

## 3.5 Reproducibility

All analyses were implemented in Python. Random subsampling and cross-validation procedures use fixed or deterministic seeds (`seed = 42` base, derived deterministically from sound type and brain area strings where varied). The shared `.npz` firing-rate arrays are built once by `shared/build_firing_rate_arrays.py` and reused across all analysis modules.

---

## Run Order

```bash
python shared/build_firing_rate_arrays.py   # build .npz arrays from HDF databases (run once)
python methods/run_all.py                   # Figure 2
python pca/run_all.py                       # Figures 3–6, S2–S5
python ridge/run_all.py                     # Figures 7–8, S6–S8
python discriminability/run_all.py          # Figures 9–11, S9–S13  (~30–40 min)
```

---

## Environment

Requires Python ≥ 3.10, a local `jaratoolbox` copy (bundled under `jaratoolbox/`), and the HDF databases referenced in `shared/params.py`. Figures are written to `params.figSavePath`; `.npz` arrays go to `params.dbSavePath`. Neither path is committed to this repository.
