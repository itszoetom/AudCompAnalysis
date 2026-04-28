# Methods Figures

Descriptive figures used in the Methods section of the thesis (Figure 2, Supplemental Figure S1).

## Run

```bash
python methods/run_all.py
```

## Scripts

### `plot_data_info.py` → Supplemental Figure S1
Histogram of neurons recorded per session for each subregion, for both the non-speech and speech datasets. Shows that neuron counts are sufficient for the fixed-count subsampling used in all population analyses.

### `plot_single_mouse_psth.py` → Figure 2
Example single-unit raster plots and peri-stimulus time histograms (PSTHs) for one representative neuron in each stimulus category. The combined 2×2 figure (`combined_psth_figure2`) shows PT and AM (top row), natural sounds and speech (bottom row).

- Trials are color-coded by stimulus identity using a viridis colormap.
- PSTHs use 10 ms bins with 5-bin (50 ms) boxcar smoothing.
- Shaded regions mark the three analysis windows: onset (yellow), sustained (teal), offset (orange).
- The best example neuron per category is selected by a scoring function that rewards strong evoked responses and stimulus-selective tuning.
- Saves both `.png` and `.svg` for each figure.

## Output

Figures are written to `figSavePath/methods/psth/`.

## Notes

- Requires local access to `jaratoolbox` and the HDF databases referenced in `shared/params.py`.
- PSTH figures are descriptive only; no statistical comparisons are applied.
- Speech examples use a secondary candidate cell (rank 1) to avoid hyper-reactive edge cases.
