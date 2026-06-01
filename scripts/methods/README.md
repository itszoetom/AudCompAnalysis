# Methods Figures

Descriptive figures for the Methods section (Figure 2, Supplemental Figure S1).

## Run

```bash
python -m scripts.methods.run_all
```

## Scripts

- `plot_data_info.py` → **Supplemental Figure S1.** Histogram of neurons per session per subregion, for both the non-speech and speech datasets.
- `plot_single_mouse_psth.py` → **Figure 2.** Example single-unit rasters and PSTHs for one representative neuron in each stimulus category. Trials colored by stimulus identity; PSTHs use 10 ms bins with 5-bin boxcar smoothing; shaded bands mark the onset (yellow), sustained (teal), and offset (orange) analysis windows.

## Output

Figures are written to `figSavePath/methods/psth/`. Requires `jaratoolbox` and the HDF databases referenced in `shared/params.py`.
