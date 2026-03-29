# Methods Figures

Descriptive methods figures for dataset summaries and example single-cell responses.

## Canonical entry points
- `python methods/run_all.py`
  runs the full methods figure set

## Individual figure scripts
- `plot_data_info.py`
  combined non-speech and speech session, mouse, and total-neuron summary figures with per-session cutoff lines
- `plot_single_mouse_psth.py`
  canonical example single-cell figure script with raster and PSTH panels side by side across speech and non-speech datasets using a shared viridis stimulus colormap
- `plot_single_mouse_spikerate.py`
  one example single-neuron firing-rate figure with raster, PSTH, and onset/sustained/offset rate bars

## Outputs
- figures are written to `figSavePath/methods/`
- these are descriptive figures and do not add inferential statistics
- long-running scripts include `tqdm` progress bars and stage prints where useful

## Notes
- the canonical combined raster-plus-PSTH script includes natural sounds, AM, pure tones, FT, and VOT example panels
- these scripts require local `jaratoolbox` and database access
