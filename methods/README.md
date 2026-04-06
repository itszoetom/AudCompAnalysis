# Methods Figures

Descriptive methods figures for dataset summaries and example single-cell responses.

## Canonical entry points
- `python methods/run_all.py`
  runs the full methods figure set

## Individual figure scripts
- `plot_data_info.py`
  combined non-speech and speech session, mouse, and total-neuron summary figures with per-session cutoff lines
- `plot_single_mouse_psth.py`
  canonical example single-cell figure script that now writes one clean raster-plus-PSTH figure per thesis sound type

## Outputs
- figures are written to `figSavePath/methods/`
- these are descriptive figures and do not add inferential statistics
- long-running scripts include `tqdm` progress bars and stage prints where useful

## Notes
- the canonical raster-plus-PSTH script writes separate speech, pure-tone, AM, and natural-sound example figures
- these scripts require local `jaratoolbox` and database access
