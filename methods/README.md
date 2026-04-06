# Methods Figures

Descriptive figures for dataset summaries and example single-cell responses.

## Run

- `python methods/run_all.py`

## Main Scripts

- `plot_data_info.py`
  session, mouse, and neuron-count summary figures
- `plot_single_mouse_psth.py`
  example raster-plus-PSTH figures for speech, PT, AM, and natural sounds

## Output

- figures are written to `figSavePath/methods/`
- these figures are descriptive and do not run inferential statistics

## Notes

- the PSTH script writes separate figures for each thesis sound type
- local `jaratoolbox` and database access are required
