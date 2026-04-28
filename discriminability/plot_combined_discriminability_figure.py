"""Compound discriminability figures — one per sound category.

Natural sounds figure
---------------------
  A (heatmaps: sustained+offset × 3 regions)  |  B (region boxplots: sustained+offset)
                                               |  C (within-vs-between: sustained+offset)

Speech figure
-------------
  D (heatmaps: onset+sustained × 3 regions)  |  E (region boxplots: onset+sustained, centred)

Simple sounds figure
--------------------
  A (PT heatmaps: onset+sustained × 4)   |  B (PT boxplots: all 3 windows)
  C (AM heatmaps: onset+sustained × 4)   |  D (AM boxplots: all 3 windows)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import funcs, params  # noqa: E402
from discriminability.discriminability_analysis import (  # noqa: E402
    draw_heatmap_grid,
    draw_region_boxplot_panels,
    draw_within_between_panels,
    get_figure_dir,
    get_tuning_path,
    load_method_results,
    plot_heatmaps,
    plot_linear_svm_example,
    plot_natural_within_between_boxplots,
    plot_region_boxplots,
    plot_svm_hyperparameter_tuning,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

METHOD_KEY = "linearSVM"
METHOD_LABEL = "Linear SVM Accuracy"
VALUE_COL = "Accuracy"
YLABEL = "Linear SVM Accuracy"

# Natural sounds: sustained + offset (drop onset)
NAT_HEATMAP_WINDOWS = ["sustained", "offset"]
NAT_BOX_WINDOWS = ["sustained", "offset"]

# Speech: onset + sustained (drop offset)
SPE_HEATMAP_WINDOWS = ["onset", "sustained"]
SPE_BOX_WINDOWS = ["onset", "sustained"]

# Natural-sound regions: Primary, Dorsal, Posterior (no Ventral)
NAT_HEATMAP_REGIONS = [
    "Primary auditory area",
    "Dorsal auditory area",
    "Posterior auditory area",
]

# Speech regions: Primary, Ventral, Posterior (no Dorsal)
SPE_REGIONS = [
    "Primary auditory area",
    "Ventral auditory area",
    "Posterior auditory area",
]

# Simple sounds (PT + AM): all 4 brain regions
# Heatmaps: onset + sustained;  Boxplots: all three windows
SIMPLE_HEATMAP_WINDOWS = ["onset", "sustained"]
SIMPLE_BOX_WINDOWS = ["onset", "sustained", "offset"]
SIMPLE_REGIONS = [
    "Primary auditory area",
    "Dorsal auditory area",
    "Ventral auditory area",
    "Posterior auditory area",
]

CMAP = "viridis"
VMIN = 0.0
VMAX = 1.0

# Font sizes — hierarchy: titles (largest) > region/window labels > tick labels
FS_PANEL_LETTER = 70   # A / B / C / D / E — stamped above group titles
FS_HM_TITLE = 51       # heatmap group title
FS_SECTION_TITLE = 47  # boxplot section title
FS_HM_COL_HEADER = 43  # brain-region column header
FS_HM_ROW_LABEL = 41   # window-name row label
FS_HM_TICK = 42        # stimulus tick labels (+5 extra, more space now)
FS_CBAR = 39           # colorbar label
FS_CBAR_TICK = 40      # colorbar tick labels (+5 extra)
FS_BOX_TITLE = 45      # individual window title on boxplots
FS_BOX_LABEL = 41      # y-axis label on boxplots
FS_BOX_TICK = 39       # axis tick labels on boxplots


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _add_panel_letter(
    fig: plt.Figure,
    ax: plt.Axes,
    letter: str,
    pad: float = 0.028,
    x_offset: float = -0.025,
    fontsize: int | None = None,
) -> None:
    """Stamp a bold panel letter above the group title for *ax*.

    Uses figure coordinates so the letter always sits higher than the
    section title (which is placed at y_top + 0.020).  Call this
    **after** fig.canvas.draw() so ax.get_position() is accurate.
    """
    bb = ax.get_position()
    fig.text(
        bb.x0 + x_offset, bb.y1 + pad,
        letter,
        ha="left", va="bottom",
        fontsize=fontsize if fontsize is not None else FS_PANEL_LETTER,
        fontweight="bold",
        clip_on=False,
    )


def _add_group_title(
    fig: plt.Figure,
    axes: np.ndarray | list[plt.Axes],
    title: str,
    fontsize: int,
    pad: float = 0.010,
) -> None:
    """Place a bold title just above a group of axes using figure coordinates."""
    ax_flat = np.asarray(axes).ravel().tolist()
    x_vals: list[float] = []
    y_vals: list[float] = []
    for ax in ax_flat:
        bb = ax.get_position()
        x_vals.extend([bb.x0, bb.x1])
        y_vals.append(bb.y1)
    x_center = (min(x_vals) + max(x_vals)) / 2
    y_top = max(y_vals)
    fig.text(
        x_center, y_top + pad,
        title,
        ha="center", va="bottom",
        fontsize=fontsize, fontweight="bold",
    )


# --------------------------------------------------------------------------- #
# Complex sounds figures — natural sounds and speech as separate figures
# --------------------------------------------------------------------------- #

def make_natural_sounds_figure(results_df: pd.DataFrame) -> None:
    """Build and save the natural sounds discriminability figure.

    Layout
    ------
      A (heatmaps: sustained+offset × 3 regions)  |  B (region boxplots: sustained+offset)
                                                   |  C (within-vs-between: sustained+offset)
    """
    funcs.apply_figure_style()

    # Local font sizes — larger since this is now a standalone figure
    _fs_panel_letter = FS_PANEL_LETTER + 8
    _fs_hm_title = FS_HM_TITLE + 8
    _fs_section_title = FS_SECTION_TITLE + 8
    _fs_hm_col_header = FS_HM_COL_HEADER + 8
    _fs_hm_row_label = FS_HM_ROW_LABEL + 8
    _fs_hm_tick = FS_HM_TICK + 8
    _fs_cbar = FS_CBAR + 8
    _fs_cbar_tick = FS_CBAR_TICK + 8
    _fs_box_title = FS_BOX_TITLE + 8
    _fs_box_label = FS_BOX_LABEL + 8
    _fs_box_tick = FS_BOX_TICK + 8

    nat_hm_regions = NAT_HEATMAP_REGIONS      # Primary, Dorsal, Posterior
    n_nat_hm_regions = len(nat_hm_regions)    # 3
    n_nat_hm_windows = len(NAT_HEATMAP_WINDOWS)  # 2 (sustained, offset)
    n_nat_box_windows = len(NAT_BOX_WINDOWS)     # 2

    # Generous cells since there's only one sound type per figure
    cell_w = 7.0   # inches per heatmap cell
    cell_h = 8.0
    boxplot_col_w = 18.0  # 2 windows — wider for better separation

    hm_w = cell_w * n_nat_hm_regions   # 8.5 × 3 = 25.5
    hm_h = cell_h * n_nat_hm_windows   # 8.0 × 2 = 16.0

    total_w = hm_w + boxplot_col_w + 4.5
    total_h = hm_h + 6.0

    fig = plt.figure(figsize=(total_w, total_h))

    outer_gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[hm_w, boxplot_col_w],
        left=0.06, right=0.99,
        top=0.88, bottom=0.08,
        wspace=0.35,
    )

    # Left: Panel A — heatmaps
    gs_A = gridspec.GridSpecFromSubplotSpec(
        n_nat_hm_windows, n_nat_hm_regions,
        subplot_spec=outer_gs[0, 0],
        hspace=0.01, wspace=0.06,
    )
    axes_A = np.array(
        [[fig.add_subplot(gs_A[r, c]) for c in range(n_nat_hm_regions)]
         for r in range(n_nat_hm_windows)]
    )

    # Right: Panel B (top) and Panel C (bottom)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0, 1],
        hspace=0.90,
    )
    gs_B = gridspec.GridSpecFromSubplotSpec(
        1, n_nat_box_windows, subplot_spec=gs_right[0], wspace=0.22,
    )
    axes_B = [fig.add_subplot(gs_B[0, i]) for i in range(n_nat_box_windows)]

    gs_C = gridspec.GridSpecFromSubplotSpec(
        1, n_nat_box_windows, subplot_spec=gs_right[1], wspace=0.22,
    )
    axes_C = [fig.add_subplot(gs_C[0, i]) for i in range(n_nat_box_windows)]

    nat_df = results_df[results_df["Sound Type"] == "naturalSound"].copy()

    # Draw A — heatmaps
    cbar_src_A = draw_heatmap_grid(
        "naturalSound", nat_df, axes_A, nat_hm_regions,
        windows=NAT_HEATMAP_WINDOWS,
        value_col=VALUE_COL, cmap=CMAP, vmin=VMIN, vmax=VMAX,
        fs_col_header=_fs_hm_col_header,
        fs_row_label=_fs_hm_row_label,
        fs_tick=_fs_hm_tick,
    )
    if cbar_src_A is not None:
        cbar_A = fig.colorbar(
            cbar_src_A, ax=axes_A,
            location="right", fraction=0.015, pad=0.02, aspect=40, shrink=0.7,
        )
        cbar_A.ax.tick_params(labelsize=_fs_cbar_tick)

    # Draw B — region boxplots (legend added manually below)
    draw_region_boxplot_panels(
        "naturalSound", nat_df, axes_B,
        windows=NAT_BOX_WINDOWS,
        value_col=VALUE_COL, ylabel=YLABEL,
        fs_title=_fs_box_title, fs_label=_fs_box_label, fs_tick=_fs_box_tick,
        show_legend=False,
    )

    # Draw C — within-vs-between (legend added manually below)
    draw_within_between_panels(
        nat_df, axes_C,
        windows=NAT_BOX_WINDOWS,
        value_col=VALUE_COL, ylabel=YLABEL,
        fs_title=_fs_box_title, fs_label=_fs_box_label, fs_tick=_fs_box_tick,
        show_legend=False,
    )

    fig.canvas.draw()

    # Horizontal region legend below axes_B
    _nat_regions = funcs.get_plot_brain_regions("naturalSound")
    _nat_pal = sns.color_palette("viridis", n_colors=len(_nat_regions))
    _nat_handles_B = [
        mpatches.Patch(facecolor=_nat_pal[i], edgecolor="black", linewidth=1.0,
                       label=params.short_names.get(r, r))
        for i, r in enumerate(_nat_regions)
    ]
    _b_x0 = min(ax.get_position().x0 for ax in axes_B)
    _b_x1 = max(ax.get_position().x1 for ax in axes_B)
    _b_y0 = min(ax.get_position().y0 for ax in axes_B)
    fig.legend(
        handles=_nat_handles_B,
        loc="upper center",
        bbox_to_anchor=((_b_x0 + _b_x1) / 2, _b_y0 - 0.02),
        ncol=len(_nat_regions),
        fontsize=_fs_box_tick,
        frameon=True,
        facecolor="white",
        edgecolor="black",
    )

    # Horizontal within-vs-between legend below axes_C
    _nat_handles_C = [
        mpatches.Patch(facecolor="#2C3E7A", edgecolor="black", linewidth=1.0, label="Within category"),
        mpatches.Patch(facecolor="#5BA85A", edgecolor="black", linewidth=1.0, label="Between category"),
    ]
    _c_x0 = min(ax.get_position().x0 for ax in axes_C)
    _c_x1 = max(ax.get_position().x1 for ax in axes_C)
    _c_y0 = min(ax.get_position().y0 for ax in axes_C)
    fig.legend(
        handles=_nat_handles_C,
        loc="upper center",
        bbox_to_anchor=((_c_x0 + _c_x1) / 2, _c_y0 - 0.02),
        ncol=2,
        fontsize=_fs_box_tick,
        frameon=True,
        facecolor="white",
        edgecolor="black",
    )

    _add_group_title(fig, axes_A, f"Pairwise {METHOD_LABEL} Heatmaps for Natural Sounds",
                     fontsize=_fs_hm_title, pad=0.030)
    _add_group_title(fig, axes_B, f"Pairwise {METHOD_LABEL} for Natural Sounds",
                     fontsize=_fs_section_title, pad=0.030)
    _add_group_title(fig, axes_C, "Within vs. Between Category",
                     fontsize=_fs_section_title, pad=0.030)

    _add_panel_letter(fig, axes_A[0, 0], "A", x_offset=-0.050, fontsize=_fs_panel_letter)
    _add_panel_letter(fig, axes_B[0], "B", fontsize=_fs_panel_letter)
    _add_panel_letter(fig, axes_C[0], "C", fontsize=_fs_panel_letter)

    output_dir = get_figure_dir()
    out_path = output_dir / "natural_sounds_discriminability_figure.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def make_speech_figure(results_df: pd.DataFrame) -> None:
    """Build and save the speech discriminability figure.

    Layout
    ------
      D (heatmaps: onset+sustained × 3 regions)  |  E (region boxplots: onset+sustained, centred)
    """
    funcs.apply_figure_style()

    # Local font sizes — larger since this is now a standalone figure
    _fs_panel_letter = FS_PANEL_LETTER + 18
    _fs_hm_title = FS_HM_TITLE + 18
    _fs_section_title = FS_SECTION_TITLE + 18
    _fs_hm_col_header = FS_HM_COL_HEADER + 18
    _fs_hm_row_label = FS_HM_ROW_LABEL + 18
    _fs_hm_tick = FS_HM_TICK + 18
    _fs_cbar = FS_CBAR + 18
    _fs_cbar_tick = FS_CBAR_TICK + 18
    _fs_box_title = FS_BOX_TITLE + 18
    _fs_box_label = FS_BOX_LABEL + 18
    _fs_box_tick = FS_BOX_TICK + 18

    spe_regions = SPE_REGIONS              # Primary, Ventral, Posterior
    n_spe_regions = len(spe_regions)       # 3
    n_spe_hm_windows = len(SPE_HEATMAP_WINDOWS)  # 2 (onset, sustained)
    n_spe_box_windows = len(SPE_BOX_WINDOWS)     # 2

    cell_w = 7.0
    cell_h = 8.0
    boxplot_col_w = 18.0

    hm_w = cell_w * n_spe_regions        # 7.0 × 3 = 21.0
    hm_h = cell_h * n_spe_hm_windows     # 8.0 × 2 = 16.0

    total_w = hm_w + boxplot_col_w + 4.5
    total_h = hm_h + 6.0

    fig = plt.figure(figsize=(total_w, total_h))

    outer_gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[hm_w, boxplot_col_w],
        left=0.06, right=0.99,
        top=0.90, bottom=0.08,
        wspace=0.35,
    )

    # Left: Panel D — heatmaps
    gs_D = gridspec.GridSpecFromSubplotSpec(
        n_spe_hm_windows, n_spe_regions,
        subplot_spec=outer_gs[0, 0],
        hspace=0, wspace=0.06,
    )
    axes_D = np.array(
        [[fig.add_subplot(gs_D[r, c]) for c in range(n_spe_regions)]
         for r in range(n_spe_hm_windows)]
    )

    # Right: Panel E — region boxplots, centred vertically so they don't
    # stretch to fill the full column height (only one boxplot row here).
    gs_E_outer = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer_gs[0, 1],
        height_ratios=[1, 2, 1],
    )
    gs_E = gridspec.GridSpecFromSubplotSpec(
        1, n_spe_box_windows, subplot_spec=gs_E_outer[1], wspace=0.22,
    )
    axes_E = [fig.add_subplot(gs_E[0, i]) for i in range(n_spe_box_windows)]

    speech_df = results_df[results_df["Sound Type"] == "speech"].copy()

    # Draw D — heatmaps
    cbar_src_D = draw_heatmap_grid(
        "speech", speech_df, axes_D, spe_regions,
        windows=SPE_HEATMAP_WINDOWS,
        value_col=VALUE_COL, cmap=CMAP, vmin=VMIN, vmax=VMAX,
        fs_col_header=_fs_hm_col_header,
        fs_row_label=_fs_hm_row_label,
        fs_tick=_fs_hm_tick,
    )
    if cbar_src_D is not None:
        cbar_D = fig.colorbar(
            cbar_src_D, ax=axes_D,
            location="right", fraction=0.015, pad=0.02, aspect=40, shrink=0.7,
        )
        cbar_D.ax.tick_params(labelsize=_fs_cbar_tick)

    # Draw E — region boxplots; legend added manually below after canvas.draw()
    draw_region_boxplot_panels(
        "speech", speech_df, axes_E,
        windows=SPE_BOX_WINDOWS,
        brain_regions_override=spe_regions,
        value_col=VALUE_COL, ylabel=YLABEL,
        fs_title=_fs_box_title, fs_label=_fs_box_label, fs_tick=_fs_box_tick,
        show_legend=False,
    )

    fig.canvas.draw()

    # Region legend — horizontal row centered below Panel E
    _e_x0 = min(ax.get_position().x0 for ax in axes_E)
    _e_x1 = max(ax.get_position().x1 for ax in axes_E)
    _e_y0 = min(ax.get_position().y0 for ax in axes_E)
    _e_pal = sns.color_palette("viridis", n_colors=len(spe_regions))
    _e_handles = [
        mpatches.Patch(facecolor=_e_pal[i], edgecolor="black", linewidth=1.0,
                       label=params.short_names.get(r, r))
        for i, r in enumerate(spe_regions)
    ]
    fig.legend(
        handles=_e_handles,
        loc="upper center",
        bbox_to_anchor=((_e_x0 + _e_x1) / 2, _e_y0 - 0.03),
        ncol=len(spe_regions),
        fontsize=_fs_box_tick,
        frameon=True,
        facecolor="white",
        edgecolor="black",
    )

    _add_group_title(fig, axes_D, f"Pairwise {METHOD_LABEL} Heatmaps for Speech",
                     fontsize=_fs_hm_title, pad=0.030)
    _add_group_title(fig, axes_E, f"Pairwise {METHOD_LABEL} for Speech",
                     fontsize=_fs_section_title, pad=0.030)

    # D anchors to heatmap top-left; E anchors to first boxplot panel
    y_row = max(ax.get_position().y1 for ax in axes_D.ravel())
    x_left = axes_D[0, 0].get_position().x0 - 0.050
    x_right = axes_E[0].get_position().x0 - 0.025
    letter_pad = 0.028
    fig.text(x_left,  y_row + letter_pad, "A",
             ha="left", va="bottom", fontsize=_fs_panel_letter,
             fontweight="bold", clip_on=False)
    fig.text(x_right, y_row + letter_pad, "B",
             ha="left", va="bottom", fontsize=_fs_panel_letter,
             fontweight="bold", clip_on=False)

    output_dir = get_figure_dir()
    out_path = output_dir / "speech_discriminability_figure.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def make_simple_sounds_figure(results_df: pd.DataFrame) -> None:
    """Build and save the simple sounds (PT + AM) discriminability figure.

    Layout
    ------
      A (PT heatmaps: onset+sustained × 4 regions)  |  B (PT boxplots: all 3 windows)
      C (AM heatmaps: onset+sustained × 4 regions)  |  D (AM boxplots: all 3 windows)

    Boxplots are centered vertically within their row (not stretched to fill)
    so they sit at the same mid-height as the heatmaps.
    """
    funcs.apply_figure_style()

    # Local font-size overrides — larger than the module-level defaults so
    # the simple-sounds figure reads well at the larger heatmap cell scale.
    _fs_panel_letter = FS_PANEL_LETTER + 18   # 88
    _fs_hm_title = FS_HM_TITLE + 18          # 69
    _fs_section_title = FS_SECTION_TITLE + 18 # 65
    _fs_hm_col_header = FS_HM_COL_HEADER + 18 # 61
    _fs_hm_row_label = FS_HM_ROW_LABEL + 18  # 59
    _fs_hm_tick = FS_HM_TICK + 18            # 60
    _fs_cbar = FS_CBAR + 18                  # 57
    _fs_cbar_tick = FS_CBAR_TICK + 18        # 58
    _fs_box_title = FS_BOX_TITLE + 18        # 63
    _fs_box_label = FS_BOX_LABEL + 18        # 59
    _fs_box_tick = FS_BOX_TICK + 18          # 57

    simple_regions = SIMPLE_REGIONS
    n_simple_regions = len(simple_regions)              # 4
    n_simple_hm_windows = len(SIMPLE_HEATMAP_WINDOWS)  # 2
    n_simple_box_windows = len(SIMPLE_BOX_WINDOWS)     # 3

    # ------------------------------------------------------------------ #
    # Figure sizing — much larger heatmap cells so tick labels are big and
    # readable.  Boxplots will be centered (not stretched) in each row.
    # ------------------------------------------------------------------ #
    cell_w = 5.5    # heatmap cell width per region
    cell_h = 9.5    # tall cells → generous vertical space per window row
    boxplot_col_w = 26.0  # 3 windows — wider to give boxplots more room

    hm_w = cell_w * n_simple_regions          # 5.5 × 4 = 22.0
    hm_h = cell_h * n_simple_hm_windows       # 9.5 × 2 = 19.0

    total_w = hm_w + boxplot_col_w + 4.5
    total_h = 2 * hm_h + 8.0

    fig = plt.figure(figsize=(total_w, total_h))

    # ------------------------------------------------------------------ #
    # Outer 1 × 2 grid (left = heatmaps, right = boxplots)
    # ------------------------------------------------------------------ #
    outer_gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        width_ratios=[hm_w, boxplot_col_w],
        left=0.06, right=0.99,
        top=0.90, bottom=0.06,
        wspace=0.35,
    )

    # Left column: PT (top) and AM (bottom) heatmaps.
    # Large hspace creates intentional white space between the two sound types.
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer_gs[0, 0],
        height_ratios=[hm_h, hm_h],
        hspace=1.0,
    )

    # Right column: PT (top) and AM (bottom) boxplot slots — equal height.
    # Matches the hspace of the left column so boxplots align with heatmaps.
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer_gs[0, 1],
        hspace=1.0,
    )

    # ------------------------------------------------------------------ #
    # Panel A — PT heatmaps (n_simple_hm_windows × n_simple_regions)
    # ------------------------------------------------------------------ #
    gs_A = gridspec.GridSpecFromSubplotSpec(
        n_simple_hm_windows, n_simple_regions,
        subplot_spec=gs_left[0],
        hspace=0, wspace=0.04,
    )
    axes_A = np.array(
        [[fig.add_subplot(gs_A[r, c]) for c in range(n_simple_regions)]
         for r in range(n_simple_hm_windows)]
    )

    # ------------------------------------------------------------------ #
    # Panel B — PT boxplots, centered vertically inside the row slot.
    # A 3-row outer spec pads top & bottom so the actual plots occupy
    # only the middle ~50 % of the row height.
    # ------------------------------------------------------------------ #
    gs_B_outer = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=gs_right[0],
        height_ratios=[1, 2, 1],
    )
    gs_B = gridspec.GridSpecFromSubplotSpec(
        1, n_simple_box_windows,
        subplot_spec=gs_B_outer[1],
        wspace=0.22,
    )
    axes_B = [fig.add_subplot(gs_B[0, i]) for i in range(n_simple_box_windows)]

    # ------------------------------------------------------------------ #
    # Panel C — AM heatmaps (n_simple_hm_windows × n_simple_regions)
    # ------------------------------------------------------------------ #
    gs_C = gridspec.GridSpecFromSubplotSpec(
        n_simple_hm_windows, n_simple_regions,
        subplot_spec=gs_left[1],
        hspace=0, wspace=0.04,
    )
    axes_C = np.array(
        [[fig.add_subplot(gs_C[r, c]) for c in range(n_simple_regions)]
         for r in range(n_simple_hm_windows)]
    )

    # ------------------------------------------------------------------ #
    # Panel D — AM boxplots, centered vertically inside the row slot.
    # ------------------------------------------------------------------ #
    gs_D_outer = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=gs_right[1],
        height_ratios=[1, 2, 1],
    )
    gs_D = gridspec.GridSpecFromSubplotSpec(
        1, n_simple_box_windows,
        subplot_spec=gs_D_outer[1],
        wspace=0.22,
    )
    axes_D = [fig.add_subplot(gs_D[0, i]) for i in range(n_simple_box_windows)]

    # ------------------------------------------------------------------ #
    # Split data
    # ------------------------------------------------------------------ #
    pt_df = results_df[results_df["Sound Type"] == "PT"].copy()
    am_df = results_df[results_df["Sound Type"] == "AM"].copy()

    # ------------------------------------------------------------------ #
    # Draw Panel A — PT heatmaps (onset + sustained)
    # ------------------------------------------------------------------ #
    cbar_src_A = draw_heatmap_grid(
        "PT", pt_df, axes_A, simple_regions,
        windows=SIMPLE_HEATMAP_WINDOWS,
        value_col=VALUE_COL, cmap=CMAP, vmin=VMIN, vmax=VMAX,
        fs_col_header=_fs_hm_col_header,
        fs_row_label=_fs_hm_row_label,
        fs_tick=_fs_hm_tick - 10,
    )
    if cbar_src_A is not None:
        cbar_A = fig.colorbar(
            cbar_src_A, ax=axes_A,
            location="right", fraction=0.012, pad=0.02, aspect=30, shrink=0.5,
        )
        cbar_A.ax.tick_params(labelsize=_fs_cbar_tick)

    # ------------------------------------------------------------------ #
    # Draw Panel C — AM heatmaps (onset + sustained)
    # ------------------------------------------------------------------ #
    cbar_src_C = draw_heatmap_grid(
        "AM", am_df, axes_C, simple_regions,
        windows=SIMPLE_HEATMAP_WINDOWS,
        value_col=VALUE_COL, cmap=CMAP, vmin=VMIN, vmax=VMAX,
        fs_col_header=_fs_hm_col_header,
        fs_row_label=_fs_hm_row_label,
        fs_tick=_fs_hm_tick,
    )
    if cbar_src_C is not None:
        cbar_C = fig.colorbar(
            cbar_src_C, ax=axes_C,
            location="right", fraction=0.012, pad=0.02, aspect=30, shrink=0.5,
        )
        cbar_C.ax.tick_params(labelsize=_fs_cbar_tick)

    # ------------------------------------------------------------------ #
    # Draw Panel B — PT boxplots; legend added manually below
    # ------------------------------------------------------------------ #
    draw_region_boxplot_panels(
        "PT", pt_df, axes_B,
        windows=SIMPLE_BOX_WINDOWS,
        brain_regions_override=simple_regions,
        value_col=VALUE_COL, ylabel=YLABEL,
        fs_title=_fs_box_title, fs_label=_fs_box_label, fs_tick=_fs_box_tick,
        show_legend=False,
    )

    # ------------------------------------------------------------------ #
    # Draw Panel D — AM boxplots; legend added manually below
    # ------------------------------------------------------------------ #
    draw_region_boxplot_panels(
        "AM", am_df, axes_D,
        windows=SIMPLE_BOX_WINDOWS,
        brain_regions_override=simple_regions,
        value_col=VALUE_COL, ylabel=YLABEL,
        fs_title=_fs_box_title, fs_label=_fs_box_label, fs_tick=_fs_box_tick,
        show_legend=False,
    )

    # ------------------------------------------------------------------ #
    # Titles — after canvas.draw() so positions are finalised
    # ------------------------------------------------------------------ #
    fig.canvas.draw()

    # Horizontal region legend below each boxplot row
    _simple_pal = sns.color_palette("viridis", n_colors=len(simple_regions))
    _simple_handles = [
        mpatches.Patch(facecolor=_simple_pal[i], edgecolor="black", linewidth=1.0,
                       label=params.short_names.get(r, r))
        for i, r in enumerate(simple_regions)
    ]
    _legend_kw = dict(
        handles=_simple_handles,
        ncol=len(simple_regions),
        fontsize=_fs_box_tick,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        loc="upper center",
    )
    _b_x0 = min(ax.get_position().x0 for ax in axes_B)
    _b_x1 = max(ax.get_position().x1 for ax in axes_B)
    _b_y0 = min(ax.get_position().y0 for ax in axes_B)
    fig.legend(**_legend_kw, bbox_to_anchor=((_b_x0 + _b_x1) / 2, _b_y0 - 0.02))

    _d_x0 = min(ax.get_position().x0 for ax in axes_D)
    _d_x1 = max(ax.get_position().x1 for ax in axes_D)
    _d_y0 = min(ax.get_position().y0 for ax in axes_D)
    fig.legend(**_legend_kw, bbox_to_anchor=((_d_x0 + _d_x1) / 2, _d_y0 - 0.02))

    _add_group_title(
        fig, axes_A,
        f"Pairwise {METHOD_LABEL} Heatmaps for Pure Tones",
        fontsize=_fs_hm_title, pad=0.030,
    )
    _add_group_title(
        fig, axes_C,
        f"Pairwise {METHOD_LABEL} Heatmaps for AM",
        fontsize=_fs_hm_title, pad=0.030,
    )
    _add_group_title(
        fig, axes_B,
        f"Pairwise {METHOD_LABEL} for Pure Tones",
        fontsize=_fs_section_title, pad=0.030,
    )
    _add_group_title(
        fig, axes_D,
        f"Pairwise {METHOD_LABEL} for AM",
        fontsize=_fs_section_title, pad=0.030,
    )

    # ------------------------------------------------------------------ #
    # Panel letters — placed so A/B are at the same height and C/D are
    # at the same height, forming a clean 2×2 rectangle of letters.
    # x_left is anchored to the first heatmap column; x_right to the
    # first boxplot panel.
    # ------------------------------------------------------------------ #
    y_row1 = max(ax.get_position().y1 for ax in axes_A.ravel())
    y_row2 = max(ax.get_position().y1 for ax in axes_C.ravel())
    x_left = axes_A[0, 0].get_position().x0 - 0.050
    x_right = axes_B[0].get_position().x0 - 0.025
    letter_pad = 0.028

    fig.text(x_left,  y_row1 + letter_pad, "A",
             ha="left", va="bottom", fontsize=_fs_panel_letter,
             fontweight="bold", clip_on=False)
    fig.text(x_right, y_row1 + letter_pad, "B",
             ha="left", va="bottom", fontsize=_fs_panel_letter,
             fontweight="bold", clip_on=False)
    fig.text(x_left,  y_row2 + letter_pad, "C",
             ha="left", va="bottom", fontsize=_fs_panel_letter,
             fontweight="bold", clip_on=False)
    fig.text(x_right, y_row2 + letter_pad, "D",
             ha="left", va="bottom", fontsize=_fs_panel_letter,
             fontweight="bold", clip_on=False)

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    output_dir = get_figure_dir()
    out_path = output_dir / "simple_sounds_discriminability_figure.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main() -> None:
    """Run all linearSVM plots (identical to plot_linear_svm.py) plus the combined figure."""
    results_df = load_method_results(METHOD_KEY)

    # --- Same outputs as plot_linear_svm.py ---
    plot_heatmaps(
        results_df,
        method_key=METHOD_KEY,
        method_label=METHOD_LABEL,
        value_col=VALUE_COL,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
    )
    plot_region_boxplots(
        results_df,
        method_key=METHOD_KEY,
        method_label=METHOD_LABEL,
        value_col=VALUE_COL,
        ylabel=YLABEL,
    )
    plot_natural_within_between_boxplots(
        results_df,
        method_key=METHOD_KEY,
        method_label=METHOD_LABEL,
        value_col=VALUE_COL,
        ylabel=YLABEL,
    )
    plot_linear_svm_example(results_df)
    tuning_path = get_tuning_path()
    if tuning_path.exists():
        plot_svm_hyperparameter_tuning(pd.read_csv(tuning_path))

    # --- Compound figures (one per sound category) ---
    make_natural_sounds_figure(results_df)
    make_speech_figure(results_df)
    make_simple_sounds_figure(results_df)


if __name__ == "__main__":
    main()
