"""Generate the C panel (schematic stimulus waveforms) for Figure 1.

Produces a 2-row × 2-column waveform diagram:
  Column 0 — Pure Tones : low (20 kHz) on top, high (40 kHz) on bottom
  Column 1 — AM Noise   : slow (4 Hz AM) on top, fast (64 Hz AM) on bottom

PT panels use a short time window (~0.3 ms) at high sample rate so individual
sine cycles are visible.  AM panels span a full 1 second so all modulation
cycles are visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

SAVE_PATH = Path(__file__).resolve().parent / "c_panel_waveforms.png"
DPI = 300

# Font sizes — thesis figure
FONTSIZE_LABEL = 20
FONTSIZE_TICK  = 17
FONTSIZE_COL   = 22

# Viridis accent colors for low vs. high
COLOR_LOW  = plt.cm.viridis(0.20)
COLOR_HIGH = plt.cm.viridis(0.75)

# ── Pure Tone time axis ────────────────────────────────────────────────────────
# High sample rate + very short window so individual cycles at 20/40 kHz are visible
FS_PT   = 200_000                        # 200 kHz
T_PT    = 0.0003                         # 0.3 ms — ~6 cycles at 20 kHz
t_pt    = np.linspace(0, T_PT, int(FS_PT * T_PT), endpoint=False)
t_pt_ms = t_pt * 1_000                  # display in milliseconds

PT_LOW_HZ  = 20_000   # 20 kHz
PT_HIGH_HZ = 40_000   # 40 kHz

# ── AM time axis ───────────────────────────────────────────────────────────────
FS_AM  = 8_000
T_AM   = 1.0
t_am   = np.linspace(0, T_AM, int(FS_AM * T_AM), endpoint=False)

AM_LOW_HZ  = 4    # 4 modulations per second
AM_HIGH_HZ = 16   # 16 modulations per second


# ── Waveform generators ────────────────────────────────────────────────────────

def pure_tone(freq_hz: float, t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * freq_hz * t)


def am_noise(mod_rate_hz: float, t: np.ndarray, carrier_bandwidth: float = 600.0,
             rng_seed: int = 0) -> np.ndarray:
    """Amplitude-modulated broadband noise carrier."""
    rng = np.random.default_rng(rng_seed)
    freqs  = np.linspace(100, carrier_bandwidth, 30)
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    carrier = sum(np.sin(2 * np.pi * f * t + ph) for f, ph in zip(freqs, phases))
    carrier /= np.max(np.abs(carrier))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * mod_rate_hz * t)
    return carrier * envelope


# ── Axis styling ───────────────────────────────────────────────────────────────

def _style_ax(
    ax: plt.Axes,
    *,
    show_x: bool = False,
    x_label: str = "Time",
) -> None:
    ax.set_ylim(-1.45, 1.45)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(show_x)
    if show_x:
        ax.set_xlabel(x_label, fontsize=FONTSIZE_TICK)
        ax.tick_params(axis="x", labelsize=FONTSIZE_TICK)
    else:
        ax.set_xticks([])


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })

    fig, axes = plt.subplots(
        2, 2,
        figsize=(5.6, 3.6),
        constrained_layout=True,
    )

    # ── Column headers ─────────────────────────────────────────────────────────
    axes[0, 0].set_title("Pure Tones", fontsize=FONTSIZE_COL, fontweight="bold")
    axes[0, 1].set_title("AM Noise",   fontsize=FONTSIZE_COL, fontweight="bold")

    # ── Row 0: Low ─────────────────────────────────────────────────────────────
    axes[0, 0].plot(t_pt_ms, pure_tone(PT_LOW_HZ, t_pt),       color=COLOR_LOW, linewidth=2.2)
    axes[0, 1].plot(t_am,    am_noise(AM_LOW_HZ, t_am, rng_seed=1), color=COLOR_LOW, linewidth=1.4)

    # row label
    axes[0, 0].text(
        -0.18, 0.5, "Low",
        transform=axes[0, 0].transAxes,
        fontsize=FONTSIZE_COL, fontweight="bold",
        va="center", ha="right",
    )

    for ax in axes[0]:
        _style_ax(ax, show_x=False)

    # ── Row 1: High ────────────────────────────────────────────────────────────
    axes[1, 0].plot(t_pt_ms, pure_tone(PT_HIGH_HZ, t_pt),       color=COLOR_HIGH, linewidth=2.2)
    axes[1, 1].plot(t_am,    am_noise(AM_HIGH_HZ, t_am, rng_seed=2), color=COLOR_HIGH, linewidth=1.4)

    axes[1, 0].text(
        -0.18, 0.5, "High",
        transform=axes[1, 0].transAxes,
        fontsize=FONTSIZE_COL, fontweight="bold",
        va="center", ha="right",
    )

    _style_ax(axes[1, 0], show_x=True, x_label="Time (ms)")
    _style_ax(axes[1, 1], show_x=True, x_label="Time (s)")

    # ── Frequency / rate annotations inside each panel ─────────────────────────
    label_kwargs = dict(transform=None, fontsize=FONTSIZE_TICK, va="top", ha="left")

    for ax, label, color in [
        (axes[0, 0], "20 kHz",  COLOR_LOW),
        (axes[1, 0], "40 kHz",  COLOR_HIGH),
        (axes[0, 1], f"{AM_LOW_HZ} Hz AM",  COLOR_LOW),
        (axes[1, 1], f"{AM_HIGH_HZ} Hz AM", COLOR_HIGH),
    ]:
        ax.text(
            0.03, 0.99, label,
            transform=ax.transAxes,
            fontsize=FONTSIZE_TICK,
            va="top", ha="left",
            color=color,
        )

    fig.savefig(SAVE_PATH, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved C panel to {SAVE_PATH}")


if __name__ == "__main__":
    main()
