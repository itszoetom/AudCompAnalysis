"""Plot per-mouse mean spike-rate profiles by brain region and spike window for each sound."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
    from .methods_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        available_mice,
        build_dataset,
        get_brain_regions,
        get_figure_dir,
        list_available_sound_types,
    )
except ImportError:
    from methods_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        available_mice,
        build_dataset,
        get_brain_regions,
        get_figure_dir,
        list_available_sound_types,
    )

import params


def stimulus_labels(sound_type: str, unique_y: np.ndarray, unique_labels: np.ndarray | None = None) -> list[str]:
    """Return written-out stimulus labels for one sound type."""
    if unique_labels is not None:
        return [str(label) for label in unique_labels]
    if sound_type == "speech":
        labels = []
        for ft_value, vot_value in unique_y:
            labels.append(f"VOT={int(vot_value)} FT={int(ft_value)}")
        return labels
    if sound_type == "naturalSound":
        return [f"{params.SOUND_CATEGORIES[int(value) // 4]}_{int(value) % 4 + 1}" for value in unique_y]
    if sound_type == "AM":
        return [f"AM White Noise - {int(value)} Hz" for value in unique_y]
    return [f"Pure Tones - {int(value)} Hz" for value in unique_y]


def average_by_stimulus(dataset: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Average one dataset across repeated stimulus identities."""
    y = np.asarray(dataset["Y"])
    y_labels = np.asarray(dataset.get("Y_labels")) if dataset.get("Y_labels") is not None else None
    x = np.asarray(dataset["X"])

    if dataset["sound_type"] == "speech":
        unique_y, inverse = np.unique(y, axis=0, return_inverse=True)
        x_axis = np.arange(len(unique_y))
    else:
        unique_y, inverse = np.unique(y, return_inverse=True)
        x_axis = unique_y.astype(float)

    mean_response = np.array([x[inverse == idx].mean() for idx in range(len(unique_y))])
    sem_response = np.array([x[inverse == idx].std(ddof=0) / np.sqrt(np.sum(inverse == idx)) for idx in range(len(unique_y))])
    unique_label_values = None
    if y_labels is not None:
        unique_label_values = np.array([y_labels[np.flatnonzero(inverse == idx)[0]] for idx in range(len(unique_y))], dtype=object)
    return x_axis, mean_response, sem_response, stimulus_labels(dataset["sound_type"], unique_y, unique_label_values)


def main() -> None:
    apply_figure_style()
    for sound_type in list_available_sound_types():
        brain_regions = get_brain_regions(sound_type)
        for mouse_id in available_mice(sound_type):
            fig, axes = plt.subplots(
                len(brain_regions),
                len(WINDOW_ORDER),
                figsize=(3.3 * len(WINDOW_ORDER), 2.9 * len(brain_regions)),
                squeeze=False,
                constrained_layout=True,
            )
            fig.suptitle(f"{mouse_id} spike-rate profiles ({sound_type})", fontsize=16)

            for row_index, brain_area in enumerate(brain_regions):
                for col_index, window_name in enumerate(WINDOW_ORDER):
                    ax = axes[row_index, col_index]
                    dataset = build_dataset(sound_type, window_name, brain_area, mouse_id=mouse_id)
                    if dataset is None:
                        ax.axis("off")
                        continue

                    x_axis, mean_response, sem_response, x_labels = average_by_stimulus(dataset)
                    ax.plot(x_axis, mean_response, marker="o", linewidth=1.8, color="black")
                    ax.fill_between(x_axis, mean_response - sem_response, mean_response + sem_response, alpha=0.2)
                    ax.set_title(f"{params.short_names.get(brain_area, brain_area)}\n{window_name.capitalize()}", fontweight="bold")
                    ax.set_ylabel("Mean firing rate (spk/s)")
                    ax.set_xlabel("Stimulus")
                    ax.set_xticks(x_axis)
                    ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=7)
                    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

            fig.savefig(get_figure_dir() / f"{mouse_id}_{sound_type}_spike_rate_profiles.png", dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    main()
