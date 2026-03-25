"""Plot per-mouse mean spike-rate profiles by brain region and spike window for each sound."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
    from .single_mouse_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        available_mice,
        build_dataset,
        get_brain_regions,
        get_figure_dir,
        list_available_sound_types,
    )
except ImportError:
    from single_mouse_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        available_mice,
        build_dataset,
        get_brain_regions,
        get_figure_dir,
        list_available_sound_types,
    )


def average_by_stimulus(dataset: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(dataset["Y"])
    x = np.asarray(dataset["X"])

    if dataset["sound_type"] == "speech":
        unique_y, inverse = np.unique(y, axis=0, return_inverse=True)
        x_axis = np.arange(len(unique_y))
    else:
        unique_y, inverse = np.unique(y, return_inverse=True)
        x_axis = unique_y.astype(float)

    mean_response = np.array([x[inverse == idx].mean() for idx in range(len(unique_y))])
    sem_response = np.array([x[inverse == idx].std(ddof=0) / np.sqrt(np.sum(inverse == idx)) for idx in range(len(unique_y))])
    return x_axis, mean_response, sem_response


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

                    x_axis, mean_response, sem_response = average_by_stimulus(dataset)
                    ax.plot(x_axis, mean_response, marker="o", linewidth=1.8, color="black")
                    ax.fill_between(x_axis, mean_response - sem_response, mean_response + sem_response, alpha=0.2)
                    ax.set_title(f"{brain_area}\n{window_name.capitalize()}")
                    ax.set_ylabel("Mean firing rate")
                    ax.set_xlabel("Stimulus")
                    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.3)

            fig.savefig(get_figure_dir() / f"{mouse_id}_{sound_type}_spike_rate_profiles.png", dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    main()
