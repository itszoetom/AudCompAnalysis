"""Per-session PCA figures arranged as brain regions by spike windows for each sound."""

import matplotlib.pyplot as plt

try:
    from .single_mouse_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        available_mice,
        available_sessions,
        build_dataset,
        compute_pca_summary,
        get_brain_regions,
        get_figure_dir,
        labels_for_sound,
        list_available_sound_types,
    )
except ImportError:
    from single_mouse_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        available_mice,
        available_sessions,
        build_dataset,
        compute_pca_summary,
        get_brain_regions,
        get_figure_dir,
        labels_for_sound,
        list_available_sound_types,
    )


def main() -> None:
    apply_figure_style()
    for sound_type in list_available_sound_types():
        brain_regions = get_brain_regions(sound_type)
        for mouse_id in available_mice(sound_type):
            for session_id in available_sessions(sound_type, mouse_id=mouse_id):
                fig, axes = plt.subplots(
                    len(brain_regions),
                    len(WINDOW_ORDER),
                    figsize=(3.2 * len(WINDOW_ORDER), 3.0 * len(brain_regions)),
                    squeeze=False,
                    constrained_layout=True,
                )
                fig.suptitle(f"{session_id} PCA ({sound_type})", fontsize=16)

                for row_index, brain_area in enumerate(brain_regions):
                    for col_index, window_name in enumerate(WINDOW_ORDER):
                        ax = axes[row_index, col_index]
                        dataset = build_dataset(
                            sound_type,
                            window_name,
                            brain_area,
                            mouse_id=mouse_id,
                            session_id=session_id,
                        )
                        if dataset is None:
                            ax.axis("off")
                            continue

                        summary = compute_pca_summary(dataset["X"])
                        scatter = ax.scatter(
                            summary["scores"][:, 0],
                            summary["scores"][:, 1],
                            c=labels_for_sound(sound_type, dataset["Y"]),
                            cmap="viridis",
                            s=24,
                            alpha=0.8,
                        )
                        ax.set_title(f"{brain_area}\n{window_name.capitalize()}")
                        ax.set_xlabel("PC1")
                        ax.set_ylabel("PC2")
                        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
                        plt.colorbar(scatter, ax=ax)

                safe_session = session_id.replace("/", "_")
                fig.savefig(get_figure_dir() / f"{safe_session}_{sound_type}_pca.png", dpi=200)
                plt.close(fig)


if __name__ == "__main__":
    main()
