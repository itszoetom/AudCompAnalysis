"""Create population ridge predicted-versus-actual figures for each sound type across regions and windows."""

from __future__ import annotations

import matplotlib.pyplot as plt
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

from shared import params

try:
    from .ridge_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        build_sampled_dataset,
        build_target_datasets,
        fit_best_ridge,
        get_output_dir,
        get_plot_brain_regions,
        get_target_neuron_count,
        list_available_sound_types,
        plot_ridge_summary,
        run_population_ridge,
    )
except ImportError:
    from ridge_analysis import (
        WINDOW_ORDER,
        apply_figure_style,
        build_sampled_dataset,
        build_target_datasets,
        fit_best_ridge,
        get_output_dir,
        get_plot_brain_regions,
        get_target_neuron_count,
        list_available_sound_types,
        plot_ridge_summary,
        run_population_ridge,
    )


def plot_panel(ax: plt.Axes, x, y, title: str, log_target: bool) -> None:
    """Plot one predicted-versus-actual ridge panel."""
    fit = fit_best_ridge(x, y, log_target=log_target)
    y_true = fit["y_test"]
    y_pred = fit["y_pred"]
    if log_target:
        y_true = 10 ** y_true
        y_pred = 10 ** y_pred
    ax.scatter(y_true, y_pred, s=28, alpha=0.75, color="black")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.5, color="tab:red")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\n$R^2$={fit['r2_test']:.2f}", fontweight="bold")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)


def plot_scatter_grid(sound_type: str, target_name: str) -> None:
    """Plot one predicted-vs-actual scatter grid for a sound and target."""
    brain_regions = get_plot_brain_regions(sound_type)
    fig, axes = plt.subplots(
        len(brain_regions),
        len(WINDOW_ORDER),
        figsize=(4.0 * len(WINDOW_ORDER), 3.5 * len(brain_regions)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        f"{sound_type} population ridge {target_name} (n={get_target_neuron_count(sound_type)} neurons per region)",
        fontsize=16,
        fontweight="bold",
    )
    panel_conditions = [(row_index, brain_area, col_index, window_name) for row_index, brain_area in enumerate(brain_regions) for col_index, window_name in enumerate(WINDOW_ORDER)]
    for row_index, brain_area, col_index, window_name in tqdm(
        panel_conditions,
        desc=f"Ridge panels ({sound_type} {target_name})",
        unit="panel",
        dynamic_ncols=True,
    ):
        ax = axes[row_index, col_index]
        dataset = build_sampled_dataset(
            sound_type,
            window_name,
            brain_area,
            n_neurons=get_target_neuron_count(sound_type),
        )
        if dataset is None:
            ax.axis("off")
            continue
        target_dataset = next(
            (candidate for candidate in build_target_datasets(dataset) if candidate["target_name"] == target_name),
            None,
        )
        if target_dataset is None:
            ax.axis("off")
            continue
        title = f"{params.short_names[brain_area]} {window_name.capitalize()}"
        if sound_type == "speech":
            title = f"{title} {target_name}"
        plot_panel(
            ax,
            target_dataset["X"],
            target_dataset["Y"],
            title,
            log_target=bool(target_dataset["log_target"]),
        )
    fig.savefig(get_output_dir() / f"{sound_type}_{target_name}_ridge_population.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Run population ridge scatter and summary figures."""
    apply_figure_style()
    print("Running population ridge regression...")
    summary_df = run_population_ridge()
    for sound_type in tqdm(list_available_sound_types(), desc="Ridge predicted-vs-actual", unit="sound", dynamic_ncols=True):
        print(f"Building population ridge plots for {sound_type}...")
        sound_df = summary_df[summary_df["Sound Type"] == sound_type].copy()
        if not sound_df.empty:
            plot_ridge_summary(
                sound_type,
                sound_df,
                title=f"{sound_type} population ridge $R^2$",
                filename=f"{sound_type}_ridge_population_boxplots.png",
                pair_cols=["Iteration"],
            )
        target_names = ["FT", "VOT"] if sound_type == "speech" else [sound_type]
        for target_name in target_names:
            plot_scatter_grid(sound_type, target_name)


if __name__ == "__main__":
    main()
