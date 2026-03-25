"""Run equal-neuron ridge regression and save the subset summary table."""

try:
    from .ridge_analysis import get_output_dir, minimum_common_neuron_count, run_subset_ridge
except ImportError:
    from ridge_analysis import get_output_dir, minimum_common_neuron_count, run_subset_ridge


def main() -> None:
    subset_size = minimum_common_neuron_count()
    results_df = run_subset_ridge(subset_size=subset_size, iterations=30)
    if results_df.empty:
        return
    output_path = get_output_dir() / "ridge_subset_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Using {subset_size} neurons per condition")
    print(results_df.groupby(["Window", "Brain Area", "Target"])["R2 Test"].mean())
    print(f"Saved subset ridge summary to {output_path}")


if __name__ == "__main__":
    main()
