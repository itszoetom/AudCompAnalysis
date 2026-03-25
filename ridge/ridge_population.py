"""Run ridge regression on all available population datasets and save a summary table."""

try:
    from .ridge_analysis import get_output_dir, run_population_ridge
except ImportError:
    from ridge_analysis import get_output_dir, run_population_ridge


def main() -> None:
    results_df = run_population_ridge()
    if results_df.empty:
        return
    output_path = get_output_dir() / "ridge_population_results.csv"
    results_df.to_csv(output_path, index=False)
    print(results_df.groupby(["Window", "Brain Area", "Target"])["R2 Test"].mean())
    print(f"Saved ridge summary to {output_path}")


if __name__ == "__main__":
    main()
