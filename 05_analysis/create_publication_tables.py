
import argparse
import itertools
import logging
import pathlib
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates Cliff's Delta, a non-parametric effect size."""
    if len(x) == 0 or len(y) == 0:
        return np.nan
    n_x, n_y = len(x), len(y)
    # This is faster than list comprehensions for large arrays
    gt = np.sum(x[:, None] > y)
    lt = np.sum(x[:, None] < y)
    return (gt - lt) / (n_x * n_y)


class PublicationTableGenerator:
    """
    Generates publication-ready tables of statistical comparisons
    from a master metrics DataFrame.
    """

    def __init__(self, data_path: pathlib.Path, output_dir: pathlib.Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        LOG.info(f"Loading master metrics from {data_path}")
        try:
            self.df = pd.read_csv(data_path)
        except FileNotFoundError:
            LOG.error(f"FATAL: Data file not found at {data_path}")
            raise

        self.results = []
        # Define the metrics and their column name stems
        self.metrics = {
            "Shannon Entropy": "entropy",
            "Flesch Reading Ease": "flesch",
            "Coleman-Liau Index": "cl",
            "Lexical Diversity": "lex_div",
            "GPT-2 Surprisal": "surprisal",
            "Sentiment Polarity": "sentiment_polarity",
            "Forward Flow": "ff",
        }
        # Map the logical prefixes used in comparisons to the actual column prefixes
        self.prefix_map = {
            "popular": "popular",
            "sample": "sample",
            "t50": "t50_quote",
            "snippet": "matched_snippet",
            "t50freelength": "t50_quote_free_context_length_matched",
            "nonlit": "nonlit_baseline"
        }
        # Define the comparisons
        self.comparisons = self._define_comparisons()
        LOG.info(f"Defined {len(self.comparisons)} comparison sets.")

    def _define_comparisons(self) -> dict:
        """Defines all pooled group comparisons for the analysis."""
        return {
            "Quotes vs. Non-Quotes": {
                "test": ["popular", "sample", "t50"],
                "base": ["snippet", "t50freelength", "nonlit"],
            },
            "GPQ vs. NLB": {"test": ["popular"], "base": ["nonlit"]},
            "GSQ vs. NLB": {"test": ["sample"], "base": ["nonlit"]},
            "T50Q vs. NLB": {"test": ["t50"], "base": ["nonlit"]},
        }

    def run_all_analyses(self):
        """Runs all defined comparisons for all metrics."""
        LOG.info("Starting analysis for all metrics and comparisons...")
        for metric_name, metric_stem in self.metrics.items():
            for comp_name, groups in self.comparisons.items():
                self._run_single_comparison(metric_name, metric_stem, comp_name, groups)

        # Apply FDR correction to all p-values at once
        self._apply_fdr_correction()

        # Save the final results table
        self.save_results()

    def _run_single_comparison(self, metric_name, metric_stem, comp_name, groups):
        """Runs a single pairwise statistical comparison."""
        test_cols = [f"{self.prefix_map[p]}_{metric_stem}" for p in groups["test"]]
        base_cols = [f"{self.prefix_map[p]}_{metric_stem}" for p in groups["base"]]

        # Check if all required columns exist
        if not all(c in self.df.columns for c in test_cols + base_cols):
            LOG.warning(f"Skipping '{comp_name}' for '{metric_name}': missing one or more data columns.")
            return

        # Pool the data from the respective columns
        test_vals = pd.concat([self.df[c] for c in test_cols]).dropna().values
        base_vals = pd.concat([self.df[c] for c in base_cols]).dropna().values

        if len(test_vals) == 0 or len(base_vals) == 0:
            LOG.warning(f"Skipping '{comp_name}' for '{metric_name}': one group is empty after dropping NaNs.")
            return

        # Perform Mann-Whitney U test
        # For readability, higher Flesch is better (less complex), so we expect it to be greater.
        # For Coleman-Liau, lower is better. For others, "greater" indicates more of the measured quality.
        alternative = "two-sided"
        if metric_name == "Flesch Reading Ease":
            alternative = "greater"
        elif metric_name == "Coleman-Liau Index":
            alternative = "less"

        u_stat, p_val = mannwhitneyu(test_vals, base_vals, alternative=alternative, use_continuity=True)
        delta = cliffs_delta(test_vals, base_vals)

        self.results.append({
            "Comparison": comp_name,
            "Metric": metric_name,
            "U-statistic": u_stat,
            "p_value_raw": p_val,
            "Cliff_Delta": delta,
        })
        LOG.info(f"Completed: {metric_name} | {comp_name} | Cliff's δ = {delta:+.3f}")

    def _apply_fdr_correction(self):
        """Applies Benjamini-Hochberg FDR correction to the collected p-values."""
        if not self.results:
            LOG.warning("No results to process for FDR correction.")
            return

        p_values = [res["p_value_raw"] for res in self.results]
        _, p_adj, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)

        for i, res in enumerate(self.results):
            res["p_value_fdr_bh"] = p_adj[i]

    def save_results(self):
        """Saves the final, formatted results to a CSV file."""
        if not self.results:
            LOG.error("No results were generated. Cannot save table.")
            return

        results_df = pd.DataFrame(self.results)
        
        # Create the pivot table for easy viewing, similar to the thesis table
        pivot_df = results_df.pivot_table(
            index="Metric",
            columns="Comparison",
            values="Cliff_Delta"
        ).reindex(self.metrics.keys()) # Keep original metric order

        # Add significance stars to the pivot table
        def add_stars(val, p_val):
            if pd.isna(val): return ""
            stars = ""
            if p_val < 0.001: stars = "***"
            elif p_val < 0.01: stars = "**"
            elif p_val < 0.05: stars = "*"
            return f"{val:+.2f}{stars}"

        for comp in results_df['Comparison'].unique():
            for metric in results_df['Metric'].unique():
                p_val_fdr = results_df.loc[(results_df.Comparison == comp) & (results_df.Metric == metric), 'p_value_fdr_bh'].values
                if len(p_val_fdr) > 0:
                    delta_val = pivot_df.at[metric, comp]
                    pivot_df.at[metric, comp] = add_stars(delta_val, p_val_fdr[0])

        output_path = self.output_dir / "publication_table_data.csv"
        pivot_df.to_csv(output_path)
        LOG.info(f"Publication-ready table saved to: {output_path}")
        
        # Save the raw, unpivoted data as well for full transparency
        raw_output_path = self.output_dir / "publication_table_data_raw.csv"
        results_df.to_csv(raw_output_path, index=False)
        LOG.info(f"Raw results data saved to: {raw_output_path}")
        
        print("\n--- Final Publication Table ---")
        print(pivot_df)


def main():
    """Main function to run the table generation."""
    parser = argparse.ArgumentParser(description="Generate publication-ready tables from ALOFT master metrics.")
    parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        default=pathlib.Path("data/outputs/master_metrics/ALOFT_master_metrics.csv"),
        help="Path to the master metrics CSV file."
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("data/outputs/analysis/publication_tables"),
        help="Directory to save the final table CSVs."
    )
    args = parser.parse_args()

    generator = PublicationTableGenerator(args.data_path, args.output_dir)
    generator.run_all_analyses()


if __name__ == "__main__":
    main() 