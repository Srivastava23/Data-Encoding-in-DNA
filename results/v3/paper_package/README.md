# DNA Data Encoding RL - Paper Package

This package contains all finalized data, statistics, methods, findings, and figures required to author the manuscript for the DNA Data Encoding RL project.

## Directory Structure
- `METHODS_FACTS.md`: Exact implementation details, parameters, architectures, and dataset configurations used in the final evaluation.
- `FINDINGS.md`: Cautious, formal prose outlining the headline results, supported claims, and unsupported claims. Explicitly addresses the failure of single-copy RL compared to redundancy.
- `LIMITATIONS.md`: Formal explanation of the architectural, state space, and redundancy limitations identified in the study.
- `tables_csv/`: Raw and aggregated CSV data generated from the 7-seed evaluation runs.
  - `results.csv`: Per-length, per-method accuracy and error rates.
  - `statistics_summary.csv`: Aggregated means, standard deviations, and p-values for RL vs. baseline comparisons.
  - `redundancy_scaling.csv`: Accuracy improvements scaling from 1 to 10 copies.
  - `ablation_results.csv`: Contribution of specific reward components.
- `figures_png/`: High-resolution PNG plots generated directly from the tables. Includes training curves, error rates, and redundancy scaling plots.
- `figures_pdf/`: (If generated) Vector PDF formats for the figures.
