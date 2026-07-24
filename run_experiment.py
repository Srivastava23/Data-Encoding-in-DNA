"""
Main entry-point for the full experimental pipeline.

Usage:
    python run_experiment.py

Trains reinforcement learning agents (with and without predictor) across
multiple seeds, evaluates against baseline and random encoding, runs
ablation studies, redundancy scaling, computes statistics with t-tests,
and exports CSVs + plots.
"""

import numpy as np
import torch
import sys
import os

from config import Config
from training_pipeline import train
from evaluation import (
    run_comparison,
    run_ablation_comparison,
    compute_statistics,
    export_results_csv,
    export_statistics_csv,
    run_redundancy_scaling,
    aggregate_redundancy_scaling,
    export_redundancy_csv,
)
from visualization import generate_all_plots


def set_global_seeds(seed: int):
    """Set seeds for numpy, torch, and Python's random module."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_banner():
    print("=" * 75)
    print("  Predictive Reinforcement Learning-Based Optimisation of DNA Data Encoding")
    print("  for Error-Resilient Archival Storage")
    print("=" * 75)
    print()


def print_summary_table(statistics, title="Summary"):
    """Pretty-print the statistical summary."""
    print(f"\n{'=' * 95}")
    print(f"  {title}")
    print(f"{'=' * 95}")
    print(f"{'Method':<30} {'SeqLen':>6}  {'ErrorRate':>14}  {'DecodeAcc':>14}  "
          f"{'GC Score':>12}  {'Efficiency':>14}")
    print("-" * 95)
    for statistic in statistics:
        error_mean = statistic.get("average_error_rate_mean", 0)
        error_standard_deviation = statistic.get("average_error_rate_std", 0)
        decode_mean = statistic.get("decode_accuracy_mean", 0)
        decode_standard_deviation = statistic.get("decode_accuracy_std", 0)
        gc_mean = statistic.get("gc_score_mean", 0)
        gc_standard_deviation = statistic.get("gc_score_std", 0)
        efficiency_mean = statistic.get("encoding_efficiency_mean", 0)
        efficiency_standard_deviation = statistic.get("encoding_efficiency_std", 0)

        line = (f"{statistic['method']:<30} {statistic['seq_length']:>6}  "
                f"{error_mean:>6.4f}+/-{error_standard_deviation:<6.4f}  "
                f"{decode_mean:>6.2f}+/-{decode_standard_deviation:<6.2f}  "
                f"{gc_mean:>5.3f}+/-{gc_standard_deviation:<5.3f}  "
                f"{efficiency_mean:>6.4f}+/-{efficiency_standard_deviation:<6.4f}")
        print(line)

        # significance annotations
        for label, t_key, p_key in [
            ("baseline", "t_statistic_versus_baseline", "p_value_versus_baseline"),
            ("reinforcement_learning", "t_statistic_versus_reinforcement_learning", "p_value_versus_reinforcement_learning"),
        ]:
            if p_key in statistic:
                p_value = statistic[p_key]
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "not significant"
                print(f"{'':>38} t-test versus {label}: t={statistic[t_key]:.3f}, "
                      f"p={p_value:.4f} ({significance})")

    print("=" * 95)


def main():
    print_banner()
    config = Config()

    os.makedirs(config.output_directory, exist_ok=True)

    all_rows = []
    all_ablation_rows = []
    last_epoch_rewards = []
    last_rolling_rewards = []
    last_smoothed_rewards = []
    last_predictor_losses = []
    last_validation_rewards = []
    last_predictive_agent = None
    all_seeds = []

    for seed_index in range(config.number_of_seeds):
        seed = config.seed + seed_index
        all_seeds.append(seed)
        set_global_seeds(seed)
        print(f"\n{'-' * 60}")
        print(f"  Seed {seed_index + 1}/{config.number_of_seeds}  (seed={seed})")
        print(f"{'-' * 60}")

        # --- Phase 1: Train reinforcement learning agent (no predictor) ---
        print("\nTraining reinforcement learning agent (without predictor)...")
        config_no_predictor = Config(seed=seed, use_error_predictor=False)
        reinforcement_learning_result = train(config_no_predictor, seed=seed, verbose=True)
        agent_reinforcement_learning = reinforcement_learning_result["agent"]

        # --- Phase 2: Train predictive reinforcement learning agent (with predictor) ---
        print("\nTraining predictive reinforcement learning agent (with predictor)...")
        config_with_predictor = Config(seed=seed, use_error_predictor=True)
        predictive_result = train(config_with_predictor, seed=seed, verbose=True)
        agent_predictive = predictive_result["agent"]
        predictor = predictive_result["predictor"]

        # --- Phase 3: Train ablation agents ---
        print("\nTraining ablation: reinforcement learning without GC reward...")
        config_no_gc = Config(
            seed=seed, use_error_predictor=False,
            reward_gc_bonus_weight=0.0,
        )
        agent_no_gc = train(config_no_gc, seed=seed, verbose=False)["agent"]

        print("Training ablation: reinforcement learning without homopolymer penalty...")
        config_no_homopolymer = Config(
            seed=seed, use_error_predictor=False,
            reward_homopolymer_penalty_weight=0.0,
        )
        agent_no_homopolymer = train(config_no_homopolymer, seed=seed, verbose=False)["agent"]

        # store training curves from last seed for plotting
        last_epoch_rewards = predictive_result["epoch_rewards"]
        last_rolling_rewards = predictive_result["rolling_rewards"]
        last_smoothed_rewards = predictive_result.get("smoothed_rewards", [])
        last_predictor_losses = predictive_result["predictor_losses"]
        last_validation_rewards = predictive_result.get("validation_rewards", [])
        last_predictive_agent = agent_predictive

        # --- Phase 4: Evaluate standard methods ---
        print("\nEvaluating all methods...")
        rows = run_comparison(config, agent_reinforcement_learning, agent_predictive, predictor, seed)
        all_rows.extend(rows)

        for row in rows:
            print(f"    {row['method']:<40} length={row['seq_length']:>3}  "
                  f"error_rate={row['average_error_rate']:.4f}  accuracy={row['decode_accuracy']:.2f}%")

        # --- Phase 5: Ablation comparison on frozen datasets ---
        print("\nRunning ablation comparison...")
        ablation_agents = {
            "predictive_reinforcement_learning": agent_predictive,
            "reinforcement_learning": agent_reinforcement_learning,
            "reinforcement_learning_no_gc": agent_no_gc,
            "reinforcement_learning_no_homopolymer": agent_no_homopolymer,
        }
        ablation_rows = run_ablation_comparison(config, ablation_agents, seed)
        all_ablation_rows.extend(ablation_rows)

        for row in ablation_rows:
            print(f"    {row['method']:<40} length={row['seq_length']:>3}  "
                  f"error_rate={row['average_error_rate']:.4f}  accuracy={row['decode_accuracy']:.2f}%")

    # --- Phase 6: Redundancy scaling evaluation ---
    print("\n" + "=" * 60)
    print("  Phase 6: Redundancy Scaling Evaluation (1-10 copies)")
    print("=" * 60)

    redundancy_rows = run_redundancy_scaling(
        config, last_predictive_agent, all_seeds,
        copy_counts=[1, 2, 3, 4, 5, 7, 10],
    )
    redundancy_summary = aggregate_redundancy_scaling(redundancy_rows)

    for entry in redundancy_summary:
        print(f"  Length {entry['seq_length']:>4}  {entry['number_of_copies']:>2} copies: "
              f"{entry['decode_accuracy_mean']:.2f}% ± {entry['decode_accuracy_std']:.2f}%")

    # --- Phase 7: Statistics ---
    print("\nComputing statistics...")
    statistics = compute_statistics(all_rows, config)
    ablation_statistics = compute_statistics(all_ablation_rows, config)

    # --- Phase 8: Export ---
    csv_path = config.results_path(config.csv_filename)
    statistics_path = config.results_path(config.statistics_filename)
    ablation_path = config.results_path("ablation_results.csv")
    redundancy_path = config.results_path("redundancy_scaling.csv")
    redundancy_raw_path = config.results_path("redundancy_scaling_raw.csv")
    export_results_csv(all_rows, csv_path)
    export_statistics_csv(statistics, statistics_path)
    export_statistics_csv(ablation_statistics, ablation_path)
    export_redundancy_csv(redundancy_summary, redundancy_path)
    export_redundancy_csv(redundancy_rows, redundancy_raw_path)
    print(f"\nResults CSV:          {csv_path}")
    print(f"Statistics CSV:       {statistics_path}")
    print(f"Ablation CSV:         {ablation_path}")
    print(f"Redundancy CSV:       {redundancy_path}")
    print(f"Redundancy Raw CSV:   {redundancy_raw_path}")

    # --- Phase 9: Plots ---
    generate_all_plots(
        epoch_rewards=last_epoch_rewards,
        rolling_rewards=last_rolling_rewards,
        smoothed_rewards=last_smoothed_rewards,
        predictor_losses=last_predictor_losses,
        statistics_summary=statistics,
        all_rows=all_rows,
        config=config,
        ablation_statistics=ablation_statistics,
        validation_rewards=last_validation_rewards,
        redundancy_summary=redundancy_summary,
    )

    # --- Phase 10: Final summary ---
    print_summary_table(statistics, title="Main Results")
    print_summary_table(ablation_statistics, title="Ablation Results")

    print(f"\n{'=' * 60}")
    print("  Redundancy Scaling Summary")
    print(f"{'=' * 60}")
    for entry in redundancy_summary:
        marker = " ★" if entry["decode_accuracy_mean"] >= 99.99 else ""
        print(f"  Length {entry['seq_length']:>4}  {entry['number_of_copies']:>2} copies: "
              f"{entry['decode_accuracy_mean']:>7.2f}% ± {entry['decode_accuracy_std']:.2f}%{marker}")

    print("\nExperiment complete. All outputs saved to:", config.output_directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
