"""
Local pipeline — runs AFTER downloading trained models from Google Colab.

This script:
  1. Loads trained model weights from Colab export
  2. Runs full evaluation (baseline, random, RL, predictive RL)
  3. Runs multi-copy redundancy comparison
  4. Tests on unseen data (overfitting check)
  5. Generates all plots (publication quality)
  6. Builds the policy lookup table for the website
  7. Exports website/demo_data.json
  8. Prints summary tables for the research paper

Usage:
    py run_full_pipeline.py --input results/colab_export.json

Prerequisites:
    - Trained model weights in results/checkpoints/ (from Colab)
    - colab_export.json in results/ (from Colab)
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

from config import Config
from encoding import (
    ALL_MAPPINGS, BASELINE_MAPPING, decode_action,
    generate_biased_binary, encode_binary_to_dna,
)
from rl_agent import PPOAgent
from rl_environment import DNAEncodingEnvironment, _compute_dibit_distribution
from error_predictor import ErrorPredictor
from evaluation import (
    run_comparison, run_ablation_comparison,
    compute_statistics, export_results_csv, export_statistics_csv,
    generate_frozen_dataset, evaluate_reinforcement_learning, _aggregate,
    run_redundancy_scaling, aggregate_redundancy_scaling, export_redundancy_csv,
)
from redundancy import evaluate_redundancy_comparison
from visualization import generate_all_plots
from sequence_metrics import gc_content, max_homopolymer, shannon_entropy


def load_agent(checkpoint_path: str, config: Config) -> PPOAgent:
    """Load a trained PPO agent from a checkpoint file."""
    agent = PPOAgent(
        DNAEncodingEnvironment.STATE_DIMENSION,
        DNAEncodingEnvironment.ACTION_DIMENSION,
        config,
    )
    agent.load(checkpoint_path)
    return agent


def build_policy_table(agent: PPOAgent) -> list:
    """Build a policy lookup table mapping data distributions to actions."""
    policy_table = []
    for dibit_00 in np.linspace(0.1, 0.5, 5):
        for dibit_01 in np.linspace(0.1, 0.5, 5):
            dibit_10 = max(0.05, 1.0 - dibit_00 - dibit_01 - 0.25)
            dibit_11 = max(0.05, 1.0 - dibit_00 - dibit_01 - dibit_10)
            total = dibit_00 + dibit_01 + dibit_10 + dibit_11
            state = np.array([
                dibit_00 / total, dibit_01 / total,
                dibit_10 / total, dibit_11 / total,
                max(dibit_00, dibit_01, dibit_10, dibit_11) / total,
                0.5,
            ], dtype=np.float32)
            action, _, _ = agent.select_action(state)
            primary_index, secondary_index = decode_action(action)
            policy_table.append({
                "dibit_distribution": [
                    float(state[0]), float(state[1]),
                    float(state[2]), float(state[3]),
                ],
                "selected_action": int(action),
                "primary_mapping_index": int(primary_index),
                "secondary_mapping_index": int(secondary_index),
            })
    return policy_table


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline using Colab-trained models")
    parser.add_argument("--input", required=True, help="Path to colab_export.json")
    arguments = parser.parse_args()

    # Load Colab export
    print(f"Loading Colab export from: {arguments.input}")
    with open(arguments.input, "r") as file:
        colab_export = json.load(file)

    metadata = colab_export["metadata"]
    print(f"  Training epochs: {metadata['total_training_epochs']}")
    print(f"  Number of seeds: {metadata['number_of_seeds']}")
    print(f"  Training time: {metadata.get('total_training_time_minutes', 'unknown')} minutes")

    config = Config(
        number_of_copies=metadata.get("number_of_copies", 3),
        ppo_training_epochs=metadata.get("total_training_epochs", 100000),
        ppo_hidden_dimension=metadata.get("ppo_hidden_dimension", 64),
        dropout_rate=metadata.get("dropout_rate", 0.1),
    )
    os.makedirs(config.output_directory, exist_ok=True)

    seeds = metadata.get("seeds", [config.seed + i for i in range(metadata["number_of_seeds"])])
    checkpoint_directory = os.path.join(config.output_directory, "checkpoints")

    # ================================================================
    # Phase 1: Load all trained models
    # ================================================================
    print("\n" + "=" * 60)
    print("  Phase 1: Loading trained models")
    print("=" * 60)

    all_rows = []
    all_ablation_rows = []

    for seed_index, seed in enumerate(seeds):
        print(f"\n  Seed {seed_index + 1}/{len(seeds)} (seed={seed})")

        # Load agents
        agent_reinforcement_learning = load_agent(
            os.path.join(checkpoint_directory, f"reinforcement_learning_seed{seed}.pt"), config)
        agent_predictive = load_agent(
            os.path.join(checkpoint_directory, f"predictive_seed{seed}.pt"), config)
        agent_no_gc = load_agent(
            os.path.join(checkpoint_directory, f"ablation_no_gc_seed{seed}.pt"), config)
        agent_no_homopolymer = load_agent(
            os.path.join(checkpoint_directory, f"ablation_no_homopolymer_seed{seed}.pt"), config)

        # Load predictor if available
        predictor = None
        predictor_path = os.path.join(checkpoint_directory, f"predictor_seed{seed}.pt")
        if os.path.exists(predictor_path):
            predictor = ErrorPredictor(config)
            predictor.load(predictor_path)

        print("    All models loaded.")

        # ================================================================
        # Phase 2: Evaluate all methods
        # ================================================================
        print("    Evaluating all methods...")
        rows = run_comparison(
            config, agent_reinforcement_learning, agent_predictive, predictor, seed)
        all_rows.extend(rows)

        for row in rows:
            print(f"      {row['method']:<45} length={row['seq_length']:>3}  "
                  f"error_rate={row['average_error_rate']:.4f}  "
                  f"accuracy={row['decode_accuracy']:.2f}%")

        # ================================================================
        # Phase 3: Ablation comparison
        # ================================================================
        print("    Running ablation comparison...")
        ablation_agents = {
            "predictive_reinforcement_learning": agent_predictive,
            "reinforcement_learning": agent_reinforcement_learning,
            "reinforcement_learning_no_gc": agent_no_gc,
            "reinforcement_learning_no_homopolymer": agent_no_homopolymer,
        }
        ablation_rows = run_ablation_comparison(config, ablation_agents, seed)
        all_ablation_rows.extend(ablation_rows)

    # ================================================================
    # Phase 4: Overfitting check
    # ================================================================
    print("\n" + "=" * 60)
    print("  Phase 4: Overfitting Check (Unseen Data)")
    print("=" * 60)

    test_seed = 99999
    unseen_results = {}
    last_seed = seeds[-1]
    agent_for_test = load_agent(
        os.path.join(checkpoint_directory, f"predictive_seed{last_seed}.pt"), config)

    for sequence_length in config.sequence_lengths:
        test_dataset = generate_frozen_dataset(sequence_length, 200, test_seed)
        test_results = _aggregate(evaluate_reinforcement_learning(
            test_dataset, agent_for_test, config, sequence_length,
            np.random.default_rng(test_seed + sequence_length)
        ))
        unseen_results[str(sequence_length)] = {
            key: float(value) if isinstance(value, (np.floating, float)) else value
            for key, value in test_results.items()
        }
        print(f"  Length {sequence_length}: "
              f"error_rate={test_results['average_error_rate']:.4f}  "
              f"decode_accuracy={test_results['decode_accuracy']:.2f}%")

    # Compare with training data
    for sequence_length in config.sequence_lengths:
        training_row = next(
            (row for row in all_rows
             if row["method"] == "predictive_reinforcement_learning"
             and row["seq_length"] == sequence_length), None)
        if training_row:
            gap = training_row["decode_accuracy"] - unseen_results[str(sequence_length)]["decode_accuracy"]
            status = "OK" if abs(gap) < 5.0 else "WARNING: possible overfitting"
            print(f"  Length {sequence_length}: train={training_row['decode_accuracy']:.2f}%  "
                  f"test={unseen_results[str(sequence_length)]['decode_accuracy']:.2f}%  "
                  f"gap={gap:.2f}%  [{status}]")

    # ================================================================
    # Phase 5: Multi-copy redundancy comparison
    # ================================================================
    print("\n" + "=" * 60)
    print("  Phase 5: Multi-Copy Redundancy Comparison")
    print("=" * 60)

    redundancy_comparison = {}
    for sequence_length in config.sequence_lengths:
        binary_data = generate_frozen_dataset(sequence_length, 1, 42)[0]
        dibit_distribution = _compute_dibit_distribution(binary_data)
        state = np.array([
            dibit_distribution[0], dibit_distribution[1],
            dibit_distribution[2], dibit_distribution[3],
            float(np.max(dibit_distribution)),
            sequence_length / 256.0,
        ], dtype=np.float32)
        action, _, _ = agent_for_test.select_action(state)
        comparison = evaluate_redundancy_comparison(
            binary_data, action, np.random.default_rng(12345), config, [1, 2, 3, 4, 5]
        )
        redundancy_comparison[str(sequence_length)] = comparison
        print(f"  Length {sequence_length}:")
        for entry in comparison:
            print(f"    {entry['number_of_copies']} copies: "
                  f"{entry['redundancy_accuracy']*100:.2f}% accuracy")

    # ================================================================
    # Phase 5.5: Systematic Redundancy Scaling (1-10 copies)
    # ================================================================
    print("\n" + "=" * 60)
    print("  Phase 5.5: Systematic Redundancy Scaling (1-10 copies)")
    print("=" * 60)

    redundancy_rows = run_redundancy_scaling(
        config, agent_for_test, seeds,
        copy_counts=[1, 2, 3, 4, 5, 7, 10],
    )
    redundancy_summary = aggregate_redundancy_scaling(redundancy_rows)

    for entry in redundancy_summary:
        marker = " ★" if entry["decode_accuracy_mean"] >= 99.99 else ""
        print(f"  Length {entry['seq_length']:>4}  {entry['number_of_copies']:>2} copies: "
              f"{entry['decode_accuracy_mean']:>7.2f}% ± {entry['decode_accuracy_std']:.2f}%{marker}")

    # ================================================================
    # Phase 6: Compute statistics
    # ================================================================
    print("\n" + "=" * 60)
    print("  Phase 6: Computing Statistics")
    print("=" * 60)

    statistics = compute_statistics(all_rows, config)
    ablation_statistics = compute_statistics(all_ablation_rows, config)

    # Export CSVs
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
    print(f"  Results CSV:          {csv_path}")
    print(f"  Statistics CSV:       {statistics_path}")
    print(f"  Ablation CSV:         {ablation_path}")
    print(f"  Redundancy CSV:       {redundancy_path}")
    print(f"  Redundancy Raw CSV:   {redundancy_raw_path}")

    # ================================================================
    # Phase 7: Generate plots
    # ================================================================
    print("\n" + "=" * 60)
    print("  Phase 7: Generating Plots")
    print("=" * 60)

    training_curves = colab_export.get("training_curves", {})
    # Find predictive training curves
    last_training_data = None
    for key in training_curves:
        if "predictive" in key:
            last_training_data = training_curves[key]
    if last_training_data is None and training_curves:
        last_training_data = list(training_curves.values())[-1]

    if last_training_data:
        generate_all_plots(
            epoch_rewards=last_training_data.get("epoch_rewards", []),
            rolling_rewards=last_training_data.get("rolling_rewards", []),
            smoothed_rewards=[],
            predictor_losses=last_training_data.get("predictor_losses", []),
            statistics_summary=statistics,
            all_rows=all_rows,
            config=config,
            ablation_statistics=ablation_statistics,
            validation_rewards=last_training_data.get("validation_rewards", []),
            redundancy_summary=redundancy_summary,
        )

    # ================================================================
    # Phase 8: Build website demo data
    # ================================================================
    print("\n" + "=" * 60)
    print("  Phase 8: Building Website Data")
    print("=" * 60)

    policy_table = build_policy_table(agent_for_test)

    encoding_mappings = {}
    for index, mapping in enumerate(ALL_MAPPINGS):
        encoding_mappings[str(index)] = {
            f"{dibit[0]}{dibit[1]}": nucleotide
            for dibit, nucleotide in mapping.items()
        }

    demo_data = {
        "metadata": metadata,
        "policy_lookup_table": policy_table,
        "encoding_mappings": encoding_mappings,
        "training_curves": training_curves,
        "main_results": [
            {key: (float(value) if isinstance(value, (np.floating, float)) else value)
             for key, value in stat.items()}
            for stat in statistics
        ],
        "ablation_results": [
            {key: (float(value) if isinstance(value, (np.floating, float)) else value)
             for key, value in stat.items()}
            for stat in ablation_statistics
        ],
        "redundancy_comparison": redundancy_comparison,
        "redundancy_scaling": [
            {key: (float(value) if isinstance(value, (np.floating, float)) else value)
             for key, value in entry.items()}
            for entry in redundancy_summary
        ],
        "unseen_test_results": unseen_results,
    }

    demo_path = os.path.join("website", "demo_data.json")
    os.makedirs("website", exist_ok=True)
    with open(demo_path, "w") as file:
        json.dump(demo_data, file, indent=2)
    print(f"  Website demo data: {demo_path} ({os.path.getsize(demo_path)/1024:.1f} KB)")

    # ================================================================
    # Phase 9: Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<50} {'Length':>6}  {'Error Rate':>12}  {'Accuracy':>10}")
    print("-" * 80)
    for stat in statistics:
        print(f"  {stat['method']:<48} {stat['seq_length']:>6}  "
              f"{stat.get('average_error_rate_mean', 0):>12.4f}  "
              f"{stat.get('decode_accuracy_mean', 0):>9.2f}%")

    print(f"\n  Redundancy improvement:")
    for sequence_length, comparisons in redundancy_comparison.items():
        single = comparisons[0]["redundancy_accuracy"] * 100
        triple = next((c["redundancy_accuracy"]*100 for c in comparisons if c["number_of_copies"]==3), single)
        print(f"    Length {sequence_length}: 1 copy = {single:.2f}%  ->  3 copies = {triple:.2f}%  "
              f"(+{triple-single:.2f}%)")

    print("\n" + "=" * 70)
    print("  DONE! Open website/index.html to see the full demo.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
