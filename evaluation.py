"""
Evaluate and compare encoding strategies with statistical validation.

Four methods are compared:
    1. Fixed baseline  (canonical 00->A, 01->C, 10->G, 11->T)
    2. Random encoding (random mapping permutation each trial)
    3. Reinforcement learning encoding (PPO agent, no predictor)
    4. Predictive reinforcement learning (PPO agent + error predictor in reward)

Additionally, multi-copy redundancy is evaluated for each method.

Evaluation uses frozen datasets: the same binary data is used
across all four methods for fair comparison. Data is generated once
per (seed, sequence_length) pair and reused across all methods.
"""

import csv
import os
import numpy as np
from typing import Dict, List, Optional
from scipy import stats as scipy_stats

from training_pipeline_v3 import _compute_block_state
from encoding import get_block_splits, decode_block_sequence
import torch
from config import Config
from encoding import (
    BASELINE_MAPPING, BASELINE_MAPPING_INDEX, ALL_MAPPINGS,
    generate_biased_binary, encode_binary_to_dna, encode_with_action,
    decode_with_action, decode_action,
    NUMBER_OF_ACTIONS, NUMBER_OF_MAPPINGS,
    DATA_PROFILES,
)
from error_simulation import simulate_errors, compute_error_rate
from decoder import decode_dna_to_binary, compute_bit_accuracy
from sequence_metrics import gc_content, stability_score, normalized_max_homopolymer
from rl_agent_v3 import PPOAgentV3 as PPOAgent
from error_predictor import ErrorPredictor
from redundancy import encode_with_redundancy


# frozen dataset generation

def generate_frozen_dataset(
    sequence_length: int, number_of_trials: int, seed: int,
) -> List[np.ndarray]:
    """
    Generate a fixed set of binary data samples for evaluation.

    Uses a mix of all three data profiles (high_gc, low_gc, balanced)
    with a deterministic random number generator seeded per (seed, sequence_length).
    The same dataset is reused for all methods to eliminate data variance.
    """
    # unique deterministic seed per (seed, sequence_length) pair
    evaluation_random_generator = np.random.default_rng(seed * 10000 + sequence_length)
    number_of_bits = sequence_length * 2
    dataset: List[np.ndarray] = []

    for trial_index in range(number_of_trials):
        # cycle through profiles so each gets equal representation
        profile_function = DATA_PROFILES[trial_index % len(DATA_PROFILES)]
        dataset.append(profile_function(number_of_bits, evaluation_random_generator))

    return dataset


# single-trial helpers

def _evaluate_baseline_trial(
    binary_bits: np.ndarray, config: Config, random_generator: np.random.Generator,
) -> Dict[str, float]:
    """Encode with fixed baseline, simulate errors, decode, return metrics."""
    dna_sequence = encode_binary_to_dna(binary_bits, BASELINE_MAPPING)
    mutated_sequence, _ = simulate_errors(dna_sequence, config.substitution_rate,
                                          config.insertion_rate, config.deletion_rate, random_generator)
    decoded_bits = decode_dna_to_binary(mutated_sequence, BASELINE_MAPPING)
    return _compute_metrics(binary_bits, dna_sequence, mutated_sequence, decoded_bits)


def _evaluate_random_trial(
    binary_bits: np.ndarray, config: Config, random_generator: np.random.Generator,
) -> Dict[str, float]:
    """Encode with a random mapping, simulate errors, decode."""
    mapping_index = random_generator.integers(0, NUMBER_OF_MAPPINGS)
    mapping = ALL_MAPPINGS[mapping_index]
    dna_sequence = encode_binary_to_dna(binary_bits, mapping)
    mutated_sequence, _ = simulate_errors(dna_sequence, config.substitution_rate,
                                          config.insertion_rate, config.deletion_rate, random_generator)
    decoded_bits = decode_dna_to_binary(mutated_sequence, mapping)
    return _compute_metrics(binary_bits, dna_sequence, mutated_sequence, decoded_bits)


def _evaluate_reinforcement_learning_trial(
    binary_bits: np.ndarray, config: Config, random_generator: np.random.Generator,
    agent: PPOAgent, sequence_length: int,
) -> Dict[str, float]:
    splits = get_block_splits(len(binary_bits), config)
    N = len(splits)
    block_actions = []
    dna_so_far = ""
    actor_hx, critic_hx = None, None
    agent.network.eval()
    with torch.no_grad():
        for k in range(N):
            state = _compute_block_state(binary_bits, splits, k, N, block_actions, dna_so_far)
            state_t = torch.tensor(state, dtype=torch.float32).to(agent.device)
            logits, val, actor_hx, critic_hx = agent.network(state_t, actor_hx, critic_hx)
            action_int = torch.argmax(logits).item()
            
            b_start, b_end = splits[k]
            block_bits = binary_bits[b_start:b_end]
            mapping_idx = action_int
            if config.use_header_free_encoding and k > 0:
                prev_s, prev_e = splits[k - 1]
                offset = int(np.sum(binary_bits[prev_s:prev_e])) % 24
                mapping_idx = (mapping_idx + offset) % 24
            block_dna = encode_binary_to_dna(block_bits, ALL_MAPPINGS[mapping_idx])
            dna_so_far += block_dna
            block_actions.append(action_int)

    mutated_sequence, _ = simulate_errors(dna_so_far, config.substitution_rate,
                                          config.insertion_rate, config.deletion_rate, random_generator)
    decoded_bits = decode_block_sequence(mutated_sequence, block_actions, config)
    return _compute_metrics(binary_bits, dna_so_far, mutated_sequence, decoded_bits)



def _evaluate_redundancy_trial(
    binary_bits: np.ndarray, config: Config, random_generator: np.random.Generator,
    agent: PPOAgent, sequence_length: int,
) -> Dict[str, float]:
    result = encode_with_redundancy(binary_bits, None, random_generator, config, agent=agent)
    return {
        "bit_accuracy": result["redundancy_accuracy"],
        "error_rate": result["consensus_error_rate"],
        "gc_content": gc_content(result["original_dna"]),
        "stability": stability_score(result["original_dna"]),
        "homopolymer": normalized_max_homopolymer(result["original_dna"]),
        "single_copy_accuracy": result["single_copy_accuracy"],
        "redundancy_accuracy": result["redundancy_accuracy"],
        "number_of_copies": result["number_of_copies"],
    }



def _compute_metrics(original_bits, dna_sequence, mutated_sequence, decoded_bits) -> Dict[str, float]:
    bit_accuracy = compute_bit_accuracy(original_bits, decoded_bits)
    error_rate = compute_error_rate(dna_sequence, mutated_sequence)
    gc_value = gc_content(dna_sequence)
    stability = stability_score(dna_sequence)
    homopolymer = normalized_max_homopolymer(dna_sequence)
    return {
        "bit_accuracy": bit_accuracy,
        "error_rate": error_rate,
        "gc_content": gc_value,
        "stability": stability,
        "homopolymer": homopolymer,
    }


# method evaluators (using frozen datasets)

def evaluate_baseline(dataset: List[np.ndarray], config: Config,
                      random_generator: np.random.Generator) -> List[Dict[str, float]]:
    return [_evaluate_baseline_trial(binary_bits, config, random_generator) for binary_bits in dataset]


def evaluate_random(dataset: List[np.ndarray], config: Config,
                    random_generator: np.random.Generator) -> List[Dict[str, float]]:
    return [_evaluate_random_trial(binary_bits, config, random_generator) for binary_bits in dataset]


def evaluate_reinforcement_learning(dataset: List[np.ndarray], agent: PPOAgent, config: Config,
                                    sequence_length: int, random_generator: np.random.Generator) -> List[Dict[str, float]]:
    return [_evaluate_reinforcement_learning_trial(binary_bits, config, random_generator, agent, sequence_length) for binary_bits in dataset]


def evaluate_redundancy(dataset: List[np.ndarray], agent: PPOAgent, config: Config,
                        sequence_length: int, random_generator: np.random.Generator) -> List[Dict[str, float]]:
    return [_evaluate_redundancy_trial(binary_bits, config, random_generator, agent, sequence_length) for binary_bits in dataset]


# aggregation

def _aggregate(results: List[Dict[str, float]]) -> Dict[str, float]:
    keys = results[0].keys()
    aggregated = {}
    for key in keys:
        values = [result[key] for result in results]
        aggregated[f"average_{key}"] = float(np.mean(values))
    aggregated["decode_accuracy"] = aggregated["average_bit_accuracy"] * 100.0
    aggregated["gc_score"] = 1.0 - 2.0 * abs(aggregated["average_gc_content"] - 0.5)
    aggregated["stability_score"] = aggregated["average_stability"]
    aggregated["encoding_efficiency"] = aggregated["average_bit_accuracy"] * (1.0 - aggregated["average_error_rate"])
    return aggregated


# full comparison run

def run_comparison(
    config: Config,
    agent_reinforcement_learning: PPOAgent,
    agent_predictive: Optional[PPOAgent],
    predictor: Optional[ErrorPredictor],
    seed: int,
) -> List[Dict]:
    rows = []

    for sequence_length in config.sequence_lengths:
        # generate frozen dataset once, reused across ALL methods
        dataset = generate_frozen_dataset(
            sequence_length, config.trials_per_length, seed
        )

        # 1. baseline
        baseline_results = _aggregate(evaluate_baseline(
            dataset, config, np.random.default_rng(seed + 2000 + sequence_length)))
        baseline_results.update({"seq_length": sequence_length, "method": "baseline", "seed": seed})
        rows.append(baseline_results)

        # 2. random
        random_results = _aggregate(evaluate_random(
            dataset, config, np.random.default_rng(seed + 3000 + sequence_length)))
        random_results.update({"seq_length": sequence_length, "method": "random", "seed": seed})
        rows.append(random_results)

        # 3. reinforcement learning (no predictor)
        reinforcement_learning_results = _aggregate(evaluate_reinforcement_learning(
            dataset, agent_reinforcement_learning, config, sequence_length,
            np.random.default_rng(seed + 4000 + sequence_length)))
        reinforcement_learning_results.update({"seq_length": sequence_length, "method": "reinforcement_learning", "seed": seed})
        rows.append(reinforcement_learning_results)

        # 4. predictive reinforcement learning
        if agent_predictive is not None:
            predictive_results = _aggregate(evaluate_reinforcement_learning(
                dataset, agent_predictive, config, sequence_length,
                np.random.default_rng(seed + 5000 + sequence_length)))
            predictive_results.update({"seq_length": sequence_length, "method": "predictive_reinforcement_learning", "seed": seed})
            rows.append(predictive_results)

            # 5. predictive reinforcement learning with redundancy
            redundancy_results = _aggregate(evaluate_redundancy(
                dataset, agent_predictive, config, sequence_length,
                np.random.default_rng(seed + 6000 + sequence_length)))
            redundancy_results.update({"seq_length": sequence_length, "method": "predictive_reinforcement_learning_redundancy", "seed": seed})
            rows.append(redundancy_results)

    return rows


# ablation comparison

def run_ablation_comparison(
    config: Config,
    agents: Dict[str, PPOAgent],
    seed: int,
) -> List[Dict]:
    """
    Evaluate ablation conditions on the same frozen datasets.

    agents: dictionary mapping ablation name -> trained PPOAgent.
    Expected keys: "predictive_reinforcement_learning", "reinforcement_learning",
    "reinforcement_learning_no_gc", "reinforcement_learning_no_homopolymer".
    The baseline is computed without any agent.
    """
    rows = []

    for sequence_length in config.sequence_lengths:
        dataset = generate_frozen_dataset(
            sequence_length, config.trials_per_length, seed
        )

        # fixed baseline (no agent)
        baseline_results = _aggregate(evaluate_baseline(
            dataset, config, np.random.default_rng(seed + 2000 + sequence_length)
        ))
        baseline_results.update({"seq_length": sequence_length, "method": "baseline", "seed": seed})
        rows.append(baseline_results)

        # each ablation agent
        for random_offset, (name, agent) in enumerate(agents.items(), start=4000):
            result = _aggregate(evaluate_reinforcement_learning(
                dataset, agent, config, sequence_length,
                np.random.default_rng(seed + random_offset + sequence_length)
            ))
            result.update({"seq_length": sequence_length, "method": name, "seed": seed})
            rows.append(result)

    return rows


# statistical summary

def compute_statistics(all_rows: List[Dict], config: Config) -> List[Dict]:
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in all_rows:
        key = (row["method"], row["seq_length"])
        grouped[key].append(row)

    summary_rows = []
    metric_keys = ["average_error_rate", "decode_accuracy", "gc_score",
                    "stability_score", "encoding_efficiency", "average_bit_accuracy"]

    for (method, sequence_length), rows in sorted(grouped.items()):
        summary = {"method": method, "seq_length": sequence_length}
        for metric_key in metric_keys:
            values = [row[metric_key] for row in rows if metric_key in row]
            summary[f"{metric_key}_mean"] = float(np.mean(values)) if values else 0.0
            summary[f"{metric_key}_std"] = float(np.std(values)) if values else 0.0
        summary_rows.append(summary)

    # t-tests: baseline versus reinforcement learning, baseline versus predictive, reinforcement learning versus predictive
    ttest_pairs = [
        ("baseline", "reinforcement_learning",
         "t_statistic_versus_baseline", "p_value_versus_baseline"),
        ("baseline", "predictive_reinforcement_learning",
         "t_statistic_versus_baseline", "p_value_versus_baseline"),
        ("reinforcement_learning", "predictive_reinforcement_learning",
         "t_statistic_versus_reinforcement_learning", "p_value_versus_reinforcement_learning"),
    ]

    for sequence_length in config.sequence_lengths:
        for method_a, method_b, t_column, p_column in ttest_pairs:
            key_a = (method_a, sequence_length)
            key_b = (method_b, sequence_length)
            if key_a in grouped and key_b in grouped:
                error_rates_a = [row["average_error_rate"] for row in grouped[key_a]]
                error_rates_b = [row["average_error_rate"] for row in grouped[key_b]]
                if len(error_rates_a) >= 2 and len(error_rates_b) >= 2:
                    t_statistic, p_value = scipy_stats.ttest_ind(error_rates_a, error_rates_b)
                    for summary in summary_rows:
                        if summary["method"] == method_b and summary["seq_length"] == sequence_length:
                            summary[t_column] = float(t_statistic)
                            summary[p_column] = float(p_value)

    return summary_rows


# CSV export

def export_results_csv(rows: List[Dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_statistics_csv(rows: List[Dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# redundancy scaling evaluation

def run_redundancy_scaling(
    config: Config,
    agent: PPOAgent,
    seeds: list,
    copy_counts: list = None,
) -> List[Dict]:
    """
    Systematically evaluate decode accuracy across different numbers of
    redundant copies, averaged over multiple seeds and trials.

    Returns a list of dictionaries with columns:
        seq_length, number_of_copies, decode_accuracy_mean, decode_accuracy_std,
        error_rate_mean, error_rate_std, seed (per-seed rows for aggregation)
    """
    from rl_environment import _compute_dibit_distribution

    if copy_counts is None:
        copy_counts = [1, 2, 3, 4, 5, 7, 10]

    all_rows = []

    for seed in seeds:
        for sequence_length in config.sequence_lengths:
            dataset = generate_frozen_dataset(
                sequence_length, config.trials_per_length, seed
            )

            for n_copies in copy_counts:
                test_config = Config(
                    number_of_copies=n_copies,
                    substitution_rate=config.substitution_rate,
                    insertion_rate=config.insertion_rate,
                    deletion_rate=config.deletion_rate,
                )

                trial_accuracies = []
                trial_error_rates = []
                random_generator = np.random.default_rng(
                    seed * 100000 + sequence_length * 100 + n_copies
                )

                for binary_bits in dataset:
                    result = encode_with_redundancy(
                        binary_bits, None, random_generator, test_config, agent=agent
                    )
                    trial_accuracies.append(result["redundancy_accuracy"])
                    trial_error_rates.append(result["consensus_error_rate"])

                all_rows.append({
                    "seed": seed,
                    "seq_length": sequence_length,
                    "number_of_copies": n_copies,
                    "decode_accuracy": float(np.mean(trial_accuracies)) * 100.0,
                    "error_rate": float(np.mean(trial_error_rates)),
                })

    return all_rows


def aggregate_redundancy_scaling(rows: List[Dict]) -> List[Dict]:
    """Aggregate per-seed redundancy scaling rows into mean±std summaries."""
    from collections import defaultdict

    grouped = defaultdict(list)
    for row in rows:
        key = (row["seq_length"], row["number_of_copies"])
        grouped[key].append(row)

    summary = []
    for (seq_length, n_copies), seed_rows in sorted(grouped.items()):
        accuracies = [r["decode_accuracy"] for r in seed_rows]
        error_rates = [r["error_rate"] for r in seed_rows]
        summary.append({
            "seq_length": seq_length,
            "number_of_copies": n_copies,
            "decode_accuracy_mean": float(np.mean(accuracies)),
            "decode_accuracy_std": float(np.std(accuracies)),
            "error_rate_mean": float(np.mean(error_rates)),
            "error_rate_std": float(np.std(error_rates)),
            "num_seeds": len(seed_rows),
        })

    return summary


def export_redundancy_csv(rows: List[Dict], path: str):
    """Export redundancy scaling results to CSV."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fieldnames = sorted(rows[0].keys())
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
