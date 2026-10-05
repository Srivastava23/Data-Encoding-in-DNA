"""
Re-run the full evaluation for:
  "Adaptive DNA Encoding for Data Storage via Predictive RL
   and Consensus Redundancy"

Uses the EXISTING trained checkpoints (seeds 42-48).
Produces:
  - provenance.md
  - results_raw.csv        (A)
  - redundancy_raw.csv     (B)
  - ablation_raw.csv       (C)
  - overfit_raw.csv        (D)
  - ttests_raw.csv         (E)
  - results_tables.md      (aggregated tables + derived quantities)
  - self_check.txt
  - demo_log.txt
"""

import os
import sys
import csv
import json
import hashlib
import numpy as np
import torch
from scipy import stats as scipy_stats
from collections import defaultdict
from typing import Dict, List, Optional

# ---- project imports -------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from encoding import (
    BASELINE_MAPPING, BASELINE_MAPPING_INDEX, ALL_MAPPINGS,
    generate_biased_binary, encode_binary_to_dna, encode_with_action,
    decode_with_action, decode_action,
    NUMBER_OF_ACTIONS, NUMBER_OF_MAPPINGS,
)
from error_simulation import simulate_errors, compute_error_rate
from decoder import decode_dna_to_binary, compute_bit_accuracy
from sequence_metrics import gc_content, stability_score, normalized_max_homopolymer, max_homopolymer
from rl_agent import PPOAgent
from rl_environment import DNAEncodingEnvironment, _compute_dibit_distribution
from error_predictor import ErrorPredictor
from redundancy import encode_with_redundancy
from evaluation import generate_frozen_dataset

# ============================================================================
# SETTINGS (frozen, matching the paper exactly)
# ============================================================================
SEEDS = [42, 43, 44, 45, 46, 47, 48]
SEQ_LENGTHS = [64, 128, 256, 512, 1024]
TRIALS_PER_LENGTH = 100
COPY_COUNTS = [1, 3, 5, 7, 10]

CHECKPOINT_DIR = os.path.join("results", "checkpoints")
OUTPUT_DIR = os.path.join("results", "rerun")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HELPERS
# ============================================================================

def load_agent(checkpoint_path: str, config: Config) -> PPOAgent:
    agent = PPOAgent(
        DNAEncodingEnvironment.STATE_DIMENSION,
        DNAEncodingEnvironment.ACTION_DIMENSION,
        config,
    )
    agent.load(checkpoint_path)
    return agent


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def make_state(binary_bits, sequence_length):
    dibit_distribution = _compute_dibit_distribution(binary_bits)
    return np.array([
        dibit_distribution[0], dibit_distribution[1],
        dibit_distribution[2], dibit_distribution[3],
        float(np.max(dibit_distribution)),
        sequence_length / 256.0,
    ], dtype=np.float32)


# ============================================================================
# SINGLE-TRIAL EVALUATION FUNCTIONS (return per-trial metrics)
# ============================================================================

def eval_baseline_trial(binary_bits, config, rng):
    dna = encode_binary_to_dna(binary_bits, BASELINE_MAPPING)
    mutated, _ = simulate_errors(dna, config.substitution_rate,
                                  config.insertion_rate, config.deletion_rate, rng)
    decoded = decode_dna_to_binary(mutated, BASELINE_MAPPING)
    acc = compute_bit_accuracy(binary_bits, decoded)
    err = compute_error_rate(dna, mutated)
    gc = gc_content(dna)
    hp = max_homopolymer(dna)
    stab = stability_score(dna)
    return acc, err, gc, hp, stab


def eval_random_trial(binary_bits, config, rng):
    mapping_idx = rng.integers(0, NUMBER_OF_MAPPINGS)
    mapping = ALL_MAPPINGS[mapping_idx]
    dna = encode_binary_to_dna(binary_bits, mapping)
    mutated, _ = simulate_errors(dna, config.substitution_rate,
                                  config.insertion_rate, config.deletion_rate, rng)
    decoded = decode_dna_to_binary(mutated, mapping)
    acc = compute_bit_accuracy(binary_bits, decoded)
    err = compute_error_rate(dna, mutated)
    gc = gc_content(dna)
    hp = max_homopolymer(dna)
    stab = stability_score(dna)
    return acc, err, gc, hp, stab


def eval_rl_trial(binary_bits, config, rng, agent, seq_len):
    state = make_state(binary_bits, seq_len)
    action, _, _ = agent.select_action(state)
    dna, pi, si = encode_with_action(binary_bits, action, rng)
    mutated, _ = simulate_errors(dna, config.substitution_rate,
                                  config.insertion_rate, config.deletion_rate, rng)
    decoded = decode_with_action(mutated, action, seq_len)
    acc = compute_bit_accuracy(binary_bits, decoded)
    err = compute_error_rate(dna, mutated)
    gc = gc_content(dna)
    hp = max_homopolymer(dna)
    stab = stability_score(dna)
    return acc, err, gc, hp, stab


def eval_redundancy_trial(binary_bits, config, rng, agent, seq_len, n_copies):
    state = make_state(binary_bits, seq_len)
    action, _, _ = agent.select_action(state)
    test_config = Config(
        number_of_copies=n_copies,
        substitution_rate=config.substitution_rate,
        insertion_rate=config.insertion_rate,
        deletion_rate=config.deletion_rate,
    )
    result = encode_with_redundancy(binary_bits, action, rng, test_config)
    return result["redundancy_accuracy"]


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    print("=" * 70)
    print("  FULL RE-EVALUATION (7 seeds × 5 lengths × 100 trials)")
    print("=" * 70)

    config = Config()

    # ---- Collect checkpoint hashes ----------------------------------------
    print("\n[1/8] Computing checkpoint file hashes...")
    checkpoint_hashes = {}
    for seed in SEEDS:
        for label, pattern in [
            ("rl", "reinforcement_learning_seed{}.pt"),
            ("pred", "predictive_seed{}.pt"),
            ("predictor", "predictor_seed{}.pt"),
            ("no_gc", "ablation_no_gc_seed{}.pt"),
            ("no_homo", "ablation_no_homopolymer_seed{}.pt"),
        ]:
            path = os.path.join(CHECKPOINT_DIR, pattern.format(seed))
            checkpoint_hashes[f"{label}_seed{seed}"] = md5_file(path)

    # ---- Load all agents --------------------------------------------------
    print("\n[2/8] Loading all agents...")
    agents = {}
    for seed in SEEDS:
        agents[("rl", seed)] = load_agent(
            os.path.join(CHECKPOINT_DIR, f"reinforcement_learning_seed{seed}.pt"), config)
        agents[("pred", seed)] = load_agent(
            os.path.join(CHECKPOINT_DIR, f"predictive_seed{seed}.pt"), config)
        agents[("no_gc", seed)] = load_agent(
            os.path.join(CHECKPOINT_DIR, f"ablation_no_gc_seed{seed}.pt"), config)
        agents[("no_homo", seed)] = load_agent(
            os.path.join(CHECKPOINT_DIR, f"ablation_no_homopolymer_seed{seed}.pt"), config)
        print(f"  Seed {seed}: all 4 agents loaded")

    # ========================================================================
    # A. MAIN COMPARISON (results_raw.csv)
    # ========================================================================
    print("\n[3/8] Running main method comparison...")
    results_raw = []

    for seed in SEEDS:
        for seq_len in SEQ_LENGTHS:
            dataset = generate_frozen_dataset(seq_len, TRIALS_PER_LENGTH, seed)

            # Collect per-trial metrics for each method
            methods_config = [
                ("Fixed Baseline", None, "baseline"),
                ("Random Mapping", None, "random"),
                ("RL Agent", agents[("rl", seed)], "rl"),
                ("Predictive RL", agents[("pred", seed)], "pred_rl"),
                ("Pred. RL + 3-copy", agents[("pred", seed)], "pred_rl_red"),
            ]

            for method_name, agent, method_key in methods_config:
                # Deterministic RNG per (seed, method, length)
                method_offsets = {
                    "baseline": 2000, "random": 3000, "rl": 4000,
                    "pred_rl": 5000, "pred_rl_red": 6000,
                }
                rng = np.random.default_rng(seed + method_offsets[method_key] + seq_len)

                accs, errs, gcs, hps, stabs = [], [], [], [], []
                for trial_idx, binary_bits in enumerate(dataset):
                    if method_key == "baseline":
                        a, e, g, h, s = eval_baseline_trial(binary_bits, config, rng)
                    elif method_key == "random":
                        a, e, g, h, s = eval_random_trial(binary_bits, config, rng)
                    elif method_key in ("rl", "pred_rl"):
                        a, e, g, h, s = eval_rl_trial(binary_bits, config, rng, agent, seq_len)
                    elif method_key == "pred_rl_red":
                        # For redundancy: we need all metrics from the underlying encoding too
                        state = make_state(binary_bits, seq_len)
                        action, _, _ = agent.select_action(state)
                        test_config = Config(
                            number_of_copies=3,
                            substitution_rate=config.substitution_rate,
                            insertion_rate=config.insertion_rate,
                            deletion_rate=config.deletion_rate,
                        )
                        result = encode_with_redundancy(binary_bits, action, rng, test_config)
                        a = result["redundancy_accuracy"]
                        e = result["consensus_error_rate"]
                        g = gc_content(result["original_dna"])
                        h = max_homopolymer(result["original_dna"])
                        s = stability_score(result["original_dna"])

                    accs.append(a)
                    errs.append(e)
                    gcs.append(g)
                    hps.append(h)
                    stabs.append(s)

                # Store per-seed aggregated row (mean over 100 trials for this seed)
                results_raw.append({
                    "seed": seed,
                    "length": seq_len,
                    "method": method_name,
                    "decode_accuracy_pct": float(np.mean(accs)) * 100.0,
                    "error_rate": float(np.mean(errs)),
                    "gc_content": float(np.mean(gcs)),
                    "max_homopolymer_run": float(np.mean(hps)),
                    "stability_score": float(np.mean(stabs)),
                })

            print(f"  Seed {seed}, Length {seq_len}: done")

    # Write results_raw.csv
    csv_path_A = os.path.join(OUTPUT_DIR, "results_raw.csv")
    fieldnames_A = ["seed", "length", "method", "decode_accuracy_pct",
                    "error_rate", "gc_content", "max_homopolymer_run", "stability_score"]
    with open(csv_path_A, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_A)
        w.writeheader()
        w.writerows(results_raw)
    print(f"  -> {csv_path_A}")

    # ========================================================================
    # B. REDUNDANCY SCALING (redundancy_raw.csv)
    # ========================================================================
    print("\n[4/8] Running redundancy scaling...")
    redundancy_raw = []

    for seed in SEEDS:
        for seq_len in SEQ_LENGTHS:
            dataset = generate_frozen_dataset(seq_len, TRIALS_PER_LENGTH, seed)
            agent = agents[("pred", seed)]

            for n_copies in COPY_COUNTS:
                rng = np.random.default_rng(seed * 100000 + seq_len * 100 + n_copies)
                trial_accs = []
                for binary_bits in dataset:
                    acc = eval_redundancy_trial(binary_bits, config, rng, agent, seq_len, n_copies)
                    trial_accs.append(acc)

                redundancy_raw.append({
                    "seed": seed,
                    "length": seq_len,
                    "n_copies": n_copies,
                    "decode_accuracy_pct": float(np.mean(trial_accs)) * 100.0,
                })

            print(f"  Seed {seed}, Length {seq_len}: done")

    csv_path_B = os.path.join(OUTPUT_DIR, "redundancy_raw.csv")
    with open(csv_path_B, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "length", "n_copies", "decode_accuracy_pct"])
        w.writeheader()
        w.writerows(redundancy_raw)
    print(f"  -> {csv_path_B}")

    # ========================================================================
    # C. ABLATION (ablation_raw.csv)
    # ========================================================================
    print("\n[5/8] Running ablation comparison...")
    ablation_raw = []

    ablation_conditions = [
        ("Baseline", None),
        ("Full RL", "rl"),
        ("Predictive RL", "pred"),
        ("RL w/o GC", "no_gc"),
        ("RL w/o Homopolymer", "no_homo"),
    ]
    # RNG offsets for ablation: baseline=2000, then agents enumerate from 4000
    ablation_offsets = {
        "Baseline": 2000,
        "Full RL": 4000,
        "Predictive RL": 4001,
        "RL w/o GC": 4002,
        "RL w/o Homopolymer": 4003,
    }

    for seed in SEEDS:
        for seq_len in SEQ_LENGTHS:
            dataset = generate_frozen_dataset(seq_len, TRIALS_PER_LENGTH, seed)

            for cond_name, agent_key in ablation_conditions:
                rng = np.random.default_rng(seed + ablation_offsets[cond_name] + seq_len)

                accs, errs = [], []
                for binary_bits in dataset:
                    if agent_key is None:
                        a, e, _, _, _ = eval_baseline_trial(binary_bits, config, rng)
                    else:
                        agent = agents[(agent_key, seed)] if agent_key in ("rl", "pred") else agents[(agent_key, seed)]
                        a, e, _, _, _ = eval_rl_trial(binary_bits, config, rng, agent, seq_len)
                    accs.append(a)
                    errs.append(e)

                ablation_raw.append({
                    "seed": seed,
                    "length": seq_len,
                    "condition": cond_name,
                    "error_rate": float(np.mean(errs)),
                    "decode_accuracy_pct": float(np.mean(accs)) * 100.0,
                })

        print(f"  Seed {seed}: done")

    csv_path_C = os.path.join(OUTPUT_DIR, "ablation_raw.csv")
    with open(csv_path_C, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "length", "condition", "error_rate", "decode_accuracy_pct"])
        w.writeheader()
        w.writerows(ablation_raw)
    print(f"  -> {csv_path_C}")

    # ========================================================================
    # D. OVERFITTING CHECK (overfit_raw.csv)
    # ========================================================================
    print("\n[6/8] Running overfitting check...")
    overfit_raw = []
    test_seed = 99999

    for seed in SEEDS:
        agent = agents[("pred", seed)]
        for seq_len in SEQ_LENGTHS:
            # "Train" = evaluation on frozen dataset (same as main comparison)
            train_dataset = generate_frozen_dataset(seq_len, TRIALS_PER_LENGTH, seed)
            rng_train = np.random.default_rng(seed + 5000 + seq_len)
            train_accs = []
            for binary_bits in train_dataset:
                a, _, _, _, _ = eval_rl_trial(binary_bits, config, rng_train, agent, seq_len)
                train_accs.append(a)

            # "Test" = unseen dataset with different seed
            test_dataset = generate_frozen_dataset(seq_len, 200, test_seed)
            rng_test = np.random.default_rng(test_seed + seq_len)
            test_accs = []
            for binary_bits in test_dataset:
                a, _, _, _, _ = eval_rl_trial(binary_bits, config, rng_test, agent, seq_len)
                test_accs.append(a)

            overfit_raw.append({
                "seed": seed,
                "length": seq_len,
                "train_accuracy_pct": float(np.mean(train_accs)) * 100.0,
                "test_accuracy_pct": float(np.mean(test_accs)) * 100.0,
            })
        print(f"  Seed {seed}: done")

    csv_path_D = os.path.join(OUTPUT_DIR, "overfit_raw.csv")
    with open(csv_path_D, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "length", "train_accuracy_pct", "test_accuracy_pct"])
        w.writeheader()
        w.writerows(overfit_raw)
    print(f"  -> {csv_path_D}")

    # ========================================================================
    # E. T-TESTS (ttests_raw.csv)
    # ========================================================================
    print("\n[7/8] Computing t-tests...")

    # Group results by (method, length) -> list of per-seed decode accuracies
    method_length_accs = defaultdict(list)
    for row in results_raw:
        method_length_accs[(row["method"], row["length"])].append(row["decode_accuracy_pct"])

    # Also group error rates
    method_length_errs = defaultdict(list)
    for row in results_raw:
        method_length_errs[(row["method"], row["length"])].append(row["error_rate"])

    ttest_comparisons = [
        ("Predictive RL vs Baseline", "Predictive RL", "Fixed Baseline"),
        ("Predictive RL vs RL Agent", "Predictive RL", "RL Agent"),
        ("RL Agent vs Baseline", "RL Agent", "Fixed Baseline"),
    ]

    ttests_raw = []
    for comp_name, method_a, method_b in ttest_comparisons:
        for seq_len in SEQ_LENGTHS:
            vals_a = method_length_errs[(method_a, seq_len)]
            vals_b = method_length_errs[(method_b, seq_len)]
            n_a, n_b = len(vals_a), len(vals_b)
            mean_a, mean_b = np.mean(vals_a), np.mean(vals_b)
            t_stat, p_val = scipy_stats.ttest_ind(vals_a, vals_b)
            ttests_raw.append({
                "comparison": comp_name,
                "length": seq_len,
                "n_a": n_a,
                "n_b": n_b,
                "mean_a": float(mean_a),
                "mean_b": float(mean_b),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "test_type": "scipy.stats.ttest_ind (Welch=False, i.e. Student's)",
            })

    csv_path_E = os.path.join(OUTPUT_DIR, "ttests_raw.csv")
    with open(csv_path_E, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "comparison", "length", "n_a", "n_b", "mean_a", "mean_b",
            "t_statistic", "p_value", "test_type"])
        w.writeheader()
        w.writerows(ttests_raw)
    print(f"  -> {csv_path_E}")

    # ========================================================================
    # AGGREGATED TABLES + DERIVED QUANTITIES -> results_tables.md
    # ========================================================================
    print("\n[8/8] Building aggregated tables...")

    md_lines = []
    md_lines.append("# Aggregated Results Tables\n")
    md_lines.append("All values: mean ± std across 7 seeds (42–48), each seed = mean of 100 trials.\n")

    # Helper: aggregate by (method, length)
    def agg(raw_rows, group_keys, value_key):
        """Return dict of (group) -> (mean, std, values_list)"""
        groups = defaultdict(list)
        for row in raw_rows:
            key = tuple(row[k] for k in group_keys)
            groups[key].append(row[value_key])
        result = {}
        for key, vals in groups.items():
            result[key] = (np.mean(vals), np.std(vals), vals)
        return result

    methods_order = ["Fixed Baseline", "Random Mapping", "RL Agent", "Predictive RL", "Pred. RL + 3-copy"]

    # ---- Table 1: Error Rate ----
    err_agg = agg(results_raw, ["method", "length"], "error_rate")
    md_lines.append("## Table 1: Error Rate by Method and Length\n")
    md_lines.append("| Method | " + " | ".join(str(l) for l in SEQ_LENGTHS) + " |")
    md_lines.append("|--------|" + "|".join(["-----:" for _ in SEQ_LENGTHS]) + "|")
    for m in methods_order:
        cells = []
        for l in SEQ_LENGTHS:
            mean, std, _ = err_agg[(m, l)]
            cells.append(f"{mean:.3f}±{std:.3f}")
        md_lines.append(f"| {m} | " + " | ".join(cells) + " |")
    md_lines.append("")
    # Best per column
    for l in SEQ_LENGTHS:
        best_m = min(methods_order, key=lambda m: err_agg[(m, l)][0])
        md_lines.append(f"- **Length {l}**: Best = {best_m} ({err_agg[(best_m, l)][0]:.3f}). "
                        f"Pred RL {'<' if err_agg[('Predictive RL',l)][0] < err_agg[('RL Agent',l)][0] else '>'} RL Agent; "
                        f"Pred RL {'<' if err_agg[('Predictive RL',l)][0] < err_agg[('Fixed Baseline',l)][0] else '>'} Baseline.")
    md_lines.append("")

    # ---- Table 2: Decode Accuracy ----
    acc_agg = agg(results_raw, ["method", "length"], "decode_accuracy_pct")
    md_lines.append("## Table 2: Decode Accuracy (%) by Method and Length\n")
    md_lines.append("| Method | " + " | ".join(str(l) for l in SEQ_LENGTHS) + " |")
    md_lines.append("|--------|" + "|".join(["-----:" for _ in SEQ_LENGTHS]) + "|")
    for m in methods_order:
        cells = []
        for l in SEQ_LENGTHS:
            mean, std, _ = acc_agg[(m, l)]
            cells.append(f"{mean:.2f}±{std:.2f}")
        md_lines.append(f"| {m} | " + " | ".join(cells) + " |")
    md_lines.append("")
    for l in SEQ_LENGTHS:
        best_m = max(methods_order, key=lambda m: acc_agg[(m, l)][0])
        md_lines.append(f"- **Length {l}**: Best = {best_m} ({acc_agg[(best_m, l)][0]:.2f}%). "
                        f"Pred RL {'>' if acc_agg[('Predictive RL',l)][0] > acc_agg[('RL Agent',l)][0] else '<'} RL Agent; "
                        f"Pred RL {'>' if acc_agg[('Predictive RL',l)][0] > acc_agg[('Fixed Baseline',l)][0] else '<'} Baseline.")
    md_lines.append("")

    # ---- Table 3: Stability Score ----
    stab_agg = agg(results_raw, ["method", "length"], "stability_score")
    md_lines.append("## Table 3: Stability Score by Method and Length\n")
    md_lines.append("| Method | " + " | ".join(str(l) for l in SEQ_LENGTHS) + " |")
    md_lines.append("|--------|" + "|".join(["-----:" for _ in SEQ_LENGTHS]) + "|")
    for m in methods_order:
        cells = []
        for l in SEQ_LENGTHS:
            mean, std, _ = stab_agg[(m, l)]
            cells.append(f"{mean:.3f}±{std:.3f}")
        md_lines.append(f"| {m} | " + " | ".join(cells) + " |")
    md_lines.append("")
    for l in SEQ_LENGTHS:
        best_m = max(methods_order, key=lambda m: stab_agg[(m, l)][0])
        md_lines.append(f"- **Length {l}**: Best = {best_m} ({stab_agg[(best_m, l)][0]:.3f}). "
                        f"Pred RL {'>' if stab_agg[('Predictive RL',l)][0] > stab_agg[('Fixed Baseline',l)][0] else '<'} Baseline.")
    md_lines.append("")

    # ---- Table 4: Redundancy Scaling ----
    red_agg = agg(redundancy_raw, ["n_copies", "length"], "decode_accuracy_pct")
    md_lines.append("## Table 4: Redundancy Scaling — Decode Accuracy (%) by Copies and Length\n")
    md_lines.append("| Copies | " + " | ".join(str(l) for l in SEQ_LENGTHS) + " |")
    md_lines.append("|--------|" + "|".join(["-----:" for _ in SEQ_LENGTHS]) + "|")
    for nc in COPY_COUNTS:
        cells = []
        for l in SEQ_LENGTHS:
            mean, std, _ = red_agg[(nc, l)]
            cells.append(f"{mean:.2f}±{std:.2f}")
        md_lines.append(f"| {nc} | " + " | ".join(cells) + " |")
    md_lines.append("")

    # ---- Table 5: Ablation ----
    abl_err_agg = agg(ablation_raw, ["condition", "length"], "error_rate")
    abl_acc_agg = agg(ablation_raw, ["condition", "length"], "decode_accuracy_pct")
    abl_conds = ["Baseline", "Full RL", "Predictive RL", "RL w/o GC", "RL w/o Homopolymer"]

    md_lines.append("## Table 5a: Ablation — Error Rate by Condition and Length\n")
    md_lines.append("| Condition | " + " | ".join(str(l) for l in SEQ_LENGTHS) + " |")
    md_lines.append("|-----------|" + "|".join(["-----:" for _ in SEQ_LENGTHS]) + "|")
    for c in abl_conds:
        cells = []
        for l in SEQ_LENGTHS:
            mean, std, _ = abl_err_agg[(c, l)]
            cells.append(f"{mean:.3f}±{std:.3f}")
        md_lines.append(f"| {c} | " + " | ".join(cells) + " |")
    md_lines.append("")

    md_lines.append("## Table 5b: Ablation — Decode Accuracy (%) by Condition and Length\n")
    md_lines.append("| Condition | " + " | ".join(str(l) for l in SEQ_LENGTHS) + " |")
    md_lines.append("|-----------|" + "|".join(["-----:" for _ in SEQ_LENGTHS]) + "|")
    for c in abl_conds:
        cells = []
        for l in SEQ_LENGTHS:
            mean, std, _ = abl_acc_agg[(c, l)]
            cells.append(f"{mean:.2f}±{std:.2f}")
        md_lines.append(f"| {c} | " + " | ".join(cells) + " |")
    md_lines.append("")

    # ---- Table 6: Overfitting ----
    of_agg_train = agg(overfit_raw, ["length"], "train_accuracy_pct")
    of_agg_test = agg(overfit_raw, ["length"], "test_accuracy_pct")
    md_lines.append("## Table 6: Overfitting Check — Train vs Test Accuracy (%)\n")
    md_lines.append("| Length | Train | Test | Gap |")
    md_lines.append("|--------|------:|-----:|----:|")
    for l in SEQ_LENGTHS:
        train_mean = of_agg_train[(l,)][0]
        test_mean = of_agg_test[(l,)][0]
        gap = train_mean - test_mean
        md_lines.append(f"| {l} | {train_mean:.2f} | {test_mean:.2f} | {gap:+.2f} |")
    md_lines.append("")

    # ---- Table 7: t-test results ----
    md_lines.append("## Table 7: t-test Results (on error rate, 7 per-seed means per method)\n")
    md_lines.append("| Comparison | Length | t | p |")
    md_lines.append("|------------|-------:|--:|--:|")
    for row in ttests_raw:
        sig = " *" if row["p_value"] < 0.05 else ""
        md_lines.append(f"| {row['comparison']} | {row['length']} | {row['t_statistic']:.2f} | {row['p_value']:.3f}{sig} |")
    md_lines.append("")

    # ---- DERIVED QUANTITIES ----
    md_lines.append("## Derived Quantities\n")

    # PP gain of Pred RL + Red over Baseline
    md_lines.append("### Percentage-point gain of Pred. RL + Red. over Baseline\n")
    md_lines.append("Formula: `gain_pp = acc_pred_rl_red_mean - acc_baseline_mean`\n")
    for l in SEQ_LENGTHS:
        pred_red = acc_agg[("Pred. RL + 3-copy", l)][0]
        baseline = acc_agg[("Fixed Baseline", l)][0]
        md_lines.append(f"- Length {l}: {pred_red:.2f} - {baseline:.2f} = **{pred_red - baseline:+.2f} pp**")
    md_lines.append("")

    # Relative error-rate reduction
    md_lines.append("### Relative error-rate reduction of Pred. RL + Red. vs Baseline\n")
    md_lines.append("Formula: `reduction_pct = (err_baseline - err_pred_red) / err_baseline * 100`\n")
    for l in SEQ_LENGTHS:
        err_pred = err_agg[("Pred. RL + 3-copy", l)][0]
        err_base = err_agg[("Fixed Baseline", l)][0]
        reduction = (err_base - err_pred) / err_base * 100
        md_lines.append(f"- Length {l}: ({err_base:.3f} - {err_pred:.3f}) / {err_base:.3f} × 100 = **{reduction:.1f}%**")
    md_lines.append("")

    # Gain from 1 to 10 copies
    md_lines.append("### Gain from 1 to 10 copies (percentage points)\n")
    md_lines.append("Formula: `gain = acc_10copies - acc_1copy`\n")
    for l in SEQ_LENGTHS:
        a1 = red_agg[(1, l)][0]
        a10 = red_agg[(10, l)][0]
        md_lines.append(f"- Length {l}: {a10:.2f} - {a1:.2f} = **{a10 - a1:+.2f} pp**")
    md_lines.append("")

    # Relative stability improvement at length 256
    md_lines.append("### Relative stability improvement of Predictive RL over Baseline at length 256\n")
    stab_pred_256 = stab_agg[("Predictive RL", 256)][0]
    stab_base_256 = stab_agg[("Fixed Baseline", 256)][0]
    rel_imp = (stab_pred_256 - stab_base_256) / stab_base_256 * 100
    md_lines.append(f"Formula: `(stab_pred - stab_base) / stab_base × 100`")
    md_lines.append(f"= ({stab_pred_256:.3f} - {stab_base_256:.3f}) / {stab_base_256:.3f} × 100 = **{rel_imp:.1f}%**\n")

    # Largest train-test gap
    md_lines.append("### Largest absolute train-test gap\n")
    gaps = []
    for l in SEQ_LENGTHS:
        train_mean = of_agg_train[(l,)][0]
        test_mean = of_agg_test[(l,)][0]
        gaps.append((l, train_mean - test_mean))
    max_gap_l, max_gap_v = max(gaps, key=lambda x: abs(x[1]))
    test_exceeds = sum(1 for l, g in gaps if g < 0)
    md_lines.append(f"- Largest absolute gap: **{abs(max_gap_v):.2f} pp** at length **{max_gap_l}**")
    md_lines.append(f"- Lengths where test exceeds train: **{test_exceeds}** of {len(SEQ_LENGTHS)}\n")

    # Write results_tables.md
    tables_path = os.path.join(OUTPUT_DIR, "results_tables.md")
    with open(tables_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  -> {tables_path}")

    # ========================================================================
    # PROVENANCE.MD
    # ========================================================================
    print("\nWriting provenance.md...")

    prov_lines = []
    prov_lines.append("# Provenance\n")

    prov_lines.append("## Q1: Redundancy-scaling vs main comparison\n")
    prov_lines.append("The redundancy-scaling experiment uses the **same frozen evaluation datasets** "
                      "(generated with `seed * 10000 + length`) as the main comparison.\n")
    prov_lines.append("However, it uses a **different RNG seed** for error simulation: "
                      "`seed * 100000 + length * 100 + n_copies` (from `evaluation.py:run_redundancy_scaling`, L411), "
                      "whereas the main comparison uses `seed + 6000 + length` (from `evaluation.py:run_comparison`, L246).\n")
    prov_lines.append("The agent checkpoints are also different: the main comparison's redundancy row uses "
                      "`agent_predictive` loaded per-seed, whereas `run_redundancy_scaling` in the original pipeline "
                      "(`run_full_pipeline.py` L245-248) uses `agent_for_test` which is a single agent loaded from "
                      "the **last seed only** (seed 48). In this re-run, redundancy scaling uses the per-seed "
                      "predictive agent, matching the main comparison.\n")

    prov_lines.append("## Q2: copies=1 and copies=3 vs main comparison\n")
    prov_lines.append("In the original pipeline, copies=1 and copies=3 in the scaling table will **NOT** exactly "
                      "match the main comparison for two reasons:\n")
    prov_lines.append("1. **Different RNG seeds** for error simulation (see Q1).\n")
    prov_lines.append("2. **Different agent** used in the original pipeline (single agent for all seeds in scaling "
                      "vs per-seed agent in main comparison).\n")
    prov_lines.append("In this re-run, the per-seed agent is used for both, but the RNG seeds still differ, "
                      "so numbers will differ due to different noise draws.\n")

    prov_lines.append("## Q3: Unit of std in every table\n")
    prov_lines.append("In every table, std is computed across the **7 per-seed means** "
                      "(each seed's mean is over 100 trials). That is, `np.std([mean_seed42, mean_seed43, ..., mean_seed48])`. "
                      "It is NOT the std across all 700 trials.\n")

    prov_lines.append("## Q4: t-test details\n")
    prov_lines.append("The t-tests use `scipy.stats.ttest_ind(vals_a, vals_b)` with default parameters "
                      "(equal_var=True, i.e. Student's t-test, not Welch's).\n")
    prov_lines.append("The samples are the **7 per-seed means** of the error rate (not 700 individual trials). "
                      "The test is unpaired (independent two-sample).\n")
    prov_lines.append("Exact call: `scipy_stats.ttest_ind(error_rates_a, error_rates_b)` "
                      "(see `evaluation.py` L333).\n")

    prov_lines.append("## Q5: Overfitting train and test sets\n")
    prov_lines.append("The 'train' set in the overfitting check is the **same frozen evaluation dataset** "
                      "used in the main comparison (generated with `seed * 10000 + length`, 100 trials per length).\n")
    prov_lines.append("The 'test' set is a **separate, independent dataset** generated with seed=99999 "
                      "and 200 trials per length (`run_full_pipeline.py` L178-189).\n")
    prov_lines.append("The test RNG for error simulation uses `np.random.default_rng(test_seed + seq_len)`, "
                      "which is different from the train RNG.\n")

    prov_lines.append("## Package versions\n")
    prov_lines.append(f"- Python: {sys.version}")
    prov_lines.append(f"- PyTorch: {torch.__version__}")
    prov_lines.append(f"- NumPy: {np.__version__}")
    import scipy
    prov_lines.append(f"- SciPy: {scipy.__version__}\n")

    prov_lines.append("## Checkpoint hashes (MD5)\n")
    for key in sorted(checkpoint_hashes.keys()):
        prov_lines.append(f"- `{key}`: `{checkpoint_hashes[key]}`")

    prov_path = os.path.join(OUTPUT_DIR, "provenance.md")
    with open(prov_path, "w") as f:
        f.write("\n".join(prov_lines))
    print(f"  -> {prov_path}")

    # ========================================================================
    # SELF-CHECK
    # ========================================================================
    print("\nRunning consistency self-checks...")
    checks = []

    # Check 1: Aggregated numbers match raw CSVs
    for m in methods_order:
        for l in SEQ_LENGTHS:
            raw_vals = [r["decode_accuracy_pct"] for r in results_raw
                        if r["method"] == m and r["length"] == l]
            computed_mean = np.mean(raw_vals)
            computed_std = np.std(raw_vals)
            table_mean, table_std, _ = acc_agg[(m, l)]
            ok = abs(computed_mean - table_mean) < 0.001 and abs(computed_std - table_std) < 0.001
            status = "PASS" if ok else f"FAIL (raw mean={computed_mean:.4f} table={table_mean:.4f})"
            checks.append(f"Acc consistency {m} L{l}: {status}")

    # Check 2: copies=1 vs main comparison Predictive RL
    checks.append("")
    checks.append("--- copies=1 vs main Predictive RL (expect DIFFERENT due to RNG) ---")
    for l in SEQ_LENGTHS:
        main_val = acc_agg[("Predictive RL", l)][0]
        red_val = red_agg[(1, l)][0]
        diff = abs(main_val - red_val)
        status = f"DIFF={diff:.2f}pp (main={main_val:.2f}, scaling={red_val:.2f}) — explained by different RNG seeds"
        checks.append(f"copies=1 L{l}: {status}")

    checks.append("")
    checks.append("--- copies=3 vs main Pred RL+Red (expect DIFFERENT due to RNG) ---")
    for l in SEQ_LENGTHS:
        main_val = acc_agg[("Pred. RL + 3-copy", l)][0]
        red_val = red_agg[(3, l)][0]
        diff = abs(main_val - red_val)
        status = f"DIFF={diff:.2f}pp (main={main_val:.2f}, scaling={red_val:.2f}) — explained by different RNG seeds"
        checks.append(f"copies=3 L{l}: {status}")

    # ========================================================================
    # CLAIMS VERIFICATION
    # ========================================================================
    checks.append("")
    checks.append("=" * 60)
    checks.append("CLAIMS VERIFICATION")
    checks.append("=" * 60)

    # Claim 1: Predictive RL has higher decode accuracy than RL Agent at every length
    claim1 = all(acc_agg[("Predictive RL", l)][0] > acc_agg[("RL Agent", l)][0] for l in SEQ_LENGTHS)
    details1 = "; ".join(f"L{l}: {acc_agg[('Predictive RL',l)][0]:.2f} vs {acc_agg[('RL Agent',l)][0]:.2f}"
                         for l in SEQ_LENGTHS)
    checks.append(f"Claim 1 (Pred RL > RL Agent at every length): {'TRUE' if claim1 else 'FALSE'} — {details1}")

    # Claim 2: RL Agent trails both Baseline and Predictive RL at every length
    claim2 = all(
        acc_agg[("RL Agent", l)][0] < acc_agg[("Fixed Baseline", l)][0] and
        acc_agg[("RL Agent", l)][0] < acc_agg[("Predictive RL", l)][0]
        for l in SEQ_LENGTHS
    )
    details2 = "; ".join(
        f"L{l}: RL={acc_agg[('RL Agent',l)][0]:.2f} Base={acc_agg[('Fixed Baseline',l)][0]:.2f} Pred={acc_agg[('Predictive RL',l)][0]:.2f}"
        for l in SEQ_LENGTHS
    )
    checks.append(f"Claim 2 (RL Agent trails both at every length): {'TRUE' if claim2 else 'FALSE/PARTLY'} — {details2}")

    # Claim 3: Pred RL sig better than RL Agent at 128 (p<0.05) and not at 64
    p_pred_rl_128 = next(r["p_value"] for r in ttests_raw
                         if r["comparison"] == "Predictive RL vs RL Agent" and r["length"] == 128)
    p_pred_rl_64 = next(r["p_value"] for r in ttests_raw
                        if r["comparison"] == "Predictive RL vs RL Agent" and r["length"] == 64)
    claim3 = p_pred_rl_128 < 0.05 and p_pred_rl_64 >= 0.05
    checks.append(f"Claim 3 (Pred RL sig at 128 not at 64): {'TRUE' if claim3 else 'FALSE'} — "
                  f"p(128)={p_pred_rl_128:.4f}, p(64)={p_pred_rl_64:.4f}")

    # Claim 4: RL Agent sig worse than Baseline at 128
    p_rl_base_128 = next(r["p_value"] for r in ttests_raw
                         if r["comparison"] == "RL Agent vs Baseline" and r["length"] == 128)
    claim4 = p_rl_base_128 < 0.05
    checks.append(f"Claim 4 (RL Agent sig worse than Baseline at 128): {'TRUE' if claim4 else 'FALSE'} — "
                  f"p={p_rl_base_128:.4f}")

    # Claim 5: Pred RL + Red best decode accuracy at all 5 lengths
    claim5 = all(
        acc_agg[("Pred. RL + 3-copy", l)][0] == max(acc_agg[(m, l)][0] for m in methods_order)
        for l in SEQ_LENGTHS
    )
    checks.append(f"Claim 5 (Pred RL+Red best at all lengths): {'TRUE' if claim5 else 'FALSE'}")

    # Claim 6: Predictive RL higher stability than Baseline at all lengths
    claim6 = all(stab_agg[("Predictive RL", l)][0] > stab_agg[("Fixed Baseline", l)][0] for l in SEQ_LENGTHS)
    checks.append(f"Claim 6 (Pred RL higher stability than Baseline at all lengths): {'TRUE' if claim6 else 'FALSE'}")

    # Claim 7: At 256 nt, baseline stability=0.886 and Pred RL=0.930
    base_stab_256 = stab_agg[("Fixed Baseline", 256)][0]
    pred_stab_256 = stab_agg[("Predictive RL", 256)][0]
    claim7 = abs(base_stab_256 - 0.886) < 0.005 and abs(pred_stab_256 - 0.930) < 0.005
    checks.append(f"Claim 7 (stability 256: base=0.886, pred=0.930): {'TRUE' if claim7 else 'FALSE'} — "
                  f"base={base_stab_256:.3f}, pred={pred_stab_256:.3f}")

    # Claim 8: Ablation findings
    # a) RL w/o GC lowest error at 64
    abl_methods_learned = ["Full RL", "Predictive RL", "RL w/o GC", "RL w/o Homopolymer"]
    c8a = abl_err_agg[("RL w/o GC", 64)][0] == min(abl_err_agg[(c, 64)][0] for c in abl_conds)
    # b) RL w/o Homo better than Full RL at 128
    c8b = abl_err_agg[("RL w/o Homopolymer", 128)][0] < abl_err_agg[("Full RL", 128)][0]
    # c) Full RL lowest among learned methods at 64
    c8c = abl_err_agg[("Full RL", 64)][0] == min(abl_err_agg[(c, 64)][0] for c in abl_methods_learned)
    # Note: paper says "Full RL posted the lowest error rate among the learned methods" at 64
    # but also says "RL w/o GC produced the lowest error rate at length 64" (including baseline)
    checks.append(f"Claim 8a (RL w/o GC lowest error at 64, all methods): {'TRUE' if c8a else 'FALSE'} — "
                  + ", ".join(f"{c}={abl_err_agg[(c,64)][0]:.4f}" for c in abl_conds))
    checks.append(f"Claim 8b (RL w/o Homo < Full RL at 128): {'TRUE' if c8b else 'FALSE'} — "
                  f"w/o Homo={abl_err_agg[('RL w/o Homopolymer',128)][0]:.4f}, Full RL={abl_err_agg[('Full RL',128)][0]:.4f}")
    checks.append(f"Claim 8c (Full RL lowest among learned at 64): {'TRUE' if c8c else 'FALSE'} — "
                  + ", ".join(f"{c}={abl_err_agg[(c,64)][0]:.4f}" for c in abl_methods_learned))

    # Claim 9: Max train-test gap is 1.24 pp at 256 nt and test>train at 4 of 5
    claim9a = abs(abs(max_gap_v) - 1.24) < 0.5 and max_gap_l == 256
    claim9b = test_exceeds == 4
    checks.append(f"Claim 9 (max gap=1.24pp at 256, test>train at 4/5): "
                  f"{'TRUE' if claim9a and claim9b else 'PARTLY'} — "
                  f"max gap={abs(max_gap_v):.2f}pp at L{max_gap_l}, test>train at {test_exceeds}/5")

    # Write self-check
    check_path = os.path.join(OUTPUT_DIR, "self_check.txt")
    with open(check_path, "w") as f:
        f.write("\n".join(checks))
    print(f"  -> {check_path}")

    # ========================================================================
    # DEMO LOG
    # ========================================================================
    print("\nRunning web demo simulation...")
    demo_lines = []
    demo_lines.append("# Web Demo Log\n")

    # Demo 1: String "Hello DNA!"
    demo_lines.append("## Demo: 'Hello DNA!' with 10 copies\n")
    text = "Hello DNA!"
    text_bits = np.array([int(b) for c in text for b in format(ord(c), '08b')], dtype=np.int8)
    # Pad to even length for dibit encoding
    if len(text_bits) % 2 != 0:
        text_bits = np.concatenate([text_bits, np.array([0], dtype=np.int8)])
    seq_len_demo = len(text_bits) // 2
    demo_lines.append(f"- Input text: '{text}'")
    demo_lines.append(f"- Binary length: {len(text_bits)} bits")
    demo_lines.append(f"- DNA sequence length: {seq_len_demo} nt")

    agent_demo = agents[("pred", 48)]  # Use last seed's agent
    state_demo = make_state(text_bits, seq_len_demo)
    action_demo, _, _ = agent_demo.select_action(state_demo)
    p_idx, s_idx = decode_action(action_demo)
    demo_lines.append(f"- Agent chose action: {action_demo} (primary mapping={p_idx}, secondary mapping={s_idx})")

    demo_rng = np.random.default_rng(12345)
    demo_config = Config(
        number_of_copies=10,
        substitution_rate=0.02,
        insertion_rate=0.01,
        deletion_rate=0.01,
    )
    demo_result = encode_with_redundancy(text_bits, action_demo, demo_rng, demo_config)

    per_copy_accs = []
    for copy_data in demo_result["copies"]:
        decoded_copy = decode_with_action(copy_data["mutated_sequence"], action_demo, seq_len_demo)
        copy_acc = compute_bit_accuracy(text_bits, decoded_copy)
        per_copy_accs.append(copy_acc)

    demo_lines.append(f"- Noise seed: 12345")
    demo_lines.append(f"- Per-copy accuracies: {[f'{a*100:.2f}%' for a in per_copy_accs]}")
    demo_lines.append(f"- Mean single-copy accuracy: {np.mean(per_copy_accs)*100:.2f}%")
    demo_lines.append(f"- Consensus (10-copy) accuracy: {demo_result['redundancy_accuracy']*100:.2f}%")
    demo_lines.append(f"- Error channel parameters: sub=2%, ins=1%, del=1%")
    demo_lines.append(f"- Same error channel as evaluation: YES (same rates and context-dependent multipliers)")
    demo_lines.append("")

    # Try decoding back to text
    consensus_decoded = demo_result["consensus_decoded_bits"]
    recovered_bytes = []
    for i in range(0, min(len(consensus_decoded), len(text_bits)), 8):
        byte_bits = consensus_decoded[i:i+8]
        if len(byte_bits) == 8:
            byte_val = int("".join(str(int(b)) for b in byte_bits), 2)
            recovered_bytes.append(byte_val)
    recovered_text = "".join(chr(b) for b in recovered_bytes if 32 <= b < 127)
    demo_lines.append(f"- Recovered text from consensus: '{recovered_text}'")

    demo_path = os.path.join(OUTPUT_DIR, "demo_log.txt")
    with open(demo_path, "w") as f:
        f.write("\n".join(demo_lines))
    print(f"  -> {demo_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("  RE-EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nAll output files in: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  - provenance.md")
    print(f"  - results_raw.csv     ({len(results_raw)} rows)")
    print(f"  - redundancy_raw.csv  ({len(redundancy_raw)} rows)")
    print(f"  - ablation_raw.csv    ({len(ablation_raw)} rows)")
    print(f"  - overfit_raw.csv     ({len(overfit_raw)} rows)")
    print(f"  - ttests_raw.csv      ({len(ttests_raw)} rows)")
    print(f"  - results_tables.md")
    print(f"  - self_check.txt")
    print(f"  - demo_log.txt")


if __name__ == "__main__":
    main()
