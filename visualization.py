"""
Generate research-quality matplotlib plots.

Produces PNG figures saved to the results directory:
    - reward_versus_epochs.png
    - rolling_reward.png
    - smoothed_reward.png
    - predictor_loss.png
    - error_rate_comparison.png
    - decode_accuracy.png
    - gc_content_distribution.png
    - encoding_efficiency.png
    - ablation_comparison.png (optional)
    - validation_reward.png (optional)
    - redundancy_comparison.png (optional)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List

from config import Config


METHOD_COLORS = {
    "baseline": "#e74c3c",
    "random": "#f39c12",
    "reinforcement_learning": "#3498db",
    "predictive_reinforcement_learning": "#2ecc71",
    "predictive_reinforcement_learning_redundancy": "#9b59b6",
}

METHOD_LABELS = {
    "baseline": "Fixed Baseline",
    "random": "Random Mapping",
    "reinforcement_learning": "Reinforcement Learning (PPO)",
    "predictive_reinforcement_learning": "Predictive Reinforcement Learning",
    "predictive_reinforcement_learning_redundancy": "Predictive RL + Redundancy",
}


def _save_figure(figure, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {path}")


def plot_reward_versus_epochs(rewards: List[float], config: Config):
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(rewards, alpha=0.3, linewidth=0.5, color="#3498db", label="Epoch reward")
    # rolling average overlay
    window = min(100, len(rewards) // 5) if len(rewards) > 5 else 1
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        axis.plot(range(window - 1, len(rewards)), smoothed,
                linewidth=2, color="#2c3e50", label=f"Rolling average (window={window})")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Reward")
    axis.set_title("Reward versus Training Epochs")
    axis.legend()
    axis.grid(True, alpha=0.3)
    _save_figure(figure, config.results_path("reward_versus_epochs.png"))


def plot_rolling_reward(rolling: List[float], config: Config):
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(rolling, linewidth=2, color="#8e44ad")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Rolling Average Reward")
    axis.set_title("Rolling Average Reward Curve")
    axis.grid(True, alpha=0.3)
    _save_figure(figure, config.results_path("rolling_reward.png"))


def plot_smoothed_reward(smoothed: List[float], config: Config):
    if not smoothed:
        return
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(smoothed, linewidth=2, color="#27ae60")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("EMA Smoothed Reward")
    axis.set_title("Exponential Moving Average of Reward (alpha=0.02)")
    axis.grid(True, alpha=0.3)
    _save_figure(figure, config.results_path("smoothed_reward.png"))


def plot_predictor_loss(losses: List[float], config: Config):
    if not losses:
        return
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(losses, linewidth=2, color="#e67e22")
    axis.set_xlabel("Training Step")
    axis.set_ylabel("Mean Squared Error Loss")
    axis.set_title("Error Predictor Training Loss")
    axis.grid(True, alpha=0.3)
    _save_figure(figure, config.results_path("predictor_loss.png"))


def plot_validation_reward(validation_rewards: List[float], config: Config):
    if not validation_rewards:
        return
    figure, axis = plt.subplots(figsize=(10, 5))
    x_values = [(index + 1) * 500 for index in range(len(validation_rewards))]
    axis.plot(x_values, validation_rewards, linewidth=2, color="#e74c3c", marker="o", markersize=3)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation Reward")
    axis.set_title("Validation Reward Over Training (Anti-Overfitting Monitor)")
    axis.grid(True, alpha=0.3)
    _save_figure(figure, config.results_path("validation_reward.png"))


# grouped bar chart helper

def _grouped_bar(statistics: List[Dict], metric_mean: str, metric_standard_deviation: str,
                 y_label: str, title: str, filename: str, config: Config):
    """Draw a grouped bar chart: x-axis = sequence length, groups = methods."""
    from collections import defaultdict

    methods_in_data = []
    sequence_lengths = sorted(set(statistic["seq_length"] for statistic in statistics))
    data = defaultdict(dict)
    for statistic in statistics:
        method = statistic["method"]
        sequence_length = statistic["seq_length"]
        data[method][sequence_length] = (
            statistic.get(metric_mean, 0),
            statistic.get(metric_standard_deviation, 0),
        )
        if method not in methods_in_data:
            methods_in_data.append(method)

    number_of_methods = len(methods_in_data)
    x_positions = np.arange(len(sequence_lengths))
    bar_width = 0.8 / number_of_methods

    figure, axis = plt.subplots(figsize=(10, 6))
    for method_index, method in enumerate(methods_in_data):
        means = [data[method].get(sequence_length, (0, 0))[0] for sequence_length in sequence_lengths]
        standard_deviations = [data[method].get(sequence_length, (0, 0))[1] for sequence_length in sequence_lengths]
        offset = (method_index - number_of_methods / 2 + 0.5) * bar_width
        axis.bar(x_positions + offset, means, bar_width, yerr=standard_deviations, capsize=3,
               color=METHOD_COLORS.get(method, "#95a5a6"),
               label=METHOD_LABELS.get(method, method),
               alpha=0.85)

    axis.set_xticks(x_positions)
    axis.set_xticklabels([str(sequence_length) for sequence_length in sequence_lengths])
    axis.set_xlabel("Sequence Length")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    _save_figure(figure, config.results_path(filename))


def plot_error_rate_comparison(statistics: List[Dict], config: Config):
    _grouped_bar(statistics, "average_error_rate_mean", "average_error_rate_std",
                 "Average Error Rate", "Error Rate Comparison",
                 "error_rate_comparison.png", config)


def plot_decode_accuracy(statistics: List[Dict], config: Config):
    _grouped_bar(statistics, "decode_accuracy_mean", "decode_accuracy_std",
                 "Decode Accuracy (%)", "Decode Accuracy Comparison",
                 "decode_accuracy.png", config)


def plot_gc_distribution(all_rows: List[Dict], config: Config):
    """Histogram overlay of GC content for each method (all sequence lengths combined)."""
    from collections import defaultdict
    method_gc_values = defaultdict(list)
    for row in all_rows:
        if "average_gc_content" in row:
            method_gc_values[row["method"]].append(row["average_gc_content"])

    figure, axis = plt.subplots(figsize=(10, 6))
    for method, values in method_gc_values.items():
        axis.hist(values, bins=20, alpha=0.5,
                color=METHOD_COLORS.get(method, "#95a5a6"),
                label=METHOD_LABELS.get(method, method))

    axis.axvline(0.40, color="red", linestyle="--", alpha=0.5, label="Target range")
    axis.axvline(0.60, color="red", linestyle="--", alpha=0.5)
    axis.set_xlabel("GC Content")
    axis.set_ylabel("Frequency")
    axis.set_title("GC Content Distribution by Encoding Method")
    axis.legend()
    axis.grid(True, alpha=0.3)
    _save_figure(figure, config.results_path("gc_content_distribution.png"))


def plot_encoding_efficiency(statistics: List[Dict], config: Config):
    _grouped_bar(statistics, "encoding_efficiency_mean", "encoding_efficiency_std",
                 "Encoding Efficiency", "Encoding Efficiency Comparison",
                 "encoding_efficiency.png", config)


# ablation comparison

ABLATION_COLORS = {
    "baseline":                                      "#e74c3c",
    "reinforcement_learning":                        "#3498db",
    "predictive_reinforcement_learning":              "#2ecc71",
    "reinforcement_learning_no_gc":                  "#9b59b6",
    "reinforcement_learning_no_homopolymer":          "#f39c12",
}

ABLATION_LABELS = {
    "baseline":                                      "Fixed Baseline",
    "reinforcement_learning":                        "Reinforcement Learning (PPO)",
    "predictive_reinforcement_learning":              "Predictive Reinforcement Learning",
    "reinforcement_learning_no_gc":                  "Reinforcement Learning without GC Reward",
    "reinforcement_learning_no_homopolymer":          "Reinforcement Learning without Homopolymer Penalty",
}


def plot_ablation_comparison(statistics: List[Dict], config: Config):
    """Side-by-side error rate and decode accuracy for all ablation conditions."""
    from collections import defaultdict

    if not statistics:
        return

    methods_in_data = []
    sequence_lengths = sorted(set(statistic["seq_length"] for statistic in statistics))
    error_rate_data = defaultdict(dict)
    accuracy_data = defaultdict(dict)
    for statistic in statistics:
        method = statistic["method"]
        sequence_length = statistic["seq_length"]
        error_rate_data[method][sequence_length] = (
            statistic.get("average_error_rate_mean", 0),
            statistic.get("average_error_rate_std", 0),
        )
        accuracy_data[method][sequence_length] = (
            statistic.get("decode_accuracy_mean", 0),
            statistic.get("decode_accuracy_std", 0),
        )
        if method not in methods_in_data:
            methods_in_data.append(method)

    number_of_methods = len(methods_in_data)
    x_positions = np.arange(len(sequence_lengths))
    bar_width = 0.8 / number_of_methods

    figure, (axis_1, axis_2) = plt.subplots(1, 2, figsize=(16, 6))

    for method_index, method in enumerate(methods_in_data):
        offset = (method_index - number_of_methods / 2 + 0.5) * bar_width
        color = ABLATION_COLORS.get(method, "#95a5a6")
        label = ABLATION_LABELS.get(method, method)

        # error rate
        means = [error_rate_data[method].get(sequence_length, (0, 0))[0] for sequence_length in sequence_lengths]
        standard_deviations = [error_rate_data[method].get(sequence_length, (0, 0))[1] for sequence_length in sequence_lengths]
        axis_1.bar(x_positions + offset, means, bar_width, yerr=standard_deviations, capsize=3,
                color=color, label=label, alpha=0.85)

        # decode accuracy
        means = [accuracy_data[method].get(sequence_length, (0, 0))[0] for sequence_length in sequence_lengths]
        standard_deviations = [accuracy_data[method].get(sequence_length, (0, 0))[1] for sequence_length in sequence_lengths]
        axis_2.bar(x_positions + offset, means, bar_width, yerr=standard_deviations, capsize=3,
                color=color, label=label, alpha=0.85)

    for axis, y_label, title in [
        (axis_1, "Average Error Rate", "Ablation: Error Rate Comparison"),
        (axis_2, "Decode Accuracy (%)", "Ablation: Decode Accuracy Comparison"),
    ]:
        axis.set_xticks(x_positions)
        axis.set_xticklabels([str(sequence_length) for sequence_length in sequence_lengths])
        axis.set_xlabel("Sequence Length")
        axis.set_ylabel(y_label)
        axis.set_title(title)
        axis.legend(fontsize=8)
        axis.grid(True, axis="y", alpha=0.3)

    figure.tight_layout()
    _save_figure(figure, config.results_path("ablation_comparison.png"))


# redundancy scaling plot

LENGTH_COLORS = {
    64: "#e74c3c",
    128: "#3498db",
    256: "#2ecc71",
    512: "#f39c12",
    1024: "#9b59b6",
}

LENGTH_MARKERS = {
    64: "o",
    128: "s",
    256: "D",
    512: "^",
    1024: "v",
}


def plot_redundancy_scaling(redundancy_summary: List[Dict], config: Config):
    """
    Line chart: decode accuracy (y) vs number of copies (x),
    one line per sequence length with error bars.
    """
    from collections import defaultdict

    if not redundancy_summary:
        return

    # group by sequence length
    data_by_length = defaultdict(lambda: {"copies": [], "means": [], "stds": []})
    for entry in redundancy_summary:
        sl = entry["seq_length"]
        data_by_length[sl]["copies"].append(entry["number_of_copies"])
        data_by_length[sl]["means"].append(entry["decode_accuracy_mean"])
        data_by_length[sl]["stds"].append(entry["decode_accuracy_std"])

    figure, axis = plt.subplots(figsize=(10, 6))

    for seq_length in sorted(data_by_length.keys()):
        d = data_by_length[seq_length]
        color = LENGTH_COLORS.get(seq_length, "#95a5a6")
        marker = LENGTH_MARKERS.get(seq_length, "o")
        axis.errorbar(
            d["copies"], d["means"], yerr=d["stds"],
            marker=marker, linewidth=2, markersize=8, capsize=4,
            color=color, label=f"Length {seq_length}",
            alpha=0.85,
        )

    axis.set_xlabel("Number of Redundant Copies", fontsize=12)
    axis.set_ylabel("Decode Accuracy (%)", fontsize=12)
    axis.set_title("Decode Accuracy versus Number of Redundant Copies", fontsize=14)
    axis.legend(fontsize=10)
    axis.grid(True, alpha=0.3)
    axis.set_xticks(sorted(set(e["number_of_copies"] for e in redundancy_summary)))

    # annotate 100% points
    for entry in redundancy_summary:
        if entry["decode_accuracy_mean"] >= 99.99:
            axis.annotate(
                "100%",
                xy=(entry["number_of_copies"], entry["decode_accuracy_mean"]),
                xytext=(5, 10), textcoords="offset points",
                fontsize=9, fontweight="bold", color="#27ae60",
            )

    figure.tight_layout()
    _save_figure(figure, config.results_path("redundancy_scaling.png"))


def generate_all_plots(
    epoch_rewards: List[float],
    rolling_rewards: List[float],
    smoothed_rewards: List[float],
    predictor_losses: List[float],
    statistics_summary: List[Dict],
    all_rows: List[Dict],
    config: Config,
    ablation_statistics: List[Dict] = None,
    validation_rewards: List[float] = None,
    redundancy_summary: List[Dict] = None,
):
    """Generate all plots including optional ablation, validation, and redundancy charts."""
    print("\nGenerating plots...")
    plot_reward_versus_epochs(epoch_rewards, config)
    plot_rolling_reward(rolling_rewards, config)
    plot_smoothed_reward(smoothed_rewards, config)
    plot_predictor_loss(predictor_losses, config)
    plot_error_rate_comparison(statistics_summary, config)
    plot_decode_accuracy(statistics_summary, config)
    plot_gc_distribution(all_rows, config)
    plot_encoding_efficiency(statistics_summary, config)
    if ablation_statistics:
        plot_ablation_comparison(ablation_statistics, config)
    if validation_rewards:
        plot_validation_reward(validation_rewards, config)
    if redundancy_summary:
        plot_redundancy_scaling(redundancy_summary, config)
    print("  All plots saved.\n")
