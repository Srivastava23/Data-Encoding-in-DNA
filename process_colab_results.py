"""
Process results exported from Google Colab training.

After downloading colab_export.json from the Colab notebook,
run this script to:
  1. Validate the exported JSON
  2. Generate website/demo_data.json for the interactive demo
  3. Update results/_data.json with the new format
  4. Print a summary table of all results

Usage:
    python process_colab_results.py --input results/colab_export.json
"""

import json
import os
import sys
import argparse
import numpy as np


def validate_export(data: dict) -> bool:
    """Check that all required keys exist in the export."""
    required_keys = [
        "metadata", "policy_lookup_table", "training_curves",
        "main_results", "statistics", "ablation_statistics",
        "redundancy_comparison", "unseen_test_results",
    ]
    missing = [key for key in required_keys if key not in data]
    if missing:
        print(f"ERROR: Missing keys in export: {missing}")
        return False
    print("Validation passed: all required keys present.")
    return True


def generate_demo_data(export_data: dict) -> dict:
    """
    Transform the Colab export into the format needed by the website.
    This creates the demo_data.json that the website loads.
    """
    # Build a mapping table for the website to use
    # Shows how each of the 24 mappings translates dibits to nucleotides
    encoding_mappings = export_data.get("encoding_mappings", {})

    # Sample training curves (subsample for reasonable JSON size)
    training_curves = {}
    for key, curve_data in export_data.get("training_curves", {}).items():
        training_curves[key] = {
            "epoch_rewards": curve_data.get("epoch_rewards", []),
            "rolling_rewards": curve_data.get("rolling_rewards", []),
            "validation_rewards": curve_data.get("validation_rewards", []),
            "predictor_losses": curve_data.get("predictor_losses", []),
            "final_epoch": curve_data.get("final_epoch", 0),
            "stopped_early": curve_data.get("stopped_early", False),
        }

    demo_data = {
        "metadata": export_data["metadata"],
        "policy_lookup_table": export_data["policy_lookup_table"],
        "encoding_mappings": encoding_mappings,
        "training_curves": training_curves,
        "main_results": export_data["statistics"],
        "ablation_results": export_data["ablation_statistics"],
        "redundancy_comparison": export_data["redundancy_comparison"],
        "unseen_test_results": export_data["unseen_test_results"],
    }

    return demo_data


def generate_legacy_data(export_data: dict) -> dict:
    """Generate the _data.json format used by the existing website charts."""
    main_stats = export_data.get("statistics", [])
    ablation_stats = export_data.get("ablation_statistics", [])

    def format_row(row):
        formatted = {}
        for key, value in row.items():
            if isinstance(value, float):
                formatted[key] = str(value)
            else:
                formatted[key] = str(value) if value is not None else ""
        return formatted

    return {
        "main": [format_row(row) for row in main_stats],
        "ablation": [format_row(row) for row in ablation_stats],
    }


def print_summary(export_data: dict):
    """Print a human-readable summary of the results."""
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    metadata = export_data.get("metadata", {})
    print(f"\n  Training epochs:    {metadata.get('total_training_epochs', 'unknown')}")
    print(f"  Number of seeds:    {metadata.get('number_of_seeds', 'unknown')}")
    print(f"  Number of copies:   {metadata.get('number_of_copies', 'unknown')}")
    print(f"  Dropout rate:       {metadata.get('dropout_rate', 'unknown')}")

    # Main results
    print(f"\n{'Method':<50} {'Length':>6}  {'Error Rate':>12}  {'Accuracy':>10}")
    print("-" * 80)
    for stat in export_data.get("statistics", []):
        method = stat.get("method", "")
        sequence_length = stat.get("seq_length", "")
        error_mean = stat.get("average_error_rate_mean", 0)
        accuracy_mean = stat.get("decode_accuracy_mean", 0)
        print(f"  {method:<48} {sequence_length:>6}  {error_mean:>12.4f}  {accuracy_mean:>9.2f}%")

    # Redundancy comparison
    print(f"\n\n  REDUNDANCY COMPARISON (Accuracy by Number of Copies)")
    print("-" * 60)
    redundancy_data = export_data.get("redundancy_comparison", {})
    for sequence_length, comparisons in redundancy_data.items():
        print(f"\n  Sequence Length {sequence_length}:")
        for entry in comparisons:
            copies = entry["number_of_copies"]
            accuracy = entry["redundancy_accuracy"] * 100
            print(f"    {copies} copies: {accuracy:.2f}% accuracy")

    # Overfitting check
    print(f"\n\n  OVERFITTING CHECK (Unseen Test Data)")
    print("-" * 60)
    for sequence_length, results in export_data.get("unseen_test_results", {}).items():
        accuracy = results.get("decode_accuracy", 0)
        print(f"  Sequence length {sequence_length}: {accuracy:.2f}% accuracy on unseen data")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Process Colab training results")
    parser.add_argument("--input", required=True, help="Path to colab_export.json")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, do not generate outputs")
    arguments = parser.parse_args()

    # Load export
    print(f"Loading export from: {arguments.input}")
    with open(arguments.input, "r") as file:
        export_data = json.load(file)

    # Validate
    if not validate_export(export_data):
        sys.exit(1)

    if arguments.validate_only:
        print("Validation complete (--validate-only mode).")
        return

    # Print summary
    print_summary(export_data)

    # Generate demo_data.json for website
    demo_data = generate_demo_data(export_data)
    demo_path = os.path.join("website", "demo_data.json")
    os.makedirs("website", exist_ok=True)
    with open(demo_path, "w") as file:
        json.dump(demo_data, file, indent=2)
    print(f"\nWebsite demo data saved to: {demo_path}")
    print(f"  File size: {os.path.getsize(demo_path) / 1024:.1f} KB")

    # Generate legacy _data.json
    legacy_data = generate_legacy_data(export_data)
    legacy_path = os.path.join("results", "_data.json")
    with open(legacy_path, "w") as file:
        json.dump(legacy_data, file, indent=2)
    print(f"Legacy results data saved to: {legacy_path}")

    print("\n  DONE! Open website/index.html to see the demo.")


if __name__ == "__main__":
    main()
