import json
import csv
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------
# Mapping dictionaries
# ---------------------------------------------------------
METHOD_MAP = {
    "Fixed Baseline": "baseline",
    "Random Mapping": "random",
    "RL Agent": "reinforcement_learning",
    "Predictive RL": "predictive_reinforcement_learning",
    "Pred. RL + 3-copy": "predictive_reinforcement_learning_redundancy"
}

ABLATION_MAP = {
    "Baseline": "baseline",
    "Full RL": "reinforcement_learning",
    "Predictive RL": "predictive_reinforcement_learning",
    "RL w/o GC": "reinforcement_learning_no_gc",
    "RL w/o Homopolymer": "reinforcement_learning_no_homopolymer"
}

# ---------------------------------------------------------
# Load original data
# ---------------------------------------------------------
with open('website/demo_data.json', 'r') as f:
    demo_data = json.load(f)

# ---------------------------------------------------------
# Update main_results
# ---------------------------------------------------------
with open('results/rerun/results_raw.csv', 'r') as f:
    results_raw = list(csv.DictReader(f))

# Group by (method, length)
main_grouped = defaultdict(lambda: defaultdict(list))
for row in results_raw:
    m = METHOD_MAP[row["method"]]
    l = int(row["length"])
    
    acc = float(row["decode_accuracy_pct"])
    err = float(row["error_rate"])
    gc = float(row["gc_content"])
    stab = float(row["stability_score"])
    
    gc_score = 1.0 - 2.0 * abs(gc - 0.5)
    eff = (acc / 100.0) * (1.0 - err)
    
    main_grouped[(m, l)]["decode_accuracy"].append(acc)
    main_grouped[(m, l)]["error_rate"].append(err)
    main_grouped[(m, l)]["gc_score"].append(gc_score)
    main_grouped[(m, l)]["stability_score"].append(stab)
    main_grouped[(m, l)]["encoding_efficiency"].append(eff)
    main_grouped[(m, l)]["bit_accuracy"].append(acc / 100.0)

new_main = []
for (m, l), data in main_grouped.items():
    entry = {
        "method": m,
        "seq_length": l,
        "average_error_rate_mean": float(np.mean(data["error_rate"])),
        "average_error_rate_std": float(np.std(data["error_rate"], ddof=0)),
        "decode_accuracy_mean": float(np.mean(data["decode_accuracy"])),
        "decode_accuracy_std": float(np.std(data["decode_accuracy"], ddof=0)),
        "gc_score_mean": float(np.mean(data["gc_score"])),
        "gc_score_std": float(np.std(data["gc_score"], ddof=0)),
        "stability_score_mean": float(np.mean(data["stability_score"])),
        "stability_score_std": float(np.std(data["stability_score"], ddof=0)),
        "encoding_efficiency_mean": float(np.mean(data["encoding_efficiency"])),
        "encoding_efficiency_std": float(np.std(data["encoding_efficiency"], ddof=0)),
        "average_bit_accuracy_mean": float(np.mean(data["bit_accuracy"])),
        "average_bit_accuracy_std": float(np.std(data["bit_accuracy"], ddof=0)),
    }
    new_main.append(entry)

demo_data["main_results"] = new_main

# ---------------------------------------------------------
# Update ablation_results
# ---------------------------------------------------------
with open('results/rerun/ablation_raw.csv', 'r') as f:
    ablation_raw = list(csv.DictReader(f))

ablation_grouped = defaultdict(lambda: defaultdict(list))
for row in ablation_raw:
    m = ABLATION_MAP[row["condition"]]
    l = int(row["length"])
    acc = float(row["decode_accuracy_pct"])
    err = float(row["error_rate"])
    
    ablation_grouped[(m, l)]["decode_accuracy"].append(acc)
    ablation_grouped[(m, l)]["error_rate"].append(err)

new_ablation = []
for (m, l), data in ablation_grouped.items():
    entry = {
        "method": m,
        "seq_length": l,
        "average_error_rate_mean": float(np.mean(data["error_rate"])),
        "average_error_rate_std": float(np.std(data["error_rate"], ddof=0)),
        "decode_accuracy_mean": float(np.mean(data["decode_accuracy"])),
        "decode_accuracy_std": float(np.std(data["decode_accuracy"], ddof=0))
    }
    new_ablation.append(entry)

demo_data["ablation_results"] = new_ablation

# ---------------------------------------------------------
# Update redundancy_scaling
# ---------------------------------------------------------
with open('results/rerun/redundancy_raw.csv', 'r') as f:
    redundancy_raw = list(csv.DictReader(f))

redundancy_grouped = defaultdict(lambda: defaultdict(list))
for row in redundancy_raw:
    c = int(row["n_copies"])
    l = int(row["length"])
    acc = float(row["decode_accuracy_pct"])
    redundancy_grouped[(c, l)]["decode_accuracy"].append(acc)

new_redundancy = []
for (c, l), data in redundancy_grouped.items():
    entry = {
        "number_of_copies": c,
        "seq_length": l,
        "decode_accuracy_mean": float(np.mean(data["decode_accuracy"])),
        "decode_accuracy_std": float(np.std(data["decode_accuracy"], ddof=0))
    }
    new_redundancy.append(entry)

demo_data["redundancy_scaling"] = new_redundancy

# ---------------------------------------------------------
# Update unseen_test_results
# ---------------------------------------------------------
with open('results/rerun/overfit_raw.csv', 'r') as f:
    overfit_raw = list(csv.DictReader(f))

unseen_grouped = defaultdict(list)
for row in overfit_raw:
    l = str(row["length"])
    test_acc = float(row["test_accuracy_pct"])
    unseen_grouped[l].append(test_acc)

for l, accs in unseen_grouped.items():
    mean_acc = float(np.mean(accs))
    if "unseen_test_results" not in demo_data:
        demo_data["unseen_test_results"] = {}
    if l not in demo_data["unseen_test_results"]:
        demo_data["unseen_test_results"][l] = {}
        
    demo_data["unseen_test_results"][l]["decode_accuracy"] = mean_acc
    demo_data["unseen_test_results"][l]["average_bit_accuracy"] = mean_acc / 100.0

# ---------------------------------------------------------
# Save updated data
# ---------------------------------------------------------
with open('website/demo_data.json', 'w') as f:
    json.dump(demo_data, f)
    
print("Successfully updated website/demo_data.json with the new 7-seed metrics!")
