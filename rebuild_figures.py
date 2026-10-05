import os
import sys
import csv
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. SETUP
# ----------------------------------------------------------------------------
OUTPUT_DIR = "figures_for_paper_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROVENANCE_FILE = os.path.join(OUTPUT_DIR, "figure_provenance.md")

RERUN_DIR = "results/rerun"
ZIP_PATH = "results/colab_trained_models.zip"

METHOD_COLORS = {
    "Fixed Baseline": "#e74c3c",
    "Random Mapping": "#f39c12",
    "RL Agent": "#3498db",
    "Predictive RL": "#2ecc71",
    "Pred. RL + 3-copy": "#9b59b6",
}

ABLATION_COLORS = {
    "Baseline": "#e74c3c",
    "Full RL": "#3498db",
    "Predictive RL": "#2ecc71",
    "RL w/o GC": "#9b59b6",
    "RL w/o Homopolymer": "#f39c12",
}

LENGTH_COLORS = {
    64: "#e74c3c",
    128: "#3498db",
    256: "#2ecc71",
    512: "#f39c12",
    1024: "#9b59b6",
}
LENGTH_MARKERS = {64: "o", 128: "s", 256: "D", 512: "^", 1024: "v"}

SEEDS = [42, 43, 44, 45, 46, 47, 48]
LENGTHS = [64, 128, 256, 512, 1024]

# We will collect checks to print later
CHECKS = []
PROVENANCE = []

# Initialize provenance file
with open(PROVENANCE_FILE, "w") as f:
    f.write("# Figure Provenance\n\n")
    f.write("Generated using `rebuild_figures.py`.\n")
    f.write("Data sources: `results/rerun/` CSV files.\n\n")

def add_provenance(fig_name, source, seeds, n_per_point, error_bars, extra=""):
    with open(PROVENANCE_FILE, "a") as f:
        f.write(f"## {fig_name}\n")
        f.write(f"- **Source Data**: {source}\n")
        f.write(f"- **Seeds Included**: {seeds}\n")
        f.write(f"- **N per point**: {n_per_point}\n")
        f.write(f"- **Error bars**: {error_bars}\n")
        if extra:
            f.write(f"- **Details**: {extra}\n")
        f.write("\n")

def save_fig_data(name, headers, rows):
    path = os.path.join(OUTPUT_DIR, f"figure_data_{name}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def _save_figure(figure, name):
    path = os.path.join(OUTPUT_DIR, name)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)

# ----------------------------------------------------------------------------
# 2. LOAD DATA
# ----------------------------------------------------------------------------
def load_csv(filename):
    path = os.path.join(RERUN_DIR, filename)
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

results_raw = load_csv("results_raw.csv")
redundancy_raw = load_csv("redundancy_raw.csv")
ablation_raw = load_csv("ablation_raw.csv")

# ----------------------------------------------------------------------------
# 3. GENERATE FIGURES
# ----------------------------------------------------------------------------
def generate_grouped_bar(data_rows, y_col, fig_name, title, ylabel, methods, colors, save_data_name):
    # Calculate means and stds
    stats = {}
    for m in methods:
        stats[m] = {l: [] for l in LENGTHS}
    
    for row in data_rows:
        m = row["method"]
        if m in stats:
            l = int(row["length"])
            stats[m][l].append(float(row[y_col]))
    
    # Check that we have 7 seeds per point
    for m in methods:
        for l in LENGTHS:
            if len(stats[m][l]) != 7:
                CHECKS.append(f"FAIL: {fig_name} {m} {l} has {len(stats[m][l])} seeds instead of 7")
                
    means = {m: [np.mean(stats[m][l]) for l in LENGTHS] for m in methods}
    stds = {m: [np.std(stats[m][l], ddof=0) for l in LENGTHS] for m in methods}
    
    # Save data
    headers = ["Method", "Length", "Mean", "Std"]
    out_rows = []
    for m in methods:
        for i, l in enumerate(LENGTHS):
            out_rows.append([m, l, means[m][i], stds[m][i]])
    save_fig_data(save_data_name, headers, out_rows)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(LENGTHS))
    width = 0.8 / len(methods)
    
    for i, m in enumerate(methods):
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, means[m], width, yerr=stds[m], capsize=3,
               color=colors.get(m, "#000"), label=m, alpha=0.85)
               
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in LENGTHS], fontfamily="serif")
    ax.set_xlabel("Sequence Length (nt)", fontfamily="serif", fontsize=12)
    ax.set_ylabel(ylabel, fontfamily="serif", fontsize=12)
    ax.set_title(title, fontfamily="serif", fontsize=14)
    # Legend outside to avoid overlap
    ax.legend(prop={'family': 'serif', 'size': 10}, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis="y", alpha=0.3)
    
    _save_figure(fig, fig_name)
    
    # Print out values for checks
    for m in methods:
        for i, l in enumerate(LENGTHS):
            # We round to match markdown
            val = means[m][i]
            std = stds[m][i]
            # Will be checked visually or by script output
            
# Fig 5: Error Rate
methods_main = ["Fixed Baseline", "Random Mapping", "RL Agent", "Predictive RL", "Pred. RL + 3-copy"]
generate_grouped_bar(results_raw, "error_rate", "fig5_error_rate_comparison.png",
                    "Error Rate Comparison", "Average Error Rate", methods_main, METHOD_COLORS, "fig5")
add_provenance("fig5_error_rate_comparison.png", "results_raw.csv (error_rate)", "42-48", "7 means (each mean of 100 trials)", "Standard deviation across the 7 per-seed means")

# Fig 6: Decode Accuracy
generate_grouped_bar(results_raw, "decode_accuracy_pct", "fig6_decode_accuracy_comparison.png",
                    "Decode Accuracy Comparison", "Decode Accuracy (%)", methods_main, METHOD_COLORS, "fig6")
add_provenance("fig6_decode_accuracy_comparison.png", "results_raw.csv (decode_accuracy_pct)", "42-48", "7 means (each mean of 100 trials)", "Standard deviation across the 7 per-seed means", "Legend moved outside plot to avoid hiding data.")

# Fig 7: GC Content
fig, ax = plt.subplots(figsize=(10, 6))
headers = ["Method", "GC_Content_Mean_Of_100_Trials"]
out_rows = []
for m in methods_main:
    gc_vals = [float(r["gc_content"]) for r in results_raw if r["method"] == m]
    for v in gc_vals: out_rows.append([m, v])
    ax.hist(gc_vals, bins=20, alpha=0.6, color=METHOD_COLORS.get(m, "#000"), label=m)
save_fig_data("fig7", headers, out_rows)
ax.axvline(0.40, color="red", linestyle="--", alpha=0.5, label="Target range")
ax.axvline(0.60, color="red", linestyle="--", alpha=0.5)
ax.set_xlabel("GC Content", fontfamily="serif", fontsize=12)
ax.set_ylabel("Frequency (of per-seed means)", fontfamily="serif", fontsize=12)
ax.set_title("GC Content Distribution", fontfamily="serif", fontsize=14)
ax.legend(prop={'family': 'serif', 'size': 10}, bbox_to_anchor=(1.05, 1), loc='upper left')
_save_figure(fig, "fig7_gc_content_distribution.png")
add_provenance("fig7_gc_content_distribution.png", "results_raw.csv (gc_content)", "42-48", "35 means per method (7 seeds x 5 lengths)", "N/A (Histogram)", "Distribution is over the per-seed means, aggregated across all lengths.")

# Fig 8: Encoding Efficiency
# Formula: average_bit_accuracy * (1.0 - average_error_rate)
# From script: decode_accuracy_pct is bit accuracy * 100
with open(PROVENANCE_FILE, "a") as f:
    f.write("## Composite Efficiency Score Formula\n")
    f.write("Formula: `encoding_efficiency = (decode_accuracy_pct / 100.0) * (1.0 - error_rate)`\n")
    f.write("Justification: The efficiency score is simply the product of the bit-level accuracy and the sequence retention rate (1 - error rate), representing the total fraction of correct information successfully transmitted.\n\n")

eff_rows = []
for r in results_raw:
    nr = dict(r)
    eff = (float(r["decode_accuracy_pct"]) / 100.0) * (1.0 - float(r["error_rate"]))
    nr["encoding_efficiency"] = eff
    eff_rows.append(nr)
generate_grouped_bar(eff_rows, "encoding_efficiency", "fig8_encoding_efficiency_comparison.png",
                    "Encoding Efficiency Comparison", "Encoding Efficiency", methods_main, METHOD_COLORS, "fig8")
add_provenance("fig8_encoding_efficiency_comparison.png", "results_raw.csv (computed)", "42-48", "7 means", "Standard deviation across the 7 per-seed means")

# Fig 9: Ablation
methods_abl = ["Baseline", "Full RL", "Predictive RL", "RL w/o GC", "RL w/o Homopolymer"]
stats_err = {m: {l: [] for l in LENGTHS} for m in methods_abl}
stats_acc = {m: {l: [] for l in LENGTHS} for m in methods_abl}
for row in ablation_raw:
    m = row["condition"]
    l = int(row["length"])
    stats_err[m][l].append(float(row["error_rate"]))
    stats_acc[m][l].append(float(row["decode_accuracy_pct"]))
    
means_err = {m: [np.mean(stats_err[m][l]) for l in LENGTHS] for m in methods_abl}
stds_err = {m: [np.std(stats_err[m][l], ddof=0) for l in LENGTHS] for m in methods_abl}
means_acc = {m: [np.mean(stats_acc[m][l]) for l in LENGTHS] for m in methods_abl}
stds_acc = {m: [np.std(stats_acc[m][l], ddof=0) for l in LENGTHS] for m in methods_abl}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(LENGTHS))
width = 0.8 / len(methods_abl)

out_rows = []
for i, m in enumerate(methods_abl):
    offset = (i - len(methods_abl) / 2 + 0.5) * width
    ax1.bar(x + offset, means_err[m], width, yerr=stds_err[m], capsize=3, color=ABLATION_COLORS[m], label=m)
    ax2.bar(x + offset, means_acc[m], width, yerr=stds_acc[m], capsize=3, color=ABLATION_COLORS[m], label=m)
    for j, l in enumerate(LENGTHS):
        out_rows.append([m, l, means_err[m][j], stds_err[m][j], means_acc[m][j], stds_acc[m][j]])
save_fig_data("fig9", ["Method", "Length", "Err_Mean", "Err_Std", "Acc_Mean", "Acc_Std"], out_rows)

ax1.set_xticks(x); ax1.set_xticklabels([str(l) for l in LENGTHS], fontfamily="serif")
ax2.set_xticks(x); ax2.set_xticklabels([str(l) for l in LENGTHS], fontfamily="serif")
ax1.set_ylabel("Average Error Rate", fontfamily="serif")
ax2.set_ylabel("Decode Accuracy (%)", fontfamily="serif")
ax1.set_title("Ablation: Error Rate", fontfamily="serif")
ax2.set_title("Ablation: Decode Accuracy", fontfamily="serif")
ax2.legend(prop={'family': 'serif', 'size': 10}, bbox_to_anchor=(1.05, 1), loc='upper left')
_save_figure(fig, "fig9_ablation_study_comparison.png")
add_provenance("fig9_ablation_study_comparison.png", "ablation_raw.csv", "42-48", "7 means", "Standard deviation across the 7 per-seed means")

# Fig 11: Redundancy Scaling
copies = sorted(list(set(int(r["n_copies"]) for r in redundancy_raw)))
stats_red = {l: {c: [] for c in copies} for l in LENGTHS}
for row in redundancy_raw:
    l = int(row["length"])
    c = int(row["n_copies"])
    stats_red[l][c].append(float(row["decode_accuracy_pct"]))

fig, ax = plt.subplots(figsize=(10, 6))
out_rows = []
for l in LENGTHS:
    ms = [np.mean(stats_red[l][c]) for c in copies]
    ss = [np.std(stats_red[l][c], ddof=0) for c in copies]
    for i, c in enumerate(copies): out_rows.append([l, c, ms[i], ss[i]])
    ax.errorbar(copies, ms, yerr=ss, marker=LENGTH_MARKERS[l], color=LENGTH_COLORS[l], label=f"Length {l}", capsize=4)
save_fig_data("fig11", ["Length", "Copies", "Acc_Mean", "Acc_Std"], out_rows)

ax.set_xticks(copies)
ax.set_xlabel("Number of Redundant Copies", fontfamily="serif")
ax.set_ylabel("Decode Accuracy (%)", fontfamily="serif")
ax.set_title("Redundancy Scaling", fontfamily="serif")
ax.legend(prop={'family': 'serif'})
_save_figure(fig, "fig11_redundancy_scaling.png")
add_provenance("fig11_redundancy_scaling.png", "redundancy_raw.csv", "42-48", "7 means", "Standard deviation across the 7 per-seed means")

# ----------------------------------------------------------------------------
# 4. TRAINING CURVES (From colab export in ZIP)
# ----------------------------------------------------------------------------
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    export_data = json.loads(z.read("results/colab_export.json"))
curves = export_data["training_curves"]
meta = export_data["metadata"]

# EMA Smoothing
def apply_ema(data, alpha=0.02):
    res = []
    for x in data:
        if not res: res.append(x)
        else: res.append(alpha * x + (1 - alpha) * res[-1])
    return res

# RL Agent curves
fig, ax = plt.subplots(figsize=(10, 6))
out_rows = []
for s in SEEDS:
    k = f"seed_{s}_reinforcement_learning"
    if k in curves:
        raw_rw = curves[k]["epoch_rewards"]
        ema_rw = apply_ema(raw_rw)
        ax.plot(ema_rw, label=f"Seed {s}")
        for ep, val in enumerate(ema_rw): out_rows.append([s, ep, val])
save_fig_data("fig3", ["Seed", "Epoch", "EMA_Reward"], out_rows)
ax.set_xlabel("Epoch", fontfamily="serif")
ax.set_ylabel("Reward", fontfamily="serif")
ax.set_title("Training Reward: RL Agent", fontfamily="serif")
ax.legend(prop={'family': 'serif'})
_save_figure(fig, "fig3_training_reward_rl.png")
add_provenance("fig3_training_reward_rl.png", "colab_export.json inside ZIP", "42-48", "1 per epoch", "N/A", "Unnormalised. Smoothed with EMA alpha=0.02.")

# Predictive RL curves
fig, ax = plt.subplots(figsize=(10, 6))
out_rows = []
for s in SEEDS:
    k = f"seed_{s}_predictive"
    if k in curves:
        raw_rw = curves[k]["epoch_rewards"]
        ema_rw = apply_ema(raw_rw)
        ax.plot(ema_rw, label=f"Seed {s}")
        for ep, val in enumerate(ema_rw): out_rows.append([s, ep, val])
save_fig_data("fig4", ["Seed", "Epoch", "EMA_Reward"], out_rows)
ax.set_xlabel("Epoch", fontfamily="serif")
ax.set_ylabel("Reward", fontfamily="serif")
ax.set_title("Training Reward: Predictive RL", fontfamily="serif")
ax.legend(prop={'family': 'serif'})
_save_figure(fig, "fig4_training_reward_pred.png")
add_provenance("fig4_training_reward_pred.png", "colab_export.json inside ZIP", "42-48", "1 per epoch", "N/A", "Unnormalised. Smoothed with EMA alpha=0.02.")

# Predictor Loss
fig, ax = plt.subplots(figsize=(10, 6))
out_rows = []
for s in SEEDS:
    k = f"seed_{s}_predictive"
    if k in curves:
        raw_loss = curves[k].get("predictor_losses", [])
        if raw_loss:
            ema_loss = apply_ema(raw_loss)
            ax.plot(ema_loss, label=f"Seed {s}")
            for ep, val in enumerate(ema_loss): out_rows.append([s, ep, val])
save_fig_data("fig10", ["Seed", "Epoch", "EMA_Loss"], out_rows)
ax.set_xlabel("Epoch", fontfamily="serif")
ax.set_ylabel("MSE Loss", fontfamily="serif")
ax.set_title("Predictor Training Loss", fontfamily="serif")
ax.legend(prop={'family': 'serif'})
_save_figure(fig, "fig10_predictor_loss.png")
add_provenance("fig10_predictor_loss.png", "colab_export.json inside ZIP", "42-48", "1 per epoch", "N/A", "Unnormalised MSE. Smoothed with EMA alpha=0.02.")

# Printing training stats
print("=== TRAINING METRICS ===")
time_mins = meta.get("total_training_time_minutes", -1)
print(f"Total training time: {time_mins} minutes (Paper claimed ~19.9 mins)")
all_stops = []
for s in SEEDS:
    k1 = f"seed_{s}_reinforcement_learning"
    k2 = f"seed_{s}_predictive"
    if k1 in curves and k2 in curves:
        stop1 = curves[k1].get("final_epoch")
        stop2 = curves[k2].get("final_epoch")
        all_stops.extend([stop1, stop2])
        
        # RL metrics
        r1 = curves[k1]["epoch_rewards"]
        e1 = apply_ema(r1)
        # Pred metrics
        r2 = curves[k2]["epoch_rewards"]
        e2 = apply_ema(r2)
        l2 = curves[k2].get("predictor_losses", [])
        em_l2 = apply_ema(l2) if l2 else []
        
        print(f"Seed {s}:")
        print(f"  RL final epoch: {stop1}")
        print(f"  Pred final epoch: {stop2}")
        print(f"  Pred MSE: first={l2[0]:.4f} last={l2[-1]:.4f}")
        print(f"  RL EMA reward: first={e1[0]:.4f} last={e1[-1]:.4f}")
        print(f"  Pred EMA reward: first={e2[0]:.4f} last={e2[-1]:.4f}")

min_stop = min(all_stops)
max_stop = max(all_stops)
print(f"Early stopping range: {min_stop} to {max_stop} (Paper claimed 11,000 to 25,500)")
print("=== CHECKS ===")
for c in CHECKS: print(c)
if not CHECKS: print("PASS: All structural data checks passed.")

# PROPOSED CAPTIONS
with open(PROVENANCE_FILE, "a") as f:
    f.write("## Proposed Captions\n")
    f.write("1. **Fig 5**: Error rate across all five encoding methods for five sequence lengths (64 to 1024 nt). Error bars represent the standard deviation across the 7 training seeds.\n")
    f.write("2. **Fig 6**: Decode accuracy comparison across all methods and lengths. Error bars show standard deviation over the 7 independent seeds.\n")
    f.write("3. **Fig 7**: GC content distribution of encoded sequences aggregated across all evaluation lengths, plotted over the per-seed means (n=35 per method). Red lines indicate the target optimal biological range (40-60%).\n")
    f.write("4. **Fig 8**: Encoding efficiency score (Bit Accuracy × Sequence Retention) across methods. Error bars denote the standard deviation across 7 seeds.\n")
    f.write("5. **Fig 9**: Ablation study evaluating the contribution of individual reward components to the overall error rate and decode accuracy. Error bars represent standard deviation across 7 seeds.\n")
    f.write("6. **Fig 11**: Decode accuracy as a function of the number of redundant copies used for consensus voting in the Predictive RL agent. Error bars reflect standard deviation across 7 seeds.\n")

print("Generated all files in figures_for_paper_v2/")
