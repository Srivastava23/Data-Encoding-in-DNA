# %% [markdown]
# # DNA Data Encoding — Colab Training Only
#
# This notebook ONLY trains the reinforcement learning agents and exports
# the model weights + training curves. All evaluation, redundancy testing,
# plot generation, and website data processing is done locally.
#
# **Steps:**
# 1. Upload project files
# 2. Train agents (100,000 epochs on GPU)
# 3. Download model weights + training curves
# 4. Run `py run_full_pipeline.py --input results/colab_export.json` locally

# %% [markdown]
# ## Cell 1: Setup

# %%
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "numpy", "scipy", "matplotlib", "-q"])

print("Dependencies installed.")

import torch
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected. Training will be slower.")

# %% [markdown]
# ## Cell 2: Upload Project Files
#
# Run this cell, then click "Choose Files" and select your `dna_project.zip`.

# %%
import os
import zipfile

from google.colab import files
print("Select your dna_project.zip file:")
uploaded = files.upload()

zip_filename = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_filename, 'r') as zip_reference:
    zip_reference.extractall('/content/dna_project')

os.chdir('/content/dna_project')
print("\nProject files extracted:")
for filename in sorted(os.listdir('.')):
    print(f"  {filename}")

# %% [markdown]
# ## Cell 3: Configure Training

# %%
import numpy as np
from config import Config

config = Config(
    ppo_training_epochs=100000,
    number_of_seeds=7,
    ppo_learning_rate=3e-4,
    ppo_batch_size=64,
    ppo_hidden_dimension=64,
    dropout_rate=0.1,
    use_learning_rate_scheduling=True,
    early_stopping_patience=10000,
    validation_split=0.2,
    number_of_copies=3,
    use_error_predictor=True,
)

print("=" * 60)
print("  Training Configuration")
print("=" * 60)
for field_name in sorted(vars(config)):
    if not field_name.startswith("_"):
        print(f"  {field_name:<40} = {getattr(config, field_name)}")
print("=" * 60)

# %% [markdown]
# ## Cell 4: Train All Agents
#
# This is the main training cell. It trains 4 agents per seed:
# 1. Reinforcement learning agent (without predictor)
# 2. Predictive reinforcement learning agent (with predictor)
# 3. Ablation: without GC reward
# 4. Ablation: without homopolymer penalty
#
# **Estimated time: 1–1.5 hours on T4 GPU.**

# %%
from training_pipeline import train
import json
import time

def set_global_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

os.makedirs(config.output_directory, exist_ok=True)
os.makedirs(os.path.join(config.output_directory, "checkpoints"), exist_ok=True)

all_training_data = {}
all_model_paths = {}
overall_start_time = time.time()

for seed_index in range(config.number_of_seeds):
    seed = config.seed + seed_index
    set_global_seeds(seed)
    print(f"\n{'=' * 60}")
    print(f"  Seed {seed_index + 1}/{config.number_of_seeds}  (seed={seed})")
    print(f"{'=' * 60}")

    # --- Agent 1: Reinforcement learning (no predictor) ---
    print("\n  [1/4] Training reinforcement learning agent (without predictor)...")
    start_time = time.time()
    config_no_predictor = Config(
        seed=seed, use_error_predictor=False,
        ppo_training_epochs=config.ppo_training_epochs,
        dropout_rate=config.dropout_rate,
        use_learning_rate_scheduling=config.use_learning_rate_scheduling,
        early_stopping_patience=config.early_stopping_patience,
    )
    result_reinforcement_learning = train(config_no_predictor, seed=seed, verbose=True)
    elapsed = time.time() - start_time
    print(f"  Done in {elapsed/60:.1f} minutes. "
          f"Final epoch: {result_reinforcement_learning['final_epoch']}  "
          f"Early stopped: {result_reinforcement_learning['stopped_early']}")

    # Save model
    model_path = os.path.join(config.output_directory, "checkpoints", f"reinforcement_learning_seed{seed}.pt")
    result_reinforcement_learning["agent"].save(model_path)
    all_model_paths[f"reinforcement_learning_seed{seed}"] = model_path

    all_training_data[f"seed_{seed}_reinforcement_learning"] = {
        "epoch_rewards": result_reinforcement_learning["epoch_rewards"],
        "rolling_rewards": result_reinforcement_learning["rolling_rewards"],
        "validation_rewards": result_reinforcement_learning["validation_rewards"],
        "policy_losses": result_reinforcement_learning["policy_losses"],
        "value_losses": result_reinforcement_learning["value_losses"],
        "learning_rates": result_reinforcement_learning["learning_rates"],
        "final_epoch": result_reinforcement_learning["final_epoch"],
        "stopped_early": result_reinforcement_learning["stopped_early"],
    }

    # --- Agent 2: Predictive reinforcement learning (with predictor) ---
    print("\n  [2/4] Training predictive reinforcement learning agent (with predictor)...")
    start_time = time.time()
    config_with_predictor = Config(
        seed=seed, use_error_predictor=True,
        ppo_training_epochs=config.ppo_training_epochs,
        dropout_rate=config.dropout_rate,
        use_learning_rate_scheduling=config.use_learning_rate_scheduling,
        early_stopping_patience=config.early_stopping_patience,
    )
    result_predictive = train(config_with_predictor, seed=seed, verbose=True)
    elapsed = time.time() - start_time
    print(f"  Done in {elapsed/60:.1f} minutes. "
          f"Final epoch: {result_predictive['final_epoch']}  "
          f"Early stopped: {result_predictive['stopped_early']}")

    model_path = os.path.join(config.output_directory, "checkpoints", f"predictive_seed{seed}.pt")
    result_predictive["agent"].save(model_path)
    all_model_paths[f"predictive_seed{seed}"] = model_path

    if result_predictive["predictor"] is not None:
        predictor_path = os.path.join(config.output_directory, "checkpoints", f"predictor_seed{seed}.pt")
        result_predictive["predictor"].save(predictor_path)
        all_model_paths[f"predictor_seed{seed}"] = predictor_path

    all_training_data[f"seed_{seed}_predictive"] = {
        "epoch_rewards": result_predictive["epoch_rewards"],
        "rolling_rewards": result_predictive["rolling_rewards"],
        "validation_rewards": result_predictive["validation_rewards"],
        "predictor_losses": result_predictive["predictor_losses"],
        "policy_losses": result_predictive["policy_losses"],
        "value_losses": result_predictive["value_losses"],
        "learning_rates": result_predictive["learning_rates"],
        "final_epoch": result_predictive["final_epoch"],
        "stopped_early": result_predictive["stopped_early"],
    }

    # --- Agent 3: Ablation without GC reward ---
    print("\n  [3/4] Training ablation: without GC reward...")
    start_time = time.time()
    config_no_gc = Config(
        seed=seed, use_error_predictor=False,
        ppo_training_epochs=config.ppo_training_epochs,
        reward_gc_bonus_weight=0.0,
        dropout_rate=config.dropout_rate,
        use_learning_rate_scheduling=config.use_learning_rate_scheduling,
        early_stopping_patience=config.early_stopping_patience,
    )
    result_no_gc = train(config_no_gc, seed=seed, verbose=False)
    elapsed = time.time() - start_time
    print(f"  Done in {elapsed/60:.1f} minutes.")

    model_path = os.path.join(config.output_directory, "checkpoints", f"ablation_no_gc_seed{seed}.pt")
    result_no_gc["agent"].save(model_path)
    all_model_paths[f"ablation_no_gc_seed{seed}"] = model_path

    # --- Agent 4: Ablation without homopolymer penalty ---
    print("\n  [4/4] Training ablation: without homopolymer penalty...")
    start_time = time.time()
    config_no_homopolymer = Config(
        seed=seed, use_error_predictor=False,
        ppo_training_epochs=config.ppo_training_epochs,
        reward_homopolymer_penalty_weight=0.0,
        dropout_rate=config.dropout_rate,
        use_learning_rate_scheduling=config.use_learning_rate_scheduling,
        early_stopping_patience=config.early_stopping_patience,
    )
    result_no_homopolymer = train(config_no_homopolymer, seed=seed, verbose=False)
    elapsed = time.time() - start_time
    print(f"  Done in {elapsed/60:.1f} minutes.")

    model_path = os.path.join(config.output_directory, "checkpoints", f"ablation_no_homopolymer_seed{seed}.pt")
    result_no_homopolymer["agent"].save(model_path)
    all_model_paths[f"ablation_no_homopolymer_seed{seed}"] = model_path

total_time = time.time() - overall_start_time
print(f"\n{'=' * 60}")
print(f"  ALL TRAINING COMPLETE — Total time: {total_time/60:.1f} minutes")
print(f"{'=' * 60}")

# %% [markdown]
# ## Cell 5: Export Training Data + Model Weights

# %%
import shutil

def subsample_list(data_list, maximum_points=1000):
    """Keep at most maximum_points entries for manageable JSON size."""
    if len(data_list) <= maximum_points:
        return data_list
    step = len(data_list) // maximum_points
    return data_list[::step]

# Build export (training curves only — evaluation is done locally)
colab_export = {
    "metadata": {
        "total_training_epochs": config.ppo_training_epochs,
        "number_of_seeds": config.number_of_seeds,
        "seeds": [config.seed + i for i in range(config.number_of_seeds)],
        "dropout_rate": config.dropout_rate,
        "learning_rate_scheduling": config.use_learning_rate_scheduling,
        "early_stopping_patience": config.early_stopping_patience,
        "number_of_copies": config.number_of_copies,
        "sequence_lengths": config.sequence_lengths,
        "ppo_learning_rate": config.ppo_learning_rate,
        "ppo_batch_size": config.ppo_batch_size,
        "ppo_hidden_dimension": config.ppo_hidden_dimension,
        "total_training_time_minutes": round(total_time / 60, 1),
    },
    "training_curves": {},
    "model_paths": all_model_paths,
}

# Subsample training curves for reasonable file size
for key, data in all_training_data.items():
    colab_export["training_curves"][key] = {
        "epoch_rewards": subsample_list(data["epoch_rewards"]),
        "rolling_rewards": subsample_list(data["rolling_rewards"]),
        "validation_rewards": data["validation_rewards"],
        "predictor_losses": data.get("predictor_losses", []),
        "policy_losses": subsample_list(data.get("policy_losses", [])),
        "value_losses": subsample_list(data.get("value_losses", [])),
        "learning_rates": subsample_list(data.get("learning_rates", [])),
        "final_epoch": data["final_epoch"],
        "stopped_early": data["stopped_early"],
    }

export_path = os.path.join(config.output_directory, "colab_export.json")
with open(export_path, "w") as file:
    json.dump(colab_export, file, indent=2)

print(f"Training data exported to: {export_path}")
print(f"  File size: {os.path.getsize(export_path) / 1024:.1f} KB")
print(f"\nModel weights saved in: {config.output_directory}/checkpoints/")
for name, path in all_model_paths.items():
    size = os.path.getsize(path) / 1024
    print(f"  {name}: {size:.1f} KB")

# %% [markdown]
# ## Cell 6: Download Everything
#
# This creates a single zip with model weights + training curves.

# %%
archive_name = "colab_trained_models"
shutil.make_archive(archive_name, "zip", ".", config.output_directory)
print(f"Archive created: {archive_name}.zip")
print(f"Size: {os.path.getsize(f'{archive_name}.zip') / (1024*1024):.1f} MB")

from google.colab import files
files.download(f"{archive_name}.zip")

print("\n" + "=" * 60)
print("  NEXT STEPS (on your local machine):")
print("=" * 60)
print("  1. Extract colab_trained_models.zip into your project's results/ folder")
print("  2. Run:  py run_full_pipeline.py --input results/colab_export.json")
print("  3. Open website/index.html to see the full demo")
print("=" * 60)
