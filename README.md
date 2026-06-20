# Predictive Reinforcement Learning-Based DNA Data Encoding

**🟢 Live Demo:** [https://adaptive-dna-encoding.netlify.app/](https://adaptive-dna-encoding.netlify.app/)

This project demonstrates a complete, visual pipeline for encoding digital binary data into DNA sequences. It uses Reinforcement Learning (Proximal Policy Optimisation) combined with an Error Predictor to learn the most robust mappings from bits to nucleotides (A, C, G, T) while maximising biological stability (GC content and minimising homopolymers).

To combat the inherent noise in DNA synthesis and sequencing, the pipeline utilises **multi-copy redundancy** (storing up to 10 copies of the DNA sequence) and a consensus majority-voting algorithm to dramatically improve data retrieval accuracy.

## Key Findings

| Metric | Value |
|--------|-------|
| Decode Accuracy (10 copies) | **100%** — zero data loss |
| Decode Accuracy (3 copies) | **~98.5%** |
| Decode Accuracy (1 copy) | **~97.5%** |
| Training Epochs | 100,000 per seed |
| Independent Seeds | 3 (seed 42, 43, 44) |
| Total Training Time | ~8.7 minutes (Google Colab T4 GPU) |

The Predictive Reinforcement Learning agent with multi-copy redundancy achieves **100% decode accuracy** on short sequences (64 nucleotides) using 10 redundant copies with majority voting — effectively zero data loss.

## Results Summary

| Method | Length 64 | Length 128 | Length 256 |
|--------|-----------|------------|------------|
| Fixed Baseline | 76.56% | 70.87% | 64.67% |
| Random Mapping | 76.13% | 70.51% | 64.41% |
| Reinforcement Learning | 76.94% | 71.18% | 64.93% |
| Predictive Reinforcement Learning | 76.62% | 71.14% | 64.03% |
| Predictive Reinforcement Learning + Redundancy | **82.23%** | **77.93%** | **71.15%** |

*Decode accuracy (%) averaged across 3 independent seeds with 100 evaluation trials per condition.*

## Features

- **Proximal Policy Optimisation Reinforcement Learning:** Dynamically chooses encoding mappings based on the specific binary data distribution.
- **Error Predictor:** Anticipates potential sequencing errors (substitutions, insertions, deletions) to penalise fragile sequences during training.
- **Multi-Copy Redundancy:** Encodes multiple distinct variations of the same data (1 to 10 copies). During decoding, a majority vote recovers the original data with high fidelity, heavily mitigating substitution errors.
- **Text and Image Encoding:** The interactive web demo supports encoding both plain text and image files (PNG, JPEG, GIF, BMP, WebP, and more) into DNA, demonstrating data-type agnostic operation.
- **Google Colab Training:** Slimmed-down notebook (`DNA_Training_Colab.ipynb`) for fast GPU training of the Reinforcement Learning agents.
- **Local Evaluation Pipeline:** Automated script (`run_full_pipeline.py`) that handles all testing on unseen data, ablation comparisons, and statistical chart generation.
- **Interactive Web Dashboard:** A responsive web interface (`website/index.html`) that visualises the entire encoding process, simulates errors, and displays the trained model's performance graphs.

## System Architecture

```
┌────────────────────────────┐
│  Layer 1: Reinforcement    │
│  Learning Agent (PPO)      │
│  ─ 6-dim state vector      │
│  ─ 96 composite actions    │
│  ─ 2×64 hidden, Tanh       │
└──────────┬─────────────────┘
           │ predicted error
           ▼
┌────────────────────────────┐
│  Layer 2: Error Predictor  │
│  ─ MLP estimates risk from │
│    GC content, entropy,    │
│    homopolymers, length    │
└──────────┬─────────────────┘
           │ store copies
           ▼
┌────────────────────────────┐
│  Redundancy: Multi-Copy    │
│  Storage + Majority Voting │
│  ─ 1–10 independent copies  │
│  ─ Position-wise consensus │
└────────────────────────────┘
```

## Repository Structure

- `DNA_Training_Colab.ipynb` — Ready-to-use notebook for GPU training on Google Colab.
- `run_full_pipeline.py` — Main execution script to process Colab results and generate the website data.
- `rl_agent.py`, `rl_environment.py` — Reinforcement learning model and environment.
- `redundancy.py` — Logic for multi-copy encoding and consensus voting.
- `error_simulation.py`, `error_predictor.py` — Sequence mutation and neural error prediction.
- `encoding.py`, `decoder.py` — Core logic for bit-to-DNA translation.
- `website/` — Contains the frontend interface (`index.html`, `style.css`, `script.js`).
- `results/` — Directory where generated charts, statistics, and model weights are stored.

## How to Run

### 1. Train the Models
1. Zip the Python scripts and `dataset/` folder into `dna_project.zip`.
2. Upload and run `DNA_Training_Colab.ipynb` on Google Colab (using a T4 GPU).
3. The notebook will automatically download `colab_trained_models.zip` when finished (takes approximately 10-15 minutes).

### 2. Evaluate Locally
1. Extract `colab_trained_models.zip` into the `results/` folder of this repository.
2. Run the evaluation pipeline:
   ```bash
   python run_full_pipeline.py --input results/colab_export.json
   ```
   *This evaluates accuracy, builds charts, and generates `website/demo_data.json`.*

### 3. View the Interactive Dashboard
Because web browsers block local files from fetching data for security reasons, you must run a simple local server:
```bash
python -m http.server 8000
```
Then, open your web browser and navigate to:
[http://localhost:8000/website/index.html](http://localhost:8000/website/index.html)
