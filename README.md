# Predictive Reinforcement Learning-Based DNA Data Encoding

**🟢 Live Demo:** [https://adaptive-dna-encoding.netlify.app/](https://adaptive-dna-encoding.netlify.app/)

This project demonstrates a complete, visual pipeline for encoding digital binary data into DNA sequences. It uses Reinforcement Learning (Proximal Policy Optimisation) combined with an Error Predictor to learn the most robust mappings from bits to nucleotides (A, C, G, T) while maximising biological stability (GC content and minimising homopolymers).

To combat the inherent noise in DNA synthesis and sequencing, the pipeline utilises **multi-copy redundancy** (storing up to 10 copies of the DNA sequence) and a consensus majority-voting algorithm to dramatically improve data retrieval accuracy.

## Key Findings

| Metric | Value |
|--------|-------|
| Decode Accuracy (10 copies, 64 nt) | **88.72% ± 0.20%** |
| Decode Accuracy (3 copies, 64 nt) | **81.96% ± 1.81%** |
| Decode Accuracy (1 copy, 64 nt) | **77.14% ± 1.09%** |
| Pred RL vs RL (128 nt) | **p = 0.008** (statistically significant) |
| Overfitting Gap | < 1.24 pp (no overfitting) |
| Sequence Lengths Evaluated | 64, 128, 256, 512, 1024 nt |
| Independent Seeds | 7 (seeds 42–48) |
| Total Training Time | ~19.9 minutes (Google Colab T4 GPU) |

The Predictive RL agent with 10-copy redundancy achieves **88.72% ± 0.20%** decode accuracy at 64 nt — a **12.43 pp** improvement over a single copy. The error predictor is statistically proven to be necessary: without it, the RL agent performs significantly *worse* than baseline at 128 nt (p = 0.002).

## Results Summary

### Decode Accuracy (%, mean across 7 seeds)

| Method | 64 nt | 128 nt | 256 nt | 512 nt | 1024 nt |
|--------|-------|--------|--------|--------|---------|
| Fixed Baseline | 76.56 | 70.87 | 64.67 | 59.73 | 57.47 |
| Random Mapping | 75.94 | 68.99 | 62.72 | 59.57 | 58.19 |
| RL (no predictor) | 75.07 | 67.27 | 64.62 | 59.80 | 57.66 |
| Predictive RL | **77.14** | **71.95** | 64.38 | 59.92 | 57.46 |
| Pred RL + 3 copies | **81.96** | **73.66** | **67.27** | **61.94** | **59.16** |

*100 evaluation trials per condition per seed on frozen datasets.*

### Redundancy Scaling (Predictive RL)

| Copies | 64 nt | 128 nt | 256 nt | 512 nt | 1024 nt |
|--------|-------|--------|--------|--------|---------|
| 1 | 76.29 | 70.72 | 62.37 | 59.85 | 57.16 |
| 3 | 81.96 | 72.99 | 67.49 | 61.70 | 58.98 |
| 5 | 85.03 | 76.20 | 69.69 | 64.93 | 60.55 |
| 7 | 86.16 | 78.17 | 71.82 | 66.36 | 61.80 |
| 10 | **88.72** | **79.39** | **73.61** | **67.27** | **62.96** |

### Statistical Significance (independent t-tests, 7 seeds)

| Comparison | 64 nt | 128 nt |
|-----------|-------|--------|
| Pred RL vs RL Agent | p = 0.061 | **p = 0.008** ✓ |
| RL Agent vs Baseline | p = 0.13 | **p = 0.002** ✓ |
| Pred RL vs Baseline | p = 0.57 | p = 0.39 |

**Key insight:** The RL agent without the error predictor is significantly *worse* than the fixed baseline at 128 nt (p = 0.002). The error predictor corrects this, making the predictive RL agent significantly better than the RL-only agent (p = 0.008). This proves the predictor is a necessary component.

### Overfitting Check (Predictive RL)

| Length | Train Acc | Test Acc | Gap |
|--------|-----------|----------|-----|
| 64 nt | 78.46% | 78.92% | −0.46 |
| 128 nt | 71.99% | 72.72% | −0.73 |
| 256 nt | 64.16% | 62.92% | +1.24 |
| 512 nt | 59.97% | 60.17% | −0.19 |
| 1024 nt | 57.25% | 57.66% | −0.41 |

Max gap: 1.24 pp. Four of five lengths have negative gaps (test > train).

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

## Features

- **Proximal Policy Optimisation RL:** Dynamically chooses encoding mappings based on binary data distribution.
- **Error Predictor:** Neural model that anticipates sequencing errors to penalise fragile sequences during training.
- **Multi-Copy Redundancy:** 1–10 copies with position-wise majority voting for error correction.
- **Text & Image Encoding:** The web demo encodes text, PNG, JPEG, GIF, BMP, and WebP into DNA.
- **Google Colab Training:** GPU notebook (`DNA_Training_Colab.ipynb`) trains 28 models in ~20 minutes.
- **Local Evaluation Pipeline:** `run_full_pipeline.py` runs all evaluation, ablation, and chart generation.
- **Interactive Web Dashboard:** Visualises encoding, error simulation, and performance at [adaptive-dna-encoding.netlify.app](https://adaptive-dna-encoding.netlify.app/).

## Repository Structure

```
├── DNA_Training_Colab.ipynb     # Colab notebook for GPU training
├── run_full_pipeline.py         # Local evaluation pipeline (8 phases)
├── config.py                    # Central hyperparameters
├── rl_agent.py                  # PPO actor-critic network
├── rl_environment.py            # RL environment (state, reward)
├── error_predictor.py           # Neural error prediction model
├── error_simulation.py          # DNA error simulation (sub/ins/del)
├── encoding.py                  # 24 mappings, binary↔DNA conversion
├── decoder.py                   # DNA→binary decoding
├── redundancy.py                # Multi-copy storage + majority voting
├── sequence_metrics.py          # GC content, homopolymer, entropy
├── evaluation.py                # Evaluation framework + statistics
├── visualization.py             # Publication-quality figure generation
├── training_pipeline.py         # Training loop with early stopping
├── paper/main.tex               # IEEE conference paper
├── figures_for_paper/           # 11 publication figures
├── website/                     # Interactive dashboard
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── demo_data.json           # Evaluation data for website
└── results/                     # CSVs, plots, model checkpoints
    ├── results.csv
    ├── statistics_summary.csv
    ├── ablation_results.csv
    └── redundancy_scaling.csv
```

## How to Run

### 1. Train the Models (Google Colab)
1. Zip the Python source files and `dataset/` folder into `dna_project.zip`.
2. Upload and run `DNA_Training_Colab.ipynb` on Google Colab with a T4 GPU.
3. The notebook trains 7 seeds × 4 agent variants = 28 models (~20 minutes).
4. Download `colab_trained_models.zip` and `colab_export.json`.

### 2. Evaluate Locally
1. Extract `colab_trained_models.zip` into the `results/` folder.
2. Run the evaluation pipeline:
   ```bash
   python run_full_pipeline.py --input results/colab_export.json
   ```
   This runs 8 phases: model loading → evaluation → ablation → statistics → overfitting check → redundancy scaling → plot generation → website data export.

### 3. View the Dashboard
```bash
python -m http.server 8000
```
Open [http://localhost:8000/website/index.html](http://localhost:8000/website/index.html).

## Paper

The accompanying IEEE conference paper is in `paper/main.tex`. It covers the full methodology, 7-seed experimental results across 5 sequence lengths, ablation study, statistical significance analysis, and redundancy scaling evaluation.

## License

This project was developed as part of academic research. See the repository for details.
