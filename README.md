# Predictive Reinforcement Learning-Based DNA Data Encoding

**🟢 Live Demo:** [https://adaptive-dna-encoding.netlify.app/](https://adaptive-dna-encoding.netlify.app/)

This project demonstrates a complete, visual pipeline for encoding digital binary data into DNA sequences. It uses Reinforcement Learning (Proximal Policy Optimisation) combined with an Error Predictor to learn the most robust mappings from bits to nucleotides (A, C, G, T) while maximising biological stability (GC content and minimising homopolymers).

To combat the inherent noise in DNA synthesis and sequencing, the pipeline utilises **multi-copy redundancy** (storing up to 10 copies of the DNA sequence) and a consensus majority-voting algorithm to dramatically improve data retrieval accuracy.

## Key Findings

| Metric | Value |
|--------|-------|
| Decode Accuracy (10 copies, 64 nt) | **88.24% ± 0.92%** |
| Decode Accuracy (3 copies, 64 nt) | **81.75% ± 0.93%** |
| Decode Accuracy (1 copy, 64 nt) | **75.68% ± 1.35%** |
| Overfitting Gap | < 1.24 pp (no overfitting) |
| Sequence Lengths Evaluated | 64, 128, 256, 512, 1024 nt |
| Independent Seeds | 7 (seeds 42–48) |
| Total Training Time | ~19.9 minutes (Google Colab T4 GPU) |

The Predictive RL agent with 10-copy redundancy achieves **88.24% ± 0.92%** decode accuracy at 64 nt. Multi-copy consensus voting consistently improves accuracy across all sequence lengths, with diminishing returns as length increases — confirming that redundancy is the strongest error-correction mechanism.

## Results Summary

### Decode Accuracy (%, mean ± std across 7 seeds)

| Method | 64 nt | 128 nt | 256 nt | 512 nt | 1024 nt |
|--------|-------|--------|--------|--------|---------|
| Fixed Baseline | 74.61±2.18 | 69.79±1.20 | 64.01±0.86 | 60.02±0.54 | 57.39±0.21 |
| Random Mapping | 76.04±1.32 | 69.25±0.69 | 63.48±1.01 | 59.83±0.77 | 57.71±0.64 |
| RL Agent | 76.32±1.78 | 69.87±1.08 | 64.16±0.32 | 59.81±0.74 | 57.82±0.63 |
| Predictive RL | 76.21±1.86 | 69.54±0.99 | 64.57±0.78 | 60.19±0.54 | 57.36±0.31 |
| Pred RL + 3 copies | **81.25±0.90** | **73.50±0.38** | **67.26±0.41** | **62.71±0.56** | **59.21±0.39** |

*100 evaluation trials per condition per seed on frozen datasets.*

### Redundancy Scaling (Predictive RL)

| Copies | 64 nt | 128 nt | 256 nt | 512 nt | 1024 nt |
|--------|-------|--------|--------|--------|---------|
| 1 | 75.68±1.35 | 70.21±1.30 | 64.30±1.10 | 60.31±0.78 | 57.29±0.52 |
| 3 | 81.75±0.93 | 72.81±1.04 | 67.08±0.44 | 62.16±0.76 | 58.92±0.58 |
| 5 | 84.07±0.67 | 76.66±0.65 | 69.84±0.54 | 64.71±0.31 | 60.57±0.29 |
| 7 | 86.41±0.69 | 78.78±0.86 | 71.61±0.96 | 65.69±0.59 | 61.84±0.46 |
| 10 | **88.24±0.92** | **80.34±1.01** | **72.27±0.73** | **67.22±0.39** | **62.99±0.31** |

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
- **Local Evaluation Pipeline:** `rerun_evaluation.py` runs all evaluation, ablation, and chart generation.
- **Interactive Web Dashboard:** Visualises encoding, error simulation, and performance at [adaptive-dna-encoding.netlify.app](https://adaptive-dna-encoding.netlify.app/).

## Repository Structure

```
├── DNA_Training_Colab.ipynb     # Colab notebook for GPU training
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
├── rerun_evaluation.py          # Full 7-seed evaluation pipeline
├── rebuild_figures.py           # Figure regeneration from raw CSVs
├── paper/main.tex               # IEEE conference paper
├── website/                     # Interactive dashboard
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── demo_data.json           # Evaluation data for website
└── results/                     # Model checkpoints + trained models
    ├── checkpoints/             # Per-seed .pt model files
    ├── colab_trained_models.zip # Trained models from Colab
    └── colab_export.json        # Training metadata + curves
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
   python rerun_evaluation.py
   ```
   This generates all raw CSVs, statistical tables, and self-check reports in `results/rerun/`.

### 3. View the Dashboard
```bash
python -m http.server 8000
```
Open [http://localhost:8000/website/index.html](http://localhost:8000/website/index.html).

## Paper

The accompanying IEEE conference paper is in `paper/main.tex`. It covers the full methodology, 7-seed experimental results across 5 sequence lengths (64–1024 nt), ablation study, and redundancy scaling evaluation.

## License

This project was developed as part of academic research. See the repository for details.
