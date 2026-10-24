# Reinforcement-Learning Mapping Selection and Consensus Redundancy for DNA Data Storage: A Multi-Seed Simulation Study

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Journal: Entropy](https://img.shields.io/badge/Submitted%20to-Entropy%20(MDPI)-orange.svg)](https://www.mdpi.com/journal/entropy)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Netlify-00C7B7.svg)](https://adaptive-dna-encoding.netlify.app/)

> **Authors:** Ananya Chandra and Dr. P. Nithyanandam (Corresponding Author: `nithyanandam.p@vit.ac.in`)  
> **Affiliation:** School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Chennai 600127, India.  
> **Status:** Submitted as a Research Article to *Entropy* (MDPI).  
> **Interactive Web Demonstration:** [https://adaptive-dna-encoding.netlify.app/](https://adaptive-dna-encoding.netlify.app/)

---

## 📌 Abstract & Research Focus

Digital data storage in synthetic deoxyribonucleic acid (DNA) offers extreme information density (~215 PB/gram) and physical longevity, but synthesis and sequencing introduce stochastic substitution, insertion, and deletion errors. This project investigates a fundamental question in DNA channel coding:

> **Can a reinforcement learning (RL) agent dynamically selecting dibit-to-nucleotide mappings block-by-block provide a meaningful error-correction advantage compared to multi-copy consensus redundancy with majority voting?**

Using a multi-seed simulation framework (7 independent random seeds, 5 sequence lengths from 64 to 1024 nt), we evaluate:
1. **Recurrent PPO Agent:** A Proximal Policy Optimisation policy with a Gated Recurrent Unit (GRU) core that sequentially assigns mappings across 4 chained sequence blocks.
2. **Predictive Error Model:** A neural Multi-Layer Perceptron (MLP) regressing anticipated sequencing error from structural features to guide reward shaping.
3. **Consensus Redundancy:** Physical replica storage (1 to 10 copies) reconstructed via position-wise majority voting.

### Key Finding
**Physical consensus redundancy accounts for the overwhelming majority of performance gains.**
- At 64 nucleotides, 3-copy voting raises decode accuracy from **76.23% to 81.69%**, and 10 copies reach **88.96%** (a 12.65 percentage-point gain over single copy).
- In contrast, learned mappings showed **no statistically significant difference** in error rate compared to a fixed canonical baseline ($p \ge 0.081$ unadjusted; all Holm–Bonferroni adjusted $p = 1.000$) and performed within 1 percentage point of a random-mapping control.
- While the RL agent successfully learned to balance GC content across individual sequences (cutting GC deviation from 50% by more than half), this structural optimization did not translate into decode accuracy gains under our simulated channel.

---

## 🏗️ System Architecture

```
                      ┌───────────────────────────────────────────────┐
                      │            Binary Input Bitstream             │
                      └──────────────────────┬────────────────────────┘
                                             │ 2-bit dibits
                                             ▼
                      ┌───────────────────────────────────────────────┐
                      │    Layer 1: Recurrent PPO Agent (GRU)        │
                      │  • State: 8-dim local block statistics        │
                      │  • Action: 24 dibit-to-base permutations      │
                      │  • K = 4 chained blocks with parity offset    │
                      └──────────────┬─────────────────┬──────────────┘
                                     │                 │
              Candidate Block Encoding│                 │ Predicted Risk
                                     ▼                 ▼
                      ┌───────────────────────────────────────────────┐
                      │    Layer 2: Neural Error Predictor (MLP)      │
                      │  • Estimates sequencing risk from GC, entropy, │
                      │    homopolymers, length, and mapping choice   │
                      └──────────────────────┬────────────────────────┘
                                             │
                                             ▼ Encoded DNA Sequence
                      ┌───────────────────────────────────────────────┐
                      │    Layer 3: Multi-Copy Consensus Redundancy   │
                      │  • Replicated into N copies (1 to 10)         │
                      │  • Context-Dependent Channel Simulation       │
                      │    (2% sub, 1% ins, 1% del, local penalties)  │
                      │  • Position-wise consensus majority voting    │
                      └──────────────────────┬────────────────────────┘
                                             │
                                             ▼ Consensus Sequence
                      ┌───────────────────────────────────────────────┐
                      │             Block-by-Block Decoder            │
                      │  • Inverse mapping using recovered offsets    │
                      │  • Recovered Binary Payload                   │
                      └───────────────────────────────────────────────┘
```

---

## 📊 Summary of Experimental Results

All results are aggregated over **7 independent training seeds** (seeds 42–48) evaluated over **100 frozen trials per condition** at each sequence length (700 trials total per method/length).

### 1. Decode Accuracy (%, Mean ± Std across 7 Seeds)
*Higher is better. Best result per column in **bold**.*

| Method | 64 nt | 128 nt (Train Length) | 256 nt | 512 nt | 1024 nt (Held-Out) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fixed Baseline** | 75.91 ± 1.15 | 68.94 ± 0.57 | 64.04 ± 0.79 | 59.93 ± 0.62 | 57.46 ± 0.40 |
| **Random Mapping** | 76.28 ± 1.18 | 69.30 ± 1.24 | 64.42 ± 0.64 | 60.40 ± 0.63 | 57.75 ± 0.45 |
| **RL Agent** | 76.57 ± 1.24 | 69.35 ± 1.06 | 64.93 ± 0.96 | 59.70 ± 0.83 | 57.88 ± 0.50 |
| **Predictive RL** | 76.23 ± 1.98 | 69.77 ± 1.89 | 64.55 ± 0.73 | 60.42 ± 0.57 | 57.62 ± 0.48 |
| **Pred. RL + 3 Copies** | **81.69 ± 0.89** | **73.56 ± 1.12** | **68.08 ± 1.00** | **62.66 ± 0.61** | **59.25 ± 0.35** |

### 2. Multi-Copy Consensus Scaling (Predictive RL, % Accuracy)
*Measured across independent scaling trials (1 to 10 redundant copies).*

| Copies | 64 nt | 128 nt | 256 nt | 512 nt | 1024 nt |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Copy** | 76.31 ± 1.31 | 70.61 ± 1.49 | 65.37 ± 0.67 | 60.16 ± 0.34 | 57.33 ± 0.41 |
| **2 Copies** | 77.01 ± 0.66 | 68.97 ± 1.60 | 64.15 ± 0.82 | 60.11 ± 0.75 | 57.54 ± 0.50 |
| **3 Copies** | 81.24 ± 0.81 | 73.66 ± 0.86 | 67.55 ± 0.48 | 62.88 ± 0.56 | 59.34 ± 0.49 |
| **4 Copies** | 82.41 ± 0.50 | 76.00 ± 1.15 | 68.51 ± 0.50 | 63.95 ± 0.58 | 60.61 ± 0.35 |
| **5 Copies** | 85.39 ± 0.94 | 77.05 ± 1.04 | 69.96 ± 0.75 | 64.76 ± 0.64 | 60.67 ± 0.27 |
| **7 Copies** | 87.17 ± 1.22 | 79.05 ± 1.26 | 71.79 ± 0.78 | 66.01 ± 0.73 | 62.19 ± 0.33 |
| **10 Copies** | **88.96 ± 0.92** | **80.59 ± 0.92** | **72.83 ± 0.53** | **67.29 ± 0.53** | **63.04 ± 0.40** |

*Note on 2 copies: With two copies, every disagreement results in a tie resolved in favor of the first copy, yielding negligible or slightly negative gains. Solid improvements emerge from 3 copies onward.*

### 3. Hypothesis Testing (Positional Error Rate)
*Independent two-sample $t$-tests ($t$ / unadjusted $p$). All 15 tests yield Holm–Bonferroni adjusted $p = 1.000$.*

| Comparison | 64 nt | 128 nt | 256 nt | 512 nt | 1024 nt |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pred. RL vs. Baseline** | −0.48 / 0.639 | −1.14 / 0.276 | −1.25 / 0.234 | −1.60 / 0.135 | −0.66 / 0.521 |
| **Pred. RL vs. RL Agent** | +0.43 / 0.677 | −0.46 / 0.657 | +0.83 / 0.425 | −1.91 / 0.081 | +1.01 / 0.331 |
| **RL Agent vs. Baseline** | −1.16 / 0.267 | −1.03 / 0.325 | −1.84 / 0.090 | +0.51 / 0.622 | −1.72 / 0.111 |

### 4. Sequence Stability & GC Deviation
*Stability Score $S \in [0, 1]$ (higher is better) and Implied Mean Absolute GC Deviation $|g - 0.5|$ (lower is better).*

| Method | Stability Score (256 nt) | Stability Score (1024 nt) | Implied GC Deviation (64 nt) | Implied GC Deviation (1024 nt) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** | 0.887 ± 0.005 | 0.898 ± 0.002 | 0.113 ± 0.007 | 0.099 ± 0.002 |
| **Random** | 0.927 ± 0.006 | 0.935 ± 0.006 | 0.077 ± 0.008 | 0.062 ± 0.006 |
| **RL Agent** | **0.956 ± 0.004** | **0.971 ± 0.005** | 0.052 ± 0.006 | **0.026 ± 0.005** |
| **Predictive RL** | **0.956 ± 0.008** | 0.970 ± 0.011 | **0.051 ± 0.006** | 0.027 ± 0.011 |

The learned agents consistently cut individual sequence GC deviation by more than half compared to fixed mapping.

---

## 📂 Repository Structure

```
.
├── config.py                  # Global hyperparameters & simulation settings
├── encoding.py                # 24 dibit-to-base bijections & block chaining
├── decoder.py                 # Block-by-block inverse mapping decoder
├── error_simulation.py        # Context-dependent DNA channel (sub/ins/del)
├── sequence_metrics.py        # GC content, homopolymers, entropy, stability
├── redundancy.py              # Multi-copy simulation & majority consensus voting
├── rl_agent_v3.py             # Recurrent GRU PPO policy network & value head
├── rl_environment_v3.py       # Sequential multi-block environment
├── error_predictor.py         # Multi-Layer Perceptron sequencing risk model
├── training_pipeline_v3.py    # Multi-seed training loop with curriculum learning
├── evaluation.py              # Benchmarking framework & hypothesis testing
├── run_full_pipeline.py       # Full evaluation script for exported checkpoints
├── patch_v3.py                # Pipeline patching & validation utilities
├── rebuild_figures.py         # Publication figure regeneration from CSV data
├── visualization.py           # Matplotlib styling & plotting routines
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
│
├── paper/                     # LaTeX manuscript submitted to Entropy (MDPI)
│   ├── main.tex               # Complete LaTeX source
│   ├── Definitions/           # Official MDPI LaTeX class & styles
│   └── *.png                  # Manuscript figures
│
├── results/                   # Raw empirical data & checkpoints
│   ├── statistics_summary.csv # Main comparison metrics & t-tests
│   ├── results.csv            # Raw per-trial evaluation logs
│   ├── redundancy_scaling.csv # Consensus scaling results (1-10 copies)
│   ├── ablation_results.csv   # Reward ablation experimental data
│   └── checkpoints/           # Trained PyTorch agent models (.pt) for seeds 42-48
│
└── website/                   # Interactive dashboard (HTML5/CSS3/Vanilla JS)
    ├── index.html             # UI layout
    ├── style.css              # Responsive dark-theme styling
    ├── script.js              # In-browser pipeline simulation
    └── demo_data.json         # Evaluation benchmarks payload
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Srivastava23/Data-Encoding-in-DNA.git
cd Data-Encoding-in-DNA
pip install -r requirements.txt
```

### 2. Re-evaluate Checkpoints & Reproduce Tables

To evaluate the trained checkpoints across all 7 seeds (seeds 42–48) and 5 sequence lengths:

```bash
python run_full_pipeline.py
```

This re-generates the statistical summary files matching Tables 2, 3, 4, 5, 6, and 7 of the manuscript.

### 3. Rebuild Publication Figures

To re-render all publication plots from the raw CSV data into `results/`:

```bash
python rebuild_figures.py
```

### 4. Train New Agents

To train recurrent PPO agents from scratch using curriculum learning:

```bash
python training_pipeline_v3.py
```

### 5. Launch the Web Demonstration Locally

```bash
python -m http.server 8000
```
Then navigate to `http://localhost:8000/website/index.html` in your web browser.

---

## 🔍 Scientific Limitations & Takeaways

1. **Simulation Channel:** Synthesis and sequencing noise are simulated using fixed error rates (2% substitution, 1% insertion, 1% deletion) with heuristic context modulation. Real sequencing platforms (Illumina, Oxford Nanopore, PacBio) exhibit platform-specific positional biases.
2. **Action Space Expressivity:** Permuting 4 dibits across 24 bijections cannot break internal homopolymer runs, which are determined by the underlying binary data.
3. **Naive Consensus:** The majority voting mechanism compares copies position-wise without Multiple Sequence Alignment (MSA). As sequences grow, unaligned indels compound, explaining the performance decline from 64 nt (88.96%) to 1024 nt (63.04%).
4. **Action Index Storage:** The decoder currently assumes action indices are known. Storing 4 indices requires $\approx 18.3$ bits per sequence (14.3% overhead at 64 nt, 0.9% at 1024 nt).
5. **Practical Coding Implication:** Without multi-seed validation, random-mapping controls, and multi-comparison correction, learned encoders can easily appear artificially superior.

---

## 📖 Citation

If you find this codebase or research useful, please cite:

```bibtex
@article{chandra2026reinforcement,
  title={Reinforcement-Learning Mapping Selection and Consensus Redundancy for DNA Data Storage: A Multi-Seed Simulation Study},
  author={Chandra, Ananya and Nithyanandam, P.},
  journal={Entropy (Submitted)},
  year={2026},
  publisher={MDPI},
  url={https://github.com/Srivastava23/Data-Encoding-in-DNA}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
