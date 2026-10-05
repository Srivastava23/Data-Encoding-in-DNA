# Project Status — Final Reference

**Last updated:** 2026-10-01  
**Paper title:** "Adaptive DNA Encoding for Data Storage via Predictive RL and Consensus Redundancy"  
**Target venue:** IEEE ACCESS

---

## Training Summary

- **Platform:** Google Colab (T4 GPU)
- **Total seeds trained:** 7 (seeds 42, 43, 44, 45, 46, 47, 48)
- **Training epochs:** Up to 100,000 per seed (early stopping at 10,000 epochs without improvement)
- **Actual training time:** ~19.9 minutes total across all 7 seeds × 4 agent types
- **Training sequence length:** 128 nt (models are evaluated at 64, 128, 256, 512, 1024 nt)

### Important Note on colab_export.json

The file `results/colab_export.json` metadata says `"number_of_seeds": 3` and `"seeds": [42, 43, 44]`.  
This is because the Colab export was saved **partway through** (after the first 3 seeds).  
The remaining 4 seeds (45–48) were trained in a subsequent Colab session.  
**All 7 seeds' checkpoints are stored in `results/colab_trained_models.zip`.**

If checkpoints for seeds 45–48 are missing from `results/checkpoints/`, extract them from the zip:
```
results/colab_trained_models.zip → results/checkpoints/
```

---

## Trained Checkpoints (per seed)

For each of the 7 seeds (42–48), the following 5 checkpoint files exist:

| File Pattern | Agent Type | Description |
|-------------|-----------|-------------|
| `reinforcement_learning_seed{S}.pt` | RL Agent (no predictor) | PPO agent trained without error predictor in reward |
| `predictive_seed{S}.pt` | Predictive RL | PPO agent trained WITH error predictor in reward |
| `predictor_seed{S}.pt` | Error Predictor | MLP that predicts sequencing error rate |
| `ablation_no_gc_seed{S}.pt` | RL w/o GC | PPO agent with GC reward weight set to 0 |
| `ablation_no_homopolymer_seed{S}.pt` | RL w/o Homopolymer | PPO agent with homopolymer penalty set to 0 |

Total: 7 seeds × 5 files = **35 checkpoint files**

---

## Evaluation Configuration

| Parameter | Value |
|-----------|-------|
| Seeds | 42, 43, 44, 45, 46, 47, 48 |
| Sequence lengths | 64, 128, 256, 512, 1024 nt |
| Trials per length per seed | 100 |
| Frozen dataset seed formula | `seed * 10000 + length` |
| Error rates | 2% substitution, 1% insertion, 1% deletion |
| Redundancy (main comparison) | 3 copies |
| Redundancy scaling | 1, 3, 5, 7, 10 copies |
| Overfitting test seed | 99999 (200 trials per length) |

### Methods Compared (Main)
1. **Fixed Baseline** — canonical mapping (00→A, 01→C, 10→G, 11→T)
2. **Random Mapping** — random permutation per trial
3. **RL Agent** — PPO without error predictor
4. **Predictive RL** — PPO with error predictor
5. **Pred. RL + Redundancy** — Predictive RL + 3-copy consensus voting

### Ablation Conditions
1. Baseline (no agent)
2. Full RL (all reward components)
3. Predictive RL (all components + predictor)
4. RL w/o GC (GC reward weight = 0)
5. RL w/o Homopolymer (homopolymer penalty weight = 0)

---

## Key Results (from paper)

| Length | Baseline Acc | Pred RL Acc | Pred RL+Red Acc | Gain over Baseline |
|--------|-------------|-------------|-----------------|-------------------|
| 64 | 76.56% | 77.14% | 81.96% | +5.40 pp |
| 128 | 70.87% | 71.95% | 73.66% | +2.79 pp |
| 256 | 64.67% | 64.38% | 67.27% | +2.60 pp |
| 512 | 59.73% | 59.92% | 61.94% | +2.21 pp |
| 1024 | 57.47% | 57.46% | 59.16% | +1.69 pp |

**Key statistical result:** Predictive RL vs RL Agent at length 128: t=4.97, p=0.008 (significant)

---

## File Structure

```
Data in DNA/
├── config.py                  # Central configuration (defaults to 7 seeds)
├── encoding.py                # Binary↔DNA encoding (24 mappings, 96 actions)
├── decoder.py                 # DNA→binary decoding
├── error_simulation.py        # Context-dependent error channel
├── error_predictor.py         # MLP error predictor (Layer 2)
├── rl_agent.py                # PPO actor-critic agent (Layer 1)
├── rl_environment.py          # Gym-style RL environment
├── redundancy.py              # Multi-copy consensus voting (Layer 3)
├── sequence_metrics.py        # GC content, homopolymer, stability
├── training_pipeline.py       # Training loop (used on Colab)
├── colab_training_notebook.py # Colab training script
├── evaluation.py              # Frozen-dataset evaluation framework
├── run_full_pipeline.py       # Full local evaluation pipeline
├── run_experiment.py          # Older experiment runner
├── visualization.py           # Plot generation
├── rerun_evaluation.py        # Re-evaluation script (for paper verification)
├── requirements.txt           # torch, numpy, scipy, matplotlib, websockets
├── results/
│   ├── checkpoints/           # All 35 trained .pt files (7 seeds × 5 types)
│   ├── colab_export.json      # Training metadata + curves (NOTE: only shows 3 seeds)
│   ├── colab_trained_models.zip # ALL 7 seeds' checkpoints (authoritative source)
│   ├── results.csv            # Per-seed evaluation results
│   ├── statistics_summary.csv # Aggregated statistics
│   ├── ablation_results.csv   # Ablation comparison
│   ├── redundancy_scaling.csv # Redundancy scaling summary
│   ├── redundancy_scaling_raw.csv # Per-seed redundancy data
│   └── *.png                  # Generated figures
├── paper/                     # LaTeX source
├── figures_for_paper/         # Publication figures
└── website/                   # Interactive web demo
    ├── index.html
    ├── script.js
    ├── style.css
    └── demo_data.json
```

---

## How to Reproduce Results

1. **Ensure checkpoints are extracted:** All `.pt` files from `results/colab_trained_models.zip` must be in `results/checkpoints/`
2. **Run evaluation:** `python run_full_pipeline.py --input results/colab_export.json`
   - OR use `python rerun_evaluation.py` for the detailed re-evaluation with raw CSVs
3. **No retraining needed** — all evaluation uses frozen pre-trained checkpoints

---

## Common Gotchas

1. **`colab_export.json` says 3 seeds** — This is misleading. All 7 seeds were trained; the export was saved after the first batch. The zip file has all 7.
2. **Models trained at length 128** — But evaluated at all 5 lengths (64–1024). The state vector includes normalized length, so the agent generalizes.
3. **RNG seeds differ between experiments** — The main comparison and redundancy scaling use different RNG formulas for error simulation, so copies=1 in scaling ≠ Predictive RL in main comparison (different noise draws, same agent and data).
4. **std is across 7 seed means** — Not across 700 trials. Each seed's mean is over 100 trials; std is over those 7 means.
5. **t-tests are Student's (equal variance)** — `scipy.stats.ttest_ind()` with default `equal_var=True`, on 7 per-seed error rate means.
