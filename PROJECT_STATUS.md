# PROJECT STATUS — DNA Data Encoding via Predictive RL

**Last updated:** October 2025

## Training

- **Platform:** Google Colab (T4 GPU)
- **Seeds:** 7 independent seeds (42, 43, 44, 45, 46, 47, 48)
- **Models per seed:** 4 (RL Agent, Predictive RL, Ablation no-GC, Ablation no-Homopolymer)
- **Total models trained:** 28
- **Training time:** ~19.9 minutes total
- **Trained models location:** `results/colab_trained_models.zip` (all 7 seeds, all 4 variants)
- **Training metadata:** `results/colab_export.json`

## Evaluation Settings

| Parameter | Value |
|-----------|-------|
| Seeds | 42, 43, 44, 45, 46, 47, 48 |
| Sequence lengths | 64, 128, 256, 512, 1024 nt |
| Trials per (seed, length) | 100 |
| Methods evaluated | Fixed Baseline, Random Mapping, RL Agent, Predictive RL, Pred RL + 3-copy |
| Ablation conditions | Full RL, Predictive RL, RL w/o GC, RL w/o Homopolymer |
| Redundancy copies tested | 1, 3, 5, 7, 10 |
| Error channel | 2% substitution, 1% insertion, 1% deletion |

## Key Results (7-seed means)

| Metric | Value |
|--------|-------|
| Best single-method accuracy (64 nt) | 76.32% (RL Agent) |
| Best with redundancy (10 copies, 64 nt) | 88.24% ± 0.92% |
| Pred RL + 3-copy vs single-copy improvement | +5.57 pp at 64 nt |
| Overfitting gap | < 1.24 pp |

## File Structure

```
├── config.py                    # Central hyperparameters
├── rl_agent.py                  # PPO actor-critic network
├── rl_environment.py            # RL environment (state, reward)
├── error_predictor.py           # Neural error prediction
├── error_simulation.py          # DNA error simulation
├── encoding.py                  # Binary ↔ DNA encoding (24 mappings)
├── decoder.py                   # DNA → binary decoding
├── redundancy.py                # Multi-copy + majority voting
├── sequence_metrics.py          # GC, homopolymer, entropy
├── evaluation.py                # Evaluation framework
├── training_pipeline.py         # Training loop + early stopping
├── visualization.py             # Figure generation
├── rerun_evaluation.py          # Full 7-seed evaluation script
├── rebuild_figures.py           # Paper figure regeneration
├── DNA_Training_Colab.ipynb     # Colab training notebook
├── colab_training_notebook.py   # Python version of Colab notebook
├── process_colab_results.py     # Process Colab export
├── run_experiment.py            # Single experiment runner
├── run_full_pipeline.py         # Full pipeline (8 phases)
├── paper/main.tex               # IEEE paper
├── dataset/                     # Data profiles + sample corpus
├── website/                     # Interactive web dashboard
└── results/
    ├── checkpoints/             # 7 seeds × 4 variants = 28 .pt files
    ├── colab_trained_models.zip # Trained models archive
    └── colab_export.json        # Training curves + metadata
```

## Notes

- All evaluation is inference-only using pre-trained `.pt` files from `results/checkpoints/`.
- The trained models inside `colab_trained_models.zip` were extracted to `results/checkpoints/` after download.
- Frozen evaluation datasets are generated deterministically per (seed, length) pair using `seed * 10000 + length` as the RNG seed, ensuring identical test data across all methods.
