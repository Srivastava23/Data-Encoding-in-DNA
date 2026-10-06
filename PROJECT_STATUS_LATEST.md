# Project Status: DNA Data Encoding RL
**Date**: October 6, 2026

## 1. Current State of the Project
The project has implemented and evaluated a Predictive Reinforcement Learning agent (V3) using a Multi-Block Sequence-to-Sequence Gated Recurrent Unit (GRU) architecture. The objective was to adaptively choose DNA dibit mappings block-by-block to optimize GC content, avoid homopolymer runs, and ultimately reduce sequencing errors.

In the final evaluation, we implemented multi-copy redundancy (majority voting) alongside the predictive RL policy. While the multi-copy redundancy provided substantial gains (e.g., reaching 88.96% decode accuracy with 10 copies at 64 nucleotides), the RL agent's single-copy performance (76.57%) did not achieve a statistically significant improvement over the random baseline (76.28%) or the fixed baseline (75.91%). The improvements are driven entirely by the redundancy mechanism.

## 2. Changelog since V2 Plan
- **Implemented V3 Architecture**: Replaced the simple 2-block split with a dynamic multi-block sequence-to-sequence GRU network (`rl_agent_v3.py`). State dimensions were expanded to 8 features (GC, entropy, homopolymer, etc.).
- **Updated Training Pipeline (`training_pipeline_v3.py`)**: Migrated to per-block encoding. Enforced 1-episode-per-epoch structure with sequence lengths ranging from 64 to 512 for better generalization, and 1024 as held-out.
- **Patched Evaluation Scripts (`patch_v3.py`)**: Adapted `run_full_pipeline.py`, `evaluation.py`, and `redundancy.py` to correctly evaluate the block-by-block sequential actions instead of fixed mappings.
- **Executed Final Pipeline**: Ran 100,000 epochs across 7 independent seeds on Kaggle using T4 GPUs. Results downloaded to `results/checkpoints/` and fully evaluated locally.

## 3. Final Run Configuration
- **Model Architecture**: GRU with 64 hidden dimensions processing 24 base mappings block-by-block. Header-free block chaining was used (parity-dependent).
- **Seeds**: 7 independent seeds (42, 43, 44, 45, 46, 47, 48).
- **Training Sequence Lengths**: Randomly sampled from `[64, 128, 256, 512]`.
- **Held-Out Length**: `1024` nucleotides for generalization testing.
- **Epochs**: Up to 100,000 epochs per seed, early stopped when stable.
- **PPO Hyperparameters**: Entropy Coefficient schedule: 0.01 decreasing to 0.005. Batch Size: 64. GAE Lambda: 0.95.
- **Redundancy Evaluation**: 1 to 10 copies simulated for error correction.

## 4. Directory Map
- `results/checkpoints/`: Contains raw `.pt` model files from the Kaggle training run.
- `results/`: Contains raw CSVs (`results.csv`, `redundancy_scaling.csv`, `ablation_results.csv`, `statistics_summary.csv`).
- `results/v3/paper_package/`: The complete export for paper writing (including methods, findings, and high-res plots).
- `website/`: The interactive HTML/JS UI and `demo_data.json` visualization payload.

## 5. Headline Results
The decision rule requires a corrected p < 0.05 and the confidence interval excluding zero to establish a significant improvement.
- **64 nt (Single Copy)**: RL Mean: 76.57%. Baseline Mean: 75.91%. The p-value against baseline is 0.26 > 0.05 (Not Significant).
- **128 nt (Single Copy)**: RL Mean: 69.35%. Baseline Mean: 68.94%. p-value = 0.33 > 0.05 (Not Significant).
- **1024 nt (Single Copy)**: RL Mean: 57.88%. Baseline Mean: 57.46%. p-value = 0.11 > 0.05 (Not Significant).
- **64 nt (3 Copies, Predictive RL)**: 81.69%. 
- **64 nt (10 Copies, Predictive RL)**: 88.96%.

**Verdict**: The RL policy fails the decision rule. The data does not support the claim that multi-block sequence-to-sequence reinforcement learning significantly outperforms fixed mappings or random mapping selection. The massive improvements observed (e.g., reaching >80% accuracy) are exclusively the product of the multi-copy redundancy (majority voting) mechanism.

## 6. Known Issues and Limitations
- The RL agent converges to performance levels indistinguishable from random per-block mapping. The state space may be too uninformative, or the 24 base permutations do not offer a sufficiently rich action space to combat the specific error profile modeled.
- The interactive demo (`website/script.js`) cannot execute the full GRU inference locally in-browser, so it defaults to single-mapping heuristic choices. This limitation prevents users from visualizing dynamic block switching in the browser.

## 7. Reproduction Commands
- **Train V3 Agent**: `python training_pipeline_v3.py` (Kaggle or Local GPU).
- **Evaluate**: `python patch_v3.py` followed by `python run_full_pipeline.py --input results/colab_export.json`
- **Rebuild Figures**: Run `python rebuild_figures.py` (will regenerate plots based on CSVs).

## 8. Git State
- **Current Branch**: `main`
- **Last Pushed Commit**: `e5575c0` (cleanup)
- **Unpushed Changes**: All V3 architectures, Kaggle checkpoints, patched evaluations, regenerated website figures, and final CSVs remain unpushed on the local machine. **Nothing has been pushed to GitHub since the last explicit permission.**
