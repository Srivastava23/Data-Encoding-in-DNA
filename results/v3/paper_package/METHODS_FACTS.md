# Methods & Implementation Details

## Architecture
- **Layer 1: RL Encoding Agent**
  - Uses Proximal Policy Optimization (PPO).
  - Multi-Block Sequence-to-Sequence formulation.
  - **State Representation (8-dim)**: Local block GC content, block homopolymer run length, block entropy, normalized block index, sequence length ratio, previous mapping index, previous nucleotide 1, previous nucleotide 2.
  - **Action Space**: 24 base permutations mapping the four dibits (00, 01, 10, 11) to the four nucleotides (A, C, G, T). Header-free block chaining encodes subsequent blocks using parities of the previous block's encoding.
  - **Network Topology**: A Linear projection followed by a Gated Recurrent Unit (GRU) with 64 hidden dimensions, followed by separate Actor and Critic linear heads.
- **Layer 2: Error Predictor**
  - Multi-Layer Perceptron (MLP) trained to regress the expected error rate given sequence length, GC, entropy, homopolymer, and mapped features.
- **Layer 3: Redundancy & Majority Voting**
  - Uses exact replica sequence storage (1 up to 10 copies).
  - Retrieves DNA sequences using position-wise majority voting across copies to suppress independent sequencing errors.

## Hyperparameters & Training Settings
- **PPO Batch Size**: 64
- **PPO Hidden Dimensions**: 64
- **PPO Learning Rate**: 3e-4
- **PPO Entropy Coefficient**: Scheduled linearly from 0.01 to 0.005.
- **GAE Lambda**: 0.95
- **Discount Factor (Gamma)**: 0.99
- **Epochs**: Up to 100,000 per seed.
- **Seeds**: 7 independent models trained (42, 43, 44, 45, 46, 47, 48).

## Dataset & Evaluation
- **Data Generator**: Synthesizes random binary strings conforming to three distribution profiles: High GC, Low GC, and Balanced.
- **Curriculum Learning**: Balanced profile -> Mixed profiles -> All profiles.
- **Training Sequence Lengths**: Randomly sampled blocks of length `64, 128, 256, 512` nucleotides.
- **Generalization/Test Sequence Length**: `1024` nucleotides (strictly out-of-distribution during training).
- **Error Model Simulation Rates**:
  - Substitution: 2%
  - Insertion: 1%
  - Deletion: 1%
