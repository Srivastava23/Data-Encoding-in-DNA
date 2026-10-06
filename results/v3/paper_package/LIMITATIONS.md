# Limitations

## 1. Action Space and State Space Expressivity
The primary limitation discovered in this study is that a permutation-based mapping action space (the 24 possible assignments of 4 dibits to 4 nucleotides) may be fundamentally insufficient to correct highly localized DNA errors. A sequence with poor GC content or severe homopolymer runs may not be salvageable by merely permuting the dibit mapping, since the underlying binary data may have low entropy. The agent is forced to select the "least bad" option, which often converges functionally to a random permutation. Additionally, the 8-dimension block-level state may not encapsulate enough sequential context for the GRU to infer deeper structural risks.

## 2. Naive Consensus Mechanism
The multi-copy redundancy system assumes perfectly aligned copies to execute position-wise majority voting. In reality, insertion and deletion (indel) errors shift sequence alignment. Without applying an aggressive Multiple Sequence Alignment (MSA) algorithm before majority voting, the consensus mechanism will deteriorate rapidly as sequence length increases and indels compound. This explains the steep drop in accuracy as sequence length approaches 1024 nt, even with multiple copies.

## 3. Disconnect Between Predictor and Policy
The Error Predictor (MLP) was trained to regress sequencing errors from block-level features. While the predictor achieved reasonable loss convergence, using its outputs to guide or constrain the policy (Predictive RL) failed to yield a performance advantage. This suggests that while the predictor can identify dangerous sequences, the action space (mappings) cannot always provide a route to avoid them.

## 4. Overhead Unaccounted for in Raw Accuracy
Redundancy incurs a massive physical synthesis overhead. Storing 10 copies of a sequence requires 10x the DNA synthesis cost. While this study reports an 88.96% accuracy with 10 copies, a real-world cost-benefit analysis must weigh the dramatic increase in chemical synthesis overhead against algorithmic error-correction methods (e.g., Reed-Solomon or LDPC codes) which provide robustness with far less physical redundancy.
