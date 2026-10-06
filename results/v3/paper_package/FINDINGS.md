# Key Findings

## Single-Copy Performance vs. Multi-Copy Redundancy
The core hypothesis of this research was that a Reinforcement Learning agent, capable of making sequential encoding decisions across blocks of data, would significantly outperform random mapping selection or a fixed baseline mapping. 

The evaluation data explicitly contradicts this hypothesis for single-copy storage:
- **Baseline Method** (single copy, length 64): 75.91% decode accuracy.
- **Random Policy** (single copy, length 64): 76.28% decode accuracy.
- **RL Agent** (single copy, length 64): 76.57% decode accuracy.
- **Predictive RL** (single copy, length 64): 76.23% decode accuracy.

In all pairwise comparisons between RL methods and the baseline, the p-values exceeded the alpha threshold of 0.05, meaning the differences in performance are completely within the bounds of statistical noise. The multi-block RL agent fails to provide a meaningful edge in data recovery over simply choosing random permutations block-by-block.

## The Role of Majority Voting Redundancy
While the sequence-to-sequence GRU policy failed to improve the outcome, implementing a multi-copy redundancy layer coupled with position-wise majority voting during decoding successfully mitigated errors across all sequence lengths.
- Storing **3 copies** of a 64 nt sequence raised accuracy from ~76% to **81.69%**.
- Storing **10 copies** of a 64 nt sequence raised accuracy to a robust **88.96%**.

## Claims Fully Supported by the Data
- Multi-copy redundancy with majority voting systematically suppresses DNA substitution and insertion/deletion errors, scaling log-linearly with the number of redundant copies up to 10 copies.
- Redundancy achieves a maximum accuracy of 88.96% for 64 nucleotide sequences under the specified 4% combined error rate profile.
- A GRU-based recurrent neural network can stably converge during PPO training on a multi-block DNA mapping environment across varying sequence lengths.
- The learned RL policy generalizes to out-of-distribution sequence lengths (1024 nt), avoiding catastrophic overfitting, though its performance remains equal to the baseline.

## Tempting Claims NOT Supported by the Data (Do Not Claim)
- **DO NOT CLAIM**: "RL significantly outperforms the baseline encoding." (The difference is marginal and not statistically significant).
- **DO NOT CLAIM**: "The multi-block sequence-to-sequence model achieved massive improvements." (The architecture achieved nothing measurable; the massive improvements came exclusively from the redundancy pipeline).
- **DO NOT CLAIM**: "Predictive RL improves upon standard RL." (Predictive RL performed interchangeably with or slightly worse than plain RL).

## Revisions to V1 Results (Old Claims to Discard)
- Discard the old finding: "RL achieved 88.72% decode accuracy over baseline 58%." (This was flawed data generated using unequal evaluation configurations and evaluation bugs).
- Discard any claims that RL mapping reduces error rates by over 10% on its own.
