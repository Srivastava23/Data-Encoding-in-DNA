# Figure Provenance

Generated using `rebuild_figures.py`.
Data sources: `results/rerun/` CSV files.

## fig5_error_rate_comparison.png
- **Source Data**: results_raw.csv (error_rate)
- **Seeds Included**: 42-48
- **N per point**: 7 means (each mean of 100 trials)
- **Error bars**: Standard deviation across the 7 per-seed means

## fig6_decode_accuracy_comparison.png
- **Source Data**: results_raw.csv (decode_accuracy_pct)
- **Seeds Included**: 42-48
- **N per point**: 7 means (each mean of 100 trials)
- **Error bars**: Standard deviation across the 7 per-seed means
- **Details**: Legend moved outside plot to avoid hiding data.

## fig7_gc_content_distribution.png
- **Source Data**: results_raw.csv (gc_content)
- **Seeds Included**: 42-48
- **N per point**: 35 means per method (7 seeds x 5 lengths)
- **Error bars**: N/A (Histogram)
- **Details**: Distribution is over the per-seed means, aggregated across all lengths.

## Composite Efficiency Score Formula
Formula: `encoding_efficiency = (decode_accuracy_pct / 100.0) * (1.0 - error_rate)`
Justification: The efficiency score is simply the product of the bit-level accuracy and the sequence retention rate (1 - error rate), representing the total fraction of correct information successfully transmitted.

## fig8_encoding_efficiency_comparison.png
- **Source Data**: results_raw.csv (computed)
- **Seeds Included**: 42-48
- **N per point**: 7 means
- **Error bars**: Standard deviation across the 7 per-seed means

## fig9_ablation_study_comparison.png
- **Source Data**: ablation_raw.csv
- **Seeds Included**: 42-48
- **N per point**: 7 means
- **Error bars**: Standard deviation across the 7 per-seed means

## fig11_redundancy_scaling.png
- **Source Data**: redundancy_raw.csv
- **Seeds Included**: 42-48
- **N per point**: 7 means
- **Error bars**: Standard deviation across the 7 per-seed means

## fig3_training_reward_rl.png
- **Source Data**: colab_export.json inside ZIP
- **Seeds Included**: 42-48
- **N per point**: 1 per epoch
- **Error bars**: N/A
- **Details**: Unnormalised. Smoothed with EMA alpha=0.02.

## fig4_training_reward_pred.png
- **Source Data**: colab_export.json inside ZIP
- **Seeds Included**: 42-48
- **N per point**: 1 per epoch
- **Error bars**: N/A
- **Details**: Unnormalised. Smoothed with EMA alpha=0.02.

## fig10_predictor_loss.png
- **Source Data**: colab_export.json inside ZIP
- **Seeds Included**: 42-48
- **N per point**: 1 per epoch
- **Error bars**: N/A
- **Details**: Unnormalised MSE. Smoothed with EMA alpha=0.02.

## Proposed Captions
1. **Fig 5**: Error rate across all five encoding methods for five sequence lengths (64 to 1024 nt). Error bars represent the standard deviation across the 7 training seeds.
2. **Fig 6**: Decode accuracy comparison across all methods and lengths. Error bars show standard deviation over the 7 independent seeds.
3. **Fig 7**: GC content distribution of encoded sequences aggregated across all evaluation lengths, plotted over the per-seed means (n=35 per method). Red lines indicate the target optimal biological range (40-60%).
4. **Fig 8**: Encoding efficiency score (Bit Accuracy × Sequence Retention) across methods. Error bars denote the standard deviation across 7 seeds.
5. **Fig 9**: Ablation study evaluating the contribution of individual reward components to the overall error rate and decode accuracy. Error bars represent standard deviation across 7 seeds.
6. **Fig 11**: Decode accuracy as a function of the number of redundant copies used for consensus voting in the Predictive RL agent. Error bars reflect standard deviation across 7 seeds.
