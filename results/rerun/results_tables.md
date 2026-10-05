# Aggregated Results Tables

All values: mean ± std across 7 seeds (42–48), each seed = mean of 100 trials.

## Table 1: Error Rate by Method and Length

| Method | 64 | 128 | 256 | 512 | 1024 |
|--------|-----:|-----:|-----:|-----:|-----:|
| Fixed Baseline | 0.369±0.032 | 0.442±0.017 | 0.527±0.013 | 0.588±0.008 | 0.628±0.003 |
| Random Mapping | 0.347±0.018 | 0.449±0.009 | 0.536±0.015 | 0.590±0.011 | 0.623±0.009 |
| RL Agent | 0.342±0.027 | 0.440±0.016 | 0.525±0.005 | 0.591±0.010 | 0.622±0.010 |
| Predictive RL | 0.344±0.027 | 0.445±0.015 | 0.520±0.011 | 0.585±0.008 | 0.628±0.005 |
| Pred. RL + 3-copy | 0.272±0.014 | 0.387±0.005 | 0.480±0.005 | 0.548±0.008 | 0.600±0.006 |

- **Length 64**: Best = Pred. RL + 3-copy (0.272). Pred RL > RL Agent; Pred RL < Baseline.
- **Length 128**: Best = Pred. RL + 3-copy (0.387). Pred RL > RL Agent; Pred RL > Baseline.
- **Length 256**: Best = Pred. RL + 3-copy (0.480). Pred RL < RL Agent; Pred RL < Baseline.
- **Length 512**: Best = Pred. RL + 3-copy (0.548). Pred RL < RL Agent; Pred RL < Baseline.
- **Length 1024**: Best = Pred. RL + 3-copy (0.600). Pred RL > RL Agent; Pred RL > Baseline.

## Table 2: Decode Accuracy (%) by Method and Length

| Method | 64 | 128 | 256 | 512 | 1024 |
|--------|-----:|-----:|-----:|-----:|-----:|
| Fixed Baseline | 74.61±2.18 | 69.79±1.20 | 64.01±0.86 | 60.02±0.54 | 57.39±0.21 |
| Random Mapping | 76.04±1.32 | 69.25±0.69 | 63.48±1.01 | 59.83±0.77 | 57.71±0.64 |
| RL Agent | 76.32±1.78 | 69.87±1.08 | 64.16±0.32 | 59.81±0.74 | 57.82±0.63 |
| Predictive RL | 76.21±1.86 | 69.54±0.99 | 64.57±0.78 | 60.19±0.54 | 57.36±0.31 |
| Pred. RL + 3-copy | 81.25±0.90 | 73.50±0.38 | 67.26±0.41 | 62.71±0.56 | 59.21±0.39 |

- **Length 64**: Best = Pred. RL + 3-copy (81.25%). Pred RL < RL Agent; Pred RL > Baseline.
- **Length 128**: Best = Pred. RL + 3-copy (73.50%). Pred RL < RL Agent; Pred RL < Baseline.
- **Length 256**: Best = Pred. RL + 3-copy (67.26%). Pred RL > RL Agent; Pred RL > Baseline.
- **Length 512**: Best = Pred. RL + 3-copy (62.71%). Pred RL > RL Agent; Pred RL > Baseline.
- **Length 1024**: Best = Pred. RL + 3-copy (59.21%). Pred RL < RL Agent; Pred RL < Baseline.

## Table 3: Stability Score by Method and Length

| Method | 64 | 128 | 256 | 512 | 1024 |
|--------|-----:|-----:|-----:|-----:|-----:|
| Fixed Baseline | 0.856±0.007 | 0.877±0.006 | 0.887±0.005 | 0.892±0.003 | 0.898±0.002 |
| Random Mapping | 0.888±0.004 | 0.912±0.004 | 0.926±0.005 | 0.933±0.005 | 0.938±0.004 |
| RL Agent | 0.898±0.004 | 0.920±0.003 | 0.933±0.007 | 0.943±0.004 | 0.948±0.004 |
| Predictive RL | 0.898±0.005 | 0.922±0.005 | 0.933±0.004 | 0.943±0.004 | 0.949±0.004 |
| Pred. RL + 3-copy | 0.902±0.004 | 0.923±0.007 | 0.933±0.003 | 0.941±0.004 | 0.951±0.004 |

- **Length 64**: Best = Pred. RL + 3-copy (0.902). Pred RL > Baseline.
- **Length 128**: Best = Pred. RL + 3-copy (0.923). Pred RL > Baseline.
- **Length 256**: Best = Pred. RL + 3-copy (0.933). Pred RL > Baseline.
- **Length 512**: Best = Predictive RL (0.943). Pred RL > Baseline.
- **Length 1024**: Best = Pred. RL + 3-copy (0.951). Pred RL > Baseline.

## Table 4: Redundancy Scaling — Decode Accuracy (%) by Copies and Length

| Copies | 64 | 128 | 256 | 512 | 1024 |
|--------|-----:|-----:|-----:|-----:|-----:|
| 1 | 75.68±1.35 | 70.21±1.30 | 64.30±1.10 | 60.31±0.78 | 57.29±0.52 |
| 3 | 81.75±0.93 | 72.81±1.04 | 67.08±0.44 | 62.16±0.76 | 58.92±0.58 |
| 5 | 84.07±0.67 | 76.66±0.65 | 69.84±0.54 | 64.71±0.31 | 60.57±0.29 |
| 7 | 86.41±0.69 | 78.78±0.86 | 71.61±0.96 | 65.69±0.59 | 61.84±0.46 |
| 10 | 88.24±0.92 | 80.34±1.01 | 72.27±0.73 | 67.22±0.39 | 62.99±0.31 |

## Table 5a: Ablation — Error Rate by Condition and Length

| Condition | 64 | 128 | 256 | 512 | 1024 |
|-----------|-----:|-----:|-----:|-----:|-----:|
| Baseline | 0.369±0.032 | 0.442±0.017 | 0.527±0.013 | 0.588±0.008 | 0.628±0.003 |
| Full RL | 0.333±0.019 | 0.459±0.017 | 0.519±0.018 | 0.586±0.008 | 0.626±0.007 |
| Predictive RL | 0.342±0.022 | 0.442±0.026 | 0.524±0.015 | 0.585±0.016 | 0.626±0.008 |
| RL w/o GC | 0.349±0.022 | 0.441±0.021 | 0.521±0.015 | 0.586±0.009 | 0.628±0.007 |
| RL w/o Homopolymer | 0.337±0.026 | 0.439±0.017 | 0.531±0.016 | 0.585±0.011 | 0.625±0.006 |

## Table 5b: Ablation — Decode Accuracy (%) by Condition and Length

| Condition | 64 | 128 | 256 | 512 | 1024 |
|-----------|-----:|-----:|-----:|-----:|-----:|
| Baseline | 74.61±2.18 | 69.79±1.20 | 64.01±0.86 | 60.02±0.54 | 57.39±0.21 |
| Full RL | 77.00±1.33 | 68.61±1.24 | 64.62±1.21 | 60.18±0.62 | 57.49±0.42 |
| Predictive RL | 76.46±1.55 | 69.78±1.90 | 64.22±1.01 | 60.22±1.13 | 57.53±0.55 |
| RL w/o GC | 75.84±1.48 | 69.80±1.45 | 64.45±1.04 | 60.07±0.63 | 57.37±0.48 |
| RL w/o Homopolymer | 76.71±1.72 | 69.84±1.12 | 63.79±1.10 | 60.18±0.83 | 57.59±0.41 |

## Table 6: Overfitting Check — Train vs Test Accuracy (%)

| Length | Train | Test | Gap |
|--------|------:|-----:|----:|
| 64 | 76.12 | 77.39 | -1.27 |
| 128 | 69.59 | 71.86 | -2.27 |
| 256 | 63.68 | 63.87 | -0.19 |
| 512 | 60.29 | 60.05 | +0.24 |
| 1024 | 57.44 | 57.52 | -0.08 |

## Table 7: t-test Results (on error rate, 7 per-seed means per method)

| Comparison | Length | t | p |
|------------|-------:|--:|--:|
| Predictive RL vs Baseline | 64 | -1.49 | 0.162 |
| Predictive RL vs Baseline | 128 | 0.33 | 0.748 |
| Predictive RL vs Baseline | 256 | -1.13 | 0.280 |
| Predictive RL vs Baseline | 512 | -0.62 | 0.548 |
| Predictive RL vs Baseline | 1024 | 0.32 | 0.756 |
| Predictive RL vs RL Agent | 64 | 0.09 | 0.932 |
| Predictive RL vs RL Agent | 128 | 0.58 | 0.576 |
| Predictive RL vs RL Agent | 256 | -1.20 | 0.254 |
| Predictive RL vs RL Agent | 512 | -1.29 | 0.221 |
| Predictive RL vs RL Agent | 1024 | 1.48 | 0.163 |
| RL Agent vs Baseline | 64 | -1.58 | 0.139 |
| RL Agent vs Baseline | 128 | -0.20 | 0.842 |
| RL Agent vs Baseline | 256 | -0.34 | 0.742 |
| RL Agent vs Baseline | 512 | 0.77 | 0.457 |
| RL Agent vs Baseline | 1024 | -1.38 | 0.192 |

## Derived Quantities

### Percentage-point gain of Pred. RL + Red. over Baseline

Formula: `gain_pp = acc_pred_rl_red_mean - acc_baseline_mean`

- Length 64: 81.25 - 74.61 = **+6.64 pp**
- Length 128: 73.50 - 69.79 = **+3.71 pp**
- Length 256: 67.26 - 64.01 = **+3.25 pp**
- Length 512: 62.71 - 60.02 = **+2.69 pp**
- Length 1024: 59.21 - 57.39 = **+1.82 pp**

### Relative error-rate reduction of Pred. RL + Red. vs Baseline

Formula: `reduction_pct = (err_baseline - err_pred_red) / err_baseline * 100`

- Length 64: (0.369 - 0.272) / 0.369 × 100 = **26.3%**
- Length 128: (0.442 - 0.387) / 0.442 × 100 = **12.3%**
- Length 256: (0.527 - 0.480) / 0.527 × 100 = **9.0%**
- Length 512: (0.588 - 0.548) / 0.588 × 100 = **6.7%**
- Length 1024: (0.628 - 0.600) / 0.628 × 100 = **4.3%**

### Gain from 1 to 10 copies (percentage points)

Formula: `gain = acc_10copies - acc_1copy`

- Length 64: 88.24 - 75.68 = **+12.56 pp**
- Length 128: 80.34 - 70.21 = **+10.12 pp**
- Length 256: 72.27 - 64.30 = **+7.96 pp**
- Length 512: 67.22 - 60.31 = **+6.91 pp**
- Length 1024: 62.99 - 57.29 = **+5.71 pp**

### Relative stability improvement of Predictive RL over Baseline at length 256

Formula: `(stab_pred - stab_base) / stab_base × 100`
= (0.933 - 0.887) / 0.887 × 100 = **5.2%**

### Largest absolute train-test gap

- Largest absolute gap: **2.27 pp** at length **128**
- Lengths where test exceeds train: **4** of 5
