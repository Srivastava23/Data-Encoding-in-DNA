# Provenance

## Q1: Redundancy-scaling vs main comparison

The redundancy-scaling experiment uses the **same frozen evaluation datasets** (generated with `seed * 10000 + length`) as the main comparison.

However, it uses a **different RNG seed** for error simulation: `seed * 100000 + length * 100 + n_copies` (from `evaluation.py:run_redundancy_scaling`, L411), whereas the main comparison uses `seed + 6000 + length` (from `evaluation.py:run_comparison`, L246).

The agent checkpoints are also different: the main comparison's redundancy row uses `agent_predictive` loaded per-seed, whereas `run_redundancy_scaling` in the original pipeline (`run_full_pipeline.py` L245-248) uses `agent_for_test` which is a single agent loaded from the **last seed only** (seed 48). In this re-run, redundancy scaling uses the per-seed predictive agent, matching the main comparison.

## Q2: copies=1 and copies=3 vs main comparison

In the original pipeline, copies=1 and copies=3 in the scaling table will **NOT** exactly match the main comparison for two reasons:

1. **Different RNG seeds** for error simulation (see Q1).

2. **Different agent** used in the original pipeline (single agent for all seeds in scaling vs per-seed agent in main comparison).

In this re-run, the per-seed agent is used for both, but the RNG seeds still differ, so numbers will differ due to different noise draws.

## Q3: Unit of std in every table

In every table, std is computed across the **7 per-seed means** (each seed's mean is over 100 trials). That is, `np.std([mean_seed42, mean_seed43, ..., mean_seed48])`. It is NOT the std across all 700 trials.

## Q4: t-test details

The t-tests use `scipy.stats.ttest_ind(vals_a, vals_b)` with default parameters (equal_var=True, i.e. Student's t-test, not Welch's).

The samples are the **7 per-seed means** of the error rate (not 700 individual trials). The test is unpaired (independent two-sample).

Exact call: `scipy_stats.ttest_ind(error_rates_a, error_rates_b)` (see `evaluation.py` L333).

## Q5: Overfitting train and test sets

The 'train' set in the overfitting check is the **same frozen evaluation dataset** used in the main comparison (generated with `seed * 10000 + length`, 100 trials per length).

The 'test' set is a **separate, independent dataset** generated with seed=99999 and 200 trials per length (`run_full_pipeline.py` L178-189).

The test RNG for error simulation uses `np.random.default_rng(test_seed + seq_len)`, which is different from the train RNG.

## Package versions

- Python: 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]
- PyTorch: 2.10.0+cpu
- NumPy: 2.2.6
- SciPy: 1.16.1

## Checkpoint hashes (MD5)

- `no_gc_seed42`: `2fb6a7f1d9e55a3a8b5bcc98a6783eef`
- `no_gc_seed43`: `b0a9d4540ee7f9c2357e8324985c1c60`
- `no_gc_seed44`: `0079f4cbc97bb7ec554747306f974fbe`
- `no_gc_seed45`: `6c7046b7a3f2bb1615e86755976b9535`
- `no_gc_seed46`: `9bdd057cf9fba27cff8da37b64ad9e06`
- `no_gc_seed47`: `71aa69d47e936022efd3e6c8718510aa`
- `no_gc_seed48`: `40b63909df6381fb319b655ae090d949`
- `no_homo_seed42`: `704d36594be0dae36cffedde3c9acac2`
- `no_homo_seed43`: `b2ae0725147a3199bf343ad62a40aa77`
- `no_homo_seed44`: `1ddab01e9ccdfde76daf12e04ddb61e8`
- `no_homo_seed45`: `650736c0ceaf940bc684737b864f8b88`
- `no_homo_seed46`: `bd1be0de30d5f11c2ba18e682bbdef81`
- `no_homo_seed47`: `a98141c517b3e4f32cfa8cf8b80fc6b1`
- `no_homo_seed48`: `2bc85114b626f33609d68ca5c9f867f4`
- `pred_seed42`: `76c9f1372b15e8bc8e5cc427a6af69be`
- `pred_seed43`: `d4755497ba9241f98c3034858db17560`
- `pred_seed44`: `5ec1ee4db0815c5b480bdbd3568211d6`
- `pred_seed45`: `fed372faad54c0f0074f2ebb7c912ed8`
- `pred_seed46`: `b71973bbcce3db3ee4a1b95d9455262b`
- `pred_seed47`: `c4eab6244a55d5b05bdf6c2a9f0b080d`
- `pred_seed48`: `49454a521570c84730dbe83e9571e5c8`
- `predictor_seed42`: `8e380f24f4ae9434bf2d92bb21daac97`
- `predictor_seed43`: `e96a471b0320cf3dc088a1a09c5f0a31`
- `predictor_seed44`: `195dd155709ea6f21247a52c7c4d7103`
- `predictor_seed45`: `680284fd40af6dc61473cff9a2dcc559`
- `predictor_seed46`: `892b98576b8dda59ce1344af2623cf72`
- `predictor_seed47`: `790254f96163989c31b0d3fc81a436d5`
- `predictor_seed48`: `e3a9c52ee90efbf6f088c08a173206dd`
- `rl_seed42`: `2ee2db32105e5d9d7c8c3003a54104f2`
- `rl_seed43`: `3d4bb22c6a338afed75c4c3ab6496cd6`
- `rl_seed44`: `bdf46fc0c78f4dd0ecc4dbce5222d30b`
- `rl_seed45`: `ef1b05a4ab3617c2ff88a74f13f5df33`
- `rl_seed46`: `75b2a682989e3b7515b353ad0404dbbd`
- `rl_seed47`: `d23d4dc7c4da6f175df91bbc70a9d044`
- `rl_seed48`: `7b5a7625e475d421f39806d86c2ad4c4`