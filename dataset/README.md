# Dataset for DNA Data Encoding Project

## Overview

This folder contains the dataset used to train and evaluate the reinforcement learning-based DNA encoding optimisation system. Since there is no standard public benchmark for DNA data storage, the dataset is **synthetically generated** using controlled binary data profiles that simulate real-world data distributions.

## Data Profiles

Three data distribution profiles are used to ensure the reinforcement learning agent generalises across different types of input data:

| Profile | Description | Dibit Distribution | Resulting GC Content (Baseline Mapping) |
|---------|-------------|--------------------|-----------------------------------------|
| **High GC** | Biased toward dibits 01 and 10 | 10% (00), 35% (01), 35% (10), 20% (11) | ~70% GC |
| **Low GC** | Biased toward dibits 00 and 11 | 35% (00), 10% (01), 20% (10), 35% (11) | ~30% GC |
| **Balanced** | Roughly equal dibit frequencies | 25% each | ~50% GC |

Each profile adds Dirichlet noise for variety across trials.

## Curriculum Learning Schedule

During training, data difficulty increases progressively:

- **Epochs 0–33%**: Balanced data only (easy)
- **Epochs 33–66%**: 50% balanced, 50% random profile (medium)
- **Epochs 66–100%**: All three profiles equally (hard)

## Sequence Lengths

Three DNA sequence lengths are used for evaluation:
- **64 nucleotides** (128 binary bits)
- **128 nucleotides** (256 binary bits)
- **256 nucleotides** (512 binary bits)

## Data Splits

- **Training data**: Generated on-the-fly during each epoch using the curriculum schedule
- **Validation data**: 50 trials per validation step, generated with a separate random seed (seed + 99999), used for early stopping
- **Test data (Frozen)**: 100 trials per sequence length per seed, generated deterministically for fair cross-method comparison

## Sample Text Corpus

The file `sample_text_corpus.txt` contains a curated text excerpt used for the interactive website demonstration. This text is converted to binary and encoded into DNA to show a real-world use case of the system.

## Files

| File | Description |
|------|-------------|
| `README.md` | This file |
| `sample_text_corpus.txt` | Text corpus for website demo encoding |

## Encoding Scheme

Binary data is split into 2-bit blocks called **dibits**:
- `00` → maps to one of {A, C, G, T}
- `01` → maps to one of {A, C, G, T}
- `10` → maps to one of {A, C, G, T}
- `11` → maps to one of {A, C, G, T}

The baseline mapping is: 00→A, 01→C, 10→G, 11→T

The reinforcement learning agent selects from all 24 possible permutations of this mapping to minimise sequencing errors for each data block.
