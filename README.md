# Predictive Reinforcement Learning-Based DNA Data Encoding

This project demonstrates a complete, visual pipeline for encoding digital binary data into DNA sequences. It uses Reinforcement Learning (Proximal Policy Optimization) combined with an Error Predictor to learn the most robust mappings from bits to nucleotides (A, C, G, T) while maximizing biological stability (GC content and minimizing homopolymers).

To combat the inherent noise in DNA synthesis and sequencing, the pipeline utilizes **multi-copy redundancy** (e.g., storing 3 or 5 copies of the DNA sequence) and a consensus majority-voting algorithm to dramatically improve data retrieval accuracy.

## Features

- **PPO Reinforcement Learning:** Dynamically chooses encoding mappings based on the specific binary data distribution.
- **Error Predictor:** Anticipates potential sequencing errors (substitutions, insertions, deletions) to penalize fragile sequences during training.
- **Multi-Copy Redundancy:** Encodes multiple distinct variations of the same data. During decoding, a majority vote recovers the original data with high fidelity, heavily mitigating substitution errors.
- **Google Colab Training:** Slimmed-down notebook (`DNA_Training_Colab.ipynb`) for fast GPU training of the RL agents.
- **Local Evaluation Pipeline:** Automated script (`run_full_pipeline.py`) that handles all testing on unseen data, ablation comparisons, and statistical chart generation.
- **Interactive Web Dashboard:** A beautiful, responsive web UI (`website/index.html`) that visualizes the entire encoding process, simulates errors, and displays the trained model's performance graphs.

## Repository Structure

- `DNA_Training_Colab.ipynb`: Ready-to-use notebook for GPU training.
- `run_full_pipeline.py`: Main execution script to process Colab results.
- `rl_agent.py`, `rl_environment.py`: Reinforcement learning model and environment.
- `redundancy.py`: Logic for multi-copy encoding and consensus voting.
- `error_simulation.py`, `error_predictor.py`: Sequence mutation and neural prediction.
- `encoding.py`, `decoder.py`: Core logic for bit-to-DNA translation.
- `website/`: Contains the frontend UI (`index.html`, `styles.css`, `script.js`).
- `results/`: Directory where generated charts, statistics, and model weights are stored.

## How to Run

### 1. Train the Models
1. Zip the python scripts and `dataset/` folder into `dna_project.zip`.
2. Upload and run `DNA_Training_Colab.ipynb` on Google Colab (using a T4 GPU).
3. The notebook will automatically download `colab_trained_models.zip` when finished (takes ~10-15 minutes).

### 2. Evaluate Locally
1. Extract `colab_trained_models.zip` into the `results/` folder of this repository.
2. Run the evaluation pipeline:
   ```bash
   python run_full_pipeline.py --input results/colab_export.json
   ```
   *This evaluates accuracy, builds charts, and generates `website/demo_data.json`.*

### 3. View the Interactive Dashboard
Because web browsers block local files from fetching data for security reasons, you must run a simple local server:
```bash
python -m http.server 8000
```
Then, open your web browser and navigate to:
[http://localhost:8000/website/index.html](http://localhost:8000/website/index.html)
