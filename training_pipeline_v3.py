"""
Training pipeline V3 — multi-block adaptive encoding.

Structure matches the original (fast) pipeline:
  - Each epoch = 1 episode (1 sequence through N blocks)
  - Transitions accumulate in a rollout buffer
  - PPO update fires every `ppo_batch_size` episodes
  - No per-episode baseline recomputation (uses running average)

This keeps training at ~100k epochs / few minutes on T4,
identical to the V1 speed.
"""

import os
import time
import torch
import numpy as np
from torch.distributions import Categorical
from typing import Dict, List, Optional

from config import Config
from encoding import (
    generate_curriculum_binary,
    generate_biased_binary,
    encode_binary_to_dna,
    ALL_MAPPINGS,
    BASELINE_MAPPING,
    get_block_splits,
    encode_block_sequence,
    decode_block_sequence,
)
from error_simulation import simulate_errors, compute_error_rate
from decoder import compute_bit_accuracy
from sequence_metrics import gc_content, normalized_max_homopolymer, shannon_entropy
from rl_agent_v3 import PPOAgentV3
from error_predictor import ErrorPredictor


def _compute_block_state(binary_bits, splits, block_idx, N, block_actions, dna_so_far):
    """Compute state vector for block `block_idx`."""
    start, end = splits[block_idx]
    block_bits = binary_bits[start:end]

    dummy_dna = encode_binary_to_dna(block_bits, BASELINE_MAPPING)
    gc = gc_content(dummy_dna)
    hp = normalized_max_homopolymer(dummy_dna)
    ent = shannon_entropy(dummy_dna)

    prev_map = -1.0
    j1 = -1.0
    j2 = -1.0

    if block_idx > 0:
        prev_map = float(block_actions[-1]) / 23.0
        if len(dna_so_far) >= 1:
            j1 = float("ACGT".index(dna_so_far[-1])) / 3.0
        if len(dna_so_far) >= 2:
            j2 = float("ACGT".index(dna_so_far[-2])) / 3.0

    return np.array([gc, hp, ent, float(block_idx) / max(1, N),
                     float(N) / 32.0, prev_map, j1, j2], dtype=np.float32)


def train(config: Config, seed: int, verbose: bool = True) -> Dict:
    """
    Train the multi-block RL agent.  One episode per epoch (fast).
    Returns the same dict shape as the V1 pipeline for compatibility.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    N_blocks = getattr(config, 'number_of_encoding_blocks', 4)
    config.number_of_encoding_blocks = N_blocks
    seq_len = getattr(config, 'training_sequence_length', 128)
    config.training_sequence_length = seq_len
    n_bits = seq_len * 2

    # ---- models -------------------------------------------------------
    STATE_DIM = 8
    ACTION_DIM = 24

    agent = PPOAgentV3(STATE_DIM, ACTION_DIM, config)

    predictor: Optional[ErrorPredictor] = None
    if config.use_error_predictor:
        predictor = ErrorPredictor(config)

    # ---- logging ------------------------------------------------------
    epoch_rewards: List[float] = []
    rolling_rewards: List[float] = []
    policy_losses: List[float] = []
    value_losses: List[float] = []
    predictor_losses: List[float] = []
    learning_rates: List[float] = []
    validation_rewards: List[float] = []

    rolling_window = 100
    rolling_sum = 0.0

    total_epochs = config.ppo_training_epochs

    # ---- rollout buffer (list of episode dicts) -----------------------
    rollout_buffer: List[Dict] = []

    if verbose:
        print("=== Training Pipeline V3 ===")
        print(f"Blocks: {N_blocks}")
        print(f"Sequence Length: {seq_len}")

    t0 = time.time()

    for epoch in range(1, total_epochs + 1):
        # --- generate one sequence ---
        if total_epochs > 1:
            binary_bits = generate_curriculum_binary(n_bits, rng, epoch, total_epochs)
        else:
            binary_bits = generate_biased_binary(n_bits, rng)

        splits = get_block_splits(len(binary_bits), config)
        N = len(splits)

        # --- roll out one episode (N block-steps) ---
        ep_states = []
        ep_actions = []
        ep_rewards = []
        ep_values = []
        ep_log_probs = []

        block_actions = []
        dna_so_far = ""
        actor_hx, critic_hx = None, None

        agent.network.eval()
        with torch.no_grad():
            for k in range(N):
                state = _compute_block_state(binary_bits, splits, k, N,
                                             block_actions, dna_so_far)

                state_t = torch.tensor(state, dtype=torch.float32).to(agent.device)
                logits, val, actor_hx, critic_hx = agent.network(
                    state_t, actor_hx, critic_hx)

                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action_int = int(action.item())

                # --- encode this block ---
                b_start, b_end = splits[k]
                block_bits = binary_bits[b_start:b_end]
                mapping_idx = action_int
                if config.use_header_free_encoding and k > 0:
                    prev_s, prev_e = splits[k - 1]
                    offset = int(np.sum(binary_bits[prev_s:prev_e])) % 24
                    mapping_idx = (mapping_idx + offset) % 24
                block_dna = encode_binary_to_dna(block_bits, ALL_MAPPINGS[mapping_idx])
                dna_so_far += block_dna
                block_actions.append(action_int)

                # intermediate reward = 0 (credit assignment via GAE at terminal)
                ep_states.append(state)
                ep_actions.append(action_int)
                ep_rewards.append(0.0)  # placeholder, will fill terminal
                ep_values.append(val.item())
                ep_log_probs.append(log_prob.item())

        # --- terminal: simulate errors on full DNA and compute reward ---
        mutated, _ = simulate_errors(
            dna_so_far,
            config.substitution_rate, config.insertion_rate,
            config.deletion_rate, rng)
        decoded_bits = decode_block_sequence(mutated, block_actions, config)
        bit_acc = compute_bit_accuracy(binary_bits, decoded_bits)
        error_rate = compute_error_rate(dna_so_far, mutated)

        gc = gc_content(dna_so_far)
        hp = normalized_max_homopolymer(dna_so_far)
        from sequence_metrics import gc_balance_score
        gc_bonus = gc_balance_score(dna_so_far, config.gc_target_low, config.gc_target_high)

        terminal_reward = (
            config.reward_accuracy_weight * bit_acc
            + config.reward_gc_bonus_weight * gc_bonus
            - config.reward_homopolymer_penalty_weight * hp
            - config.reward_error_rate_weight * error_rate
        )
        ep_rewards[-1] = terminal_reward   # assign to last block step

        # collect predictor training data
        if predictor is not None:
            for i in range(N):
                bs, be = splits[i]
                b_bits = binary_bits[bs:be]
                m_idx = block_actions[i]
                if config.use_header_free_encoding and i > 0:
                    ps, pe = splits[i - 1]
                    offset = int(np.sum(binary_bits[ps:pe])) % 24
                    m_idx = (m_idx + offset) % 24
                blk_dna = encode_binary_to_dna(b_bits, ALL_MAPPINGS[m_idx])
                feat = predictor.build_features(
                    gc_content(blk_dna), shannon_entropy(blk_dna),
                    normalized_max_homopolymer(blk_dna), len(blk_dna), m_idx)
                predictor.collect(feat, error_rate)

        # store episode
        rollout_buffer.append({
            'states': np.array(ep_states, dtype=np.float32),
            'actions': np.array(ep_actions, dtype=np.int64),
            'rewards': np.array(ep_rewards, dtype=np.float32),
            'values': np.array(ep_values, dtype=np.float32),
            'log_probs': np.array(ep_log_probs, dtype=np.float32),
        })

        episode_reward = terminal_reward
        epoch_rewards.append(episode_reward)
        rolling_sum += episode_reward
        if epoch > rolling_window:
            rolling_sum -= epoch_rewards[epoch - rolling_window - 1]
        rolling_rewards.append(rolling_sum / min(epoch, rolling_window))

        # --- PPO update when buffer is full ---
        if len(rollout_buffer) >= config.ppo_batch_size:
            agent.network.train()
            pl, vl, ent = agent.update(rollout_buffer)
            policy_losses.append(pl)
            value_losses.append(vl)
            learning_rates.append(agent.optimizer.param_groups[0]['lr'])
            rollout_buffer = []

            agent.step_decay(epoch / total_epochs)

        # --- train predictor periodically ---
        if predictor is not None and epoch % config.predictor_training_interval == 0:
            pred_loss = predictor.train_step(rng)
            predictor_losses.append(pred_loss)

        # --- logging ---
        if verbose and (epoch % 2000 == 0 or epoch <= 2):
            lr = agent.optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:>6d}/{total_epochs}  "
                  f"reward={episode_reward:.4f}  "
                  f"bit_acc={bit_acc:.4f}  "
                  f"error_rate={error_rate:.4f}  "
                  f"rolling={rolling_rewards[-1]:.4f}  "
                  f"lr={lr:.6f}")

    elapsed = time.time() - t0
    if verbose:
        print(f"Training complete in {elapsed / 60:.1f} minutes.")

    return {
        "agent": agent,
        "predictor": predictor,
        "epoch_rewards": epoch_rewards,
        "rolling_rewards": rolling_rewards,
        "validation_rewards": rolling_rewards,   # use rolling as proxy
        "policy_losses": policy_losses,
        "value_losses": value_losses,
        "predictor_losses": predictor_losses,
        "learning_rates": learning_rates,
        "final_epoch": total_epochs,
        "stopped_early": False,
    }


if __name__ == "__main__":
    c = Config()
    train(c, 42)
