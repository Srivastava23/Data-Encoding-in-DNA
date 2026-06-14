"""
Orchestrates reinforcement learning training with optional predictive error model.

Pipeline:
    1. Create environment and PPO agent.
    2. For each epoch:
       a. Reset environment with curriculum schedule (balanced -> mixed -> all).
       b. Agent selects action conditioned on data distribution.
       c. Environment steps: encode -> error -> decode -> reward.
       d. If predictor enabled: collect data, periodically train predictor,
          inject predicted error into next reward.
       e. Store transition; when batch is full, run PPO update.
    3. Validate on held-out data every N epochs (anti-overfitting).
    4. Early stopping if validation reward does not improve.
    5. Return training logs (rewards, smoothed rewards, losses).
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional

from config import Config
from rl_environment import DNAEncodingEnvironment
from rl_agent import PPOAgent
from error_predictor import ErrorPredictor


def _exponential_moving_average(values: List[float], alpha: float = 0.02) -> List[float]:
    """Compute EMA for stability analysis. alpha in (0,1], lower = smoother."""
    if not values:
        return []
    moving_average = [values[0]]
    for value in values[1:]:
        moving_average.append(alpha * value + (1 - alpha) * moving_average[-1])
    return moving_average


def _validate(agent: PPOAgent, config: Config, random_generator: np.random.Generator,
              number_of_trials: int = 50) -> float:
    """
    Run validation trials on held-out data to measure generalisation.
    Returns average reward over validation trials.
    """
    from encoding import generate_biased_binary, encode_with_action, decode_with_action
    from error_simulation import simulate_errors, compute_error_rate
    from decoder import compute_bit_accuracy
    from sequence_metrics import gc_content, normalized_max_homopolymer, gc_balance_score

    validation_rewards = []
    number_of_bits = config.training_sequence_length * 2

    for _ in range(number_of_trials):
        binary_bits = generate_biased_binary(number_of_bits, random_generator)

        # compute state
        from rl_environment import _compute_dibit_distribution
        dibit_distribution = _compute_dibit_distribution(binary_bits)
        state = np.array([
            dibit_distribution[0], dibit_distribution[1],
            dibit_distribution[2], dibit_distribution[3],
            float(np.max(dibit_distribution)),
            config.training_sequence_length / 256.0,
        ], dtype=np.float32)

        action, _, _ = agent.select_action(state)

        dna_sequence, _, _ = encode_with_action(binary_bits, action, random_generator)
        mutated_sequence, _ = simulate_errors(
            dna_sequence, config.substitution_rate,
            config.insertion_rate, config.deletion_rate, random_generator,
        )
        decoded_bits = decode_with_action(mutated_sequence, action, config.training_sequence_length)
        bit_accuracy = compute_bit_accuracy(binary_bits, decoded_bits)
        error_rate = compute_error_rate(dna_sequence, mutated_sequence)
        gc_bonus = gc_balance_score(dna_sequence, config.gc_target_low, config.gc_target_high)
        homopolymer_value = normalized_max_homopolymer(dna_sequence)

        reward = (
            config.reward_accuracy_weight * bit_accuracy
            + config.reward_gc_bonus_weight * gc_bonus
            - config.reward_homopolymer_penalty_weight * homopolymer_value
            - config.reward_error_rate_weight * error_rate
        )
        validation_rewards.append(reward)

    return float(np.mean(validation_rewards))


def train(
    config: Config,
    seed: int,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Run the full training loop.

    Returns a dictionary with keys: "agent", "predictor", "epoch_rewards",
    "rolling_rewards", "smoothed_rewards", "predictor_losses",
    "policy_losses", "value_losses", "entropies", "learning_rates",
    "validation_rewards", "best_validation_reward", "stopped_early",
    "final_epoch".
    """
    random_generator = np.random.default_rng(seed)
    environment = DNAEncodingEnvironment(config, random_generator)
    agent = PPOAgent(environment.STATE_DIMENSION, environment.ACTION_DIMENSION, config)

    predictor: Optional[ErrorPredictor] = None
    if config.use_error_predictor:
        predictor = ErrorPredictor(config)

    # separate random generator for validation (deterministic, independent of training)
    validation_random_generator = np.random.default_rng(seed + 99999)

    # logging containers
    epoch_rewards: List[float] = []
    rolling_rewards: List[float] = []
    predictor_losses: List[float] = []
    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []
    learning_rates: List[float] = []
    validation_rewards: List[float] = []

    rolling_window = 100
    rolling_sum = 0.0

    total_epochs = config.ppo_training_epochs
    checkpoint_interval = max(1000, total_epochs // 20)
    validation_interval = 500
    checkpoint_directory = os.path.join(config.output_directory, "checkpoints")
    os.makedirs(checkpoint_directory, exist_ok=True)

    # early stopping state
    best_validation_reward = float("-inf")
    epochs_without_improvement = 0
    stopped_early = False
    final_epoch = total_epochs

    for epoch in range(1, total_epochs + 1):
        # reset with curriculum parameters
        state = environment.reset(epoch=epoch, total_epochs=total_epochs)

        # ask predictor for pre-sequencing error estimate
        if predictor is not None and len(predictor.buffer) >= config.predictor_batch_size:
            features = ErrorPredictor.build_features(
                gc_content=state[0],                    # dibit_00 frequency (proxy)
                entropy=state[4],                       # max dibit frequency
                normalized_homopolymer=state[1],        # dibit_01 frequency
                sequence_length=config.training_sequence_length,
                mapping_index=0,
            )
            predicted_error = predictor.predict(features)
            environment.set_predicted_error(predicted_error)

        # agent selects action
        action, log_probability, value = agent.select_action(state)

        # step environment
        next_state, reward, done, information = environment.step(action)

        # store transition
        agent.store_transition(state, action, log_probability, reward, value, done)

        # collect predictor data
        if predictor is not None:
            features = ErrorPredictor.build_features(
                gc_content=information["gc_content"],
                entropy=information["entropy"],
                normalized_homopolymer=information["max_homopolymer"],
                sequence_length=config.training_sequence_length,
                mapping_index=information["primary_index"],
            )
            predictor.collect(features, information["error_rate"])

        # PPO update when batch is full
        if len(agent.buffer) >= config.ppo_batch_size:
            losses = agent.update()
            policy_losses.append(losses["policy_loss"])
            value_losses.append(losses["value_loss"])
            entropies.append(losses["entropy"])
            learning_rates.append(losses["learning_rate"])

        # train predictor periodically
        if predictor is not None and epoch % config.predictor_training_interval == 0:
            predictor_loss = predictor.train_step(random_generator)
            predictor_losses.append(predictor_loss)

        # logging
        epoch_rewards.append(reward)
        rolling_sum += reward
        if epoch > rolling_window:
            rolling_sum -= epoch_rewards[epoch - rolling_window - 1]
        rolling_average = rolling_sum / min(epoch, rolling_window)
        rolling_rewards.append(rolling_average)

        # validation evaluation
        if epoch % validation_interval == 0:
            validation_reward = _validate(agent, config, validation_random_generator)
            validation_rewards.append(validation_reward)

            if validation_reward > best_validation_reward:
                best_validation_reward = validation_reward
                epochs_without_improvement = 0
                # save best model
                best_path = os.path.join(checkpoint_directory, f"best_agent_seed{seed}.pt")
                agent.save(best_path)
            else:
                epochs_without_improvement += validation_interval

            # early stopping check
            if epochs_without_improvement >= config.early_stopping_patience:
                if verbose:
                    print(f"  [Seed {seed}] Early stopping at epoch {epoch} "
                          f"(no improvement for {epochs_without_improvement} epochs)")
                stopped_early = True
                final_epoch = epoch
                break

        if verbose and epoch % 2000 == 0:
            current_learning_rate = agent.optimizer.param_groups[0]["lr"]
            validation_display = validation_rewards[-1] if validation_rewards else 0.0
            print(f"  [Seed {seed}] Epoch {epoch:>6d}/{total_epochs}  "
                  f"rolling_reward={rolling_average:.3f}  "
                  f"bit_accuracy={information['bit_accuracy']:.3f}  "
                  f"error_rate={information['error_rate']:.4f}  "
                  f"learning_rate={current_learning_rate:.6f}  "
                  f"validation_reward={validation_display:.3f}")

        # checkpoint save
        if epoch % checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_directory, f"agent_seed{seed}_epoch{epoch}.pt"
            )
            agent.save(checkpoint_path)

    # compute smoothed rewards (EMA) for stability analysis
    smoothed_rewards = _exponential_moving_average(epoch_rewards, alpha=0.02)

    return {
        "agent": agent,
        "predictor": predictor,
        "environment": environment,
        "epoch_rewards": epoch_rewards,
        "rolling_rewards": rolling_rewards,
        "smoothed_rewards": smoothed_rewards,
        "predictor_losses": predictor_losses,
        "policy_losses": policy_losses,
        "value_losses": value_losses,
        "entropies": entropies,
        "learning_rates": learning_rates,
        "validation_rewards": validation_rewards,
        "best_validation_reward": best_validation_reward,
        "stopped_early": stopped_early,
        "final_epoch": final_epoch,
    }
