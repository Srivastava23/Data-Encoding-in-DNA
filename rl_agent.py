"""
PPO (Proximal Policy Optimization) agent implemented in PyTorch.

Actor-Critic architecture with:
    - shared two-hidden-layer backbone (64 units each)
    - separate actor (softmax) and critic (scalar value) heads
    - clipped surrogate objective + value loss + entropy bonus
    - rollout buffer for on-policy updates
    - dropout regularisation for anti-overfitting
    - save/load support for reproducibility
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Dict

from config import Config


class ActorCritic(nn.Module):
    """Two-layer MLP with separate actor and critic heads, with optional dropout."""

    def __init__(self, state_dimension: int, action_dimension: int,
                 hidden_dimension: int = 64, dropout_rate: float = 0.0):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dimension, hidden_dimension),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.actor_head = nn.Linear(hidden_dimension, action_dimension)
        self.critic_head = nn.Linear(hidden_dimension, 1)

    def forward(self, input_tensor: torch.Tensor):
        features = self.shared_backbone(input_tensor)
        action_logits = self.actor_head(features)
        state_value = self.critic_head(features).squeeze(-1)
        return action_logits, state_value

    def act(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select an action and return (action, log_probability, value)."""
        input_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_logits, state_value = self.forward(input_tensor)
        distribution = Categorical(logits=action_logits)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action).item(), state_value.item()

    def evaluate(self, states: torch.Tensor,
                 actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probabilities, values, and entropy for given states/actions."""
        action_logits, state_values = self.forward(states)
        distribution = Categorical(logits=action_logits)
        log_probabilities = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probabilities, state_values, entropy


class RolloutBuffer:
    """Stores transitions collected during rollout for PPO update."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probabilities: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def store(self, state, action, log_probability, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probabilities.append(log_probability)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)

    def get_batches(self, discount_factor: float) -> Dict[str, torch.Tensor]:
        """Compute discounted returns and advantages, return tensors."""
        # single-step epochs -> return = reward
        returns = []
        running_return = 0.0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                running_return = 0.0
            running_return = reward + discount_factor * running_return
            returns.insert(0, running_return)

        returns_tensor = torch.FloatTensor(returns)
        values_tensor = torch.FloatTensor(self.values)
        advantages = returns_tensor - values_tensor
        # normalise advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "states": torch.FloatTensor(np.array(self.states)),
            "actions": torch.LongTensor(self.actions),
            "old_log_probabilities": torch.FloatTensor(self.log_probabilities),
            "returns": returns_tensor,
            "advantages": advantages,
        }


class PPOAgent:
    """Proximal Policy Optimization agent."""

    def __init__(self, state_dimension: int, action_dimension: int, config: Config):
        self.config = config
        self.network = ActorCritic(
            state_dimension, action_dimension,
            config.ppo_hidden_dimension, config.dropout_rate,
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.ppo_learning_rate)

        # learning rate scheduler (cosine annealing)
        if config.use_learning_rate_scheduling:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.ppo_training_epochs,
                eta_min=config.minimum_learning_rate,
            )
        else:
            self.scheduler = None

        self.buffer = RolloutBuffer()

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        self.network.eval()
        result = self.network.act(state)
        self.network.train()
        return result

    def store_transition(self, state, action, log_probability, reward, value, done):
        self.buffer.store(state, action, log_probability, reward, value, done)

    def update(self) -> Dict[str, float]:
        """Run PPO update on the collected rollout buffer."""
        configuration = self.config
        batch = self.buffer.get_batches(configuration.ppo_discount_factor)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(configuration.ppo_update_passes):
            log_probabilities, values, entropy = self.network.evaluate(
                batch["states"], batch["actions"]
            )

            # clipped surrogate objective
            ratio = torch.exp(log_probabilities - batch["old_log_probabilities"])
            surrogate_1 = ratio * batch["advantages"]
            surrogate_2 = torch.clamp(ratio, 1.0 - configuration.ppo_clipping_epsilon,
                                1.0 + configuration.ppo_clipping_epsilon) * batch["advantages"]
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

            # value loss
            value_loss = nn.functional.mse_loss(values, batch["returns"])

            # combined loss
            loss = (
                policy_loss
                + configuration.ppo_value_coefficient * value_loss
                - configuration.ppo_entropy_coefficient * entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), configuration.ppo_maximum_gradient_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        number_of_passes = configuration.ppo_update_passes
        self.buffer.clear()

        # step the learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "policy_loss": total_policy_loss / number_of_passes,
            "value_loss": total_value_loss / number_of_passes,
            "entropy": total_entropy / number_of_passes,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def save(self, path: str):
        """Save network weights."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        """Load network weights."""
        self.network.load_state_dict(torch.load(path, map_location="cpu"))
