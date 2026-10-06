import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Dict
from config import Config

class ActorCriticV3(nn.Module):
    def __init__(self, state_dimension: int, action_dimension: int, hidden_dimension: int):
        super(ActorCriticV3, self).__init__()
        self.actor_fc = nn.Linear(state_dimension, hidden_dimension)
        self.actor_gru = nn.GRU(hidden_dimension, hidden_dimension, batch_first=True)
        self.actor_out = nn.Linear(hidden_dimension, action_dimension)

        self.critic_fc = nn.Linear(state_dimension, hidden_dimension)
        self.critic_gru = nn.GRU(hidden_dimension, hidden_dimension, batch_first=True)
        self.critic_out = nn.Linear(hidden_dimension, 1)

    def forward(self, state: torch.Tensor, actor_hx: torch.Tensor = None, critic_hx: torch.Tensor = None):
        """
        state shape: (batch, seq_len, state_dim) or (batch, state_dim)
        """
        is_unbatched = False
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)
            is_unbatched = True
        elif state.dim() == 2:
            state = state.unsqueeze(1)
            
        a = torch.relu(self.actor_fc(state))
        a, actor_hx = self.actor_gru(a, actor_hx)
        action_logits = self.actor_out(a)

        c = torch.relu(self.critic_fc(state))
        c, critic_hx = self.critic_gru(c, critic_hx)
        state_value = self.critic_out(c)

        if is_unbatched:
            action_logits = action_logits.squeeze(0).squeeze(0)
            state_value = state_value.squeeze(0).squeeze(0)
            
        return action_logits, state_value, actor_hx, critic_hx

class PPOAgentV3:
    def __init__(self, state_dimension: int, action_dimension: int, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = ActorCriticV3(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            hidden_dimension=config.ppo_hidden_dimension
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=config.ppo_learning_rate
        )
        
        self.entropy_coef = config.ppo_entropy_coefficient
        
    def step_decay(self, progress: float):
        # Anneal entropy
        min_e = self.config.ppo_entropy_coefficient_min
        max_e = self.config.ppo_entropy_coefficient
        self.entropy_coef = max_e - progress * (max_e - min_e)

    def update(self, rollouts: List[Dict]) -> Tuple[float, float, float]:
        """
        rollouts is a list of episodes.
        Each episode is a dict:
        {
            'states': np.ndarray (N, state_dim),
            'actions': np.ndarray (N,),
            'rewards': np.ndarray (N,),
            'values': np.ndarray (N,),
            'log_probs': np.ndarray (N,)
        }
        """
        if not rollouts:
            return 0.0, 0.0, 0.0

        all_states = []
        all_actions = []
        all_returns = []
        all_advantages = []
        all_old_log_probs = []
        
        gamma = self.config.ppo_discount_factor
        lam = self.config.ppo_gae_lambda

        for ep in rollouts:
            N = len(ep['rewards'])
            returns = np.zeros(N, dtype=np.float32)
            advantages = np.zeros(N, dtype=np.float32)
            
            last_gae = 0.0
            for t in reversed(range(N)):
                next_val = ep['values'][t+1] if t + 1 < N else 0.0
                delta = ep['rewards'][t] + gamma * next_val - ep['values'][t]
                advantages[t] = last_gae = delta + gamma * lam * last_gae
                returns[t] = advantages[t] + ep['values'][t]
                
            all_states.append(ep['states'])
            all_actions.append(ep['actions'])
            all_returns.append(returns)
            all_advantages.append(advantages)
            all_old_log_probs.append(ep['log_probs'])
            
        # pad or stack. Since N is fixed per sequence for now, we can just stack.
        # all_states: (num_episodes, N, state_dim)
        states = torch.tensor(np.stack(all_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.stack(all_actions), dtype=torch.long).to(self.device)
        returns = torch.tensor(np.stack(all_returns), dtype=torch.float32).to(self.device)
        advantages = torch.tensor(np.stack(all_advantages), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(np.stack(all_old_log_probs), dtype=torch.float32).to(self.device)
        
        # normalize advantages over the batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        num_episodes = states.shape[0]
        batch_size = self.config.ppo_batch_size
        indices = np.arange(num_episodes)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        for _ in range(self.config.ppo_update_passes):
            np.random.shuffle(indices)
            for start in range(0, num_episodes, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                b_states = states[batch_idx] # (B, N, state_dim)
                b_actions = actions[batch_idx] # (B, N)
                b_returns = returns[batch_idx]
                b_advantages = advantages[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                
                logits, values, _, _ = self.network(b_states) # logits: (B, N, action_dim), values: (B, N, 1)
                values = values.squeeze(-1)
                
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.ppo_clipping_epsilon, 1.0 + self.config.ppo_clipping_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(values, b_returns)
                
                loss = policy_loss + self.config.ppo_value_coefficient * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.ppo_maximum_gradient_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
                
        if num_updates > 0:
            return total_policy_loss / num_updates, total_value_loss / num_updates, total_entropy / num_updates
        return 0.0, 0.0, 0.0

    def save(self, filepath: str):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
