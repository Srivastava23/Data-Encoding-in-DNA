"""
Neural network that predicts sequencing error probability.

A small feed-forward network trained on data collected during
reinforcement learning interaction. It learns to estimate the expected
error rate before sequencing, based on sequence-level features.

If disabled (config.use_error_predictor = False), the rest of the system
still functions normally — the predicted-error term in the reward
is simply zero.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

from config import Config


class ErrorPredictorNetwork(nn.Module):
    """
    Simple MLP: 5 input features -> predicted error probability.

    Input features:
        [0] GC content
        [1] Shannon entropy (normalised / 2)
        [2] Normalised max homopolymer
        [3] Normalised sequence length (sequence_length / 256)
        [4] Encoding policy index (mapping_index / 24)

    Output: single scalar in [0, 1].
    """

    INPUT_DIMENSION = 5

    def __init__(self, hidden_dimension: int = 64, dropout_rate: float = 0.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.INPUT_DIMENSION, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(input_tensor).squeeze(-1)


class PredictorBuffer:
    """Fixed-capacity ring buffer storing (features, actual_error_rate) pairs."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.features: List[np.ndarray] = []
        self.targets: List[float] = []
        self._index = 0

    def add(self, feature_vector: np.ndarray, error_rate: float):
        if len(self.features) < self.capacity:
            self.features.append(feature_vector)
            self.targets.append(error_rate)
        else:
            self.features[self._index] = feature_vector
            self.targets[self._index] = error_rate
        self._index = (self._index + 1) % self.capacity

    def __len__(self):
        return len(self.features)

    def sample(self, batch_size: int, random_generator: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        total_samples = len(self)
        indices = random_generator.choice(total_samples, size=min(batch_size, total_samples), replace=False)
        feature_batch = np.array([self.features[i] for i in indices], dtype=np.float32)
        target_batch = np.array([self.targets[i] for i in indices], dtype=np.float32)
        return feature_batch, target_batch


class ErrorPredictor:
    """
    Wrapper around the predictor network + buffer.

    Usage:
        predictor = ErrorPredictor(config)
        predictor.collect(features, error_rate)
        loss = predictor.train_step(random_generator)
        prediction = predictor.predict(features)
    """

    def __init__(self, config: Config):
        self.config = config
        self.network = ErrorPredictorNetwork(
            config.predictor_hidden_dimension,
            config.dropout_rate,
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.predictor_learning_rate)
        self.buffer = PredictorBuffer(config.predictor_buffer_size)
        self.loss_history: List[float] = []

    @staticmethod
    def build_features(gc_content: float, entropy: float, normalized_homopolymer: float,
                       sequence_length: int, mapping_index: int) -> np.ndarray:
        """Construct the 5-dimension feature vector."""
        return np.array([
            gc_content,
            entropy / 2.0,
            normalized_homopolymer,
            sequence_length / 256.0,
            mapping_index / 24.0,
        ], dtype=np.float32)

    def collect(self, features: np.ndarray, error_rate: float):
        """Store one observation in the buffer."""
        self.buffer.add(features, error_rate)

    def predict(self, features: np.ndarray) -> float:
        """Return predicted error probability for one feature vector."""
        self.network.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            prediction = self.network(input_tensor).item()
        return prediction

    def train_step(self, random_generator: np.random.Generator) -> float:
        """
        Train on a mini-batch from the buffer.
        Returns the average loss over all update passes.
        """
        if len(self.buffer) < self.config.predictor_batch_size:
            return 0.0

        self.network.train()
        total_loss = 0.0

        for _ in range(self.config.predictor_update_passes):
            feature_batch, target_batch = self.buffer.sample(self.config.predictor_batch_size, random_generator)
            feature_tensor = torch.FloatTensor(feature_batch)
            target_tensor = torch.FloatTensor(target_batch)

            prediction = self.network(feature_tensor)
            loss = nn.functional.mse_loss(prediction, target_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / self.config.predictor_update_passes
        self.loss_history.append(average_loss)
        return average_loss

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path, map_location="cpu"))
