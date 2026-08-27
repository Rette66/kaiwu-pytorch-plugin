from __future__ import annotations

import torch
from torch import nn


class TinyCNN(nn.Module):
    """Small CNN used by the feature-selection classification example."""

    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * self.image_size * self.image_size, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for flattened image inputs."""
        images = inputs.view(-1, 1, self.image_size, self.image_size)
        return self.net(images)


class SimpleRNN(nn.Module):
    """Small RNN classifier used by the feature-selection example."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.rnn = nn.RNN(feature_dim, hidden_size=8, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for sequence inputs."""
        _sequence, hidden = self.rnn(inputs)
        return self.head(hidden[-1])


class SimpleLSTM(nn.Module):
    """Small LSTM classifier used by the feature-selection example."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_size=8, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for sequence inputs."""
        _sequence, (hidden, _cell) = self.lstm(inputs)
        return self.head(hidden[-1])
