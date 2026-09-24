"""PyTorch re-implementations of the two baseline architectures."""

from __future__ import annotations

import torch
from torch import nn


def _glorot(module: nn.Module):
    """TF DNNClassifier and Keras Dense both default to glorot_uniform + zero bias."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return module


class PlasFlowDNN(nn.Module):
    """tf.contrib.learn.DNNClassifier(hidden_units=..., activation_fn=relu, n_classes=2)."""

    def __init__(self, n_features: int, hidden: list[int], n_classes: int = 2):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = _glorot(nn.Sequential(*layers))

    def forward(self, x):
        return self.net(x)

    def optimizer(self, lr: float):
        # DNNClassifier's default Adagrad uses initial_accumulator_value=0.1.
        return torch.optim.Adagrad(self.parameters(), lr=lr, initial_accumulator_value=0.1)


class ZouMLP(nn.Module):
    """model_train.py: 256-256-128-128-32-10-1 with dropout and L2 on the first four layers."""

    def __init__(self, n_features: int = 2080, dropout=(0.4, 0.4, 0.5, 0.2)):
        super().__init__()
        d1, d2, d3, d4 = dropout
        self.regularised = nn.ModuleList()
        blocks, prev = [], n_features
        for width, rate in zip((256, 256, 128, 128), (d1, d2, d3, d4)):
            linear = nn.Linear(prev, width)
            self.regularised.append(linear)
            blocks += [linear, nn.ReLU(), nn.Dropout(rate)]
            prev = width
        blocks += [nn.Linear(prev, 32), nn.ReLU(), nn.Linear(32, 10), nn.ReLU(), nn.Linear(10, 1)]
        self.net = _glorot(nn.Sequential(*blocks))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def optimizer(self, lr: float, l2: float):
        # Keras l2(lambda) contributes 2*lambda*w to the gradient; torch weight_decay
        # contributes weight_decay*w, so the equivalent setting is 2*lambda.
        reg = {id(p) for m in self.regularised for p in (m.weight,)}
        decayed = [p for p in self.parameters() if id(p) in reg]
        plain = [p for p in self.parameters() if id(p) not in reg]
        return torch.optim.Adam(
            [{"params": decayed, "weight_decay": 2 * l2}, {"params": plain, "weight_decay": 0.0}],
            lr=lr,
        )
