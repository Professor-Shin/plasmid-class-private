"""The proposed classifier, transcribed from `old resource/10_esm_attention.py`."""

from __future__ import annotations

import torch
from torch import nn


class AttentionPooling(nn.Module):
    """Learned softmax weighting over a fragment's ORF embeddings."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=0)
        return torch.sum(weights * x, dim=0, keepdim=True)

    def weights(self, x):
        return torch.softmax(self.attention(x), dim=0).squeeze(-1)


class EsmAttentionClassifier(nn.Module):
    def __init__(self, hidden_size: int = 320, classifier_hidden: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.attn_pool = AttentionPooling(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(self, x):
        return self.classifier(self.attn_pool(x)).squeeze()
