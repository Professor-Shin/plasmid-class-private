"""The proposed classifier, transcribed from `old resource/10_esm_attention.py`.

Both modules accept either a single fragment, exactly as the prototype did, or a
padded batch of fragments. The single-fragment path is byte-for-byte the original
computation and is what reproduces the laptop numbers; the batched path exists
because one fragment per optimisation step cannot train on 79,094 fragments.

`verify_components.py` asserts the two paths agree to 1e-5, so the batching is a
performance change and not a modelling change.
"""

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

    def forward(self, x, mask=None):
        if x.dim() == 2:                       # (n_orfs, hidden): the original path
            weights = torch.softmax(self.attention(x), dim=0)
            return torch.sum(weights * x, dim=0, keepdim=True)

        # (batch, max_orfs, hidden): padded batch.
        scores = self.attention(x)             # (batch, max_orfs, 1)
        if mask is not None:
            # Padding rows must not receive attention mass. Masking the logits to
            # -inf before the softmax is what makes a padded batch give the same
            # answer as the unpadded fragment; zeroing the weights afterwards
            # would leave the softmax denominator wrong.
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights * x, dim=1)   # (batch, hidden)

    def weights(self, x):
        return torch.softmax(self.attention(x), dim=0).squeeze(-1)


class EsmAttentionClassifier(nn.Module):
    """ORF-attention classifier, optionally widened by whole-fragment features.

    ``aux_dim == 0`` reproduces the original ORF-only model exactly. A non-zero
    ``aux_dim`` concatenates a per-fragment vector (GC / origin-of-replication
    composition, a DNA-LM embedding, or any combination) onto the attention-pooled
    ORF representation before the classifier head. That is the only way non-coding
    sequence reaches this model: the ORF pathway sees translated peptides and
    nothing else, and an ORF-free fragment arrives as a single zero vector.
    """

    def __init__(self, hidden_size: int = 320, classifier_hidden: int = 64,
                 dropout: float = 0.3, aux_dim: int = 0):
        super().__init__()
        self.attn_pool = AttentionPooling(hidden_size)
        self.aux_dim = int(aux_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + self.aux_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(self, x, aux=None, mask=None):
        batched = x.dim() == 3
        pooled = self.attn_pool(x, mask) if batched else self.attn_pool(x)
        if self.aux_dim:
            if aux is None:
                raise ValueError("model was built with aux_dim > 0 but got aux=None")
            pooled = torch.cat(
                [pooled, aux.reshape(len(pooled), -1) if batched else aux.reshape(1, -1)],
                dim=1)
        logits = self.classifier(pooled)
        # The single-fragment path keeps the prototype's bare squeeze(), which
        # yields a 0-d tensor; the batched path must stay (batch,) even when the
        # batch happens to hold one fragment.
        return logits.squeeze(-1) if batched else logits.squeeze()
