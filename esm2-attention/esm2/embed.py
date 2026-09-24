"""ESM-2 per-ORF embeddings: mean-pool residues, one 320-d vector per ORF."""

from __future__ import annotations

import numpy as np
import torch


def load_esm(model_name: str):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    model.eval()
    return tokenizer, model


def _token_batches(peptides: list[str], batch_tokens: int):
    """Length-sorted batches capped by padded token count, to avoid padding waste."""
    order = sorted(range(len(peptides)), key=lambda i: len(peptides[i]))
    start = 0
    while start < len(order):
        stop, longest = start, 0
        while stop < len(order):
            longest = max(longest, len(peptides[order[stop]]) + 2)
            if (stop + 1 - start) * longest > batch_tokens and stop > start:
                break
            stop += 1
        yield order[start:stop]
        start = stop


def embed_peptides(peptides: list[str], tokenizer, model, hidden_size: int,
                   max_length: int = 1024, batch_tokens: int = 8192,
                   progress=None) -> np.ndarray:
    out = np.zeros((len(peptides), hidden_size), dtype=np.float32)
    done = 0
    for idx in _token_batches(peptides, batch_tokens):
        batch = [peptides[i] for i in idx]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_length)
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out[idx] = pooled.numpy()
        done += len(idx)
        if progress:
            progress(done, len(peptides))
    return out
