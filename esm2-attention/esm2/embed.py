"""ESM-2 per-ORF embeddings: mean-pool residues, one vector per ORF.

CPU and GPU share one code path. Two details matter for the numbers to stay
comparable with the laptop run:

* the hidden size is read from the loaded checkpoint rather than from config, so
  switching to a larger ESM-2 is a config change and not a code change (the old
  code hardcoded 320, which is only true of esm2_t6_8M);
* mixed precision, when enabled, applies to the transformer only. The mean pool
  is always computed in float32, because summing several hundred residues in
  bfloat16 loses precision exactly where it would matter.
"""

from __future__ import annotations

import numpy as np
import torch


def pick_device(requested: str | None = "auto") -> str:
    """Resolve 'auto' against what is actually available."""
    if requested in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "config asks for CUDA but torch.cuda.is_available() is False. Inside a "
            "Slurm allocation this usually means the job did not request a GPU "
            "(--gres) or the torch wheel does not match the driver; on a login node "
            "it is expected.")
    return requested


def describe_device(device: str) -> str:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return (f"{device} ({props.name}, {props.total_memory / 2**30:.1f} GiB, "
                f"cc {props.major}.{props.minor})")
    return f"{device} ({torch.get_num_threads()} threads)"


def load_esm(model_name: str, device: str = "cpu"):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    model.eval()
    model.to(device)
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
                   progress=None, device: str | None = None, amp: bool = False,
                   out_dtype=np.float32) -> np.ndarray:
    if device is None:
        device = str(next(model.parameters()).device)
    use_amp = bool(amp) and str(device).startswith("cuda")
    out = np.zeros((len(peptides), hidden_size), dtype=out_dtype)
    done = 0
    for idx in _token_batches(peptides, batch_tokens):
        batch = [peptides[i] for i in idx]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_length)
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        with torch.inference_mode():
            if use_amp:
                # bfloat16 rather than float16: same speed on an A100 and no loss
                # scaling needed, so there is nothing to tune and nothing to
                # overflow in a forward-only pass.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model(**inputs).last_hidden_state
            else:
                hidden = model(**inputs).last_hidden_state
            hidden = hidden.float()
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out[idx] = pooled.cpu().numpy().astype(out_dtype, copy=False)
        done += len(idx)
        if progress:
            progress(done, len(peptides))
    return out
