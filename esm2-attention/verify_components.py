"""Checks that the re-implementation matches the prototype before anything is trained.

1. ORF extraction reproduces the prototype's function exactly.
2. Length-bucketed batching gives the same embeddings as the prototype's
   per-fragment batch-of-8 loop.
3. Attention pooling equals an explicit softmax-weighted sum.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from Bio.Seq import Seq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from esm2 import data, embed, model as model_mod, orfs  # noqa: E402


def prototype_extract_orfs(dna_seq, min_len=100, max_orfs=10):
    """Verbatim from old resource/09_train_esm2.py."""
    seq_obj = Seq(dna_seq)
    found = []
    for strand, seq in [(1, seq_obj), (-1, seq_obj.reverse_complement())]:
        for frame in range(3):
            try:
                l = len(seq) - frame
                padded = seq[frame:frame + (l - l % 3)]
                peps = str(padded.translate()).split("*")
                found.extend([p for p in peps if len(p) >= min_len])
            except Exception:
                continue
    found.sort(key=len, reverse=True)
    return found[:max_orfs]


def prototype_embed(seq_orfs, tokenizer, esm, hidden):
    """Verbatim pooling from the prototype: batches of 8 within one fragment."""
    if not seq_orfs:
        return torch.zeros(1, hidden)
    chunks = []
    for i in range(0, len(seq_orfs), 8):
        batch = seq_orfs[i:i + 8]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=1024)
        with torch.no_grad():
            out = esm(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(out.size()).float()
            chunks.append((out * mask).sum(1) / mask.sum(1).clamp(min=1e-9))
    return torch.cat(chunks, dim=0)


def main():
    cfg = data.load_config(HERE / "config.yaml")
    rng = np.random.default_rng(0)

    table = data.load_table(cfg["dataset_dir"], "test")
    picked = table.iloc[np.sort(rng.choice(len(table), size=12, replace=False))]
    seqs = data.read_fasta_subset(Path(cfg["dataset_dir"]) / "test.fasta",
                                  picked["fragment_id"])
    sequences = [seqs[f] for f in picked["fragment_id"]]

    mine = [orfs.extract_orfs(s, cfg["orf"]["min_length"], cfg["orf"]["max_orfs"])
            for s in sequences]
    theirs = [prototype_extract_orfs(s, cfg["orf"]["min_length"], cfg["orf"]["max_orfs"])
              for s in sequences]
    assert mine == theirs, "ORF extraction differs from the prototype"
    print(f"  ORF extraction matches the prototype on {len(sequences)} fragments "
          f"({sum(len(o) for o in mine)} ORFs)")

    torch.set_num_threads(cfg["esm"]["torch_threads"])
    tokenizer, esm = embed.load_esm(cfg["esm"]["model_name"])
    hidden = cfg["esm"]["hidden_size"]

    flat = [p for o in mine for p in o]
    bucketed = embed.embed_peptides(flat, tokenizer, esm, hidden,
                                    cfg["esm"]["max_length"], cfg["esm"]["batch_tokens"])
    reference = torch.cat([prototype_embed(o, tokenizer, esm, hidden) for o in mine]).numpy()
    gap = np.abs(bucketed - reference).max()
    assert gap < 1e-4, f"bucketed embeddings differ by {gap}"
    print(f"  length-bucketed embeddings match the prototype loop (max |diff| {gap:.2e})")

    torch.manual_seed(0)
    net = model_mod.EsmAttentionClassifier(hidden, cfg["train"]["classifier_hidden"],
                                           cfg["train"]["dropout"])
    net.eval()
    x = torch.randn(7, hidden)
    scores = net.attn_pool.attention(x)
    manual = (torch.softmax(scores, dim=0) * x).sum(dim=0, keepdim=True)
    expected = net.classifier(manual).squeeze()
    assert torch.allclose(net(x), expected, atol=1e-6), "attention pooling mismatch"
    assert torch.allclose(net.attn_pool.weights(x).sum(), torch.tensor(1.0), atol=1e-6)
    print("  attention pooling equals an explicit softmax-weighted sum")
    print("all component checks passed")


if __name__ == "__main__":
    main()
