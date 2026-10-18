"""Checks that the re-implementation matches the prototype before anything is trained.

1. ORF extraction reproduces the prototype's function exactly.
2. Length-bucketed batching gives the same embeddings as the prototype's
   per-fragment batch-of-8 loop.
3. Attention pooling equals an explicit softmax-weighted sum.

Three checks were added for the server run, because that run changes *how* the
same computation is performed and each change could silently alter the numbers:

4. A padded, masked batch gives the same logits as one fragment at a time. This is
   what makes `train.batch_size > 1` a performance change rather than a modelling
   change.
5. Embedding on the GPU gives the same vectors as on the CPU, and the cost of
   bfloat16 autocast and float16 storage is measured rather than assumed. Skipped
   with a clear message when no GPU is allocated.
6. Removing the ORF cap only ever adds ORFs: the capped result must be a prefix of
   the uncapped one, so `max_orfs: null` cannot change which ORFs the old
   configuration saw.

    python verify_components.py                           # CPU checks
    python verify_components.py --config config_server.yaml   # + GPU checks
"""

from __future__ import annotations

import argparse
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


def check_batched_equivalence(hidden, cfg):
    """A padded masked batch must equal the per-fragment path, logit for logit."""
    torch.manual_seed(0)
    for aux_dim in (0, 12):
        net = model_mod.EsmAttentionClassifier(hidden, cfg["train"]["classifier_hidden"],
                                               cfg["train"]["dropout"], aux_dim=aux_dim)
        net.eval()   # dropout off: it is the only stochastic layer
        counts = [1, 7, 3, 64, 2]
        fragments = [torch.randn(n, hidden) for n in counts]
        auxes = [torch.randn(aux_dim) for _ in counts] if aux_dim else [None] * len(counts)

        one_at_a_time = torch.stack([
            net(x) if a is None else net(x, a) for x, a in zip(fragments, auxes)])

        width = max(counts)
        padded = torch.zeros(len(counts), width, hidden)
        mask = torch.zeros(len(counts), width, dtype=torch.bool)
        for j, x in enumerate(fragments):
            # Deliberately fill the padding with large values: if the mask were
            # wrong, the padded rows would drag the pooled vector far away and the
            # comparison below could not pass by accident.
            padded[j] = 1e3
            padded[j, :len(x)] = x
            mask[j, :len(x)] = True
        batched_aux = torch.stack(auxes) if aux_dim else None
        batched = net(padded, batched_aux, mask)

        gap = (one_at_a_time - batched).abs().max().item()
        assert gap < 1e-5, (f"padded batch differs from per-fragment by {gap:.3e} "
                            f"(aux_dim={aux_dim}); the attention mask is wrong")
        print(f"  padded masked batch == per-fragment path, aux_dim={aux_dim} "
              f"(max |diff| {gap:.2e}, padding filled with 1e3)")


def check_orf_cap_is_a_prefix(sequences, cfg):
    """Uncapped ORF calling must only ADD ORFs, keeping the capped ones unchanged."""
    worst, n_capped, totals = 0, 0, []
    for seq in sequences:
        capped = orfs.extract_orfs(seq, cfg["orf"]["min_length"], 10)
        uncapped = orfs.extract_orfs(seq, cfg["orf"]["min_length"], None)
        totals.append(len(uncapped))
        assert uncapped[:len(capped)] == capped, (
            "removing max_orfs changed which ORFs come first; the two "
            "configurations would not be comparable")
        if len(uncapped) > len(capped):
            n_capped += 1
            worst = max(worst, len(uncapped) - len(capped))
    print(f"  uncapped ORF calling is a superset: {n_capped}/{len(sequences)} fragments "
          f"had ORFs discarded by max_orfs=10, up to {worst} each "
          f"(mean {np.mean(totals):.1f} ORFs per fragment uncapped)")


def check_gpu_embedding(peptides, tokenizer, cfg, hidden):
    if not torch.cuda.is_available():
        print("  SKIPPED: no GPU visible. Run this inside a GPU allocation "
              "(srun --gres=... ) to check CPU/GPU agreement.")
        return
    print(f"  device: {embed.describe_device('cuda')}")
    _, gpu_model = embed.load_esm(cfg["esm"]["model_name"], "cuda")
    _, cpu_model = embed.load_esm(cfg["esm"]["model_name"], "cpu")
    sample = peptides[:64]
    ref = embed.embed_peptides(sample, tokenizer, cpu_model, hidden,
                               cfg["esm"]["max_length"], 8192, amp=False)
    exact = embed.embed_peptides(sample, tokenizer, gpu_model, hidden,
                                 cfg["esm"]["max_length"], cfg["esm"]["batch_tokens"],
                                 device="cuda", amp=False)
    gap = np.abs(exact - ref).max()
    assert gap < 1e-3, f"GPU float32 embeddings differ from CPU by {gap}"
    print(f"  GPU float32 == CPU float32 (max |diff| {gap:.2e})")

    if cfg["esm"].get("amp"):
        amped = embed.embed_peptides(sample, tokenizer, gpu_model, hidden,
                                     cfg["esm"]["max_length"], cfg["esm"]["batch_tokens"],
                                     device="cuda", amp=True)
        d = np.abs(amped - ref)
        scale = np.abs(ref).mean()
        print(f"  bfloat16 autocast: max |diff| {d.max():.2e}, mean |diff| "
              f"{d.mean():.2e} against a mean |value| of {scale:.3e} "
              f"({100 * d.mean() / scale:.3f}% relative)")
    if cfg["esm"].get("store_float16"):
        d = np.abs(ref.astype(np.float16).astype(np.float32) - ref)
        print(f"  float16 storage: max |diff| {d.max():.2e}, mean |diff| {d.mean():.2e} "
              "(this is what store_float16 costs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--n-fragments", type=int, default=12)
    args = ap.parse_args()

    cfg = data.load_config(args.config)
    rng = np.random.default_rng(0)

    table = data.load_table(cfg["dataset_dir"], "test")
    picked = table.iloc[np.sort(rng.choice(len(table), size=args.n_fragments,
                                           replace=False))]
    seqs = data.read_fasta_subset(Path(cfg["dataset_dir"]) / "test.fasta",
                                  picked["fragment_id"])
    sequences = [seqs[f] for f in picked["fragment_id"]]

    print("1. ORF extraction vs the prototype")
    # Compared at the prototype's own cap of 10, which is the only setting the
    # prototype has; the cap itself is checked separately in step 6.
    mine = [orfs.extract_orfs(s, cfg["orf"]["min_length"], 10) for s in sequences]
    theirs = [prototype_extract_orfs(s, cfg["orf"]["min_length"], 10) for s in sequences]
    assert mine == theirs, "ORF extraction differs from the prototype"
    print(f"  matches on {len(sequences)} fragments ({sum(len(o) for o in mine)} ORFs)")

    torch.set_num_threads(cfg["esm"]["torch_threads"])
    tokenizer, esm = embed.load_esm(cfg["esm"]["model_name"], "cpu")
    hidden = int(esm.config.hidden_size)
    if cfg["esm"].get("hidden_size") and int(cfg["esm"]["hidden_size"]) != hidden:
        raise SystemExit(f"config says hidden_size={cfg['esm']['hidden_size']} but the "
                         f"checkpoint has {hidden}")
    print(f"\n2. length-bucketed embedding vs the prototype loop (hidden {hidden})")
    flat = [p for o in mine for p in o]
    bucketed = embed.embed_peptides(flat, tokenizer, esm, hidden,
                                    cfg["esm"]["max_length"], cfg["esm"]["batch_tokens"])
    reference = torch.cat([prototype_embed(o, tokenizer, esm, hidden)
                           for o in mine]).numpy()
    gap = np.abs(bucketed - reference).max()
    assert gap < 1e-4, f"bucketed embeddings differ by {gap}"
    print(f"  matches (max |diff| {gap:.2e})")

    print("\n3. attention pooling vs an explicit softmax-weighted sum")
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
    print("  matches")

    print("\n4. padded masked batch vs per-fragment path")
    check_batched_equivalence(hidden, cfg)

    print("\n5. GPU vs CPU embedding, and the cost of reduced precision")
    check_gpu_embedding(flat, tokenizer, cfg, hidden)

    print("\n6. removing the ORF cap only adds ORFs")
    check_orf_cap_is_a_prefix(sequences, cfg)

    print("\nall component checks passed")


if __name__ == "__main__":
    main()
