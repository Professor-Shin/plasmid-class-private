"""Download the ESM-2 checkpoints into $HF_HOME on the login node.

Compute nodes may have no internet access, and `transformers` would otherwise try
to reach huggingface.co on the first forward pass inside the GPU allocation. Both
the model used for the headline run (t6_8M) and the optional larger one (t12_35M)
are cached here so switching config files later needs no network.
"""

from transformers import AutoModel, AutoTokenizer

MODELS = ("facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D")

for name in MODELS:
    AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, add_pooling_layer=False)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  cached {name}: hidden_size={model.config.hidden_size}, {params:.2f} M params")
