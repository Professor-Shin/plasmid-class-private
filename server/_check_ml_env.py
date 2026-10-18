"""Report the versions 00_setup_envs.sh just installed, and explain the one
result that always looks like a failure but is not."""

import numpy
import pandas
import sklearn
import torch
import transformers

print(f"  torch {torch.__version__} | cuda build {torch.version.cuda} | "
      f"numpy {numpy.__version__} | pandas {pandas.__version__} | "
      f"sklearn {sklearn.__version__} | transformers {transformers.__version__}")
print(f"  torch.cuda.is_available() -> {torch.cuda.is_available()}")
print("  (False on a login node is CORRECT - a login node has no GPU. The real")
print("   check runs inside the GPU allocation in 20_esm_embed.sh.)")
