"""Record the exact environment that produced the results, into results/environment.json.

Same purpose as ../baseline/capture_environment.py, plus the ESM-2 checkpoint that was
used and checksums of the prototype scripts this implementation was transcribed from.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PACKAGES = ["torch", "transformers", "tokenizers", "safetensors", "biopython", "numpy",
            "pandas", "scikit-learn", "scipy", "PyYAML"]


def sha256(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    stat = path.stat()
    return {"bytes": stat.st_size, "sha256": sha256(path),
            "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}


def powershell(expression: str) -> str | None:
    if platform.system() != "Windows":
        return None
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", expression],
                             capture_output=True, text=True, timeout=60)
        return out.stdout.strip() or None
    except Exception:
        return None


def hardware() -> dict:
    info = {"logical_cpus": os.cpu_count(), "machine": platform.machine()}
    cpu = powershell("$c=Get-CimInstance Win32_Processor;"
                     "\"$($c.Name)|$($c.NumberOfCores)|$($c.NumberOfLogicalProcessors)\"")
    if cpu and cpu.count("|") == 2:
        name, cores, logical = cpu.split("|")
        info.update({"cpu_model": name, "physical_cores": int(cores),
                     "logical_cores": int(logical)})
    ram = powershell("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
    if ram and ram.isdigit():
        info["ram_bytes"] = int(ram)
    return info


def esm_checkpoint(model_name: str) -> dict:
    """Locate the cached snapshot so the exact weights can be identified."""
    from huggingface_hub.constants import HF_HUB_CACHE
    folder = Path(HF_HUB_CACHE) / ("models--" + model_name.replace("/", "--"))
    record = {"model_name": model_name, "cache": str(folder), "found": folder.exists()}
    if folder.exists():
        snapshots = sorted((folder / "snapshots").glob("*"))
        record["revision"] = [s.name for s in snapshots]
        record["files"] = {p.name: p.stat().st_size
                           for s in snapshots for p in s.iterdir() if p.is_file()}
    import torch
    from esm2 import embed
    _, model = embed.load_esm(model_name)
    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    record["n_parameters"] = int(flat.numel())
    record["hidden_size"] = int(model.config.hidden_size)
    record["n_layers"] = int(model.config.num_hidden_layers)
    # Fingerprint of the loaded weights, so a different checkpoint cannot pass unnoticed.
    record["weight_sha256"] = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
    return record


def main():
    sys.path.insert(0, str(HERE))
    cfg_path = HERE / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    dataset_dir = (HERE / cfg["dataset_dir"]).resolve()
    results = (HERE / cfg["results_dir"]).resolve()
    cache = (HERE / cfg["cache_dir"]).resolve()
    upstream_dir = (HERE / cfg["upstream"]["directory"]).resolve()

    import torch
    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware(),
        "os": {"system": platform.system(), "version": platform.version(),
               "edition": powershell("(Get-CimInstance Win32_OperatingSystem).Caption")},
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": {p: md.version(p) for p in PACKAGES},
        "torch": {"version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                  "default_num_threads": torch.get_num_threads(),
                  "parallel_info": torch.__config__.parallel_info().strip().splitlines()},
        "thread_settings": {"esm_extraction": cfg["esm"]["torch_threads"],
                            "training": cfg["train"]["torch_threads"]},
        "esm_checkpoint": esm_checkpoint(cfg["esm"]["model_name"]),
        "prototype": {name: file_record(upstream_dir / name)
                      for name in cfg["upstream"]["files"] + [cfg["upstream"]["previous_result"]]
                      if (upstream_dir / name).exists()},
        "config": {"path": cfg_path.name, "sha256": sha256(cfg_path), "values": cfg},
        "dataset": {"directory": str(dataset_dir),
                    "files": {p.name: file_record(p)
                              for p in sorted(dataset_dir.glob("*")) if p.is_file()}},
        "source": {p.relative_to(HERE).as_posix(): file_record(p)
                   for p in sorted(HERE.rglob("*.py")) if "__pycache__" not in p.parts},
        "cache": {"files": sorted(p.name for p in cache.glob("*"))} if cache.exists() else None,
        "random_seed": cfg["random_seed"],
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "environment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    hw = manifest["hardware"]
    print(f"CPU      : {hw.get('cpu_model')} "
          f"({hw.get('physical_cores')}C/{hw.get('logical_cores')}T), "
          f"RAM {hw.get('ram_bytes', 0) / 2**30:.2f} GiB")
    print(f"Python   : {manifest['python']['version']}")
    print("Packages : " + ", ".join(f"{k} {v}" for k, v in manifest["packages"].items()))
    ck = manifest["esm_checkpoint"]
    print(f"ESM-2    : {ck['model_name']} | {ck['n_parameters']:,} params | "
          f"{ck['n_layers']} layers | hidden {ck['hidden_size']}")
    print(f"           weight sha256 {ck['weight_sha256'][:32]}...")
    print(f"written -> {results / 'environment.json'}")


if __name__ == "__main__":
    main()
