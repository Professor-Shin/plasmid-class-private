"""Record the exact environment that produced the results, into results/environment.json.

Run it in the same interpreter that ran the experiments. It captures hardware, OS,
interpreter and package versions, the PyTorch threading backend, checksums of both
the dataset files and this baseline's own source, and the upstream commits the two
architectures were transcribed from.
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
PACKAGES = ["torch", "numpy", "pandas", "scikit-learn", "scipy", "PyYAML", "joblib",
            "threadpoolctl"]


def sha256(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def file_record(path: Path, checksum=True) -> dict:
    stat = path.stat()
    rec = {"bytes": stat.st_size,
           "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
    if checksum:
        rec["sha256"] = sha256(path)
    return rec


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
    info = {"logical_cpus": os.cpu_count(), "machine": platform.machine(),
            "processor": platform.processor()}
    cpu = powershell("$c=Get-CimInstance Win32_Processor;"
                     "\"$($c.Name)|$($c.NumberOfCores)|$($c.NumberOfLogicalProcessors)"
                     "|$($c.MaxClockSpeed)\"")
    if cpu and cpu.count("|") == 3:
        name, cores, logical, mhz = cpu.split("|")
        info.update({"cpu_model": name, "physical_cores": int(cores),
                     "logical_cores": int(logical), "max_clock_mhz": int(mhz)})
    ram = powershell("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
    if ram and ram.isdigit():
        info["ram_bytes"] = int(ram)
    disk = powershell("(Get-PhysicalDisk | Select-Object -First 1).FriendlyName")
    if disk:
        info["storage"] = disk
    return info


def torch_info() -> dict:
    import torch
    return {"version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "default_num_threads": torch.get_num_threads(),
            "parallel_info": torch.__config__.parallel_info().strip().splitlines()}


def directory_usage(path: Path) -> dict:
    files = [p for p in path.rglob("*") if p.is_file()]
    return {"n_files": len(files), "total_bytes": sum(p.stat().st_size for p in files)}


def main():
    cfg_path = HERE / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    dataset_dir = (HERE / cfg["dataset_dir"]).resolve()
    results = (HERE / cfg["results_dir"]).resolve()
    cache = (HERE / cfg["cache_dir"]).resolve()

    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware(),
        "os": {"system": platform.system(), "release": platform.release(),
               "version": platform.version(), "edition": powershell(
                   "(Get-CimInstance Win32_OperatingSystem).Caption")},
        "python": {"version": platform.python_version(), "executable": sys.executable,
                   "implementation": platform.python_implementation()},
        "packages": {p: md.version(p) for p in PACKAGES},
        "torch": torch_info(),
        "thread_settings": {"plasflow": cfg["plasflow"].get("torch_threads"),
                            "zou": cfg["zou"].get("torch_threads"),
                            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
        "upstream": cfg["upstream"],
        "config": {"path": cfg_path.name, "sha256": sha256(cfg_path), "values": cfg},
        "dataset": {"directory": str(dataset_dir),
                    "files": {p.name: file_record(p)
                              for p in sorted(dataset_dir.glob("*")) if p.is_file()}},
        "source": {p.relative_to(HERE).as_posix(): file_record(p)
                   for p in sorted(HERE.rglob("*.py")) if "__pycache__" not in p.parts},
        "reference_files": {p.name: file_record(p)
                            for p in sorted((HERE / "reference").glob("*"))},
        "cache": directory_usage(cache) if cache.exists() else None,
        "random_seed": cfg["random_seed"],
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "environment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    hw = manifest["hardware"]
    print(f"CPU        : {hw.get('cpu_model')} "
          f"({hw.get('physical_cores')}C/{hw.get('logical_cores')}T)")
    print(f"RAM        : {hw.get('ram_bytes', 0) / 2**30:.2f} GiB")
    print(f"OS         : {manifest['os']['edition']} build {platform.version()}")
    print(f"Python     : {manifest['python']['version']} ({sys.executable})")
    print(f"Packages   : " + ", ".join(f"{k} {v}" for k, v in manifest["packages"].items()))
    print(f"Torch      : {manifest['torch']['version']}, "
          f"cuda={manifest['torch']['cuda_available']}, "
          f"default threads={manifest['torch']['default_num_threads']}")
    if manifest["cache"]:
        print(f"Cache      : {manifest['cache']['n_files']} files, "
              f"{manifest['cache']['total_bytes'] / 2**30:.1f} GiB")
    print(f"Dataset    : {len(manifest['dataset']['files'])} files checksummed")
    print(f"written -> {results / 'environment.json'}")


if __name__ == "__main__":
    main()
