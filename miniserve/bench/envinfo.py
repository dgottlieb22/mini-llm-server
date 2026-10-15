"""Record the environment a benchmark ran on. No GPU is a valid environment;
the record says so instead of omitting the field."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Dict, Optional


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _git_commit(cwd: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment() -> Dict[str, object]:
    import torch

    gpu: Dict[str, object]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {"present": True, "name": props.name, "memory_bytes": props.total_memory}
    else:
        gpu = {"present": False, "note": "CPU-only host; GPU utilization and GPU memory not measured"}
    try:
        import transformers

        tf_version = transformers.__version__
    except ImportError:
        tf_version = None
    return {
        "hostname_class": "dev desktop" if "dev-dsk" in platform.node() else platform.node(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "transformers": tf_version,
        "gpu": gpu,
        "git_commit": _git_commit(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    }
