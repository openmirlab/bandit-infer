"""Explicit torch device validation used by every loading path.

``auto`` is a real sentinel, not a string passed to torch. Explicit unavailable
CUDA targets fail loudly so callers never receive a silent CPU downgrade.
Reads: torch's device availability only.
"""

from __future__ import annotations

import torch


def resolve_device(value: str = "auto") -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda" or value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        device = torch.device(value)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device.index} is unavailable")
        return device
    raise ValueError("device must be auto, cpu, cuda, or cuda:N")
