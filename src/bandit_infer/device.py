"""Explicit torch device validation used by every loading path.

``auto`` is a real sentinel, not a string passed to torch: it resolves to
cuda-else-cpu and deliberately never promotes to MPS, so a Mac caller's outputs
do not move under them just because a newer torch build can see the GPU.
Explicit CUDA targets fail loudly so callers never receive a silent CPU
downgrade.

``resolve_device`` resolves Torch CPU/CUDA devices only. The pre-release Apple
MPS path was removed before publication, so an explicit ``mps`` request always
raises ``ValueError`` without probing local hardware.

Reads: torch's CUDA availability only.
"""

from __future__ import annotations

import torch


def resolve_device(value: str = "auto") -> torch.device:
    if value == "auto":
        # Legacy meaning, deliberately unchanged: cuda-else-cpu.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "mps":
        raise ValueError("device='mps' is not supported; use auto, cpu, cuda, or cuda:N")
    if value == "cuda" or value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        device = torch.device(value)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device.index} is unavailable")
        return device
    raise ValueError("device must be auto, cpu, cuda, or cuda:N")
