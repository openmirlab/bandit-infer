"""Explicit torch device validation used by every loading path.

``auto`` is a real sentinel, not a string passed to torch: it resolves to
cuda-else-cpu and deliberately never promotes to MPS, so a Mac caller's outputs
do not move under them just because a newer torch build can see the GPU. MPS is
opt-in only, mirroring ``bs-roformer-infer``'s ``_select_device`` contract.
Explicit unavailable CUDA or MPS targets fail loudly so callers never receive a
silent CPU downgrade.

``resolve_device`` resolves Torch devices only -- it has no MLX case at all.
Compute *framework* (Torch vs MLX) is a separate axis, ``api.py``'s
``backend=`` parameter, resolved by ``api.py::resolve_backend_name``; device
resolution for the MLX path lives in ``api.py::_select_mlx_device`` (MLX owns
its own execution target and never routes through this module's torch
device logic). ``mlx_available()`` lives here as a plain capability probe --
"is MLX usable at all" -- shared by both axes' resolvers.

Reads: torch's device availability only. ``mlx_available()`` imports
``mlx.core`` lazily inside the function body only -- never at module load --
so importing this module (or ``bandit_infer`` at all) never imports mlx.
"""

from __future__ import annotations

import torch


def mps_available() -> bool:
    """True when this torch build exposes a usable Apple Silicon MPS backend.

    Guarded with ``getattr`` because older torch builds have no ``backends.mps``
    attribute at all, and this is the single owner of that check so no caller
    inlines ``torch.backends.mps.is_available()`` directly.
    """
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def mlx_available() -> bool:
    """True when the optional ``mlx`` extra is importable on this machine.

    The single owner of "is MLX usable here" so no call site inlines its own
    ``import mlx.core`` probe. Imports lazily inside the function body: this
    is the only place in the package that ever imports mlx, and only when
    actually asked whether it is available.
    """
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_device(value: str = "auto") -> torch.device:
    if value == "auto":
        # Legacy meaning, deliberately unchanged: cuda-else-cpu. MPS is opt-in,
        # never auto-promoted, so it cannot silently move a Mac caller's outputs.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "mps":
        if not mps_available():
            raise RuntimeError("MPS was explicitly requested but is unavailable "
                                "(needs an Apple Silicon Mac and an arm64 torch build)")
        return torch.device("mps")
    if value == "cuda" or value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        device = torch.device(value)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device.index} is unavailable")
        return device
    raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps")
