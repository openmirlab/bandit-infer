"""Thin one-shot facade and resident lifecycle for Bandit source separation.

The session owns its loaded backend and only accepts inference after ``load``;
one-shot ``separate`` creates a fresh session. Backends remain separate at the
dispatch boundary so v1/v2 model code can never share a state-dict contract.
Reads: checkpoints, device, and backend-specific namespaces lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .checkpoints import CheckpointSpec, cache_info, get_spec, resolve_checkpoint
from .device import resolve_device


class RuntimeUnavailableError(RuntimeError):
    """The selected official model is catalogued but not yet graph-verified."""


class _Backend(Protocol):
    def infer(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]: ...


def load_runtime_backend(spec: CheckpointSpec, checkpoint: Path, device: torch.device) -> _Backend:
    """Select an independent backend; never use a shared model architecture."""
    if spec.backend == "v2":
        from ._v2.runtime import V2Runtime
        return V2Runtime(checkpoint, device)
    if spec.backend == "v1":
        from ._v1.runtime import V1Runtime
        return V1Runtime(spec, checkpoint, device)
    raise RuntimeUnavailableError(f"unknown backend {spec.backend}")


class BanditSession:
    """A reloadable, non-concurrent resident Bandit model session."""

    def __init__(self, model: str | None = None, *, backend: str | None = None, device: str = "auto", weights_dir: Path | None = None, checkpoint_path: Path | None = None, checkpoint_sha256: str | None = None) -> None:
        self.spec = get_spec(model)
        if backend is not None and backend != self.spec.backend:
            raise ValueError(f"model {self.spec.key} belongs to backend {self.spec.backend}")
        self._device_value, self._weights_dir = device, weights_dir
        self._checkpoint_path, self._checkpoint_sha256 = checkpoint_path, checkpoint_sha256
        self._runtime: _Backend | None = None
        self._status = "new"

    @property
    def status(self) -> str:
        return self._status

    def cache_info(self) -> dict[str, object]:
        return cache_info(self.spec, cache_dir=self._weights_dir)

    def load(self) -> "BanditSession":
        if self._status == "closed":
            raise RuntimeError("a closed BanditSession cannot be loaded")
        if self._status == "ready":
            return self
        try:
            device = resolve_device(self._device_value)
            checkpoint = resolve_checkpoint(self.spec, cache_dir=self._weights_dir, checkpoint_path=self._checkpoint_path, checkpoint_sha256=self._checkpoint_sha256)
            self._runtime = load_runtime_backend(self.spec, checkpoint, device)
            self._status = "ready"
        except Exception:
            self._status = "failed"
            raise
        return self

    def infer(self, audio: np.ndarray, *, sample_rate: int) -> dict[str, np.ndarray]:
        if self._status != "ready" or self._runtime is None:
            raise RuntimeError("BanditSession.infer requires a ready session; call load() first")
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        if waveform.ndim != 2:
            raise ValueError("audio must have shape (samples,) or (channels, samples)")
        return self._runtime.infer(waveform, sample_rate)

    def release(self) -> "BanditSession":
        if self._status not in {"closed", "new"}:
            self._runtime = None
            self._status = "released"
        return self

    def close(self) -> None:
        self._runtime = None
        self._status = "closed"

    def __enter__(self) -> "BanditSession":
        return self.load()

    def __exit__(self, *_: object) -> None:
        self.close()


def separate(audio: np.ndarray, *, sample_rate: int, model: str | None = None, backend: str | None = None, device: str = "auto", weights_dir: Path | None = None, checkpoint_path: Path | None = None, checkpoint_sha256: str | None = None) -> dict[str, np.ndarray]:
    """Separate one input with a fresh session; it intentionally does not cache."""
    with BanditSession(model, backend=backend, device=device, weights_dir=weights_dir, checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha256) as session:
        return session.infer(audio, sample_rate=sample_rate)
