"""Thin one-shot facade and resident lifecycle for Bandit source separation.

The session owns its loaded runtime and only accepts inference after ``load``;
one-shot ``separate`` creates a fresh session. Model families remain separate
at the dispatch boundary so v1/v2 model code can never share a state-dict
contract.

Two orthogonal axes, matching the org-wide convention six sibling packages
already use (``bs-roformer-infer``'s ``brain/decisions.md`` D1):

- ``family=`` selects the *model family* (``"v1"`` or ``"v2"``) -- this
  package's own axis, since Bandit v1 and v2 are genuinely different graphs.
- ``backend=`` selects the *compute framework* (``None``/``"torch"``
  (default), ``"mlx"``, or ``"auto"``) and ``device=`` selects *where* that
  framework runs. ``backend="mlx"`` accepts only ``"auto"`` or ``"mps"`` for
  ``device`` and refuses anything else (a Torch-only value like ``"cuda"``
  is refused, never silently ignored) -- mirroring
  ``mlx_backend.py::MLXBackend._select_device`` in every sibling that has
  this axis.

Article 7 (amendment 2026-07-17) governs why ``backend=`` was free to take
over the compute-framework meaning here: this package is not yet on PyPI, so
its pre-release surface owes no compatibility, and an earlier version of
this module spent ``backend=`` on the model-family axis instead (renamed to
``family=`` specifically so ``backend=`` could mean what it means everywhere
else in the org) -- see ``CHANGELOG.md``'s MLX entry.

Reads: checkpoints, device, and backend-specific namespaces lazily (including
``mlx``, which is imported only when ``backend="mlx"`` is actually resolved).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .checkpoints import CheckpointSpec, cache_info, get_spec, resolve_checkpoint
from .device import mlx_available, resolve_device

#: Every selectable compute-framework name, in the order `auto` prefers them.
BACKEND_NAMES = ("mlx", "torch")
DEFAULT_BACKEND = "torch"


class RuntimeUnavailableError(RuntimeError):
    """The selected official model is catalogued but not yet graph-verified."""


class BackendUnavailable(RuntimeError):
    """Raised when a compute framework is requested by name but cannot run
    here, or is asked to run with a device combination it cannot honour.

    Always raised, never swallowed into a fallback: silently substituting a
    different backend -- or silently ignoring an incompatible ``device=`` --
    discards what the caller explicitly asked for, and would only be
    discovered by noticing the wrong hardware was busy.
    """


class _Backend(Protocol):
    def infer(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]: ...


def resolve_backend_name(requested: str | None) -> str:
    """Resolve a requested compute-framework name, honouring it exactly or
    raising -- the org-wide contract (see this module's docstring).

    ``None`` and ``"torch"`` both mean the shipped Torch path -- the default
    never moves on its own. ``"auto"`` prefers MLX when it is genuinely
    importable, falling back to Torch otherwise; that is the one place a
    fallback is what the caller asked for. An *explicit* ``"mlx"`` request
    still raises when unavailable -- honoured or refused, never downgraded.
    """
    if requested is None or requested == DEFAULT_BACKEND:
        return DEFAULT_BACKEND
    if requested == "auto":
        return "mlx" if mlx_available() else DEFAULT_BACKEND
    if requested not in BACKEND_NAMES:
        raise ValueError(f"backend must be None, 'auto', or one of {BACKEND_NAMES}; got {requested!r}")
    if requested == "mlx" and not mlx_available():
        raise BackendUnavailable(
            "backend='mlx' is unavailable on this machine; install the optional "
            "extra: pip install 'bandit-infer[mlx]' (Apple Silicon only)"
        )
    return requested


def _select_mlx_device(device: str) -> str:
    """MLX owns its own execution target; a Torch device string is refused."""
    if device in ("auto", "mps"):
        return "mps"
    raise BackendUnavailable(
        f"backend='mlx' cannot honour device={device!r}; it executes on Apple "
        f"Silicon and accepts 'auto' or 'mps'. Use backend='torch' (the default) "
        f"to select a Torch device."
    )


def load_runtime_backend(spec: CheckpointSpec, checkpoint: Path, device: torch.device) -> _Backend:
    """Select an independent Torch runtime; never use a shared model architecture."""
    if spec.family == "v2":
        from ._v2.runtime import V2Runtime
        return V2Runtime(checkpoint, device)
    if spec.family == "v1":
        from ._v1.runtime import V1Runtime
        return V1Runtime(spec, checkpoint, device)
    raise RuntimeUnavailableError(f"unknown model family {spec.family}")


def load_mlx_runtime_backend(spec: CheckpointSpec, checkpoint: Path) -> _Backend:
    """Select an independent MLX runtime for ``backend="mlx"``.

    Raises rather than silently falling back to Torch when neither v1 nor v2
    (the only two model families this package has) can service the request --
    a request for ``backend="mlx"`` is honoured or refused, never quietly
    downgraded to Torch. Availability of the ``[mlx]`` extra itself is
    ``resolve_backend_name``'s job, checked before this is ever called.
    """
    import os
    # Mixed precision is an accuracy-affecting optimisation and must be opt-in,
    # never a silent default -- set explicitly, matching every sibling
    # OpenMIRLab package's MLX backend.
    os.environ.setdefault("MLX_ENABLE_AMP", "0")
    if spec.family == "v1":
        from .mlx.v1.runtime import V1RuntimeMLX
        return V1RuntimeMLX(spec, checkpoint)
    if spec.family == "v2":
        from .mlx.v2.runtime import V2RuntimeMLX
        return V2RuntimeMLX(checkpoint)
    raise RuntimeUnavailableError(f"unknown model family {spec.family}")


class BanditSession:
    """A reloadable, non-concurrent resident Bandit model session."""

    def __init__(self, model: str | None = None, *, family: str | None = None, backend: str | None = None, device: str = "auto", weights_dir: Path | None = None, checkpoint_path: Path | None = None, checkpoint_sha256: str | None = None) -> None:
        self.spec = get_spec(model)
        if family is not None and family != self.spec.family:
            raise ValueError(f"model {self.spec.key} belongs to family {self.spec.family}")
        self._backend_value = backend
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
            checkpoint = resolve_checkpoint(self.spec, cache_dir=self._weights_dir, checkpoint_path=self._checkpoint_path, checkpoint_sha256=self._checkpoint_sha256)
            backend_name = resolve_backend_name(self._backend_value)
            if backend_name == "mlx":
                _select_mlx_device(self._device_value)
                self._runtime = load_mlx_runtime_backend(self.spec, checkpoint)
            else:
                device = resolve_device(self._device_value)
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


def separate(audio: np.ndarray, *, sample_rate: int, model: str | None = None, family: str | None = None, backend: str | None = None, device: str = "auto", weights_dir: Path | None = None, checkpoint_path: Path | None = None, checkpoint_sha256: str | None = None) -> dict[str, np.ndarray]:
    """Separate one input with a fresh session; it intentionally does not cache."""
    with BanditSession(model, family=family, backend=backend, device=device, weights_dir=weights_dir, checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha256) as session:
        return session.infer(audio, sample_rate=sample_rate)
