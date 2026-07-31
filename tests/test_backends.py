"""MLX backend seam contract: dispatch, refusal, and import purity.

Offline and hardware-independent -- none of these need mlx installed or a
real checkpoint. They guard the properties the seam exists to protect: that
``backend="mlx"`` is honoured or refused (never silently ignored or
downgraded to Torch), that ``device=`` is restricted to what MLX can
actually honour when ``backend="mlx"`` (a Torch-only value like ``"cuda"``
is refused, never silently ignored -- mirroring every sibling OpenMIRLab
package's ``MLXBackend._select_device``), and that the default import path
stays mlx-free.

``family=`` (v1 vs v2 model architecture) and ``backend=`` (compute
framework: torch vs mlx) are deliberately separate axes -- see ``api.py``'s
module docstring -- so a test here never needs to touch ``family=`` at all.

Reads: bandit_infer.api, subprocess (import purity)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bandit_infer import api


def test_importing_the_package_does_not_pull_in_mlx() -> None:
    """Run in a fresh subprocess: an *earlier* test in the same process
    legitimately importing real mlx as a side effect of a real
    ``mlx_available()`` check is not the same claim as ``import bandit_infer``
    itself doing it, and process-level ``sys.modules`` state can leak
    between tests otherwise (the same isolation `mdxnet-infer` uses)."""
    result = subprocess.run(
        [sys.executable, "-c", "import sys, bandit_infer; assert not any('mlx' in name for name in sys.modules)"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_default_and_none_backend_resolve_to_torch() -> None:
    assert api.resolve_backend_name(None) == api.DEFAULT_BACKEND == "torch"
    assert api.resolve_backend_name("torch") == "torch"


def test_auto_backend_resolves_to_a_registered_backend() -> None:
    """``auto`` is the one place a fallback is what the caller asked for."""
    assert api.resolve_backend_name("auto") in api.BACKEND_NAMES


def test_unknown_backend_name_raises_value_error() -> None:
    with pytest.raises(ValueError):
        api.resolve_backend_name("onnx")


def test_unavailable_backend_raises_rather_than_substituting(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit request is honoured or fails loudly -- never downgraded.

    A silent substitution is only ever discovered by noticing the wrong
    hardware was busy.
    """
    monkeypatch.setattr(api, "mlx_available", lambda: False)
    with pytest.raises(api.BackendUnavailable):
        api.resolve_backend_name("mlx")


def test_auto_falls_back_to_torch_when_mlx_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "mlx_available", lambda: False)
    assert api.resolve_backend_name("auto") == "torch"


def test_auto_prefers_mlx_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "mlx_available", lambda: True)
    assert api.resolve_backend_name("auto") == "mlx"


def test_mlx_device_accepts_only_auto_or_mps() -> None:
    assert api._select_mlx_device("auto") == "mps"
    assert api._select_mlx_device("mps") == "mps"


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cpu"])
def test_mlx_refuses_a_torch_device_string(device: str) -> None:
    """The failure mode this axis split exists to prevent: a Torch-only
    device value paired with ``backend="mlx"`` must raise, never be
    silently ignored."""
    with pytest.raises(api.BackendUnavailable, match="cannot honour device"):
        api._select_mlx_device(device)


def test_device_cuda_with_backend_mlx_is_refused_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")
    monkeypatch.setattr(api, "resolve_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(api, "mlx_available", lambda: True)

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("load_mlx_runtime_backend must never be reached for an unsupported device")

    monkeypatch.setattr(api, "load_mlx_runtime_backend", _boom)

    session = api.BanditSession("v1-mus64-l1snr", backend="mlx", device="cuda")
    with pytest.raises(api.BackendUnavailable, match="cannot honour device"):
        session.load()
    assert session.status == "failed"


def test_backend_mlx_never_reaches_resolve_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Structural guarantee: ``backend="mlx"`` short-circuits before
    ``resolve_device`` (the Torch device resolver) is ever called. Proven by
    making ``resolve_device`` explode if it is ever reached.
    """
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")
    monkeypatch.setattr(api, "resolve_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(api, "mlx_available", lambda: True)

    def _boom(value: str) -> torch.device:
        raise AssertionError(f"resolve_device must never be called for backend='mlx' (got {value!r})")

    monkeypatch.setattr(api, "resolve_device", _boom)

    calls = {"loads": 0}

    def fake_mlx_backend(spec, checkpoint_path):
        calls["loads"] += 1

        class _Fake:
            def infer(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
                return {"speech": audio, "music": audio, "effects": audio}

        return _Fake()

    monkeypatch.setattr(api, "load_mlx_runtime_backend", fake_mlx_backend)

    session = api.BanditSession("v1-mus64-l1snr", backend="mlx")
    session.load()
    assert session.status == "ready"
    assert calls["loads"] == 1


def test_backend_mlx_dispatches_to_mlx_runtime_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")
    monkeypatch.setattr(api, "resolve_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(api, "mlx_available", lambda: True)

    seen = {}

    def fake_mlx_backend(spec, checkpoint_path):
        seen["spec"], seen["checkpoint_path"] = spec, checkpoint_path

        class _Fake:
            def infer(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
                return {"speech": audio}

        return _Fake()

    monkeypatch.setattr(api, "load_mlx_runtime_backend", fake_mlx_backend)

    with api.BanditSession("v2-multi", backend="mlx") as session:
        out = session.infer(np.zeros(8), sample_rate=48000)
    assert "speech" in out
    assert seen["checkpoint_path"] == checkpoint
    assert seen["spec"].key == "v2-multi"


def test_unknown_family_refuses_rather_than_mis_runs() -> None:
    """`load_mlx_runtime_backend` only recognises v1/v2 -- an unrecognised
    model family raises `RuntimeUnavailableError`, never a silent guess."""
    import dataclasses

    spec = api.get_spec("v1-mus64-l1snr")
    bogus = dataclasses.replace(spec, family="v3")
    with pytest.raises(api.RuntimeUnavailableError, match="unknown model family"):
        api.load_mlx_runtime_backend(bogus, Path("unused.ckpt"))
