"""Lifecycle and dispatch tests using a fake backend, never a fake weight claim.

The fake isolates session ownership: it proves load-once/release behavior while
real numerical parity remains a separate official-checkpoint gate.
Reads: bandit_infer.api.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bandit_infer import BanditSession
from bandit_infer import api


class FakeBackend:
    def infer(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        return {"speech": audio, "music": audio, "effects": audio}


def install_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, int]:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")
    calls = {"loads": 0}
    monkeypatch.setattr(api, "resolve_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(api, "resolve_device", lambda value: torch.device("cpu"))
    def load(*args: object, **kwargs: object) -> FakeBackend:
        calls["loads"] += 1
        return FakeBackend()
    monkeypatch.setattr(api, "load_runtime_backend", load)
    return calls


def test_ready_only_no_reload_release_close_and_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = install_runtime(monkeypatch, tmp_path)
    session = BanditSession()
    with pytest.raises(RuntimeError, match="requires a ready"):
        session.infer(np.zeros(8), sample_rate=44100)
    assert session.load().load() is session
    session.infer(np.zeros(8), sample_rate=44100)
    session.infer(np.zeros(8), sample_rate=44100)
    assert calls["loads"] == 1
    session.release()
    assert session.status == "released"
    session.load()
    assert calls["loads"] == 2
    session.close(); session.close()
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()


def test_backend_selection_is_not_an_alias() -> None:
    assert BanditSession("v1-mus64-l1snr").spec.backend == "v1"
    assert BanditSession("v2-multi").spec.backend == "v2"
    with pytest.raises(ValueError, match="belongs to backend"):
        BanditSession("v1-mus64-l1snr", backend="v2")
