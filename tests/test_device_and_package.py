"""Public import, package-data, and explicit-device validation tests.

Device tests use monkeypatches so an unavailable accelerator never becomes an
implicit CPU fallback. Reads: installed package resources and torch state.
"""

from __future__ import annotations

from importlib.resources import files

import pytest
import torch

import bandit_infer
from bandit_infer.device import resolve_device


def test_public_symbols_and_packaged_manifest() -> None:
    assert bandit_infer.__version__
    assert bandit_infer.BanditSession
    assert files("bandit_infer").joinpath("config/checkpoints.toml").is_file()


def test_device_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError, match="explicitly requested"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="device must"):
        resolve_device("mps")
