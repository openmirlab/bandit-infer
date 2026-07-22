"""Catalog and resolver contracts, all offline and without model weights.

These tests prove the catalog stays complete and that integrity checks do not
silently accept Zenodo's MD5 in place of a SHA-256. Reads: packaged manifest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bandit_infer.checkpoints import CheckpointConfigError, ChecksumError, cache_info, get_spec, load_manifest, resolve_checkpoint


def test_complete_registry_and_native_rates() -> None:
    default, specs = load_manifest()
    assert default in specs and len(specs) == 28
    assert sum(spec.backend == "v1" for spec in specs.values()) == 21
    assert sum(spec.backend == "v2" for spec in specs.values()) == 7
    assert {spec.sample_rate for spec in specs.values() if spec.backend == "v1"} == {44100}
    assert {spec.sample_rate for spec in specs.values() if spec.backend == "v2"} == {48000}
    assert all(spec.md5 for spec in specs.values())
    assert specs["v1-mus64-l1snr"].sha256 == "16c52a45c891fe44bccc27d2b8403398ce8dafa1df0300a0883ec316d10c21d4"
    assert specs["v2-multi"].sha256 == "abcfccf65446752a057f4a302c941479a54b7560ebf8d7bca039d2ea98e64cfc"
    assert all(not spec.sha256 for key, spec in specs.items() if key not in {"v1-mus64-l1snr", "v2-multi"})
    assert all(spec.stems == ("speech", "music", "effects") or len(spec.stems) == 1 for spec in specs.values())


def test_malformed_config_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("schema_version = 2\n", encoding="utf-8")
    with pytest.raises(CheckpointConfigError, match="schema_version"):
        load_manifest(path)


def test_cache_info_uses_resolver_candidate_without_materializing(tmp_path: Path) -> None:
    info = cache_info(get_spec(), cache_dir=tmp_path)
    assert info["exists"] is False
    assert not info["path"].exists()


def test_official_unverified_download_is_rejected() -> None:
    with pytest.raises(CheckpointConfigError, match="sha256"):
        resolve_checkpoint(get_spec("v1-mus48-l1snr"))


def test_direct_path_requires_matching_sha256(tmp_path: Path) -> None:
    path = tmp_path / "custom.ckpt"
    path.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()
    assert resolve_checkpoint(get_spec(), checkpoint_path=path, checkpoint_sha256=digest) == path
    with pytest.raises(ChecksumError):
        resolve_checkpoint(get_spec(), checkpoint_path=path, checkpoint_sha256="0" * 64)
