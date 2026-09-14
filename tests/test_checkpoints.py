"""Catalog and resolver contracts, all offline and without model weights.

These tests prove the catalog stays complete and that integrity checks do not
silently accept Zenodo's MD5 in place of a SHA-256. Reads: packaged manifest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import string
from pathlib import Path

import pytest

from bandit_infer.checkpoints import CheckpointConfigError, ChecksumError, cache_info, get_spec, load_manifest, resolve_checkpoint

# Every catalog entry carries a sha256 computed from its official download
# (cross-checked against Zenodo's size and md5); see checkpoints.toml's header.
_HEX = set(string.hexdigits.lower())


def test_complete_registry_and_native_rates() -> None:
    default, specs = load_manifest()
    assert default in specs and len(specs) == 28
    assert sum(spec.family == "v1" for spec in specs.values()) == 21
    assert sum(spec.family == "v2" for spec in specs.values()) == 7
    assert {spec.sample_rate for spec in specs.values() if spec.family == "v1"} == {44100}
    assert {spec.sample_rate for spec in specs.values() if spec.family == "v2"} == {48000}
    assert all(spec.md5 for spec in specs.values())
    assert specs["v1-mus64-l1snr"].sha256 == "16c52a45c891fe44bccc27d2b8403398ce8dafa1df0300a0883ec316d10c21d4"
    assert specs["v2-multi"].sha256 == "abcfccf65446752a057f4a302c941479a54b7560ebf8d7bca039d2ea98e64cfc"
    for key, spec in specs.items():
        assert len(spec.sha256) == 64 and set(spec.sha256) <= _HEX, f"{key} missing a verified sha256"
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


def test_official_unverified_download_is_rejected(tmp_path: Path) -> None:
    """A blank-sha256 entry must still block auto-download, even once every
    real catalog entry has a verified sha256 filled in. Synthesize one from a
    real spec (rather than relying on a specific catalog key staying blank
    forever) so this guard keeps being exercised."""
    unverified = dataclasses.replace(get_spec("v1-mus48-l1snr"), sha256="")
    with pytest.raises(CheckpointConfigError, match="sha256"):
        resolve_checkpoint(unverified, cache_dir=tmp_path)


def test_direct_path_requires_matching_sha256(tmp_path: Path) -> None:
    path = tmp_path / "custom.ckpt"
    path.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()
    assert resolve_checkpoint(get_spec(), checkpoint_path=path, checkpoint_sha256=digest) == path
    with pytest.raises(ChecksumError):
        resolve_checkpoint(get_spec(), checkpoint_path=path, checkpoint_sha256="0" * 64)
