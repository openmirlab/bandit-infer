"""Package-owned checkpoint manifest, integrity gate, and shared cache resolver.

The resolver is the only code that decides where a checkpoint lives; both model
loading and ``cache_info`` call it. It refuses official records without a
verified SHA-256 rather than treating Zenodo's MD5 as equivalent protection.
Reads: config/checkpoints.toml, local cache paths, and optional caller URLs.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tomllib
import urllib.request
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


class CheckpointConfigError(ValueError):
    """The package or a supplied registry has no safe checkpoint contract."""


class ChecksumError(ValueError):
    """A checkpoint's bytes differ from its declared SHA-256."""


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    backend: str
    variant: str
    filename: str
    size: int
    md5: str
    sha256: str
    url: str
    source_config: str
    stems: tuple[str, ...]
    sample_rate: int
    license: str
    source_revision: str
    updated: str


def _manifest_path() -> Path:
    return Path(str(files("bandit_infer").joinpath("config/checkpoints.toml")))


def load_manifest(path: Path | None = None) -> tuple[str, dict[str, CheckpointSpec]]:
    """Parse the packaged catalog and reject malformed entries before I/O."""
    try:
        raw = tomllib.loads((path or _manifest_path()).read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise CheckpointConfigError("unsupported checkpoints.toml schema_version")
        defaults = raw["checkpoint_defaults"]
        result: dict[str, CheckpointSpec] = {}
        for entry in raw["checkpoints"]:
            backend = entry["backend"]
            if backend not in {"v1", "v2"} or entry["key"] in result:
                raise CheckpointConfigError("checkpoint key/backend is invalid or duplicated")
            result[entry["key"]] = CheckpointSpec(
                key=entry["key"], backend=backend, variant=entry["variant"],
                filename=entry["filename"], size=int(entry["size"]), md5=entry["md5"],
                sha256=entry["sha256"], url=entry["url"], source_config=entry["source_config"],
                stems=tuple(entry["stems"]),
                sample_rate=int(defaults[f"{backend}_native_sample_rate"]),
                license=defaults[f"{backend}_license"],
                source_revision=defaults[f"{backend}_source_revision"], updated=defaults["updated"],
            )
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise CheckpointConfigError(f"malformed checkpoints.toml: {error}") from error
    default = raw.get("default")
    if default not in result:
        raise CheckpointConfigError("default model key is absent")
    return default, result


def get_spec(key: str | None = None) -> CheckpointSpec:
    default, specs = load_manifest()
    try:
        return specs[key or default]
    except KeyError as error:
        raise CheckpointConfigError(f"unknown Bandit model key: {key}") from error


def default_cache_dir() -> Path:
    return Path(os.environ.get("BANDIT_INFER_WEIGHTS", Path.home() / ".cache/bandit-infer"))


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate(spec: CheckpointSpec, cache_dir: Path | None) -> Path:
    return (cache_dir or default_cache_dir()) / spec.filename


def cache_info(spec: CheckpointSpec | None = None, *, cache_dir: Path | None = None) -> dict[str, Any]:
    """Return read-only cache status using the exact resolver candidate path."""
    spec = spec or get_spec()
    path = _candidate(spec, cache_dir)
    return {"key": spec.key, "path": path, "exists": path.is_file(), "verified": bool(spec.sha256) and path.is_file() and _digest(path) == spec.sha256}


def _verify(path: Path, expected: str) -> None:
    if not expected:
        raise CheckpointConfigError("official artifact has no independently verified sha256 yet")
    actual = _digest(path)
    if actual != expected:
        raise ChecksumError(f"checksum mismatch for {path.name}: expected {expected}, got {actual}")


def resolve_checkpoint(
    spec: CheckpointSpec,
    *,
    cache_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_url: str | None = None,
    checkpoint_sha256: str | None = None,
) -> Path:
    """Resolve manual/direct/cache paths, downloading atomically only with SHA-256."""
    expected = checkpoint_sha256 or spec.sha256
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        _verify(path, expected)
        return path
    target = _candidate(spec, cache_dir)
    if target.is_file():
        _verify(target, expected)
        return target
    url = checkpoint_url or spec.url
    if not checkpoint_sha256 and not spec.sha256:
        raise CheckpointConfigError("automatic official download is disabled until its sha256 is verified")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
        _verify(temporary, expected)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
