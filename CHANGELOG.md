# Changelog

## Unreleased

### Verified SHA-256 for every catalog checkpoint (2026-09-14)

- Filled in `sha256` for the 26 `checkpoints.toml` entries that had it blank,
  so all 28 official checkpoints now download and verify automatically. Each
  value was computed from a fresh download of the official Zenodo file, and
  written only after that file matched the catalog's recorded `size` and
  Zenodo `md5`. No digest was copied from elsewhere.
- This is an integrity claim, not a support claim: only `v1-mus64-l1snr` and
  `v2-multi` have load and upstream parity evidence. The other 26 are not
  load-tested.
- `tests/test_checkpoints.py` now requires a 64-character lowercase hex
  SHA-256 on every entry. The blank-SHA-256 download guard is now tested with
  a synthesized spec instead of a real catalog key, so it stays covered.

### Removed pre-release Apple runtime surface

- Removed the unpublished Apple MLX backend and optional `[mlx]` extra,
  including the from-scratch `src/bandit_infer/mlx/` implementation,
  backend dispatch resolver, MLX parity/model tests, and framework selection
  through `backend=`. The public API is now single-runtime Torch: use
  `family=` for the v1/v2 model-family selector and `device=` only for Torch
  CPU/CUDA placement.
- Removed Apple Silicon MPS support before release. `device.py::resolve_device`
  accepts `auto`, `cpu`, `cuda`, and `cuda:N`; an explicit `mps` request now
  raises `ValueError` without probing hardware availability. `auto` remains
  cuda-else-cpu.
- Preserved Torch runtime behavior by hashing the existing upstream fixtures
  and every `_v1`/`_v2` runtime file before removal, then verifying the same
  hashes after removal. No tolerance was relaxed.
- Untracked `docs/implementation-plan.md` and ignored local `docs/`; source
  distributions explicitly exclude `/docs`.
- Historical context: the removed MLX work had renamed the pre-release
  model-family argument from `backend=` to `family=` so `backend=` could mean
  compute framework. With the compute-framework axis gone, `family=` remains
  as the only public model-family selector and no deprecated alias is kept.

### Housekeeping

- Deleted dead code (`_v2/film.py`, a commented-out bark/erb filterbank
  block in `_v2/utils.py`), corrected stale headers (`_v2/__init__.py`'s
  claim that v2 has no verified runtime; `mlx/__init__.py`'s pre-rename
  `device="mlx"` reference), and added module docstrings to the nine
  upstream-ported files that had none, including the package's four
  largest. No behaviour changed.

## 0.1.0 - 2026-07-22

- Added the initial inference-only package boundary, registry, cache resolver,
  explicit device validation, and `BanditSession` lifecycle contract.
- Added independent v1 Music64 and v2 Multi runtime graphs, each validated
  stem-by-stem against an untouched-upstream fixture from an official local
  checkpoint.
- Recorded SHA-256 integrity values for the official `v1-mus64-l1snr` and
  `v2-multi` artifacts. Their cache download path is enabled; all other catalog
  entries remain fail-closed until equivalent evidence exists.
