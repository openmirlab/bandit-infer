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

### MLX backend (Apple Silicon), from-scratch port

- Added an optional `[mlx]` extra with independent v1 and v2 MLX graphs
  (`src/bandit_infer/mlx/v1/`, `src/bandit_infer/mlx/v2/`), written directly
  from this package's own Torch source -- there is no upstream MLX
  implementation of Bandit to vendor from, unlike this org's other MLX
  backends. `backend=` selects the compute framework (`None`/`"torch"`
  default, `"mlx"`, `"auto"`) and is a separate axis from `device=` (where a
  Torch backend runs): `backend="mlx"` accepts only `device="auto"`/`"mps"`
  and raises `BackendUnavailable` for anything else (a Torch-only value like
  `"cuda"` is refused, never silently ignored). `resolve_device` (the Torch
  device resolver) is never called on the `backend="mlx"` path.
  `MLX_ENABLE_AMP=0` is set explicitly before any MLX model is constructed.
- **Public-API rename, named explicitly per this org's requirement for that
  class of change:** `BanditSession`'s v1/v2 model-family selector was
  `backend=` before this entry (and `CheckpointSpec`'s field, and
  `checkpoints.toml`'s per-entry key, were both `backend` too). It is now
  `family=` / `CheckpointSpec.family` / `family="v1"`/`"v2"` in the manifest.
  This was a real design mistake caught in review, not a preemptive
  refactor: the MLX work initially spent `device="mlx"` as a fourth Torch
  device value, reasoning that `backend=` was unavailable because it already
  meant "v1 or v2" here -- but `backend` is the org-wide name for *compute
  framework* (`bs-roformer-infer`'s `brain/decisions.md` D1; six sibling
  packages already expose `backend="mlx"`), and overloading `device=`
  instead conflated two axes D1 keeps deliberately orthogonal. Renaming the
  model-family axis to `family=` frees `backend=` for its org-wide meaning
  instead. Authorized by constitution article 7 (amendment 2026-07-17,
  "unpublished surface owes no compatibility -- before a package's first
  release, delete legacy/back-compat code instead of preserving it"):
  `bandit-infer` is not on PyPI, so the original `backend=` naming choice
  owed nothing to anyone and was renamed outright, with no deprecated alias
  kept alongside it.
- Support is co-extensive with the Torch v1/v2 runtimes, measured against
  the manifest rather than declared: every one of the 28 registry `variant`
  strings resolves to the RNN-based `SeqBandModellingModule` core, so v1's
  Transformer/Conv tf-model variants and any non-`musical` v2 band type are
  out of scope for both backends equally. As with Torch, only
  `v1-mus64-l1snr` and `v2-multi` have a verified SHA-256 and can actually
  load; the checkpoint gate in `checkpoints.py` applies identically
  regardless of backend.
- Two MLX 0.31/0.32 numerical/correctness findings, applied as fixes: (1)
  `mx.fft.rfft`'s ~4.5e-07 non-zero result for an exactly-zero frame
  (`mlx/rfft_guard.py::exact_zero_safe_rfft`, applied unconditionally,
  regardless of measurement); (2) `mx.array.at[...].add()`'s silent
  corruption of large 4-D scatter-add updates, avoided in
  `mlx/v2/runtime.py`'s windowed overlap-add fold via `mx.pad` + elementwise
  add instead. MLX has no `"reflect"` pad mode (confirmed via a direct
  `ValueError`, not assumed); the v2 chunk handler's front/end padding
  round-trips through `numpy.pad(mode="reflect")` instead, verified
  byte-identical to `torch.nn.functional.pad(..., mode="reflect")` before
  being wired in.
- Bidirectional LSTM/GRU: MLX's `nn.LSTM`/`nn.GRU` are unidirectional, so
  each `ResidualRNN` holds a forward/backward pair, concatenated on the
  feature axis. The weight/bias conversion algebra that makes this match
  Torch's single bidirectional layer (`mlx/weight_utils.py::
  remap_bidirectional_rnn`) was verified against a direct, model-independent
  `torch.nn.LSTM`/`torch.nn.GRU` comparison before ever being wired into a
  model: max abs diff ~8.9e-08 (LSTM), ~1.5e-07 (GRU).
- Added the non-negotiable auditing weight loader
  (`mlx/weight_utils.py::load_converted_weights`), shared between v1 and
  v2: raises before `load_weights` if any model parameter would be left at
  random initialization or any converted tensor dropped, rather than
  `load_weights(..., strict=False)`'s silent partial load.
- Torch-vs-MLX parity, measured through the public `BanditSession` API on
  real audio with a clean, zero-padded-tail, and near-silent-tail case each
  (Apple Silicon, torch 2.13.0, mlx 0.32.0, `tests/test_mlx_parity.py`,
  `-m realweights`):

  | model | tail | max abs | relative L2 |
  | --- | --- | --- | --- |
  | v1-mus64-l1snr | signal | 8.3e-07 | 5.1e-06 |
  | v1-mus64-l1snr | zeros | 8.3e-07 | 1.7e-05 |
  | v1-mus64-l1snr | near_silent | 8.3e-07 | 1.0e-05 |
  | v2-multi | signal | 6.0e-07 | 2.0e-06 |
  | v2-multi | zeros | 6.0e-07 | 1.3e-06 |
  | v2-multi | near_silent | 6.0e-07 | 1.1e-06 |

  All three v2 tails were run end to end through the real 8s/1s
  `StandardTensorChunkedInferenceHandler` handler on both sides (not the
  faster alternate-chunk-size sanity check an earlier revision of this entry
  reported as a partial substitute for the two silence tails). v2's real
  handler forces ~22 8-second Bandit-v2 forward passes regardless of input
  length (its front-pad dominates -- see `test_mlx_parity.py`'s module
  docstring); each full-config Torch(CPU) run took 620-640s on this
  session's machine (`torch.set_num_threads(8)`, no concurrent contention --
  thread tuning and isolation shaved it from the ~14 minutes an earlier,
  contended attempt measured, but a shorter input did not, since the
  front-pad already dominates at any input length this small). Three
  sequential ~11-minute Torch runs plus their MLX counterparts (~60-100s
  each) were affordable in the foreground within a single 600000 ms-timeout
  budget per run.

  `exact_zero_safe_rfft` was measured **inert** for v1: removing it on the
  zero-padded-tail case moved worst-case max abs from 8.3e-07 (with guard)
  to 1.3e-06 (without) -- both the same ~1e-6 noise floor, not the
  order-of-magnitude jump `bs-roformer-infer`/`melband-roformer-infer`
  measured. v1's LayerNorm/InstanceNorm/GroupNorm all use `eps=1e-5`, three
  orders above the rfft artifact, and every registry checkpoint's tf-model
  is RNN-only (no attention to spread a corrupted frame across time).
  Applied anyway, per the org's standing cheap-insurance policy.
  `test_v1_rfft_guard_removal_regresses_or_is_honestly_inert` is the
  regression test that removes the fix and re-measures, not just asserts.
- Not from any of the four prior applications of this pattern: this port
  needed bidirectional LSTM/GRU (none of `bs-roformer-infer`,
  `melband-roformer-infer`, `mdxnet-infer`, or `demucs-infer` have an RNN
  core), and v2's `StandardTensorChunkedInferenceHandler` needed a from-
  scratch numerical re-implementation since Torch's `F.unfold`/`F.fold`
  im2col trick has no MLX equivalent.

### Apple Silicon (MPS) device support

- **Contract change, called out deliberately:** `device.py::resolve_device`
  previously accepted only `auto`, `cpu`, `cuda`, and `cuda:N`, and raised
  `ValueError` on `mps`. `mps` is now a valid explicit device: it resolves when
  available and raises `RuntimeError` (never a silent CPU downgrade) when
  requested but unavailable -- the same explicit-or-fail contract CUDA already
  had, per org constitution article 4b.
- `device="auto"` is **unchanged**: still resolves to cuda-else-cpu. It does
  not promote a Mac caller onto MPS, so an upgrade cannot silently move
  existing callers' numerical output. Verified by a regression test that pins
  `auto` to `cpu` even when `mps_available()` reports `True`.
- Added `mps_available()` as the single owner of "is MPS usable here," guarded
  with `getattr` for torch builds lacking `backends.mps` entirely, so no call
  site inlines `torch.backends.mps.is_available()` directly.
- Grepped the package for other CUDA-or-CPU assumptions
  (`torch.cuda.empty_cache()`, `torch.cuda.amp.autocast()`,
  `torch.backends.cudnn.benchmark`, `.cuda()`): none exist outside
  `device.py`. `_v1/tfmodel.py` and `_v2/tfmodel.py` each carry an unused
  `import torch.backends.cuda` with no call site; left as-is since removing an
  unrelated dead import is out of scope for this change.
- This mirrors the reversal `bs-roformer-infer` made on the same axis: see its
  `CHANGELOG.md` "Apple Silicon (MPS) support" entry and `brain/decisions.md`
  D8 for the evidence-driven reasoning behind treating `auto` and explicit
  `mps` as separate contracts.

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
