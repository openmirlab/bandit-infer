# bandit-infer maintainer notes

## Scope and status

This repo is an independent inference-only Bandit package. It credits Karn
Watcharasupat and collaborators' Apache-2.0 v1/v2 repositories and never
bundles, converts, or rehosts their weights. The v1 official Zenodo weights are
CC-BY-NC-4.0; v2 weights are CC-BY-SA-4.0.

The catalog accurately lists 28 intended artifacts (21 v1 Bandit and seven
v2). Zenodo publishes only MD5 metadata. `v1-mus64-l1snr` and `v2-multi` have
independently computed SHA-256 values, strict-load their independent graphs,
and pass stem-by-stem fixtures from untouched upstream implementations. All
other entries deliberately retain empty SHA-256 values. Do not fill these
values from guesses, filenames, or a different artifact.

## Layout and conventions

Every load-bearing source file starts with a nav header describing purpose and
its reads. `bandit_infer._v1` and `_v2` own separate model graphs; generic cache
or device helpers may be shared but model layers/config interpretation may not.
Production code remains inference-only: no training, evaluation, datasets,
Hydra, Lightning trainer, or experiment surface.

`CheckpointSpec.family` (`"v1"`/`"v2"`) is the model-architecture axis;
`BanditSession(model, family=...)` validates a caller-supplied value against
it. This field was named `backend` until the MLX port -- renamed to free
`backend=` for the org-wide compute-framework meaning (see "MLX backend"
below); `checkpoints.toml`'s per-entry `family="v1"`/`"v2"` key was renamed
to match.

`device.py::resolve_device` accepts `auto`, `cpu`, `cuda`, `cuda:N`, and
`mps`, resolving **Torch devices only** -- it has no MLX case. `auto` keeps
its exact legacy meaning (cuda-else-cpu) and deliberately never promotes to
MPS -- mirroring `bs-roformer-infer`'s `_select_device` contract, so two
packages in the org do not diverge on the same call. An explicitly requested
`mps` that is unavailable raises `RuntimeError`, never a silent CPU downgrade
(article 4b). `mps_available()` is the single owner of "is MPS usable here";
nothing else in the package should inline
`torch.backends.mps.is_available()`. On Apple Silicon, both `uv` and the
resolved Python interpreter must be native `arm64` builds -- an x86_64
(Rosetta) interpreter reports MPS as unavailable rather than misconfigured,
the same environment trap recorded in `bs-roformer-infer`'s
`brain/evidence.md` section 0.

## MLX backend (`src/bandit_infer/mlx/`)

Optional, additive `[mlx]` extra (Apple Silicon). `import bandit_infer` never
imports mlx (`tests/test_backends.py` asserts this via a subprocess, since an
*earlier* test in the same process legitimately importing real mlx as a side
effect of a real `mlx_available()` check is not the same claim). There is no
upstream MLX implementation of Bandit to vendor from -- unlike this org's
other MLX backends, both `mlx/v1/model.py` and `mlx/v2/model.py` are written
directly from this package's own Torch source, with every class mirroring
its Torch counterpart's attribute names 1:1 (verified against the real
`dnr-3s-mus64-l1snr.ckpt`/`checkpoint-multi.ckpt` `state_dict` keys, not
assumed from source reading).

- `mlx/rfft_guard.py`, `mlx/spectral.py`, `mlx/weight_utils.py` -- generic,
  architecture-agnostic helpers shared between v1 and v2 (the
  `exact_zero_safe_rfft` numerical guard, the STFT/iSTFT wrapper, and the
  bidirectional-RNN weight-split + auditing load gate). This is the same
  "generic cache or device helpers may be shared" carve-out named above;
  none of the three has any band-split/mask-estimation config.
- `mlx/v1/` and `mlx/v2/` -- independent model graphs, matching `_v1`/`_v2`'s
  own isolation. `v1/model.py` and `v2/model.py` are deliberately not shared
  even though several classes look near-identical (`ResidualRNN`,
  `SeqBandModellingModule`) -- `_v1/tfmodel.py` and `_v2/tfmodel.py` don't
  share either, and `BandSplitModule`'s flatten order genuinely differs
  between the two (see `mlx/v2/model.py`'s module docstring).
- Support is measured, not declared: every one of the 28 registry `variant`
  strings resolves to the RNN-based `SeqBandModellingModule` core in
  `_v1/runtime.py`/`_v2/runtime.py` -- confirmed by reading `_BAND`'s keys
  and `V2_KWARGS` against the manifest, not assumed -- so v1's
  `TransformerTimeFreqModule`/`ConvolutionalTimeFreqModule` and any
  non-`musical` v2 band type are out of scope for both backends equally,
  not an MLX-specific gap.
- `mlx/v1/runtime.py` and `mlx/v2/runtime.py` reuse `_v1.wrapper.
  get_band_specs`/`get_band_specs_map`, `_v1.runtime._BAND`, `_v2.runtime.
  V2_KWARGS`, and `_v2.utils.MusicalBandsplitSpecification` directly (pure
  Python band-index arithmetic, no Torch tensor graph) rather than
  redefining them, so band boundaries cannot drift between the Torch and
  MLX ports of the same architecture family.
- `api.py::BanditSession` has two orthogonal axes, matching six sibling
  OpenMIRLab packages (`bs-roformer-infer`'s `brain/decisions.md` D1):
  `family=` (v1 vs v2 model architecture) and `backend=` (compute framework:
  `None`/`"torch"` default, `"mlx"`, or `"auto"`, resolved by
  `api.py::resolve_backend_name`). `device=` still means a Torch device only
  when `backend="torch"`; when `backend="mlx"`, `api.py::_select_mlx_device`
  accepts only `"auto"`/`"mps"` for `device=` and raises `BackendUnavailable`
  for anything else (a Torch-only value like `"cuda"` is refused, never
  silently ignored) -- `resolve_device` (the Torch device resolver) is never
  called on the `backend="mlx"` path at all. `MLX_ENABLE_AMP=0` is set
  explicitly in `api.py::load_mlx_runtime_backend` before any MLX model is
  constructed.

  This wasn't the first shape: an earlier revision spent `device="mlx"` as a
  fourth value of the *device* axis, reasoning that `backend=` already meant
  the v1/v2 model family here and so was unavailable for the compute-
  framework meaning every sibling package uses it for. That was wrong -- it
  conflated two things D1 keeps deliberately orthogonal, and would have
  taught a user who learned `backend="mlx"` on any sibling package a
  vocabulary that means something different here. Corrected by renaming the
  model-family field/parameter to `family=` instead (freeing `backend=` for
  its org-wide meaning), authorized by article 7 (amendment 2026-07-17,
  "unpublished surface owes no compatibility"): `bandit-infer` is not yet on
  PyPI, so the original `backend=` naming choice owed nothing to anyone and
  was renamed outright rather than preserved alongside the new meaning. See
  `CHANGELOG.md`'s MLX entry for the full reasoning, named explicitly per
  this org's requirement for public-API renames.
- `mlx/v2/runtime.py`'s `ChunkedInferenceHandlerMLX` reimplements (not
  reuses) `_v2/inference.py::StandardTensorChunkedInferenceHandler`'s 8s/1s
  windowed overlap-add numerically, since `F.unfold`/`F.fold` have no MLX
  equivalent. Its front/end reflect-padding step round-trips through
  `numpy.pad(mode="reflect")` (MLX 0.31/0.32 have no `"reflect"` pad mode --
  confirmed by a direct `ValueError`, not assumed), and its windowed fold
  accumulates with `mx.pad` + elementwise add, never `mx.array.at[...]
  .add()` (the org's accumulated finding: MLX 0.31.2's scatter-add silently
  corrupts large 4-D updates -- this accumulator is exactly that scale).
- `exact_zero_safe_rfft`: measured, not assumed, against the real
  `v1-mus64-l1snr` checkpoint with a zero-padded-tail fixture -- see
  `tests/test_mlx_parity.py::test_v1_rfft_guard_removal_regresses_or_is_honestly_inert`
  for the with/without numbers. Applied unconditionally regardless of the
  measurement, per the org's standing cheap-insurance policy.

## Testing philosophy

Offline unit/regression tests run by default (`addopts = "-m 'not
realweights'"`); the `realweights` marker covers anything needing a real,
locally cached official checkpoint (never downloaded by the tests
themselves) and, for MLX, the `[mlx]` extra plus an arm64 interpreter --
`test_mlx_parity.py` skips silently rather than fails when either is
missing, so a green default run says nothing about MLX or MPS correctness.
Confirm `python -c "import platform; print(platform.machine())"` says
`arm64` before trusting a realweights run. Upstream parity (the
`verify_*_upstream_parity.py` tools below) is the accuracy gate proper: it
compares this package's output against an untouched upstream checkout, not
just against itself.

## Required verification

```bash
uv sync --all-extras --dev
uv run pytest -q
uv run python -m build
uv run python tools/verify_inference_only.py
uv run python tools/verify_wheel.py
uv run python tools/generate_v1_upstream_fixture.py --fixture tests/fixtures/v1-mus64-upstream.npz --checkpoint /path/to/dnr-3s-mus64-l1snr.ckpt --upstream-root /path/to/untouched/bandit-v1 --python /path/to/upstream-python --device cuda
uv run python tools/verify_v1_upstream_parity.py --fixture tests/fixtures/v1-mus64-upstream.npz --checkpoint /path/to/dnr-3s-mus64-l1snr.ckpt --device cuda
uv run python tools/generate_v2_upstream_fixture.py --fixture tests/fixtures/v2-multi-upstream.npz --checkpoint /path/to/checkpoint-multi.ckpt --upstream-root /path/to/untouched/bandit-v2 --python /path/to/upstream-python --device cuda
uv run python tools/verify_v2_upstream_parity.py --fixture tests/fixtures/v2-multi-upstream.npz --checkpoint /path/to/checkpoint-multi.ckpt --device cuda
uv run python tools/verify_checkpoint_compatibility.py --real
uv run python -c "import bandit_infer; print(bandit_infer.__version__); print(bandit_infer.BanditSession)"
git status --short
git log --oneline --decorate -10

# MLX backend (Apple Silicon only; needs the [mlx] extra and both real checkpoints on disk)
uv run pytest -m realweights tests/test_mlx_parity.py -v
```

The package tests, build, inference-only scan, wheel check, v1 Music64 parity,
and v2 Multi parity commands pass when supplied the recorded official
checkpoints. Fixture generation is run from a separately provisioned
environment against untouched upstream source; it is intentionally not a
package runtime dependency. The all-family real compatibility command remains
an open gate because the remaining 26 official checkpoint bytes are unverified.
