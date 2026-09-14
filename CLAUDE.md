# bandit-infer maintainer notes

## Scope and status

This repo is an independent inference-only Bandit package. It credits Karn
Watcharasupat and collaborators' Apache-2.0 v1/v2 repositories and never
bundles, converts, or rehosts their weights. The v1 official Zenodo weights are
CC-BY-NC-4.0; v2 weights are CC-BY-SA-4.0.

The catalog accurately lists 28 intended artifacts (21 v1 Bandit and seven
v2). Zenodo publishes only MD5 metadata. All 28 entries carry SHA-256 values
computed from their official Zenodo downloads (2026-09-14 for the 26 added
after the first two); each download was also confirmed to match the catalog's
recorded Zenodo `size` and `md5` before its digest was written. That makes
every entry byte-verified and safe for automatic download. It is a separate,
weaker claim than support: only `v1-mus64-l1snr` and `v2-multi` strict-load
their independent graphs and pass stem-by-stem fixtures from untouched
upstream implementations. The other 26 are integrity-verified but carry no
load or parity evidence. Never fill or change a SHA-256 value from guesses,
filenames, or a different artifact -- only from hashing the official bytes.

## Layout and conventions

Every load-bearing source file starts with a nav header describing purpose and
its reads. `bandit_infer._v1` and `_v2` own separate model graphs; generic cache
or device helpers may be shared but model layers/config interpretation may not.
Production code remains inference-only: no training, evaluation, datasets,
Hydra, Lightning trainer, or experiment surface.

`CheckpointSpec.family` (`"v1"`/`"v2"`) is the model-architecture axis;
`BanditSession(model, family=...)` validates a caller-supplied value against
it. `checkpoints.toml`'s per-entry `family="v1"`/`"v2"` key is the package's
only public selector axis beyond `model`.

`device.py::resolve_device` accepts `auto`, `cpu`, `cuda`, and `cuda:N`,
resolving Torch devices only. `auto` keeps its exact legacy meaning
(cuda-else-cpu). An explicit `mps` request raises `ValueError` without probing
hardware availability; the package intentionally does not ship an Apple
accelerator execution path.

The pre-release Apple runtime, optional extra, framework dispatch axis, and
seam tests were removed before publication. Do not reintroduce a second runtime
path unless it comes with fresh parity fixtures, package metadata, and public
API review.

## Testing philosophy

Offline unit/regression tests run by default (`addopts = "-m 'not
realweights'"`); the `realweights` marker covers anything needing a real,
locally cached official checkpoint (never downloaded by the tests
themselves). Upstream parity (the
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
```

The package tests, build, inference-only scan, wheel check, v1 Music64 parity,
and v2 Multi parity commands pass when supplied the recorded official
checkpoints. Fixture generation is run from a separately provisioned
environment against untouched upstream source; it is intentionally not a
package runtime dependency. The all-family real compatibility command remains
an open gate because the remaining 26 official checkpoint bytes are unverified.
