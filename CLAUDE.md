# bandit-infer maintainer notes

## Scope and status

This repo is an independent inference-only Bandit package. It credits Karn
Watcharasupat and collaborators' Apache-2.0 v1/v2 repositories and never
bundles, converts, or rehosts their weights. The v1 official Zenodo weights are
CC-BY-NC-4.0; v2 weights are CC-BY-SA-4.0.

The initial catalog accurately lists 28 intended artifacts (21 v1 Bandit and
seven v2) but Zenodo publishes only MD5 metadata. Each entry deliberately has
an empty SHA-256 and automatic official loading is disabled. Do not fill these
values from guesses, filenames, or a different artifact. A model becomes
supported only after a local official download is SHA-256 hashed, strict-loaded
against the correct independent graph, and compared stem-by-stem to an
untouched-upstream golden fixture.

## Layout and conventions

Every load-bearing source file starts with a nav header describing purpose and
its reads. `bandit_infer._v1` and `_v2` own separate model graphs; generic cache
or device helpers may be shared but model layers/config interpretation may not.
Production code remains inference-only: no training, evaluation, datasets,
Hydra, Lightning trainer, or experiment surface.

## Required verification

```bash
uv sync --all-extras --dev
uv run pytest -q
uv run python -m build
uv run python tools/verify_inference_only.py
uv run python tools/verify_wheel.py
uv run python tools/generate_v1_upstream_fixture.py --fixture tests/fixtures/v1-mus64.npz
uv run python tools/verify_v1_upstream_parity.py --fixture tests/fixtures/v1-mus64.npz
uv run python tools/generate_v2_upstream_fixture.py --fixture tests/fixtures/v2-multi.npz
uv run python tools/verify_v2_upstream_parity.py --fixture tests/fixtures/v2-multi.npz
uv run python tools/verify_checkpoint_compatibility.py --real
uv run python -c "import bandit_infer; print(bandit_infer.__version__); print(bandit_infer.BanditSession)"
git status --short
git log --oneline --decorate -10
```

The first five and import command are expected to pass at this stage. The four
real-model commands are intentionally explicit failing gates until their
required backend/fixture/checkpoint inputs exist; do not report them as passed.
