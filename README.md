# bandit-infer

Inference-only Bandit checkpoint registry and session contract for cinematic
audio source separation. It is an early, deliberately safety-gated package:
the catalog is complete, but official model execution is not enabled until the
upstream graphs, SHA-256 values, and golden parity evidence are complete.

## Why this exists

Bandit and Bandit v2 are research implementations for the Cinematic Audio
Source Separation (CASS) task. Their public repositories include training and
experiment systems; this standalone package is intended to preserve only a
small, installable inference surface with explicit checkpoint provenance and
model lifetime.

## Acknowledgments

- [Karn Watcharasupat](https://github.com/kwatcharasupat) and collaborators
  published [Bandit v1](https://github.com/kwatcharasupat/bandit) and
  [Bandit v2](https://github.com/kwatcharasupat/bandit-v2), both Apache-2.0.
- The official v1 weights are hosted by
  [Zenodo record 10160698](https://zenodo.org/records/10160698) under
  CC-BY-NC-4.0; the official v2 weights are hosted by
  [Zenodo record 12701995](https://zenodo.org/records/12701995) under
  CC-BY-SA-4.0. Those licenses apply to weights, not the code.

## Citation

Please cite the original work, not this packaging effort:

```bibtex
@article{watcharasupat2023bandit,
  title={A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation},
  author={Watcharasupat, Karn and Wu, Chih-Wei and Ding, Yiwei and Orife, Iroro and Hipple, Aaron J. and Williams, Phillip A. and Kramer, Scott and Lerch, Alexander and Wolcott, William},
  journal={IEEE Open Journal of Signal Processing}, year={2023},
  doi={10.1109/OJSP.2023.3339428}
}
```

## Features and current boundary

- A package-owned 28-entry catalog: 21 v1 Bandit artifacts and seven v2
  artifacts. `dnr-demucs.ckpt` and `dnr-umxhq.ckpt` are explicitly excluded.
- Independent `v1` and `v2` dispatch namespaces. No checkpoint conversion and
  no shared model-layer abstraction is permitted.
- `BanditSession` provides idempotent `load`, ready-only `infer`, reloadable
  `release`, terminal `close`, `status`, `cache_info`, and context management.
- `auto`, `cpu`, `cuda`, and `cuda:N` device validation. Explicit unavailable
  CUDA requests fail instead of silently falling back.

The public execution path intentionally remains unavailable in this prerelease.
The v1 family/config mapping is metadata-derived but not real-load verified;
the v2 seven-file common graph is likewise not yet real-load verified. No
numerical compatibility claim is made.

## Install

```bash
pip install bandit-infer
```

For development, use `uv sync --all-extras --dev`.

## Catalog and planned usage

```bash
bandit-infer --list-models
```

When the integrity and graph gates are complete, the intended public call is:

```python
from bandit_infer import BanditSession

with BanditSession("v1-mus64-l1snr", device="auto") as session:
    stems = session.infer(audio, sample_rate=44100)
```

Multi-stem models return only `speech`, `music`, and `effects`. v1 dedicated
GRU/LSTM weights return only their trained stem. v1 is natively 44.1 kHz;
v2 is natively 48 kHz. Upstream mono/channel semantics will be copied and
tested before execution is enabled; this package does not claim stereo parity.

## Weights and cache

Weights will never be bundled, committed, rehosted, or converted. The planned
default folder is `~/.cache/bandit-infer/`, overridable by `weights_dir` or
`BANDIT_INFER_WEIGHTS`; direct-path loading accepts `checkpoint_path` plus a
caller-provided SHA-256. Official URLs and sizes are in the packaged
`checkpoints.toml`.

Manual placement and auto-download are intentionally disabled for official
artifacts until each expected SHA-256 is computed from an official download.
Zenodo publishes an MD5 checksum, which this package records for provenance but
does not substitute for the required SHA-256 integrity contract.

## What this project will NEVER bundle

- model weights or a weight mirror;
- model training, evaluation metrics, datasets/dataloaders, experiments,
  Hydra, Lightning trainer surfaces, or checkpoint conversion;
- the baseline Demucs/UMXHQ artifacts in v1's Zenodo record.

## Development

Run the commands recorded in [CLAUDE.md](CLAUDE.md). The parity scripts are
present as explicit gates, but cannot produce evidence until faithful backends
and a real official checkpoint are available locally.

## License

Code is Apache-2.0. Model weights have their separate licenses stated above.
