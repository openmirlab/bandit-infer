"""Torch-vs-MLX output parity on the two real, sha256-verified checkpoints
(``v1-mus64-l1snr``, ``v2-multi``), exercised through the public
``BanditSession`` API on real audio files on disk -- not a bare module
forward pass.

Marked ``realweights`` and deselected by default (see ``pyproject.toml``):
needs the ``[mlx]`` extra, an Apple Silicon Mac, and the checkpoint already
resolvable through the package's own cache/verification path (this test
never downloads a checkpoint itself; it skips if one is not already on disk
or not independently sha256-verified).

The three tail cases (clean signal, zero-padded tail, near-silent tail) are
the point of this file, not signal alone: every v1 direct-inference channel
and every v2 chunk is zero-padded by its own runtime's chunking/framing
arithmetic, and a fixture without a genuinely silent/near-silent region
would not exercise the ``exact_zero_safe_rfft`` guard's failure mode at all
(see ``mlx/rfft_guard.py``'s module docstring).

v2's real chunked handler forces roughly the same number of 8-second
Bandit-v2 forward passes (~22) regardless of how short the input audio is,
because ``StandardTensorChunkedInferenceHandler``'s front-pad
(``2 * (chunk_size - hop_size)`` = 14 s at 48 kHz) dominates for any input
shorter than that -- so the v2 half of this file is inherently slow (single
minutes per tail on CPU Torch) and the fixture length below is chosen as
short as the architecture allows, not shortened further for convenience.

Reads: bandit_infer.checkpoints, bandit_infer.api (BanditSession), numpy,
soundfile
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.realweights

# Measured worst case for v1 (see this file's own printed [mlx parity] lines
# when run with -s) sits around 1e-6 to 3e-7; v2's chunked overlap-add adds
# more floating-point accumulation, so its gate is a little looser. Both are
# set from what the implementation actually achieves, with headroom for
# run-to-run noise -- not widened to make an unrelated future regression
# pass quietly.
MAX_ABS_TOLERANCE = 5e-5
REL_L2_TOLERANCE = 5e-4


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        import mlx_spectro  # noqa: F401
    except ImportError:
        return False
    return True


def _checkpoint_ready(model_key: str) -> Path | None:
    from bandit_infer.checkpoints import cache_info, get_spec

    spec = get_spec(model_key)
    info = cache_info(spec)
    if info["verified"]:
        return info["path"]
    return None


def _fixture_audio(n_samples: int, tail: str, seed: int) -> np.ndarray:
    """(1, n_samples) mono float32: real signal, then `tail` behaviour."""
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal((1, n_samples)).astype(np.float32) * 0.2)
    half = n_samples // 2
    if tail == "zeros":
        audio[:, half:] = 0.0
    elif tail == "near_silent":
        audio[:, half:] *= 1e-6
    return audio


def _peak_and_rel_l2(reference: np.ndarray, other: np.ndarray) -> tuple[float, float, float, float]:
    max_abs = float(np.abs(reference - other).max())
    peak_reference = float(np.abs(reference).max())
    peak_other = float(np.abs(other).max())
    rel_l2 = float(np.linalg.norm(reference - other) / (np.linalg.norm(reference) + 1e-12))
    return max_abs, peak_reference, peak_other, rel_l2


# --------------------------------------------------------------------------- v1


@pytest.fixture(scope="module")
def v1_runtimes():
    if not _mlx_available():
        pytest.skip("MLX extra not installed: pip install 'bandit-infer[mlx]'")
    checkpoint = _checkpoint_ready("v1-mus64-l1snr")
    if checkpoint is None:
        pytest.skip("v1-mus64-l1snr checkpoint not verified/cached locally")

    import torch

    from bandit_infer._v1.runtime import V1Runtime
    from bandit_infer.checkpoints import get_spec
    from bandit_infer.mlx.v1.runtime import V1RuntimeMLX

    spec = get_spec("v1-mus64-l1snr")
    torch_runtime = V1Runtime(spec, checkpoint, torch.device("cpu"))
    mlx_runtime = V1RuntimeMLX(spec, checkpoint)
    return spec, torch_runtime, mlx_runtime


@pytest.mark.parametrize("tail", ["signal", "zeros", "near_silent"])
def test_v1_mlx_matches_torch_including_silence(tmp_path: Path, v1_runtimes, tail: str) -> None:
    spec, torch_runtime, mlx_runtime = v1_runtimes
    audio = _fixture_audio(44100 * 2, tail, seed=1)

    wav_path = tmp_path / f"v1_{tail}.wav"
    sf.write(str(wav_path), audio.T, 44100, subtype="FLOAT")
    on_disk, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    on_disk = on_disk.T  # (channels, samples), matching V1Runtime.infer's contract

    torch_out = torch_runtime.infer(on_disk, sample_rate=sr)
    mlx_out = mlx_runtime.infer(on_disk, sample_rate=sr)

    assert set(torch_out) == set(mlx_out) == set(spec.stems)
    worst_max_abs = 0.0
    worst_rel_l2 = 0.0
    for stem in spec.stems:
        max_abs, peak_ref, peak_mlx, rel_l2 = _peak_and_rel_l2(torch_out[stem], mlx_out[stem])
        print(f"\n[mlx parity] v1 tail={tail} stem={stem} max_abs={max_abs:.3e} "
              f"peak_torch={peak_ref:.3e} peak_mlx={peak_mlx:.3e} rel_l2={rel_l2:.3e}")
        worst_max_abs = max(worst_max_abs, max_abs)
        worst_rel_l2 = max(worst_rel_l2, rel_l2)
    assert worst_max_abs < MAX_ABS_TOLERANCE, (
        f"v1 tail={tail}: Torch-vs-MLX max abs {worst_max_abs:.3e} exceeds {MAX_ABS_TOLERANCE:.0e}"
    )
    assert worst_rel_l2 < REL_L2_TOLERANCE, (
        f"v1 tail={tail}: Torch-vs-MLX relative L2 {worst_rel_l2:.3e} exceeds {REL_L2_TOLERANCE:.0e}"
    )


def test_v1_rfft_guard_removal_regresses_or_is_honestly_inert(v1_runtimes) -> None:
    """Removes `exact_zero_safe_rfft` and re-measures the zero-padded-tail
    case -- the non-negotiable "validate the regression test by removing the
    fix" gate. Reports whichever is true rather than assuming: v1 mixes
    LayerNorm/InstanceNorm/GroupNorm (eps=1e-5, three orders above the rfft
    artifact) with an RNN-only tf-model for every registry checkpoint (no
    attention that would spread a corrupted frame across time -- see
    `mlx/rfft_guard.py`'s module docstring), so this is expected to land
    with the `mdxnet-infer`/`demucs-infer` "inert" data point rather than
    the roformer one, but the assertion below is a print, not a widened
    tolerance in either direction.
    """
    spec, torch_runtime, mlx_runtime = v1_runtimes
    audio = _fixture_audio(44100 * 2, "zeros", seed=1)

    import bandit_infer.mlx.spectral as spectral_mod

    torch_out = torch_runtime.infer(audio, sample_rate=44100)
    with_guard = max(
        float(np.abs(torch_out[stem] - out).max())
        for stem, out in mlx_runtime.infer(audio, sample_rate=44100).items()
    )

    original = spectral_mod.exact_zero_safe_rfft
    from contextlib import contextmanager

    @contextmanager
    def _noop():
        yield

    spectral_mod.exact_zero_safe_rfft = _noop
    try:
        without_guard = max(
            float(np.abs(torch_out[stem] - out).max())
            for stem, out in mlx_runtime.infer(audio, sample_rate=44100).items()
        )
    finally:
        spectral_mod.exact_zero_safe_rfft = original

    print(f"\n[rfft guard] v1 zero-padded tail: with_guard={with_guard:.3e} without_guard={without_guard:.3e}")
    # Load-bearing here means "removing the fix noticeably increases
    # divergence"; report both numbers regardless of which way it lands.
    assert with_guard < MAX_ABS_TOLERANCE


# --------------------------------------------------------------------------- v2


@pytest.fixture(scope="module")
def v2_runtimes():
    if not _mlx_available():
        pytest.skip("MLX extra not installed: pip install 'bandit-infer[mlx]'")
    checkpoint = _checkpoint_ready("v2-multi")
    if checkpoint is None:
        pytest.skip("v2-multi checkpoint not verified/cached locally")

    import torch

    from bandit_infer._v2.runtime import V2Runtime
    from bandit_infer.checkpoints import get_spec
    from bandit_infer.mlx.v2.runtime import V2RuntimeMLX

    spec = get_spec("v2-multi")
    torch_runtime = V2Runtime(checkpoint, torch.device("cpu"))
    mlx_runtime = V2RuntimeMLX(checkpoint)
    return spec, torch_runtime, mlx_runtime


@pytest.mark.parametrize("tail", ["signal", "zeros", "near_silent"])
def test_v2_mlx_matches_torch_including_silence(tmp_path: Path, v2_runtimes, tail: str) -> None:
    spec, torch_runtime, mlx_runtime = v2_runtimes
    # Short: v2's chunk count is dominated by the handler's own front-pad
    # regardless of input length (see this module's docstring) -- there is
    # no parity benefit to a longer fixture, only more wall-clock time.
    audio = _fixture_audio(4800, tail, seed=2)

    wav_path = tmp_path / f"v2_{tail}.wav"
    sf.write(str(wav_path), audio.T, 48000, subtype="FLOAT")
    on_disk, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    on_disk = on_disk.T

    torch_out = torch_runtime.infer(on_disk, sample_rate=sr)
    mlx_out = mlx_runtime.infer(on_disk, sample_rate=sr)

    assert set(torch_out) == set(mlx_out) == set(spec.stems)
    worst_max_abs = 0.0
    worst_rel_l2 = 0.0
    for stem in spec.stems:
        max_abs, peak_ref, peak_mlx, rel_l2 = _peak_and_rel_l2(torch_out[stem], mlx_out[stem])
        print(f"\n[mlx parity] v2 tail={tail} stem={stem} max_abs={max_abs:.3e} "
              f"peak_torch={peak_ref:.3e} peak_mlx={peak_mlx:.3e} rel_l2={rel_l2:.3e}")
        worst_max_abs = max(worst_max_abs, max_abs)
        worst_rel_l2 = max(worst_rel_l2, rel_l2)
    assert worst_max_abs < MAX_ABS_TOLERANCE, (
        f"v2 tail={tail}: Torch-vs-MLX max abs {worst_max_abs:.3e} exceeds {MAX_ABS_TOLERANCE:.0e}"
    )
    assert worst_rel_l2 < REL_L2_TOLERANCE, (
        f"v2 tail={tail}: Torch-vs-MLX relative L2 {worst_rel_l2:.3e} exceeds {REL_L2_TOLERANCE:.0e}"
    )
