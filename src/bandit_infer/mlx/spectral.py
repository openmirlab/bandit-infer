"""Shared MLX STFT/iSTFT wrapper matching bandit's own torch.stft convention.

Both Torch spectral front-ends this package ships (`_v1/spectral.py`'s
`_SpectralComponent` and `_v2/bandit.py`'s inline `_Spectrogram`/
`_InverseSpectrogram`) are byte-identical copies of the same wrapper --
sharing one MLX equivalent here is a strict improvement, not new coupling,
and stays inside CLAUDE.md's "generic cache or device helpers may be shared"
carve-out: this module takes no band-split/mask-estimation config and knows
nothing about v1 or v2's architecture, only `(n_fft, hop_length, win_length,
pad_mode)`.

One convention worth stating plainly because it is *not* what `torch.stft(...,
normalized=True)` does on its own: bandit's own wrapper pre-divides the
window by its L2 norm and then calls `torch.stft`/`torch.istft` with
`normalized=False` explicitly -- not the standard `1/sqrt(n_fft)` scaling
`normalized=True` would apply. `mlx_spectro`'s own `normalized=` flag *is*
the standard `1/sqrt(n_fft)` convention, so it must stay `False` here too;
`build_transform` reproduces bandit's window-norm convention by passing an
already-normalized window array in instead. Verified by direct comparison
against `torch.stft`/`torch.istft` for both `center_pad_mode` values this
package uses (`"constant"` for v1, `"reflect"` for v2): max abs diff ~8e-07
on the STFT itself, ~1.4e-06 on an STFT/iSTFT round trip -- the same noise
floor the sibling packages' MLX STFT wrappers measure.

The Hann window itself comes from `torch.hann_window` (not a from-scratch
formula) so it is bit-identical to what produced the checkpoint's training
distribution, and torch is already a hard dependency of this package.

Reads: mlx.core, mlx_spectro, torch (window generation only), numpy,
.rfft_guard (exact_zero_safe_rfft)
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_spectro import get_transform_mlx

from .rfft_guard import exact_zero_safe_rfft


@dataclass(frozen=True)
class SpectralConfig:
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    pad_mode: str = "constant"


def _normalized_window(win_length: int) -> mx.array:
    import torch

    window = torch.hann_window(win_length, periodic=True).numpy()
    window = window / np.linalg.norm(window)
    return mx.array(window.astype(np.float32))


def build_transform(config: SpectralConfig):
    """One `mlx_spectro.SpectralTransform`, ready for `stft`/`istft` below."""
    window = _normalized_window(config.win_length)
    return get_transform_mlx(
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window_fn="hann",
        periodic=True,
        center=True,
        normalized=False,
        window=window,
        center_pad_mode=config.pad_mode,
    )


def stft(transform, audio: mx.array) -> mx.array:
    """`(..., samples) -> (..., freq, frames)` complex64, matching `torch.stft`."""
    shape = audio.shape
    flat = audio.reshape(-1, shape[-1])
    with exact_zero_safe_rfft():
        spec = transform.stft(flat)
        mx.eval(spec)
    return spec.reshape(*shape[:-1], spec.shape[-2], spec.shape[-1])


def istft(transform, spec: mx.array, length: int) -> mx.array:
    """`(..., freq, frames) -> (..., samples)`, matching `torch.istft`."""
    shape = spec.shape
    flat = spec.reshape(-1, shape[-2], shape[-1])
    with exact_zero_safe_rfft():
        audio = transform.istft(flat, length=length)
        mx.eval(audio)
    return audio.reshape(*shape[:-2], audio.shape[-1])
