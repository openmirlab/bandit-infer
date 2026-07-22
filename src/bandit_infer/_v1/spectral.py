"""v1 spectral front-end using torch's public STFT API.

The module preserves upstream transform arguments and buffer names, but avoids
the upstream torchaudio binary dependency. Reads: torch only.
"""

from typing import Dict, Optional

import torch
from torch import nn


class _Spectrogram(nn.Module):
    def __init__(self, n_fft, win_length, hop_length, window_fn, wkwargs, normalized, center, pad_mode, onesided, power=None):
        super().__init__()
        assert power is None
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length, hop_length
        self.normalized, self.center, self.pad_mode, self.onesided = normalized, center, pad_mode, onesided
        self.register_buffer("window", window_fn(win_length, **(wkwargs or {})))

    def forward(self, audio):
        shape = audio.shape
        window = self.window / self.window.norm() if self.normalized else self.window
        transformed = torch.stft(audio.reshape(-1, shape[-1]), self.n_fft, self.hop_length, self.win_length, window, self.center, self.pad_mode, False, self.onesided, return_complex=True)
        return transformed.reshape(*shape[:-1], *transformed.shape[-2:])


class _InverseSpectrogram(_Spectrogram):
    def forward(self, spectrogram, length=None):
        shape = spectrogram.shape
        window = self.window / self.window.norm() if self.normalized else self.window
        restored = torch.istft(spectrogram.reshape(-1, *shape[-2:]), self.n_fft, self.hop_length, self.win_length, window, self.center, False, self.onesided, length=length)
        return restored.reshape(*shape[:-2], restored.shape[-1])


class _SpectralComponent(nn.Module):
    def __init__(
            self,
            n_fft: int = 2048,
            win_length: Optional[int] = 2048,
            hop_length: int = 512,
            window_fn: str = "hann_window",
            wkwargs: Optional[Dict] = None,
            power: Optional[int] = None,
            center: bool = True,
            normalized: bool = True,
            pad_mode: str = "constant",
            onesided: bool = True,
            **kwargs,
    ) -> None:
        super().__init__()

        assert power is None

        window_fn = torch.__dict__[window_fn]

        self.stft = (
                _Spectrogram(
                        n_fft=n_fft,
                        win_length=win_length,
                        hop_length=hop_length,
                        pad_mode=pad_mode,
                        window_fn=window_fn,
                        wkwargs=wkwargs,
                        power=power,
                        normalized=normalized,
                        center=center,
                        onesided=onesided,
                )
        )

        self.istft = (
                _InverseSpectrogram(
                        n_fft=n_fft,
                        win_length=win_length,
                        hop_length=hop_length,
                        pad_mode=pad_mode,
                        window_fn=window_fn,
                        wkwargs=wkwargs,
                        normalized=normalized,
                        center=center,
                        onesided=onesided,
                )
        )
