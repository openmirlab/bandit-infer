"""Bandit v2 MLX runtime -- builds the one `Bandit` graph every v2 registry
checkpoint uses, strict-loads converted weights, and re-implements
`_v2/inference.py::StandardTensorChunkedInferenceHandler`'s 8s/1s windowed
overlap-add chunking in MLX (the same 8s-chunk / 1s-hop / batch-of-1
configuration `_v2/runtime.py::V2Runtime` hard-codes).

`V2_KWARGS` is reused directly from `_v2.runtime` (not copied) -- one owner
for "what does the one supported v2 architecture look like," matching how
`v1/runtime.py` reuses `_v1.runtime._BAND` rather than redefining it. Band
type is asserted `"musical"` (`_v2/bandit.py::instantiate_bandsplit` already
asserts this upstream, so every v2 registry entry is `OverlappingMaskEstimationModule`
-- there is no non-overlapping v2 path to port, unlike v1's `dnr:vox7`).

Two things worth being explicit about, since Torch's `F.unfold`/`F.fold`
im2col trick has no MLX equivalent and this port does not try to reproduce
it structurally, only its numerical result:

1. **Chunk extraction** is a plain Python slicing loop over the padded
   mixture (`padded[:, start:start+chunk_size]`) -- exactly what
   `F.unfold(kernel_size=(chunk_size,1), stride=(hop_size,1))` produces, one
   chunk at a time, without needing `mx.pad`'s `"reflect"` mode (which does
   not exist in MLX 0.31.2 -- confirmed, not assumed: `mx.pad(..., mode=
   "reflect")` raises `ValueError`. The handler's own *front/end* padding
   step is reflect-mode and still needs it, so that one step round-trips
   through `numpy.pad(mode="reflect")`, verified byte-identical to
   `torch.nn.functional.pad(..., mode="reflect")` on a direct comparison
   before being wired in here.
2. **The windowed overlap-add fold** accumulates each chunk's contribution
   with `mx.pad` + elementwise add into a running total, never `mx.array.at
   [...].add()` -- the same safe pattern `mlx/v1/model.py`'s
   `OverlappingMaskEstimationModule` uses, per the org's accumulated finding
   that MLX 0.31.2's scatter-add silently corrupts large updates. This one
   genuinely is large (`padded_samples` can run to hundreds of thousands of
   samples), so this is the case that finding was written for, not
   over-caution.

Reads: mlx.core, numpy, .model, .convert, ..weight_utils, ..spectral,
.._v2.runtime (V2_KWARGS), .._v2.utils (MusicalBandsplitSpecification),
..checkpoints (CheckpointSpec, unused directly but documents the caller
contract)
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..spectral import SpectralConfig, build_transform, istft, stft
from ..weight_utils import load_converted_weights
from .convert import convert_state_dict, load_v2_state_dict
from .model import BandSplitModule, BanditCore, OverlappingMaskEstimationModule, SeqBandModellingModule

CHUNK_SIZE_SECONDS = 8.0
HOP_SIZE_SECONDS = 1.0


def _placeholder_freq_weights(band_specs: list[tuple[int, int]]) -> list[mx.array]:
    return [mx.zeros((fend - fstart,), dtype=mx.float32) for fstart, fend in band_specs]


def _build_core() -> tuple[BanditCore, list[tuple[int, int]]]:
    from ..._v2.runtime import V2_KWARGS
    from ..._v2.utils import MusicalBandsplitSpecification

    assert V2_KWARGS["band_type"] == "musical"
    assert V2_KWARGS["rnn_type"] == "GRU"

    n_fft, fs, n_bands = V2_KWARGS["n_fft"], V2_KWARGS["fs"], V2_KWARGS["n_bands"]
    emb_dim, mlp_dim, rnn_dim = V2_KWARGS["emb_dim"], V2_KWARGS["mlp_dim"], V2_KWARGS["rnn_dim"]
    n_freq = n_fft // 2 + 1

    band_specs = MusicalBandsplitSpecification(nfft=n_fft, fs=fs, n_bands=n_bands).get_band_specs()

    band_split = BandSplitModule(band_specs=band_specs, emb_dim=emb_dim, in_channels=V2_KWARGS["in_channels"])
    tf_model = SeqBandModellingModule(n_modules=V2_KWARGS["n_sqm_modules"], emb_dim=emb_dim, rnn_dim=rnn_dim, rnn_type="GRU")
    mask_estim = {
        stem: OverlappingMaskEstimationModule(
            band_specs=band_specs, emb_dim=emb_dim, mlp_dim=mlp_dim, in_channels=V2_KWARGS["in_channels"],
            n_freq=n_freq, freq_weights=_placeholder_freq_weights(band_specs),
        )
        for stem in V2_KWARGS["stems"]
    }
    core = BanditCore(band_split=band_split, tf_model=tf_model, mask_estim=mask_estim)
    return core, band_specs


def _reflect_pad_last_axis(array: mx.array, pad_left: int, pad_right: int) -> mx.array:
    """`mx.pad` has no `"reflect"` mode in MLX 0.31.2 -- round-trip through
    `numpy.pad`, verified byte-identical to `torch.nn.functional.pad(...,
    mode="reflect")` before being wired in here."""
    np_array = np.array(array)
    widths = [(0, 0)] * (np_array.ndim - 1) + [(pad_left, pad_right)]
    return mx.array(np.pad(np_array, widths, mode="reflect"))


class ChunkedInferenceHandlerMLX:
    """MLX mirror of `StandardTensorChunkedInferenceHandler`, numerically --
    not structurally (`F.unfold`/`F.fold` have no MLX equivalent; see this
    module's docstring)."""

    def __init__(self, chunk_size_seconds: float, hop_size_seconds: float, fs: int) -> None:
        import torch

        self.fs = fs
        self.chunk_size_samples = int(chunk_size_seconds * fs)
        self.hop_size_samples = int(hop_size_seconds * fs)
        self.overlap_samples = self.chunk_size_samples - self.hop_size_samples
        self.front_pad_samples = 2 * self.overlap_samples
        scaler = self.chunk_size_samples / (2 * self.hop_size_samples)
        window = torch.hann_window(self.chunk_size_samples, periodic=True).numpy()
        self.scaled_window = mx.array((window / scaler).astype(np.float32))

    def _n_chunks(self, n_samples: int) -> int:
        return int(math.ceil((n_samples + 2 * self.front_pad_samples - self.chunk_size_samples) / self.hop_size_samples)) + 1

    def _end_pad_samples(self, n_samples: int, n_chunks: int) -> int:
        return (n_chunks - 1) * self.hop_size_samples + self.chunk_size_samples - n_samples

    def _pad_and_chunk(self, mixture: mx.array) -> tuple[list[mx.array], int, int]:
        n_channels, n_samples = mixture.shape
        n_chunks = self._n_chunks(n_samples)
        end_pad = self._end_pad_samples(n_samples, n_chunks)
        padded_samples = n_samples + 2 * self.front_pad_samples + end_pad
        front = self.front_pad_samples

        if front >= n_samples:
            reflect = (n_samples - 1, n_samples - 1)
            remaining = front - (n_samples - 1)
            constant = (remaining, remaining + end_pad)
        elif front + end_pad >= n_samples:
            reflect = (front, n_samples - 1)
            remaining = front + end_pad - (n_samples - 1)
            constant = (0, remaining)
        else:
            reflect = (front, front + end_pad)
            constant = None

        padded = _reflect_pad_last_axis(mixture, *reflect)
        if constant is not None:
            padded = mx.pad(padded, [(0, 0), constant])

        chunks = [
            padded[:, c * self.hop_size_samples : c * self.hop_size_samples + self.chunk_size_samples]
            for c in range(n_chunks)
        ]
        return chunks, n_samples, padded_samples

    def _fold(self, chunk_outputs: list[mx.array], n_samples: int, padded_samples: int, n_channels: int) -> mx.array:
        total = mx.zeros((n_channels, padded_samples), dtype=mx.float32)
        window = self.scaled_window[None, :]
        for c, chunk in enumerate(chunk_outputs):
            start = c * self.hop_size_samples
            windowed = chunk * window
            pad_right = padded_samples - start - self.chunk_size_samples
            total = total + mx.pad(windowed, [(0, 0), (start, pad_right)])
        return total[:, self.front_pad_samples : self.front_pad_samples + n_samples]

    def forward(self, mixture: mx.array, core: BanditCore, transform) -> dict[str, mx.array]:
        chunks, n_samples, padded_samples = self._pad_and_chunk(mixture)
        n_channels = mixture.shape[0]
        stems = list(core.mask_estim.keys())
        per_stem_chunks: dict[str, list[mx.array]] = {stem: [] for stem in stems}

        for chunk in chunks:
            waveform = chunk.reshape(n_channels, 1, self.chunk_size_samples)
            spectrogram = stft(transform, waveform)
            estimates = core(spectrogram)
            for stem in stems:
                wav = istft(transform, estimates[stem], self.chunk_size_samples)
                mx.eval(wav)
                per_stem_chunks[stem].append(wav.reshape(n_channels, self.chunk_size_samples))

        return {
            stem: self._fold(per_stem_chunks[stem], n_samples, padded_samples, n_channels)
            for stem in stems
        }


class V2RuntimeMLX:
    """Resident v2 MLX model with the same 8s/1s chunked overlap-add handler
    `V2Runtime` uses (`inference_batch_size=1`: every chunk is processed on
    its own, matching the batch shape the recorded upstream fixture uses)."""

    def __init__(self, checkpoint: Path) -> None:
        from ..._v2.runtime import V2_KWARGS

        self.core, self._band_specs = _build_core()
        state = load_v2_state_dict(checkpoint)
        weights = convert_state_dict(state, rnn_type="GRU")
        load_converted_weights(self.core, weights, label="v2")
        self.core.eval()

        self._transform = build_transform(
            SpectralConfig(V2_KWARGS["n_fft"], V2_KWARGS["win_length"], V2_KWARGS["hop_length"], pad_mode=V2_KWARGS["pad_mode"])
        )
        self._handler = ChunkedInferenceHandlerMLX(CHUNK_SIZE_SECONDS, HOP_SIZE_SECONDS, fs=V2_KWARGS["fs"])
        self._fs = V2_KWARGS["fs"]

    def infer(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        if sample_rate != self._fs:
            raise ValueError("v2 MLX runtime currently requires native 48000 Hz input; resampling parity is not yet recorded")
        mixture = mx.array(np.ascontiguousarray(audio, dtype=np.float32))
        output = self._handler.forward(mixture, self.core, self._transform)
        mx.eval(output)
        return {
            "speech": np.array(output["speech"]),
            "music": np.array(output["music"]),
            "effects": np.array(output["sfx"]),
        }
