"""Bandit v1 MLX runtime -- builds `model.py` graphs, strict-loads converted
checkpoint weights, and runs the same per-channel direct inference
`_v1/runtime.py::V1Runtime` does.

Support is exactly co-extensive with `_v1/runtime.py`'s own `_multi`/
`_single` dispatch -- reused directly, not redefined, from
`_v1.runtime._BAND` and `_v1.wrapper.get_band_specs`/`get_band_specs_map`
(pure Python band-index arithmetic; no Torch tensor graph is built or run)
so band boundaries cannot drift between the Torch and MLX ports. Every one
of the 21 v1 registry `variant` strings in `config/checkpoints.toml` maps
into `_multi`/`_single`'s RNN-based `SeqBandModellingModule` core --
confirmed by reading `_BAND`'s keys and `_single`'s
`variant.startswith("single-")` check against the manifest, not assumed --
so `TransformerTimeFreqModule`/`ConvolutionalTimeFreqModule` are out of
scope here exactly as they are in `_v1/runtime.py` (see `mlx/rfft_guard.py`'s
module docstring for the same point about the rfft-guard susceptibility).

Only `v1-mus64-l1snr` (multi, overlapping bands, GRU) has a real checkpoint
to measure parity against -- the other 20 v1 entries have no independently
verified SHA-256 in `config/checkpoints.toml` and so cannot load through
*either* backend (`checkpoints.resolve_checkpoint` refuses them before any
runtime is reached, identically for Torch and MLX). This module still
builds `SingleCore` for the `single-*` variants, matching `_v1/runtime.py`'s
own equally-unverified structural support -- not additional MLX-specific risk.

Reads: mlx.core, numpy, .model, .convert, .._v1.wrapper (get_band_specs,
get_band_specs_map), .._v1.runtime (_BAND), ..checkpoints (CheckpointSpec),
..spectral
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from ...checkpoints import CheckpointSpec
from ..spectral import SpectralConfig, build_transform, istft, stft
from ..weight_utils import load_converted_weights
from .convert import convert_core_state_dict, load_v1_state_dict
from .model import (
    BandSplitModule,
    MaskEstimationModule,
    MultiCore,
    OverlappingMaskEstimationModule,
    SeqBandModellingModule,
    SingleCore,
)

EMB_DIM = 128
RNN_DIM = 256
MLP_DIM = 512
N_SQM_MODULES_MULTI = 8
N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = 512
FS = 44100
N_FREQ = N_FFT // 2 + 1


def _placeholder_freq_weights(band_specs: list[tuple[int, int]]) -> list[mx.array]:
    """Correctly-shaped zero placeholders; `load_converted_weights` overwrites
    every entry from the checkpoint's own `freq_weights/{i}` buffers -- shape
    must match so `load_weights` can assign in place, values do not matter."""
    return [mx.zeros((fend - fstart,), dtype=mx.float32) for fstart, fend in band_specs]


def _build_multi(spec: CheckpointSpec) -> tuple[MultiCore, str, str]:
    from ..._v1.runtime import _BAND
    from ..._v1.wrapper import get_band_specs

    band_key, n_bands = _BAND[spec.variant]
    band_specs, _freq_weights, overlapping = get_band_specs(band_key, N_FFT, FS, n_bands)

    band_split = BandSplitModule(band_specs=band_specs, emb_dim=EMB_DIM, in_channel=1)
    tf_model = SeqBandModellingModule(n_modules=N_SQM_MODULES_MULTI, emb_dim=EMB_DIM, rnn_dim=RNN_DIM, rnn_type="GRU")
    mask_estim: dict[str, object] = {}
    for stem in spec.stems:
        if overlapping:
            mask_estim[stem] = OverlappingMaskEstimationModule(
                band_specs=band_specs, emb_dim=EMB_DIM, mlp_dim=MLP_DIM, in_channel=1,
                n_freq=N_FREQ, freq_weights=_placeholder_freq_weights(band_specs),
            )
        else:
            mask_estim[stem] = MaskEstimationModule(band_specs=band_specs, emb_dim=EMB_DIM, mlp_dim=MLP_DIM, in_channel=1)
    core = MultiCore(band_split=band_split, tf_model=tf_model, mask_estim=mask_estim)
    return core, "GRU", "bsrnn."


def _build_single(spec: CheckpointSpec) -> tuple[SingleCore, str, str]:
    from ..._v1.wrapper import get_band_specs_map

    stem = spec.stems[0]
    n_modules, rnn_type = (8, "GRU") if spec.variant == "single-gru8-vox7" else (12, "LSTM")
    band_specs_map, _freq_weights, _overlapping = get_band_specs_map(f"dnr:vox7:{stem}", N_FFT, FS, n_bands=None)
    band_specs = band_specs_map[stem]

    band_split = BandSplitModule(band_specs=band_specs, emb_dim=EMB_DIM, in_channel=1)
    tf_model = SeqBandModellingModule(n_modules=n_modules, emb_dim=EMB_DIM, rnn_dim=RNN_DIM, rnn_type=rnn_type)
    mask_estim = MaskEstimationModule(band_specs=band_specs, emb_dim=EMB_DIM, mlp_dim=MLP_DIM, in_channel=1)
    core = SingleCore(band_split=band_split, tf_model=tf_model, mask_estim=mask_estim)
    return core, rnn_type, f"bsrnn.{stem}."


class V1RuntimeMLX:
    """Resident native-rate v1 MLX graph; mirrors `V1Runtime`'s per-channel loop."""

    def __init__(self, spec: CheckpointSpec, checkpoint: Path) -> None:
        self.spec = spec
        self._is_single = spec.variant.startswith("single-")
        self.core, rnn_type, core_prefix = _build_single(spec) if self._is_single else _build_multi(spec)

        state = load_v1_state_dict(checkpoint)
        weights = convert_core_state_dict(state, core_prefix=core_prefix, rnn_type=rnn_type)
        load_converted_weights(self.core, weights, label="v1")
        self.core.eval()

        self._transform = build_transform(SpectralConfig(N_FFT, WIN_LENGTH, HOP_LENGTH, pad_mode="constant"))

    def infer(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        if sample_rate != 44100:
            raise ValueError("v1 MLX runtime currently requires native 44100 Hz input; resampling parity is not yet recorded")
        outputs: dict[str, list[np.ndarray]] = {stem: [] for stem in self.spec.stems}
        for channel in audio:
            waveform = mx.array(np.ascontiguousarray(channel, dtype=np.float32))[None, None, :]
            length = waveform.shape[-1]
            spectrogram = stft(self._transform, waveform)  # (1, 1, F, T) complex64

            if self._is_single:
                estimate = self.core(spectrogram)
                wav = istft(self._transform, estimate, length)
                mx.eval(wav)
                outputs[self.spec.stems[0]].append(np.array(wav)[0, 0])
            else:
                result = self.core(spectrogram)
                for stem in self.spec.stems:
                    wav = istft(self._transform, result[stem], length)
                    mx.eval(wav)
                    outputs[stem].append(np.array(wav)[0, 0])
        return {stem: np.stack(parts) for stem, parts in outputs.items()}
