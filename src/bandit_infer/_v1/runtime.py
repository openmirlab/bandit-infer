"""Independent Bandit v1 graph construction and direct tensor inference.

This uses the v1 wrapper/state-dict names, not the v2 model or converter.
Chunk/fader parity remains a separate v1 fixture gate; direct inference is
available for native-rate inputs that fit the caller's memory budget.
Reads: v1 wrapper, torch checkpoint bytes, and registry spec metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..checkpoints import CheckpointSpec
from .wrapper import MultiMaskMultiSourceBandSplitRNN, SingleMaskMultiSourceBandSplitRNN


_BAND = {
    "multi-mus48": ("musical", 48), "multi-mus64": ("musical", 64),
    "multi-bark48": ("bark", 48), "multi-bark64": ("bark", 64),
    "multi-erb48": ("erb", 48), "multi-erb64": ("erb", 64),
    "multi-mel48": ("dnr:mel", 48), "multi-mel64": ("dnr:mel", 64),
    "multi-tribark48": ("tribark", 48), "multi-tribark64": ("tribark", 64),
    "multi-vox7": ("dnr:vox7", None),
}


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    raw = loaded.get("state_dict", loaded)
    state = {key.removeprefix("model."): value for key, value in raw.items() if key.startswith("model.")}
    if not state:
        raise ValueError("v1 checkpoint has no Lightning model.* state_dict entries")
    return state


def _multi(spec: CheckpointSpec) -> torch.nn.Module:
    band_specs, n_bands = _BAND[spec.variant]
    return MultiMaskMultiSourceBandSplitRNN(
        in_channel=1, stems=list(spec.stems), band_specs=band_specs, n_bands=n_bands,
        fs=44100, n_sqm_modules=8, emb_dim=128, rnn_dim=256, bidirectional=True,
        rnn_type="GRU", mlp_dim=512, hidden_activation="Tanh", complex_mask=True,
        n_fft=2048, win_length=2048, hop_length=512, window_fn="hann_window",
        wkwargs=None, power=None, center=True, normalized=True, pad_mode="constant", onesided=True,
    )


def _single(spec: CheckpointSpec) -> torch.nn.Module:
    stem = spec.stems[0]
    modules, rnn_type = (8, "GRU") if spec.variant == "single-gru8-vox7" else (12, "LSTM")
    return SingleMaskMultiSourceBandSplitRNN(
        in_channel=1, band_specs_map=f"dnr:vox7:{stem}", fs=44100,
        n_sqm_modules=modules, emb_dim=128, rnn_dim=256, bidirectional=True,
        rnn_type=rnn_type, mlp_dim=512, hidden_activation="Tanh", complex_mask=True,
        n_fft=2048, win_length=2048, hop_length=512, window_fn="hann_window",
        wkwargs=None, power=None, center=True, normalized=True, pad_mode="constant", onesided=True,
    )


class V1Runtime:
    """Resident native-rate v1 graph; no v2 code or layers are used."""

    def __init__(self, spec: CheckpointSpec, checkpoint: Path, device: torch.device) -> None:
        self.spec, self.device = spec, device
        self.model = _single(spec) if spec.variant.startswith("single-") else _multi(spec)
        missing, unexpected = self.model.load_state_dict(_state_dict(checkpoint), strict=False)
        material_missing = [key for key in missing if not key.endswith(".window")]
        if material_missing or unexpected:
            raise ValueError(f"v1 state-dict mismatch; missing={material_missing[:5]}, unexpected={unexpected[:5]}")
        self.model.to(device).eval()

    def infer(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        if sample_rate != 44100:
            raise ValueError("v1 runtime currently requires native 44100 Hz input; resampling parity is not yet recorded")
        waveform = torch.from_numpy(np.ascontiguousarray(audio)).to(self.device)
        outputs: dict[str, list[np.ndarray]] = {stem: [] for stem in self.spec.stems}
        with torch.inference_mode():
            for channel in waveform:
                _, result = self.model({"audio": {"mixture": channel[None, None, :]}})
                for stem in self.spec.stems:
                    outputs[stem].append(result["audio"][stem][0, 0].cpu().numpy())
        return {stem: np.stack(parts) for stem, parts in outputs.items()}
