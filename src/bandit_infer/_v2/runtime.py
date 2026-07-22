"""Load and execute the one official Bandit v2 Mus64 graph.

The state-dict adapter removes only Lightning's outer ``model.`` namespace;
all graph names below remain those of the upstream v2 implementation. Reads:
the v2 graph, v2 chunk handler, torch checkpoint bytes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .bandit import Bandit
from .inference import StandardTensorChunkedInferenceHandler


V2_KWARGS = {
    "in_channels": 1,
    "stems": ["speech", "music", "sfx"],
    "band_type": "musical",
    "n_bands": 64,
    "normalize_channel_independently": False,
    "treat_channel_as_feature": True,
    "n_sqm_modules": 8,
    "emb_dim": 128,
    "rnn_dim": 256,
    "bidirectional": True,
    "rnn_type": "GRU",
    "mlp_dim": 512,
    "hidden_activation": "Tanh",
    "hidden_activation_kwargs": None,
    "complex_mask": True,
    "use_freq_weights": True,
    "n_fft": 2048,
    "win_length": 2048,
    "hop_length": 512,
    "window_fn": "hann_window",
    "wkwargs": None,
    "power": None,
    "center": True,
    "normalized": True,
    "pad_mode": "reflect",
    "onesided": True,
    "fs": 48000,
}


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    raw = loaded.get("state_dict", loaded)
    prefixed = {key.removeprefix("model."): value for key, value in raw.items() if key.startswith("model.")}
    state = prefixed or raw
    if not state:
        raise ValueError("v2 checkpoint has no Lightning model.* state_dict entries")
    return state


class V2Runtime:
    """Resident v2 model with upstream's 8 s / 1 s tensor overlap-add handler."""

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        self.device = device
        self.model = Bandit(**V2_KWARGS)
        missing, unexpected = self.model.load_state_dict(_state_dict(checkpoint), strict=False)
        material_missing = [key for key in missing if not key.endswith(".window")]
        if material_missing or unexpected:
            raise ValueError(f"v2 state-dict mismatch; missing={material_missing[:5]}, unexpected={unexpected[:5]}")
        self.model.to(device).eval()
        # One chunk per forward is numerically independent and avoids an OOM on
        # common GPUs; it is also the batch shape used by the recorded fixture.
        self.handler = StandardTensorChunkedInferenceHandler(
            chunk_size_seconds=8.0, hop_size_seconds=1.0, inference_batch_size=1, fs=48000
        ).to(device).eval()

    def infer(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        if sample_rate != 48000:
            raise ValueError("v2 runtime currently requires native 48000 Hz input; resampling parity is not yet recorded")
        mixture = torch.from_numpy(np.ascontiguousarray(audio)).to(self.device)[None, :, :]
        with torch.inference_mode():
            output = self.handler(mixture, self.model)["estimates"]
        return {
            "speech": output["speech"]["audio"][0].cpu().numpy(),
            "music": output["music"]["audio"][0].cpu().numpy(),
            "effects": output["sfx"]["audio"][0].cpu().numpy(),
        }
