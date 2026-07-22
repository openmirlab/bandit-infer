"""Capture a deterministic fixture with untouched Bandit v2 source.

This is a maintenance-only bridge: its caller supplies a separately provisioned
upstream checkout, interpreter, and official local checkpoint. Reads no
package graph code and never downloads or writes weights.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--upstream-root", type=Path, required=True)
parser.add_argument("--python", default=sys.executable, help="interpreter with v2 upstream dependencies")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

if not (args.upstream_root / "src/models/bandit/bandit.py").is_file():
    raise SystemExit("--upstream-root must be an untouched Bandit v2 checkout")
if not args.checkpoint.is_file():
    raise SystemExit(f"checkpoint is absent: {args.checkpoint}")

child = r'''
import sys
from pathlib import Path
import numpy as np
import torch
from src.models.bandit.bandit import Bandit
from src.system.inference_handler import StandardTensorChunkedInferenceHandler

fixture, checkpoint = map(Path, sys.argv[1:3])
device = sys.argv[3]
model = Bandit(
    in_channels=1, stems=["speech", "music", "sfx"], band_type="musical", n_bands=64,
    normalize_channel_independently=False, treat_channel_as_feature=True,
    n_sqm_modules=8, emb_dim=128, rnn_dim=256, bidirectional=True, rnn_type="GRU",
    mlp_dim=512, hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True,
    use_freq_weights=True, n_fft=2048, win_length=2048, hop_length=512,
    window_fn="hann_window", wkwargs=None, power=None, center=True, normalized=True,
    pad_mode="reflect", onesided=True, fs=48000,
)
loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
raw = loaded.get("state_dict", loaded)
state = {key.removeprefix("model."): value for key, value in raw.items() if key.startswith("model.")}
model.load_state_dict(state, strict=True)
model.to(device).eval()
handler = StandardTensorChunkedInferenceHandler(chunk_size_seconds=8.0, hop_size_seconds=1.0, inference_batch_size=1, fs=48000).to(device).eval()
samples = np.arange(2048, dtype=np.float32) / 48000
audio = (0.02 * np.sin(2 * np.pi * 440 * samples) + 0.01 * np.sin(2 * np.pi * 880 * samples))[None]
with torch.inference_mode():
    output = handler(torch.from_numpy(audio).to(device)[None], model)["estimates"]
fixture.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    fixture,
    audio=audio,
    torch_version=np.array(torch.__version__),
    device=np.array(device),
    inference_batch_size=np.array(1),
    speech=output["speech"]["audio"][0].cpu().numpy(),
    music=output["music"]["audio"][0].cpu().numpy(),
    effects=output["sfx"]["audio"][0].cpu().numpy(),
)
'''
environment = os.environ | {"PYTHONPATH": str(args.upstream_root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
subprocess.run([args.python, "-c", child, str(args.fixture), str(args.checkpoint), args.device], env=environment, check=True)
print(f"wrote untouched v2 Multi fixture: {args.fixture}")
