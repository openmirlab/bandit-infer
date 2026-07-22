"""Capture a deterministic fixture with untouched Bandit v1 source.

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
parser.add_argument("--python", default=sys.executable, help="interpreter with v1 upstream dependencies")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

if not (args.upstream_root / "core/model/bsrnn/wrapper.py").is_file():
    raise SystemExit("--upstream-root must be an untouched Bandit v1 checkout")
if not args.checkpoint.is_file():
    raise SystemExit(f"checkpoint is absent: {args.checkpoint}")

child = r'''
import sys
from pathlib import Path
import numpy as np
import torch
from core.model.bsrnn.wrapper import MultiMaskMultiSourceBandSplitRNN

fixture, checkpoint = map(Path, sys.argv[1:3])
device = sys.argv[3]
model = MultiMaskMultiSourceBandSplitRNN(
    in_channel=1, stems=["speech", "music", "effects"], band_specs="musical", n_bands=64,
    fs=44100, n_sqm_modules=8, emb_dim=128, rnn_dim=256, bidirectional=True,
    rnn_type="GRU", mlp_dim=512, hidden_activation="Tanh", complex_mask=True,
    n_fft=2048, win_length=2048, hop_length=512, window_fn="hann_window",
    wkwargs=None, power=None, center=True, normalized=True, pad_mode="constant", onesided=True,
)
loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
raw = loaded.get("state_dict", loaded)
state = {key.removeprefix("model."): value for key, value in raw.items() if key.startswith("model.")}
model.load_state_dict(state, strict=True)
model.to(device).eval()
samples = np.arange(4096, dtype=np.float32) / 44100
audio = (0.02 * np.sin(2 * np.pi * 440 * samples) + 0.01 * np.sin(2 * np.pi * 880 * samples))[None]
with torch.inference_mode():
    _, output = model({"audio": {"mixture": torch.from_numpy(audio).to(device)[:, None, :]}})
fixture.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    fixture,
    audio=audio,
    torch_version=np.array(torch.__version__),
    device=np.array(device),
    **{stem: output["audio"][stem][0].cpu().numpy() for stem in ("speech", "music", "effects")},
)
'''
environment = os.environ | {"PYTHONPATH": str(args.upstream_root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
subprocess.run([args.python, "-c", child, str(args.fixture), str(args.checkpoint), args.device], env=environment, check=True)
print(f"wrote untouched v1 Music64 fixture: {args.fixture}")
