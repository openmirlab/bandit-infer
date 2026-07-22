"""Compare every public v2 output stem with an untouched-upstream fixture.

The fixture records its upstream torch/device environment. This command uses a
caller-local official checkpoint, never downloads or bundles it. Reads:
fixture, checkpoint, and the public BanditSession.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from bandit_infer import BanditSession


parser = argparse.ArgumentParser()
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

fixture = np.load(args.fixture)
digest = hashlib.file_digest(args.checkpoint.open("rb"), "sha256").hexdigest()
with BanditSession("v2-multi", device=args.device, checkpoint_path=args.checkpoint, checkpoint_sha256=digest) as session:
    estimates = session.infer(fixture["audio"], sample_rate=48000)
for stem in ("speech", "music", "effects"):
    np.testing.assert_allclose(estimates[stem], fixture[stem], rtol=3e-5, atol=3e-6)
print("v2 parity passed: every CASS stem is within rtol=3e-5, atol=3e-6")
