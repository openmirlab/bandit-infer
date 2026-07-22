"""Install a built wheel in a throwaway venv and touch public package data.

This exercises the wheel-from-sdist artifact rather than the working tree.
Reads: dist/*.whl and uv.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    wheel = max((ROOT / "dist").glob("*.whl"), key=lambda item: item.stat().st_mtime)
    with tempfile.TemporaryDirectory(prefix="bandit-infer-wheel-") as temporary:
        environment = Path(temporary) / "venv"
        subprocess.run(["uv", "venv", str(environment)], check=True)
        python = environment / "bin" / "python"
        subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel)], check=True)
        subprocess.run([str(python), "-c", "import bandit_infer; from bandit_infer.checkpoints import load_manifest; assert bandit_infer.BanditSession; assert len(load_manifest()[1]) == 28"], check=True)
    print(f"Wheel verification passed: {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
