"""Fail if shipped source grows a training/evaluation/experiment surface.

The scan intentionally targets production source, not documentation that must
describe exclusions. Reads: src/bandit_infer Python files.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "bandit_infer"
FORBIDDEN = ("pytorch_lightning", "hydra", "trainer", "dataloader", "dataset", "wandb", "tensorboard", "evaluate(")


def main() -> int:
    hits = [f"{path}:{needle}" for path in ROOT.rglob("*.py") for needle in FORBIDDEN if needle in path.read_text(encoding="utf-8").lower()]
    if hits:
        raise SystemExit("inference-only violation:\n" + "\n".join(hits))
    print("Inference-only verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
