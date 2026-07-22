"""Minimal CLI entry point that exposes catalog facts without loading weights.

The command remains intentionally small while numerical backends are under
parity verification. Reads: checkpoints.
"""

from __future__ import annotations

import argparse

from .checkpoints import get_spec, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(prog="bandit-infer")
    parser.add_argument("--model")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        _, specs = load_manifest()
        print("\n".join(specs))
        return 0
    print(get_spec(args.model))
    return 0
