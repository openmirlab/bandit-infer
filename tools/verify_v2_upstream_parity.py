"""Guarded command placeholder for the stem-by-stem v2 parity gate.

Reads: caller fixture argument only until the independent v2 graph is ported.
"""

from __future__ import annotations
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
args = parser.parse_args()
raise SystemExit("v2 parity is blocked: no untouched-upstream fixture or faithful v2 backend")
