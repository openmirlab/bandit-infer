"""Guarded command placeholder for untouched v2 golden capture.

Reads: caller fixture argument only until the independent v2 graph is ported.
"""

from __future__ import annotations
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
args = parser.parse_args()
raise SystemExit("v2 golden capture is blocked: faithful v2 backend and official checkpoint are not yet verified")
