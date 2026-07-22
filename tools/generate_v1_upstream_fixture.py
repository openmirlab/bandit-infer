"""Guarded command placeholder for untouched v1 golden capture.

It must be implemented only beside the exact upstream environment and official
checkpoint; fabricating a fixture from this package would defeat the A/B gate.
Reads: caller fixture argument.
"""

from __future__ import annotations
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
args = parser.parse_args()
raise SystemExit("v1 golden capture is blocked: faithful v1 backend and official checkpoint are not yet verified")
