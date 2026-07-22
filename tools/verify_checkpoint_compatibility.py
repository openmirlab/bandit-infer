"""Guarded real-checkpoint compatibility gate for every registered graph family.

No filename-only compatibility claim is permitted. Reads: --real argument.
"""

from __future__ import annotations
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--real", action="store_true")
args = parser.parse_args()
raise SystemExit("real compatibility is blocked: official SHA-256 and independent backend graphs are not yet verified")
