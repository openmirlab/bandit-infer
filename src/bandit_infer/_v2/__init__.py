"""Bandit v2 namespace reserved for its faithful, independent model graph.

No v1 model layer is imported here. `runtime.py::V2Runtime` is a real,
checkpoint-verified graph (v2-multi has an independently computed SHA-256
and passes stem-by-stem parity against untouched upstream). The six other
registry entries (v2-cmn, v2-deu, v2-eng, v2-fao, v2-fra, v2-spa) share the
same graph but still lack a verified checkpoint hash.
Reads: nothing.
"""
