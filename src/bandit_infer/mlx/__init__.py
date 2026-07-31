"""Optional MLX compute path (Apple Silicon) -- never imported by `bandit_infer`
itself. Only `api.py::load_mlx_runtime_backend` imports from here, and only
when `device="mlx"` is actually requested.

`v1/` and `v2/` are independent, from-scratch model graphs (no upstream MLX
implementation of Bandit exists to vendor from); `rfft_guard.py`,
`spectral.py`, and `weight_utils.py` are the generic, architecture-agnostic
helpers shared between them.

Reads: nothing at import time -- submodules import mlx.core/mlx.nn lazily.
"""
