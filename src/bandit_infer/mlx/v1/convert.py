"""PyTorch -> MLX weight conversion for the v1 MLX graph.

Because `model.py`'s module tree mirrors `_v1`'s Torch attribute names 1:1
(verified against the real `dnr-3s-mus64-l1snr.ckpt` `state_dict`, not
assumed), conversion here is close to a rename, not a redesign. Three
exceptions:

1. **`.window` buffers are dropped.** `stft.window`/`istft.window` are
   deterministic (Torch's `hann_window`) and `mlx/spectral.py` recomputes
   the same window bit-identically at transform-construction time rather
   than loading it -- mirroring `_v1/runtime.py`'s own `material_missing`
   exception for keys ending in `.window`.
2. **Bidirectional RNN weights are split** via `..weight_utils.
   remap_bidirectional_rnn` (shared with the v2 port; see its docstring for
   the verified bias algebra).
3. **`freq_weights/{i}` -> `freq_weights.{i}`.** Torch's buffer names use
   `/`; MLX attribute paths use `.`. Values are loaded from the checkpoint
   unchanged (not recomputed from the analytic filterbank formula), matching
   `_v1/runtime.py`'s own load-not-recompute choice.

The auditing load gate itself (`load_converted_weights`) is shared with v2
in `..weight_utils` -- it is architecture-agnostic (diffs parameter key sets).

Reads: mlx.core, torch (checkpoint bytes only), ..weight_utils
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ..weight_utils import remap_bidirectional_rnn, to_mx_array


def load_v1_state_dict(path: Path) -> dict[str, object]:
    """Same Lightning `model.*` unwrap `_v1/runtime.py::_state_dict` uses."""
    import torch

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    raw = loaded.get("state_dict", loaded)
    state = {key.removeprefix("model."): value for key, value in raw.items() if key.startswith("model.")}
    if not state:
        raise ValueError("v1 checkpoint has no Lightning model.* state_dict entries")
    return state


def convert_core_state_dict(state: dict[str, object], *, core_prefix: str, rnn_type: str) -> dict[str, mx.array]:
    """`state` is the full (already Lightning-unwrapped) v1 `state_dict`;
    `core_prefix` selects and strips the one core this MLX model instance
    covers (`"bsrnn."` for a multi-mask model, `"bsrnn.<stem>."` for a
    single-mask model) so the remaining keys match `MultiCore`/`SingleCore`'s
    own attribute namespace directly.
    """
    flat: dict[str, mx.array] = {}
    for key, tensor in state.items():
        if key.endswith(".window"):
            continue
        if not key.startswith(core_prefix):
            continue
        rest = key[len(core_prefix) :].replace("freq_weights/", "freq_weights.")
        flat[rest] = to_mx_array(tensor)
    remap_bidirectional_rnn(flat, rnn_type)
    return flat
