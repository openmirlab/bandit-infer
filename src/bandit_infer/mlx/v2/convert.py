"""PyTorch -> MLX weight conversion for the v2 MLX graph.

Because `model.py`'s module tree mirrors `_v2`'s Torch attribute names 1:1
(verified against the real `checkpoint-multi.ckpt` `state_dict`), conversion
here is close to a rename, not a redesign. Same three-part shape as the v1
port (`v1/convert.py`), plus one v2-specific exception:

1. **`.window` buffers are dropped** (see `v1/convert.py`'s docstring).
2. **Bidirectional RNN weights are split** via the shared
   `..weight_utils.remap_bidirectional_rnn`.
3. **`freq_weights/{i}` -> `freq_weights.{i}`.**
4. **`mask_estim.*.combined.*` keys are dropped.** `_v2/maskestim.py`'s
   `NormMLP.combined` is a Torch shared-parameter alias of the *same*
   `norm`/`hidden`/`output` submodules (`torch.equal` confirmed
   byte-identical against the real checkpoint -- see `model.py`'s module
   docstring), so this port defines `norm`/`hidden`/`output` only and the
   `.combined.` copy carries no new information. `band_split.*.combined.*`
   is **not** dropped: `_v2/bandsplit.py::NormFC` only ever assigns
   `self.combined` (its local `norm`/`fc` variables are never `self.`
   attributes), so that is the *only* representation of those weights,
   matching this port's `NormFC.combined` list attribute directly.
5. **`tf_model.seqband.{2*i}` -> `tf_model.seqband.{i}`.** `_v2/tfmodel.py`'s
   `SeqBandModellingModule` (non-`parallel_mode`) builds one flat
   `nn.Sequential` interleaving each `ResidualRNN` with a parameter-free
   `Transpose(1, 2)` module (`[rnn_0, transpose, rnn_1, transpose, ...]`),
   so Torch's `state_dict` only has keys at even indices (confirmed against
   the real checkpoint: `tf_model.seqband.1.*` has zero keys, `tf_model.
   seqband.16.*` is the 9th `ResidualRNN`). This port's `seqband` is a plain
   Python list of `ResidualRNN` only (`model.py`'s `SeqBandModellingModule`
   does the transpose as a bare `mx.swapaxes` call in `__call__`, not a
   module in the list), so indices must be halved during conversion.

Reads: mlx.core, torch (checkpoint bytes only), ..weight_utils
"""

from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx

from ..weight_utils import remap_bidirectional_rnn, to_mx_array

_SEQBAND_INDEX = re.compile(r"^(tf_model\.seqband\.)(\d+)(\..+)$")


def _halve_seqband_index(key: str) -> str:
    match = _SEQBAND_INDEX.match(key)
    if not match:
        return key
    prefix, index, rest = match.groups()
    return f"{prefix}{int(index) // 2}{rest}"


def load_v2_state_dict(path: Path) -> dict[str, object]:
    """Same Lightning `model.*` unwrap `_v2/runtime.py::_state_dict` uses."""
    import torch

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    raw = loaded.get("state_dict", loaded)
    prefixed = {key.removeprefix("model."): value for key, value in raw.items() if key.startswith("model.")}
    state = prefixed or raw
    if not state:
        raise ValueError("v2 checkpoint has no Lightning model.* state_dict entries")
    return state


def convert_state_dict(state: dict[str, object], *, rnn_type: str) -> dict[str, mx.array]:
    flat: dict[str, mx.array] = {}
    for key, tensor in state.items():
        if key.endswith(".window"):
            continue
        if key.startswith("mask_estim.") and ".combined." in key:
            continue  # verified duplicate of norm/hidden/output; see module docstring
        new_key = _halve_seqband_index(key.replace("freq_weights/", "freq_weights."))
        flat[new_key] = to_mx_array(tensor)
    remap_bidirectional_rnn(flat, rnn_type)
    return flat
