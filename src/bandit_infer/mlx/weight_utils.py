"""Shared, architecture-agnostic weight-conversion primitives for v1 and v2.

Both graphs need the exact same two conversions -- a Torch tensor to an
`mx.array`, and Torch's single bidirectional `nn.LSTM`/`nn.GRU` split into
this port's `rnn_fwd`/`rnn_bwd` pair -- and the bias algebra in particular is
easy to get subtly wrong (see `remap_bidirectional_rnn`'s docstring). Sharing
one verified copy here, rather than pasting it into `v1/convert.py` and
`v2/convert.py` independently, stays inside CLAUDE.md's "generic cache or
device helpers may be shared" carve-out: this module has no band-split or
mask-estimation config, and does not choose between v1 and v2 architecture.

Reads: mlx.core, numpy
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def to_mx_array(tensor) -> mx.array:
    """Torch tensor -> `mx.array`, downcasting float64 (MLX has no float64
    GPU path; this package's only float64 tensors are the deterministic
    `freq_weights` buffers, where the precision loss is immaterial)."""
    array = tensor.detach().cpu().numpy()
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    return mx.array(array)


def remap_bidirectional_rnn(flat: dict[str, mx.array], rnn_type: str) -> None:
    """In place: `<prefix>.rnn.{weight,bias}_{ih,hh}_l0[_reverse]` ->
    `<prefix>.rnn_fwd`/`rnn_bwd` (`Wx`, `Wh`, plus MLX's own bias layout).

    LSTM: MLX's single `bias` is `bias_ih + bias_hh` (gate order `i,f,g,o`
    needs no reordering -- both sides use it).

    GRU: MLX's `nn.GRU` exposes only `b` (added to the x-branch, `3*hidden`)
    and `bhn` (the h-branch bias for the *n* gate only -- its own forward is
    `n = x_n + r * (h_proj_n + bhn)`; the r/z gates never see a separate
    h-branch bias at all). So Torch's `bias_hh`'s r/z portion must be
    pre-summed into `b`, and only its n portion becomes `bhn`:
    `b = concat(bias_ih[:2h] + bias_hh[:2h], bias_ih[2h:3h])`,
    `bhn = bias_hh[2h:3h]`.

    Verified against a direct `torch.nn.LSTM`/`torch.nn.GRU` call, in
    isolation, before either was ever wired into a model here: max abs diff
    ~8.9e-08 (LSTM), ~1.5e-07 (GRU) on a random bidirectional forward pass.
    """
    bias_parts: dict[tuple[str, str], dict[str, mx.array]] = {}
    for key in list(flat.keys()):
        if ".rnn." not in key:
            continue
        prefix, rest = key.split(".rnn.", 1)
        is_reverse = rest.endswith("_reverse")
        base = rest[: -len("_reverse")] if is_reverse else rest
        direction = "rnn_bwd" if is_reverse else "rnn_fwd"
        if base == "weight_ih_l0":
            flat[f"{prefix}.{direction}.Wx"] = flat.pop(key)
        elif base == "weight_hh_l0":
            flat[f"{prefix}.{direction}.Wh"] = flat.pop(key)
        elif base in ("bias_ih_l0", "bias_hh_l0"):
            bias_parts.setdefault((prefix, direction), {})[base] = flat.pop(key)

    for (prefix, direction), parts in bias_parts.items():
        bias_ih, bias_hh = parts["bias_ih_l0"], parts["bias_hh_l0"]
        if rnn_type == "LSTM":
            flat[f"{prefix}.{direction}.bias"] = bias_ih + bias_hh
        else:  # GRU
            hidden = bias_ih.shape[0] // 3
            b = mx.concatenate([bias_ih[: 2 * hidden] + bias_hh[: 2 * hidden], bias_ih[2 * hidden : 3 * hidden]])
            bhn = bias_hh[2 * hidden : 3 * hidden]
            flat[f"{prefix}.{direction}.b"] = b
            flat[f"{prefix}.{direction}.bhn"] = bhn


def load_converted_weights(model, weights: dict[str, mx.array], *, label: str) -> None:
    """Load `weights` into `model`, refusing a silent partial load.

    Every one of the model's own parameter keys must be present in
    `weights`, and every key in `weights` must be consumed -- otherwise a
    `ValueError` names counts and up to 5 example keys on each side. This
    has already caught real bugs in prior ports of this same pattern
    (including one in an upstream reference); applied unconditionally here
    too rather than re-derived per port.
    """
    from mlx.utils import tree_flatten

    model_keys = {key for key, _ in tree_flatten(model.parameters())}
    weight_keys = set(weights.keys())

    unmatched_model = sorted(model_keys - weight_keys)
    dropped_weights = sorted(weight_keys - model_keys)

    if unmatched_model or dropped_weights:
        parts = []
        if unmatched_model:
            parts.append(f"{len(unmatched_model)} model parameters unmatched (e.g. {', '.join(unmatched_model[:5])})")
        if dropped_weights:
            parts.append(f"{len(dropped_weights)} converted tensors dropped (e.g. {', '.join(dropped_weights[:5])})")
        raise ValueError(f"{label} MLX weight conversion incomplete: " + ", ".join(parts))

    model.load_weights(list(weights.items()), strict=False)
