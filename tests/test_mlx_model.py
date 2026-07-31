"""Offline MLX model tests: graph wiring, the weight-conversion audit gate,
and the bidirectional RNN weight-split algebra's own mechanics.

Skipped whole-file when the ``[mlx]`` extra is not installed. None of these
need a real checkpoint -- tiny synthetic band specs keep the graphs small
and fast, and the audit/RNN-algebra checks are exercised against direct,
model-independent comparisons, which is the strongest offline check
available. Real-checkpoint Torch-vs-MLX numeric parity, including the
silence fixtures, lives in ``test_mlx_parity.py`` (``realweights``,
deselected by default).
"""

from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_spectro")

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402


_BAND_SPECS = [(0, 3), (3, 7), (7, 12)]


def test_v1_multi_core_forward_shape() -> None:
    from bandit_infer.mlx.v1.model import BandSplitModule, MultiCore, OverlappingMaskEstimationModule, SeqBandModellingModule

    band_split = BandSplitModule(band_specs=_BAND_SPECS, emb_dim=8, in_channel=1)
    tf_model = SeqBandModellingModule(n_modules=1, emb_dim=8, rnn_dim=4, rnn_type="GRU")
    mask_estim = {
        stem: OverlappingMaskEstimationModule(
            band_specs=_BAND_SPECS, emb_dim=8, mlp_dim=16, in_channel=1, n_freq=12,
            freq_weights=[mx.ones((fend - fstart,)) for fstart, fend in _BAND_SPECS],
        )
        for stem in ("speech", "music", "effects")
    }
    core = MultiCore(band_split, tf_model, mask_estim)

    x = (mx.random.normal((1, 1, 12, 5)) + 1j * mx.random.normal((1, 1, 12, 5))).astype(mx.complex64)
    out = core(x)
    mx.eval(out)
    assert set(out) == {"speech", "music", "effects"}
    for stem_out in out.values():
        assert stem_out.shape == x.shape
        assert stem_out.dtype == mx.complex64


def test_v2_bandit_core_forward_shape() -> None:
    from bandit_infer.mlx.v2.model import BandSplitModule, BanditCore, OverlappingMaskEstimationModule, SeqBandModellingModule

    band_split = BandSplitModule(band_specs=_BAND_SPECS, emb_dim=8, in_channels=1)
    tf_model = SeqBandModellingModule(n_modules=1, emb_dim=8, rnn_dim=4, rnn_type="GRU")
    mask_estim = {
        stem: OverlappingMaskEstimationModule(
            band_specs=_BAND_SPECS, emb_dim=8, mlp_dim=16, in_channels=1, n_freq=12,
            freq_weights=[mx.ones((fend - fstart,)) for fstart, fend in _BAND_SPECS],
        )
        for stem in ("speech", "music", "sfx")
    }
    core = BanditCore(band_split, tf_model, mask_estim)

    x = (mx.random.normal((2, 1, 12, 5)) + 1j * mx.random.normal((2, 1, 12, 5))).astype(mx.complex64)
    out = core(x)
    mx.eval(out)
    assert set(out) == {"speech", "music", "sfx"}
    for stem_out in out.values():
        assert stem_out.shape == x.shape


def test_load_converted_weights_catches_a_missing_parameter() -> None:
    """The non-negotiable auditing gate: a real conversion bug (a dropped or
    misnamed key) must raise, not silently leave a layer at random init.
    """
    import mlx.nn as nn

    from bandit_infer.mlx.weight_utils import load_converted_weights

    model = nn.Linear(4, 4)
    full_weights = dict(tree_flatten(model.parameters()))
    assert set(full_weights) == {"weight", "bias"}

    # A complete, correctly-shaped weight set loads without complaint.
    load_converted_weights(model, full_weights, label="test")

    # Remove the fix: drop one key, simulating a conversion bug.
    partial_weights = {k: v for k, v in full_weights.items() if k != "bias"}
    with pytest.raises(ValueError, match="unmatched"):
        load_converted_weights(model, partial_weights, label="test")


def test_strict_false_alone_would_silently_accept_the_same_missing_parameter() -> None:
    """Demonstrates *why* the audit exists: `load_weights(..., strict=False)`
    on its own -- the naive path `load_converted_weights` replaces -- accepts
    the same incomplete weight set without error, leaving `bias` at whatever
    it was initialized to. This is the "remove the fix" half of the pair:
    with the audit (previous test) it raises; without it (this test), it
    does not.
    """
    import mlx.nn as nn

    model = nn.Linear(4, 4)
    before = mx.array(model.bias)
    full_weights = dict(tree_flatten(model.parameters()))
    partial_weights = {k: v for k, v in full_weights.items() if k != "bias"}

    model.load_weights(list(partial_weights.items()), strict=False)  # no error raised
    assert mx.array_equal(model.bias, before)  # bias silently left untouched


def test_remap_bidirectional_rnn_matches_torch_lstm_and_gru() -> None:
    """The weight-split algebra (`Wx`/`Wh`/bias for `rnn_fwd`/`rnn_bwd`) as a
    permanent regression guard, not just the throwaway script it was first
    verified with. Builds a real bidirectional `torch.nn.LSTM`/`nn.GRU`,
    converts its state dict, runs both the Torch and MLX layers on the same
    input, and compares.
    """
    torch = pytest.importorskip("torch")
    import numpy as np

    from bandit_infer.mlx.weight_utils import remap_bidirectional_rnn, to_mx_array

    batch, seq, in_dim, hidden = 2, 5, 6, 4
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((batch, seq, in_dim)).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    x_mx = mx.array(x_np)

    import mlx.nn as nn

    for rnn_type, torch_cls, mlx_cls in (("LSTM", torch.nn.LSTM, nn.LSTM), ("GRU", torch.nn.GRU, nn.GRU)):
        torch_rnn = torch_cls(input_size=in_dim, hidden_size=hidden, num_layers=1, batch_first=True, bidirectional=True)
        with torch.no_grad():
            torch_out, _ = torch_rnn(x_torch)

        # A leading prefix before ".rnn." matters: real keys always look like
        # "tf_model.seqband.0.rnn.weight_ih_l0", never a bare "rnn.*".
        flat = {f"seqband.0.rnn.{name}": to_mx_array(value) for name, value in torch_rnn.state_dict().items()}
        remap_bidirectional_rnn(flat, rnn_type)
        flat = {key.replace("seqband.0.", ""): value for key, value in flat.items()}

        fwd = mlx_cls(input_size=in_dim, hidden_size=hidden)
        bwd = mlx_cls(input_size=in_dim, hidden_size=hidden)
        fwd.update({"Wx": flat["rnn_fwd.Wx"], "Wh": flat["rnn_fwd.Wh"], **({"bias": flat["rnn_fwd.bias"]} if rnn_type == "LSTM" else {"b": flat["rnn_fwd.b"], "bhn": flat["rnn_fwd.bhn"]})})
        bwd.update({"Wx": flat["rnn_bwd.Wx"], "Wh": flat["rnn_bwd.Wh"], **({"bias": flat["rnn_bwd.bias"]} if rnn_type == "LSTM" else {"b": flat["rnn_bwd.b"], "bhn": flat["rnn_bwd.bhn"]})})

        fwd_out = fwd(x_mx)
        fwd_out = fwd_out[0] if isinstance(fwd_out, tuple) else fwd_out
        bwd_out = bwd(x_mx[:, ::-1, :])
        bwd_out = bwd_out[0] if isinstance(bwd_out, tuple) else bwd_out
        bwd_out = bwd_out[:, ::-1, :]
        mlx_out = mx.concatenate([fwd_out, bwd_out], axis=-1)
        mx.eval(mlx_out)

        max_abs = float(np.abs(np.array(mlx_out) - torch_out.numpy()).max())
        assert max_abs < 1e-5, f"{rnn_type}: max abs diff {max_abs:.3e}"
