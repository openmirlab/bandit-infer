"""Bandit v2 MLX graph -- a from-scratch port of `_v2/{bandsplit,maskestim,tfmodel,bandit}.py`.

Deliberately duplicated rather than importing `mlx.v1.model`'s near-identical
classes: `_v1` and `_v2` never share a Torch model-layer base either (see
CLAUDE.md's "`_v1` and `_v2` own separate model graphs" rule, and
`_v2/tfmodel.py`/`_v2/maskestim.py` re-implementing rather than importing
`_v1`'s versions), so this port mirrors that choice instead of introducing a
cross-version dependency the Torch source itself does not have.

Two real numeric differences from `_v1/model.py`, each confirmed against
`_v2`'s Torch source (not assumed from the classes' similar names):

1. **`BandSplitModule`'s flatten order.** v1 flattens a band's `(reim,
   band_width)` pair (reim slower-varying); v2 permutes to
   `(batch, n_time, in_chan, n_freq)` *first* and flattens `(in_chan,
   band_width, reim)` instead (reim fastest-varying) -- see
   `_v2/bandsplit.py::BandSplitModule.forward`. Getting this wrong would
   still produce a same-shaped, plausible-looking, numerically wrong mask.
2. **`NormFC`'s only attribute is `combined`** (`_v2/bandsplit.py`'s local
   `norm`/`fc` variables are never assigned to `self`), vs. v1's separate
   `self.norm`/`self.fc` -- confirmed directly against the real
   `checkpoint-multi.ckpt` `state_dict` (`band_split.norm_fc_modules.0.
   combined.{0,1}.*`, no bare `.norm.`/`.fc.` keys at all).

`NormMLP` keeps v1's separate `norm`/`hidden`/`output` naming (not
`combined`): confirmed against the same checkpoint that `_v2/maskestim.py`'s
`combined` Sequential there is a genuine Torch shared-parameter alias of
`self.norm`/`self.hidden`/`self.output` (`torch.equal` byte-identical for
all three), so `convert.py` simply drops the `.combined.` duplicate keys
rather than this model defining a second copy of the same parameters.

`checkpoint_sequential(...)` calls throughout `_v2` are Torch's activation
gradient-checkpointing wrapper -- functionally a plain sequential call in
eval/no-grad inference, which is all this module implements.

Reads: mlx.core, mlx.nn
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# --------------------------------------------------------------------------- band split


class NormFC(nn.Module):
    """`combined = [LayerNorm, Linear]`, matching `_v2/bandsplit.py::NormFC`'s
    checkpoint shape exactly (see this module's docstring)."""

    def __init__(self, emb_dim: int, bandwidth: int, in_channels: int) -> None:
        super().__init__()
        reim = 2
        self.combined = [
            nn.LayerNorm(in_channels * bandwidth * reim),
            nn.Linear(bandwidth * reim * in_channels, emb_dim),
        ]

    def __call__(self, xb: mx.array) -> mx.array:
        xb = self.combined[0](xb)
        return self.combined[1](xb)


class BandSplitModule(nn.Module):
    """`(batch, in_chan, n_freq, n_time)` complex64 -> `(batch, n_bands, n_time, emb_dim)`.

    Flatten order is `(in_chan, band_width, reim)` -- see this module's
    docstring point (1). Bands are stacked (`mx.stack`), never scatter-written.
    """

    def __init__(self, band_specs: list[tuple[int, int]], emb_dim: int, in_channels: int) -> None:
        super().__init__()
        self.band_specs = band_specs
        self.norm_fc_modules = [
            NormFC(emb_dim=emb_dim, bandwidth=fend - fstart, in_channels=in_channels)
            for fstart, fend in band_specs
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, in_chan, n_freq, n_time) complex64
        x = mx.transpose(x, (0, 3, 1, 2))  # (batch, n_time, in_chan, n_freq)
        bands = []
        for (fstart, fend), nfm in zip(self.band_specs, self.norm_fc_modules):
            xb = x[:, :, :, fstart:fend]  # (batch, n_time, in_chan, band_width) complex
            xb = mx.stack([mx.real(xb), mx.imag(xb)], axis=-1)  # (..., band_width, 2)
            batch, n_time = xb.shape[0], xb.shape[1]
            xb = xb.reshape(batch, n_time, -1)  # (batch, n_time, in_chan*band_width*2)
            bands.append(nfm(xb))
        return mx.stack(bands, axis=1)  # (batch, n_bands, n_time, emb_dim)


# --------------------------------------------------------------------------- tf model


def _run_rnn(rnn_module, x: mx.array) -> mx.array:
    out = rnn_module(x)
    return out[0] if isinstance(out, tuple) else out


class ResidualRNN(nn.Module):
    """LayerNorm -> bidirectional RNN -> Linear, with a residual add.
    `rnn_fwd`/`rnn_bwd` replace Torch's single bidirectional `nn.LSTM`/
    `nn.GRU`; see `convert.py` for the weight-splitting algebra (shared
    logic with the v1 port, verified once, applied identically here)."""

    def __init__(self, emb_dim: int, rnn_dim: int, rnn_type: str = "GRU", bidirectional: bool = True) -> None:
        super().__init__()
        cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.norm = nn.LayerNorm(emb_dim)
        self.bidirectional = bidirectional
        self.rnn_fwd = cls(input_size=emb_dim, hidden_size=rnn_dim)
        self.rnn_bwd = cls(input_size=emb_dim, hidden_size=rnn_dim) if bidirectional else None
        self.fc = nn.Linear(rnn_dim * (2 if bidirectional else 1), emb_dim)

    def __call__(self, z: mx.array) -> mx.array:
        z0 = z
        z = self.norm(z)
        batch, n_uncrossed, n_across, emb_dim = z.shape
        flat = z.reshape(batch * n_uncrossed, n_across, emb_dim)

        fwd = _run_rnn(self.rnn_fwd, flat)
        if self.bidirectional:
            bwd = _run_rnn(self.rnn_bwd, flat[:, ::-1, :])
            bwd = bwd[:, ::-1, :]
            flat = mx.concatenate([fwd, bwd], axis=-1)
        else:
            flat = fwd

        z = flat.reshape(batch, n_uncrossed, n_across, -1)
        z = self.fc(z)
        return z + z0


class SeqBandModellingModule(nn.Module):
    """`2 * n_modules` `ResidualRNN`s, transposing dims 1/2 between calls --
    matches `_v2/tfmodel.py`'s non-`parallel_mode` `nn.Sequential(rnn,
    Transpose(1,2), ...)` path (the only path `Bandit` uses)."""

    def __init__(self, n_modules: int, emb_dim: int, rnn_dim: int, rnn_type: str = "GRU", bidirectional: bool = True) -> None:
        super().__init__()
        self.seqband = [
            ResidualRNN(emb_dim=emb_dim, rnn_dim=rnn_dim, rnn_type=rnn_type, bidirectional=bidirectional)
            for _ in range(2 * n_modules)
        ]

    def __call__(self, z: mx.array) -> mx.array:
        for sbm in self.seqband:
            z = sbm(z)
            z = mx.swapaxes(z, 1, 2)
        return z


# --------------------------------------------------------------------------- mask estimation


class NormMLP(nn.Module):
    """LayerNorm -> Linear+Tanh -> Linear+GLU -> complex mask, one band's worth.
    `hidden`/`output` mirror `_v1/model.py::NormMLP` exactly (`_v2`'s
    `combined` Sequential is a verified duplicate -- see this module's
    docstring)."""

    def __init__(self, emb_dim: int, mlp_dim: int, bandwidth: int, in_channels: int, complex_mask: bool = True) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.in_channels = in_channels
        self.complex_mask = complex_mask
        self.reim = 2 if complex_mask else 1
        self.norm = nn.LayerNorm(emb_dim)
        self.hidden = [nn.Linear(emb_dim, mlp_dim), nn.Tanh()]
        self.output = [nn.Linear(mlp_dim, bandwidth * in_channels * self.reim * 2), nn.GLU(axis=-1)]

    def _reshape_output(self, mb: mx.array) -> mx.array:
        batch, n_time, _ = mb.shape
        if self.complex_mask:
            mb = mb.reshape(batch, n_time, self.in_channels, self.bandwidth, self.reim)
            mb = mb[..., 0].astype(mx.complex64) + 1j * mb[..., 1].astype(mx.complex64)
        else:
            mb = mb.reshape(batch, n_time, self.in_channels, self.bandwidth)
        return mx.transpose(mb, (0, 2, 3, 1))  # (batch, in_channels, bandwidth, n_time)

    def __call__(self, qb: mx.array) -> mx.array:
        qb = self.norm(qb)
        qb = self.hidden[0](qb)
        qb = self.hidden[1](qb)
        mb = self.output[0](qb)
        mb = self.output[1](mb)
        return self._reshape_output(mb)


class OverlappingMaskEstimationModule(nn.Module):
    """Every v2 registry checkpoint uses `band_type="musical"` (always
    overlapping) -- `_v2/bandit.py::instantiate_bandsplit` asserts
    `band_type == "musical"`, so this is the only mask estimator v2 needs
    (unlike v1, which also ships a non-overlapping path for `dnr:vox7`)."""

    def __init__(
        self,
        band_specs: list[tuple[int, int]],
        emb_dim: int,
        mlp_dim: int,
        in_channels: int,
        n_freq: int,
        freq_weights: Optional[list[mx.array]] = None,
        complex_mask: bool = True,
    ) -> None:
        super().__init__()
        self.band_specs = band_specs
        self.in_channels = in_channels
        self.n_freq = n_freq
        self.norm_mlp = [
            NormMLP(emb_dim=emb_dim, mlp_dim=mlp_dim, bandwidth=fend - fstart, in_channels=in_channels, complex_mask=complex_mask)
            for fstart, fend in band_specs
        ]
        self.freq_weights = list(freq_weights) if freq_weights is not None else None

    def __call__(self, q: mx.array) -> mx.array:
        batch, n_time = q.shape[0], q.shape[2]
        total = mx.zeros((batch, self.in_channels, self.n_freq, n_time), dtype=mx.complex64)
        for i, ((fstart, fend), nmlp) in enumerate(zip(self.band_specs, self.norm_mlp)):
            mask = nmlp(q[:, i, :, :])
            if self.freq_weights is not None:
                mask = mask * self.freq_weights[i][None, None, :, None].astype(mask.dtype)
            padded = mx.pad(mask, [(0, 0), (0, 0), (fstart, self.n_freq - fend), (0, 0)])
            total = total + padded
        return total


# --------------------------------------------------------------------------- core


class BanditCore(nn.Module):
    """One `BandSplitModule` + `SeqBandModellingModule` feeding a per-stem
    dict of mask estimators -- mirrors `_v2/bandit.py::Bandit.separate`."""

    def __init__(self, band_split: BandSplitModule, tf_model: SeqBandModellingModule, mask_estim: dict[str, OverlappingMaskEstimationModule]) -> None:
        super().__init__()
        self.band_split = band_split
        self.tf_model = tf_model
        self.mask_estim = mask_estim

    def __call__(self, x: mx.array) -> dict[str, mx.array]:
        z = self.band_split(x)
        q = self.tf_model(z)
        return {stem: x * mem(q) for stem, mem in self.mask_estim.items()}
