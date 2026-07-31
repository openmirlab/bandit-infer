"""Bandit v1 MLX graph -- a from-scratch port of `_v1/{bandsplit,maskestim,tfmodel,core}.py`.

There is no upstream MLX implementation of Bandit to vendor from, so this is
written directly from this package's own Torch source (not a translation of
a third-party MLX port, unlike the sibling packages' vendored Roformer/Demucs/
MDX23C ports). Every class here mirrors its `_v1` Torch counterpart's
attribute names 1:1 (`band_split`, `tf_model.seqband`, `mask_estim.norm_mlp`,
...) precisely so `convert.py`'s weight conversion is close to a rename, not
a redesign -- verified directly against the real `dnr-3s-mus64-l1snr.ckpt`
`state_dict` keys, not assumed from source reading alone.

Scope, matching what `_v1/runtime.py` actually constructs (`_multi`/
`_single`), not the full generality of `_v1/core.py`: `in_channel` is always
1 (every one of the 21 v1 registry entries hardcodes it), `cond_dim` is
always 0 (never passed), `mult_add_mask` is always `False`, and the tf-model
is always `SeqBandModellingModule` (RNN-based) -- `_v1/runtime.py` never
constructs `TransformerTimeFreqModule` or `ConvolutionalTimeFreqModule` for
any registry entry, so neither is ported here (see `.runtime`'s module
docstring for how this was confirmed against the manifest, not assumed).
`PatchingMaskEstimationModule`'s Torch counterpart already raises
`NotImplementedError` unconditionally -- there is nothing to port.

Bidirectional RNN: MLX's `nn.LSTM`/`nn.GRU` are unidirectional, so each
`ResidualRNN` holds two (`rnn_fwd`, `rnn_bwd`), run over the sequence and its
reverse and concatenated on the feature axis -- exactly what Torch's single
`bidirectional=True` `nn.LSTM`/`nn.GRU` produces. The bias/gate-order algebra
that makes MLX's `nn.LSTM`/`nn.GRU` produce the same numbers as Torch's is
in `convert.py`, verified there against a direct `torch.nn.LSTM`/`GRU` call
before it was ever wired into this model (max abs diff ~1e-7 for both).

Reads: mlx.core, mlx.nn
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# --------------------------------------------------------------------------- band split


class NormFC(nn.Module):
    """LayerNorm + Linear over one band's `(real, imag)` bins.

    `treat_channel_as_feature` is always True for every registry entry
    (`_v1/runtime.py` never passes it), so that branch of Torch's `NormFC`
    is the only one ported.
    """

    def __init__(self, emb_dim: int, bandwidth: int, in_channel: int) -> None:
        super().__init__()
        reim = 2
        self.norm = nn.LayerNorm(in_channel * bandwidth * reim)
        self.fc = nn.Linear(bandwidth * reim * in_channel, emb_dim)

    def __call__(self, xb: mx.array) -> mx.array:
        # xb: (batch, n_time, in_chan, reim * band_width)
        batch, n_time, in_chan, ribw = xb.shape
        xb = self.norm(xb.reshape(batch, n_time, in_chan * ribw))
        return self.fc(xb)  # (batch, n_time, emb_dim)


class BandSplitModule(nn.Module):
    """`(batch, in_chan, n_freq, n_time)` complex64 -> `(batch, n_bands, n_time, emb_dim)`.

    Builds each band's embedding independently and stacks them (`mx.stack`)
    rather than scatter-writing into a preallocated array -- MLX arrays are
    immutable and this sidesteps the whole class of scatter-add correctness
    risk flagged in the org's accumulated findings (there is nothing to
    verify here: a fresh `mx.stack` of independently computed values, not an
    accumulator).
    """

    def __init__(self, band_specs: list[tuple[int, int]], emb_dim: int, in_channel: int) -> None:
        super().__init__()
        self.band_specs = band_specs
        self.norm_fc_modules = [
            NormFC(emb_dim=emb_dim, bandwidth=fend - fstart, in_channel=in_channel)
            for fstart, fend in band_specs
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, in_chan, n_freq, n_time) complex64
        xr = mx.stack([mx.real(x), mx.imag(x)], axis=-1)  # (batch, in_chan, n_freq, n_time, 2)
        xr = mx.transpose(xr, (0, 3, 1, 4, 2))  # (batch, n_time, in_chan, 2, n_freq)
        batch, n_time, in_chan = xr.shape[0], xr.shape[1], xr.shape[2]

        bands = []
        for (fstart, fend), nfm in zip(self.band_specs, self.norm_fc_modules):
            xb = xr[..., fstart:fend]  # (batch, n_time, in_chan, 2, band_width)
            xb = xb.reshape(batch, n_time, in_chan, -1)  # (batch, n_time, in_chan, 2*band_width)
            bands.append(nfm(xb))  # (batch, n_time, emb_dim)
        return mx.stack(bands, axis=1)  # (batch, n_bands, n_time, emb_dim)


# --------------------------------------------------------------------------- tf model


def _run_rnn(rnn_module, x: mx.array) -> mx.array:
    """`nn.GRU` returns the hidden sequence directly; `nn.LSTM` returns `(hidden, cell)`."""
    out = rnn_module(x)
    return out[0] if isinstance(out, tuple) else out


class ResidualRNN(nn.Module):
    """LayerNorm -> bidirectional RNN -> Linear, with a residual add.

    `rnn_fwd`/`rnn_bwd` replace Torch's single bidirectional `nn.LSTM`/
    `nn.GRU` (`.rnn.weight_ih_l0[_reverse]`, ...); see `convert.py` for the
    weight-splitting algebra.
    """

    def __init__(self, emb_dim: int, rnn_dim: int, rnn_type: str = "GRU", bidirectional: bool = True) -> None:
        super().__init__()
        cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.norm = nn.LayerNorm(emb_dim)
        self.bidirectional = bidirectional
        self.rnn_fwd = cls(input_size=emb_dim, hidden_size=rnn_dim)
        self.rnn_bwd = cls(input_size=emb_dim, hidden_size=rnn_dim) if bidirectional else None
        self.fc = nn.Linear(rnn_dim * (2 if bidirectional else 1), emb_dim)

    def __call__(self, z: mx.array) -> mx.array:
        # z: (batch, n_uncrossed, n_across, emb_dim)
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
    """`2 * n_modules` `ResidualRNN`s, alternating band-axis and time-axis modelling
    by transposing dims 1 and 2 between calls -- matches `_v1/tfmodel.py`'s
    non-`parallel_mode` path exactly (the only path any registry checkpoint uses)."""

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

    `hidden`/`output` are two-element Python lists (`[Linear, activation]`);
    MLX flattens list attributes with `.0`/`.1` indices exactly like Torch's
    `nn.Sequential`, and the activation has no parameters, so the emitted
    keys (`hidden.0.*`, `output.0.*`) match the checkpoint's `state_dict`
    keys directly -- confirmed against the real checkpoint, not assumed.
    """

    def __init__(self, emb_dim: int, mlp_dim: int, bandwidth: int, in_channel: int, complex_mask: bool = True) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.in_channel = in_channel
        self.complex_mask = complex_mask
        self.reim = 2 if complex_mask else 1
        self.norm = nn.LayerNorm(emb_dim)
        self.hidden = [nn.Linear(emb_dim, mlp_dim), nn.Tanh()]
        self.output = [nn.Linear(mlp_dim, bandwidth * in_channel * self.reim * 2), nn.GLU(axis=-1)]

    def _reshape_output(self, mb: mx.array) -> mx.array:
        batch, n_time, _ = mb.shape
        if self.complex_mask:
            mb = mb.reshape(batch, n_time, self.in_channel, self.bandwidth, self.reim)
            mb = mb[..., 0].astype(mx.complex64) + 1j * mb[..., 1].astype(mx.complex64)
        else:
            mb = mb.reshape(batch, n_time, self.in_channel, self.bandwidth)
        return mx.transpose(mb, (0, 2, 3, 1))  # (batch, in_channel, bandwidth, n_time)

    def __call__(self, qb: mx.array) -> mx.array:
        qb = self.norm(qb)
        qb = self.hidden[0](qb)
        qb = self.hidden[1](qb)
        mb = self.output[0](qb)
        mb = self.output[1](mb)
        return self._reshape_output(mb)


class MaskEstimationBase(nn.Module):
    def __init__(self, band_specs: list[tuple[int, int]], emb_dim: int, mlp_dim: int, in_channel: int, complex_mask: bool = True) -> None:
        super().__init__()
        self.band_widths = [fend - fstart for fstart, fend in band_specs]
        self.norm_mlp = [
            NormMLP(emb_dim=emb_dim, mlp_dim=mlp_dim, bandwidth=bw, in_channel=in_channel, complex_mask=complex_mask)
            for bw in self.band_widths
        ]

    def _compute_masks(self, q: mx.array) -> list[mx.array]:
        # q: (batch, n_bands, n_time, emb_dim)
        return [nmlp(q[:, b, :, :]) for b, nmlp in enumerate(self.norm_mlp)]


class MaskEstimationModule(MaskEstimationBase):
    """Non-overlapping bands (e.g. `dnr:vox7`): concatenate along the freq axis."""

    def __init__(self, band_specs: list[tuple[int, int]], emb_dim: int, mlp_dim: int, in_channel: int, complex_mask: bool = True) -> None:
        super().__init__(band_specs, emb_dim, mlp_dim, in_channel, complex_mask)

    def __call__(self, q: mx.array) -> mx.array:
        masks = self._compute_masks(q)  # each (batch, in_channel, band_width, n_time)
        return mx.concatenate(masks, axis=2)  # (batch, in_channel, n_freq, n_time)


class OverlappingMaskEstimationModule(MaskEstimationBase):
    """Overlapping perceptual bands (musical/bark/erb/mel/tribark): each band's
    mask is zero-padded to the full freq axis and elementwise-summed --
    additive accumulation over small (2-D-ish) arrays, not the large 4-D
    scatter-add pattern the org's accumulated findings flag as unsafe in
    MLX 0.31.2; verified against a plain Python/NumPy equivalent in
    `tests/test_mlx_parity.py`'s upstream-fixture comparison regardless.
    """

    def __init__(
        self,
        band_specs: list[tuple[int, int]],
        emb_dim: int,
        mlp_dim: int,
        in_channel: int,
        n_freq: int,
        freq_weights: Optional[list[mx.array]] = None,
        complex_mask: bool = True,
    ) -> None:
        super().__init__(band_specs, emb_dim, mlp_dim, in_channel, complex_mask)
        self.band_specs = band_specs
        self.in_channel = in_channel
        self.n_freq = n_freq
        self.freq_weights = list(freq_weights) if freq_weights is not None else None

    def __call__(self, q: mx.array) -> mx.array:
        masks = self._compute_masks(q)
        batch, n_time = q.shape[0], q.shape[2]
        total = mx.zeros((batch, self.in_channel, self.n_freq, n_time), dtype=masks[0].dtype)
        for i, ((fstart, fend), mask) in enumerate(zip(self.band_specs, masks)):
            if self.freq_weights is not None:
                fw = self.freq_weights[i]
                mask = mask * fw[None, None, :, None].astype(mask.dtype)
            padded = mx.pad(mask, [(0, 0), (0, 0), (fstart, self.n_freq - fend), (0, 0)])
            total = total + padded
        return total


# --------------------------------------------------------------------------- cores


class MultiCore(nn.Module):
    """One `BandSplitModule` + `SeqBandModellingModule` feeding a per-stem
    dict of mask estimators -- mirrors `MultiSourceMultiMaskBandSplitCoreRNN`."""

    def __init__(self, band_split: BandSplitModule, tf_model: SeqBandModellingModule, mask_estim: dict[str, nn.Module]) -> None:
        super().__init__()
        self.band_split = band_split
        self.tf_model = tf_model
        self.mask_estim = mask_estim

    def __call__(self, x: mx.array) -> dict[str, mx.array]:
        z = self.band_split(x)
        q = self.tf_model(z)
        return {stem: x * mem(q) for stem, mem in self.mask_estim.items()}


class SingleCore(nn.Module):
    """One `BandSplitModule` + `SeqBandModellingModule` feeding one mask
    estimator, for the single-stem `gru8`/`lstm12` v1 recipes -- mirrors
    `SingleMaskBandsplitCoreRNN`."""

    def __init__(self, band_split: BandSplitModule, tf_model: SeqBandModellingModule, mask_estim: MaskEstimationModule) -> None:
        super().__init__()
        self.band_split = band_split
        self.tf_model = tf_model
        self.mask_estim = mask_estim

    def __call__(self, x: mx.array) -> mx.array:
        z = self.band_split(x)
        q = self.tf_model(z)
        m = self.mask_estim(q)
        return x * m
