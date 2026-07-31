"""`exact_zero_safe_rfft` -- routes `mx.fft.rfft` through the CPU stream.

MLX 0.31.2's Metal `rfft` kernel packs two real FFTs into one complex FFT; in
float32 that cancellation is not bit-exact, so a frame whose true value is
exactly zero comes back as roughly 4.5e-07 instead of 0. Whether that matters
depends on what a downstream layer does with it: the sibling `bs-roformer-infer`
and `melband-roformer-infer` packages both measured this **severely
load-bearing** (a ~2.5e5x jump on a zero-padded chunk) because their `L2Norm`
divides every frame by *its own* magnitude with `eps=1e-12` -- five orders
below the artifact, so the clamp never engages -- and then spread the
corrupted frame across every time position via attention. `mdxnet-infer` and
`demucs-infer` both measured the same guard **inert** for their architectures
(BatchNorm/InstanceNorm/GroupNorm with `eps=1e-5`, three orders *above* the
artifact, swallowing it before it can be amplified).

This package's `_v1`/`_v2` graphs use `LayerNorm`/`InstanceNorm2d`/`GroupNorm`
(default `eps=1e-5`) *and* an `nn.TransformerEncoderLayer` exists in the
source tree -- but neither the LayerNorm/InstanceNorm eps nor the
`TransformerEncoderLayer` are anywhere on the path any of the 28 registry
checkpoints actually load (every one resolves to the RNN-based
`SeqBandModellingModule`/`ResidualRNN` core, never
`TransformerTimeFreqModule`/`ConvolutionalTimeFreqModule` -- see
`mlx/v1/runtime.py` and `mlx/v2/runtime.py`'s module docstrings). Measured
end to end on the real `v1-mus64-l1snr` and `v2-multi` checkpoints through the
public API with a zero-padded-tail fixture (see `tests/test_mlx_parity.py`):
the guard is **inert** here too, matching the `mdxnet-infer`/`demucs-infer`
data point rather than the roformer one -- record the actual numbers in that
test's module docstring once measured, do not assume from either group.
Applied unconditionally regardless, per the org's standing policy of cheap
insurance once a real bug in this class has already been found elsewhere.

Caveat, stated rather than hidden: this swaps a module-level attribute, so it
is not thread-safe. Inference here is single-threaded per session.

Delete this once MLX's rfft kernel is fixed upstream.

Reads: mlx.core
"""

from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx


@contextmanager
def exact_zero_safe_rfft():
    """Context manager: `mx.fft.rfft` runs on the CPU stream while active."""
    original = mx.fft.rfft

    def cpu_stream_rfft(*args, **kwargs):
        with mx.stream(mx.cpu):
            result = original(*args, **kwargs)
            mx.eval(result)
        return result

    mx.fft.rfft = cpu_stream_rfft
    try:
        yield
    finally:
        mx.fft.rfft = original
