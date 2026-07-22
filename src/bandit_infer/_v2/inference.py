"""Faithful v2 tensor chunk overlap/add inference.

Copied from the upstream tensor handler; only its progress display is removed.
Reads: torch.
"""

import math
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


class BaseChunkedInferenceHandler(nn.Module):
    def __init__(
        self,
        chunk_size_seconds: float,
        hop_size_seconds: float,
        inference_batch_size: int,
        fs: int,
        window_fn: str = "hann_window",
        wkwargs: dict = None,
        pad_mode: str = "reflect",
        rank: int = 0,
    ):
        super().__init__()

        self.fs = fs

        self.chunk_size_samples = int(chunk_size_seconds * fs)
        self.hop_size_samples = int(hop_size_seconds * fs)
        self.overlap_samples = self.chunk_size_samples - self.hop_size_samples

        self.scaler = self.chunk_size_samples / (2 * self.hop_size_samples)

        window_fn = torch.__dict__[window_fn]

        if wkwargs is None:
            wkwargs = {}

        scaled_window = (
            window_fn(self.chunk_size_samples, **wkwargs)[None, None, :] / self.scaler
        )

        self.register_buffer("scaled_window", scaled_window)

        self.pad_mode = pad_mode
        self.inference_batch_size = inference_batch_size

        self.front_pad_samples = 2 * self.overlap_samples

        self.rank = rank

    def set_rank(self, rank: int):
        self.rank = rank

    def forward(self, mixture: torch.Tensor, model: nn.Module):
        raise NotImplementedError

    def _get_n_chunks_smart(self, n_samples: int):
        n_chunk_front_pad_only = (
            int(
                math.ceil(
                    (n_samples + self.front_pad_samples - self.chunk_size_samples)
                    / self.hop_size_samples
                )
            )
            + 1
        )

        n_end_pad_samples = self._get_end_pad_samples(n_samples, n_chunk_front_pad_only)

        if n_end_pad_samples >= self.front_pad_samples:
            return n_chunk_front_pad_only

        else:
            n_chunk_front_pad_and_end_pad = (
                int(
                    math.ceil(
                        (
                            n_samples
                            + 2 * self.front_pad_samples
                            - self.chunk_size_samples
                        )
                        / self.hop_size_samples
                    )
                )
                + 1
            )

            return n_chunk_front_pad_and_end_pad

    def _get_n_chunks(self, n_samples: int):
        return (
            int(
                math.ceil(
                    (n_samples + 2 * self.front_pad_samples - self.chunk_size_samples)
                    / self.hop_size_samples
                )
            )
            + 1
        )

    def _get_end_pad_samples(self, n_samples: int, n_chunks: int):
        return (
            (n_chunks - 1) * self.hop_size_samples + self.chunk_size_samples - n_samples
        )

    def _get_padded_samples(self, n_samples: int, n_chunks: int, end_pad_samples: int):
        return n_samples + 2 * self.front_pad_samples + end_pad_samples

    def _unfold(self, segment: torch.Tensor):
        batch_size, n_channels, _ = segment.shape

        assert batch_size == 1

        segment = segment.reshape(n_channels, 1, -1, 1)  # (n_channels, 1, n_samples, 1)

        unfolded_segment = F.unfold(
            segment,
            kernel_size=(self.chunk_size_samples, 1),
            stride=(self.hop_size_samples, 1),
        )  # (n_channels, chunk_size_samples, n_chunks)

        unfolded_segment = unfolded_segment.permute(
            0, 2, 1
        )  # (n_channels, n_chunks, chunk_size_samples)

        return unfolded_segment


class StandardTensorChunkedInferenceHandler(BaseChunkedInferenceHandler):
    def __init__(
        self,
        chunk_size_seconds: float,
        hop_size_seconds: float,
        inference_batch_size: int,
        fs: int,
        window_fn: str = "hann_window",
        wkwargs: dict = None,
        pad_mode: str = "reflect",
        rank: int = 0,
    ):
        super().__init__(
            chunk_size_seconds=chunk_size_seconds,
            hop_size_seconds=hop_size_seconds,
            inference_batch_size=inference_batch_size,
            fs=fs,
            window_fn=window_fn,
            wkwargs=wkwargs,
            pad_mode=pad_mode,
            rank=rank,
        )

    def _fold(self, stem_output: torch.Tensor, n_samples: int, padded_samples: int):
        stem_output = stem_output * self.scaled_window.to(stem_output.device)
        stem_output = torch.permute(
            stem_output, (0, 2, 1)
        )  # (n_channels, chunk_size_samples, n_chunks)


        stem_output = F.fold(
            stem_output,
            output_size=(padded_samples, 1),
            kernel_size=(self.chunk_size_samples, 1),
            stride=(self.hop_size_samples, 1),
        )  # (n_channels, 1, padded_samples, 1)

        stem_output = stem_output[
            None,
            :,
            0,
            self.front_pad_samples : self.front_pad_samples + n_samples,
            0,
        ]

        return stem_output

    def _cat_and_fold(
        self, stem_outputs: List[torch.Tensor], n_samples: int, padded_samples: int
    ):
        stem_output = torch.cat(
            stem_outputs, dim=1
        )  # (n_channels, n_chunks, chunk_size_samples)

        stem_output = self._fold(stem_output, n_samples, padded_samples)

        return stem_output

    def _pad_and_unfold(self, mixture: torch.Tensor):
        batch_size, _, n_samples = mixture.shape

        assert batch_size == 1

        n_chunks = self._get_n_chunks(n_samples)
        end_pad_samples = self._get_end_pad_samples(n_samples, n_chunks)
        padded_samples = self._get_padded_samples(n_samples, n_chunks, end_pad_samples)

        if self.front_pad_samples >= n_samples:
            reflect_pad = (n_samples - 1, n_samples - 1)
            remaining_pad = self.front_pad_samples - (n_samples - 1)
            constant_pad = (remaining_pad, remaining_pad + end_pad_samples)
        elif self.front_pad_samples + end_pad_samples >= n_samples:
            reflect_pad = (self.front_pad_samples, n_samples - 1)
            remaining_pad = self.front_pad_samples + end_pad_samples - (n_samples - 1)
            constant_pad = (0, remaining_pad)
        else:
            reflect_pad = (
                self.front_pad_samples,
                self.front_pad_samples + end_pad_samples,
            )
            constant_pad = None

        padded_mixture = F.pad(mixture, reflect_pad, mode=self.pad_mode)

        if constant_pad is not None:
            padded_mixture = F.pad(
                padded_mixture,
                constant_pad,
                mode="constant",
            )

        unfolded_mixture = self._unfold(padded_mixture)

        # (n_chunks, n_channels, chunk_size_samples)

        return unfolded_mixture, n_samples, padded_samples

    def _tensor_forward(self, mixture: torch.Tensor, model: nn.Module):
        _, n_channels, n_samples = mixture.shape

        unfolded_mixture, n_samples, padded_samples = self._pad_and_unfold(mixture)
        # (n_channels, n_chunks, chunk_size_samples)

        # print(unfolded_mixture.shape)

        n_chunks = unfolded_mixture.shape[1]
        n_batches = math.ceil(n_chunks / self.inference_batch_size)
        outputs = {stem: [] for stem in model.stems}

        for i in range(n_batches):
            start = i * self.inference_batch_size
            end = min((i + 1) * self.inference_batch_size, n_chunks)
            chunk = unfolded_mixture[:, start:end, :]
            input_dict = {
                "mixture": {"audio": chunk.reshape(-1, 1, self.chunk_size_samples)}
            }
            output = model(input_dict)

            del chunk

            for stem in model.stems:
                outputs[stem].append(
                    output["estimates"][stem]["audio"].reshape(
                        n_channels, -1, self.chunk_size_samples
                    )
                )

            del output

        final_outputs = {
            stem: {
                "audio": self._cat_and_fold(outputs[stem], n_samples, padded_samples)
            }
            for stem in model.stems
        }

        return {"estimates": final_outputs}

    def forward(self, mixture: torch.Tensor, model: nn.Module):
        return self._tensor_forward(mixture, model)
