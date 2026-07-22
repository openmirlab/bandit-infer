"""Bandit v1 graph namespace, deliberately independent from Bandit v2.

``BandsplitCoreBase`` preserves upstream's local graph contract without the
training framework it originally inherited from. Reads: torch only.
"""

from abc import ABC
from typing import Iterable, Mapping, Union

from torch import nn


class BandsplitCoreBase(nn.Module, ABC):
    """Upstream v1 base containing no training/evaluation behavior."""
    band_split: nn.Module
    tf_model: nn.Module
    mask_estim: Union[nn.Module, Mapping[str, nn.Module], Iterable[nn.Module]]

    @staticmethod
    def mask(x, m):
        return x * m
