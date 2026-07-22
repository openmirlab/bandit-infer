"""Public Bandit inference API with no eager checkpoint or model loading.

The package exposes only the thin one-shot facade and its resident session.
Reads: __about__, api.
"""

from .__about__ import __version__
from .api import BanditSession, separate

__all__ = ["BanditSession", "__version__", "separate"]

