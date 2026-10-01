"""Compatibility helpers for the FlagOS model and Kaiwu SDK environments."""

import os

from .process_sampler import KaiwuProcessSampler


def create_flagos_sampler(device):
    """Connect FlagOS models to Kaiwu in the separate Python 3.10 environment."""
    if device.type == "flagos":
        return KaiwuProcessSampler(os.environ["KAIWU_PY310"], alpha=0.95, size_limit=10)
    return None
