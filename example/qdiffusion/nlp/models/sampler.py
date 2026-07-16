"""Sampler construction for the NLP BM energy model."""

from __future__ import annotations

from typing import Any


def build_bm_sampler(
    *,
    sampler_type: str = "sa",
    sampler_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Builds the local sampler used for conditioned hidden-state sampling."""

    sampler_kwargs = dict(sampler_kwargs or {})
    if sampler_type != "sa":
        raise ValueError(
            "The initial NLP workflow supports sampler_type='sa' only; "
            "CIM integration will be added after the local baseline is verified."
        )

    from kaiwu.classical import SimulatedAnnealingOptimizer

    defaults = {"alpha": 0.95, "size_limit": 10}
    defaults.update(sampler_kwargs)
    return SimulatedAnnealingOptimizer(**defaults)
