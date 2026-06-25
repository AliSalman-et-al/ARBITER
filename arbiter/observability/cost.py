"""Token-to-cost helpers for trace instrumentation."""

from __future__ import annotations

from typing import Any

from arbiter.config import MODEL_REGISTRY


def estimate_call_cost(
    model: str,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> dict[str, Any]:
    """Estimate one call's USD cost from registry prices.

    ``None`` means unknown. A numeric ``0`` is returned only when the model has
    explicit zero pricing in ``MODEL_REGISTRY``.
    """

    info = MODEL_REGISTRY.get(model)
    if info is None:
        return {"cost": None, "pricing_unknown": True}

    total = 0.0
    known = False
    unknown = False
    components = {
        "input": (input_tokens, info.get("price_per_mtok_in")),
        "output": (output_tokens, info.get("price_per_mtok_out")),
        "cache_read": (cache_read_tokens, info.get("price_per_mtok_cache_read")),
        "cache_write": (cache_write_tokens, info.get("price_per_mtok_cache_write")),
    }

    for tokens, price in components.values():
        if tokens is None:
            continue
        if price is None:
            unknown = True
            continue
        known = True
        total += (tokens / 1_000_000) * price

    if unknown:
        return {"cost": None, "pricing_unknown": True}
    if not known:
        return {"cost": None, "pricing_unknown": True}
    return {"cost": total, "pricing_unknown": False}
