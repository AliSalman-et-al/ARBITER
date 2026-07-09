"""JSON schema helpers for provider-enforced structured output."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a strict structured-output schema derived from Pydantic JSON schema."""

    normalized = deepcopy(schema)
    _normalize_strict_schema_node(normalized)
    return normalized


def _normalize_strict_schema_node(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("default", None)
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["required"] = list(properties)
            node["additionalProperties"] = False
        for value in node.values():
            _normalize_strict_schema_node(value)
    elif isinstance(node, list):
        for item in node:
            _normalize_strict_schema_node(item)
