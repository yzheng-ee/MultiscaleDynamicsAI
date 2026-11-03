"""Internal helpers for JSON-compatible numerical values."""

import math


def _json_number(value: float | None) -> float | str | None:
    """Preserve absent values and encode infinity without conflating it with None."""
    if value is None:
        return None
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return float(value)
