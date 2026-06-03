from __future__ import annotations

GRID_PRICE_RON_PER_KWH: float = 1.2


def _safe_non_negative(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(numeric, 0.0)


def getCurrentPowerGridConsumption(required_kwh: float, generated_kwh: float) -> float:
    required = _safe_non_negative(required_kwh)
    generated = _safe_non_negative(generated_kwh)
    return max(required - generated, 0.0)


def getCurrentGridCost(consumed_kwh: float) -> float:
    consumed = _safe_non_negative(consumed_kwh)
    return consumed * GRID_PRICE_RON_PER_KWH
