"""Grid energy draw and cost calculations for the smart home optimizer."""

from __future__ import annotations

# Fixed retail rate used throughout the dashboard and RL reward logic.
GRID_PRICE_RON_PER_KWH: float = 1.2


def _safe_non_negative(value: float) -> float:
    """Treat invalid negative readings as zero to avoid bogus grid metrics."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(numeric, 0.0)


def getCurrentPowerGridConsumption(required_kwh: float, generated_kwh: float) -> float:
    """Return grid draw needed when solar generation falls short of demand.

    Example: required=4, generated=2 -> 2 kWh from the grid.
    When generation meets or exceeds demand, grid draw is zero.
    """
    required = _safe_non_negative(required_kwh)
    generated = _safe_non_negative(generated_kwh)
    return max(required - generated, 0.0)


def getCurrentGridCost(consumed_kwh: float) -> float:
    """Return the monetary cost in RON for energy taken from the grid."""
    consumed = _safe_non_negative(consumed_kwh)
    return consumed * GRID_PRICE_RON_PER_KWH
