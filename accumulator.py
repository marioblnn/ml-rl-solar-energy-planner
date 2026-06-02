"""Simple home battery model with fixed capacity bounds."""

from __future__ import annotations

# Physical limits for the household energy accumulator (kWh).
MIN_CAPACITY_KWH: float = 0.0
MAX_CAPACITY_KWH: float = 12.0


class Accumulator:
    """Stores surplus solar energy and releases it to reduce grid draw."""

    def __init__(self, initial_power_kwh: float = MIN_CAPACITY_KWH) -> None:
        # Start empty unless a caller explicitly restores persisted state.
        self._power_kwh = self._clamp(initial_power_kwh)

    def getAccumulatorPower(self) -> float:
        """Return the current stored energy in kWh."""
        return self._power_kwh

    def charge(self, kwh: float) -> float:
        """Add energy from solar surplus; returns the amount actually stored."""
        if kwh <= 0.0:
            return 0.0

        available_headroom = MAX_CAPACITY_KWH - self._power_kwh
        stored = min(kwh, available_headroom)
        self._power_kwh = self._clamp(self._power_kwh + stored)
        return stored

    def discharge(self, kwh: float) -> float:
        """Release energy to cover a deficit; returns the amount actually used."""
        if kwh <= 0.0:
            return 0.0

        released = min(kwh, self._power_kwh)
        self._power_kwh = self._clamp(self._power_kwh - released)
        return released

    def reset(self) -> None:
        """Return the accumulator to its empty starting state."""
        self._power_kwh = MIN_CAPACITY_KWH

    def _clamp(self, value: float) -> float:
        """Keep stored energy within the safe operating range at all times."""
        return max(MIN_CAPACITY_KWH, min(float(value), MAX_CAPACITY_KWH))
