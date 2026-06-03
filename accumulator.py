from __future__ import annotations


MIN_CAPACITY_KWH: float = 0.0
MAX_CAPACITY_KWH: float = 12.0


class Accumulator:
    def __init__(self, initial_power_kwh: float = MIN_CAPACITY_KWH) -> None:
        self._power_kwh = self._verify_value(initial_power_kwh)

    def getAccumulatorPower(self) -> float:
        return self._power_kwh

    def charge(self, kwh: float) -> float:
        if kwh <= 0.0:
            return 0.0

        available_headroom = MAX_CAPACITY_KWH - self._power_kwh
        stored = min(kwh, available_headroom)
        self._power_kwh = self._verify_value(self._power_kwh + stored)
        return stored

    def discharge(self, kwh: float) -> float:
        if kwh <= 0.0:
            return 0.0

        released = min(kwh, self._power_kwh)
        self._power_kwh = self._verify_value(self._power_kwh - released)
        return released

    def reset(self) -> None:
        self._power_kwh = MIN_CAPACITY_KWH

    def _verify_value(self, value: float) -> float:
        return max(MIN_CAPACITY_KWH, min(float(value), MAX_CAPACITY_KWH))
