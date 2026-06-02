from __future__ import annotations
from datetime import datetime
from typing import Optional, Tuple


PriorityWindow = Optional[Tuple[datetime, datetime]]


class Device:
    """One smart appliance.

    ``daily_runtime_hours`` is the number of hours the device should run today.
    It is the *target* the Smart Optimizer plans around: the RL agent is free to
    place those hours in the cheapest (solar/battery covered) slots, while a
    priority window forces specific hours ON regardless. Priority hours count
    toward this target, so the optimizer only adds the remaining flexible hours.
    """

    def __init__(
        self,
        name: str,
        hourly_power_consumption_kwh: float,
        is_on: bool = False,
        smart_optimizer_enabled: bool = False,
        priority_window: PriorityWindow = None,
        daily_runtime_hours: int = 0,
    ) -> None:
        self._name = name
        self._hourly_power_consumption_kwh = hourly_power_consumption_kwh
        self._is_on = is_on
        self._smart_optimizer_enabled = smart_optimizer_enabled
        self._priority_window = priority_window
        self._daily_runtime_hours = self._clamp_runtime(daily_runtime_hours)

    def get_name(self) -> str:
        return self._name

    def set_name(self, name: str) -> None:
        self._name = name

    def get_hourly_power_consumption_kwh(self) -> float:
        return self._hourly_power_consumption_kwh

    def set_hourly_power_consumption_kwh(self, kwh: float) -> None:
        self._hourly_power_consumption_kwh = max(0.0, float(kwh))

    def get_is_on(self) -> bool:
        return self._is_on

    def set_is_on(self, is_on: bool) -> None:
        self._is_on = bool(is_on)

    def get_smart_optimizer_enabled(self) -> bool:
        return self._smart_optimizer_enabled

    def set_smart_optimizer_enabled(self, enabled: bool) -> None:
        self._smart_optimizer_enabled = bool(enabled)

    def get_priority_window(self) -> PriorityWindow:
        return self._priority_window

    def set_priority_window(self, window: PriorityWindow) -> None:
        self._priority_window = window

    def clear_priority_window(self) -> None:
        self._priority_window = None

    def get_daily_runtime_hours(self) -> int:
        return self._daily_runtime_hours

    def set_daily_runtime_hours(self, hours: int) -> None:
        self._daily_runtime_hours = self._clamp_runtime(hours)

    @staticmethod
    def _clamp_runtime(hours: int) -> int:
        """Runtime targets are whole hours bounded to a single day (0..24)."""
        try:
            return max(0, min(int(hours), 24))
        except (TypeError, ValueError):
            return 0
