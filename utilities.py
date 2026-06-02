from __future__ import annotations
from datetime import datetime
from typing import Any, List, Optional
import streamlit as st

from accumulator import Accumulator
from devices import Device
from gridConsumption import getCurrentGridCost, getCurrentPowerGridConsumption
from papeg import getLocationCurrentHour, getLocationNow as _papeg_location_now


KEY_DEVICES = "devices"
KEY_SELECTED_LOCATION = "selected_location"
KEY_CURRENT_WEATHER = "current_weather"
KEY_SOLAR_FORECAST = "solar_generation_forecast"
KEY_ACCUMULATOR = "accumulator"
KEY_ACCUMULATOR_POWER = "accumulator_power"
KEY_NEEDS_REPLAN = "needs_replan"
KEY_GRID_CONSUMPTION = "grid_consumption"
KEY_GRID_COST = "grid_cost"
KEY_DAILY_GRID_KWH = "daily_grid_kwh"
KEY_DAILY_GRID_COST = "daily_grid_cost"
KEY_PLAN_SOURCE = "plan_source"


def _safe_get(key: str, default: Any = None) -> Any:
    try:
        return st.session_state.get(key, default)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce dashboard values to float so missing or bad data never crashes."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_devices() -> List[Device]:
    """Return the device list from session state, or an empty list if unset."""
    devices = _safe_get(KEY_DEVICES, [])
    if not isinstance(devices, list):
        return []
    return [d for d in devices if isinstance(d, Device)]


def _find_device(device_name: str) -> Optional[Device]:
    """Locate a device by its display name."""
    for device in _get_devices():
        try:
            if device.get_name() == device_name:
                return device
        except Exception:
            continue
    return None


def _get_current_hour() -> int:
    try:
        return getLocationCurrentHour()
    except Exception:
        return 0


def getCurrentTime() -> str:
    """Return the current clock time in the selected location's timezone."""
    try:
        return getLocationNow().strftime("%H:%M %Z")
    except Exception:
        return datetime.now().strftime("%H:%M")


def getLocationNow() -> datetime:
    """Return the current datetime in the selected location's timezone."""
    try:
        return _papeg_location_now()
    except Exception:
        return datetime.now()


def _get_solar_forecast() -> List[float]:
    raw = _safe_get(KEY_SOLAR_FORECAST, [])
    if not isinstance(raw, (list, tuple)):
        return [0.0] * 24

    forecast: List[float] = []
    for i in range(24):
        try:
            forecast.append(max(_safe_float(raw[i]), 0.0))
        except (IndexError, TypeError):
            forecast.append(0.0)
    return forecast


def _get_accumulator_power() -> float:
    """Read stored battery energy from either an Accumulator object or a float."""
    accumulator = _safe_get(KEY_ACCUMULATOR)
    if isinstance(accumulator, Accumulator):
        try:
            return max(accumulator.getAccumulatorPower(), 0.0)
        except Exception:
            pass

    return max(_safe_float(_safe_get(KEY_ACCUMULATOR_POWER, 0.0)), 0.0)


def _extract_temperature(weather: Any) -> float:
    """Support dict payloads from Open-Meteo or a plain numeric temperature."""
    if isinstance(weather, (int, float)):
        return float(weather)

    if isinstance(weather, dict):
        for key in ("temperature_2m", "temperature", "temp", "current_temp"):
            if key in weather:
                return _safe_float(weather[key])

    return 0.0


def getWeatherBadge() -> float:
    """Return the current temperature in °C for the dashboard badge."""
    try:
        weather = _safe_get(KEY_CURRENT_WEATHER)
        return _extract_temperature(weather)
    except Exception:
        return 0.0


def getLocationName() -> str:
    """Return the user-selected location label."""
    try:
        location = _safe_get(KEY_SELECTED_LOCATION, "Unknown Location")
        return str(location) if location is not None else "Unknown Location"
    except Exception:
        return "Unknown Location"


def getSolarConsumption() -> float:
    """Return estimated solar generation (kWh) for the current clock hour."""
    try:
        forecast = _get_solar_forecast()
        hour = _get_current_hour()
        return forecast[hour] if 0 <= hour < len(forecast) else 0.0
    except Exception:
        return 0.0


def getDevicesConsumption() -> float:
    """Sum hourly draw from every device that is manually turned on."""
    total = 0.0
    for device in _get_devices():
        try:
            if device.get_is_on():
                total += max(device.get_hourly_power_consumption_kwh(), 0.0)
        except Exception:
            continue
    return total


def getTodaysGeneratedEnergyApproximations() -> List[float]:
    """Return 24 hourly solar estimates (hours 0–23) for the line chart."""
    return _get_solar_forecast()


def setPriority(
    device_name: str,
    start_datetime: datetime,
    end_datetime: datetime,
) -> bool:
    """Persist a non-negotiable priority window and request an immediate replan.

    Only whole-hour boundaries are accepted so the RL planner can bind
    constraints to discrete hourly slots.
    """
    try:
        if not isinstance(start_datetime, datetime) or not isinstance(end_datetime, datetime):
            return False

        start_hour = start_datetime.hour
        end_hour = end_datetime.hour

        # Reject fractional hours — priorities align to the hourly schedule grid.
        for dt in (start_datetime, end_datetime):
            if dt.minute != 0 or dt.second != 0 or dt.microsecond != 0:
                return False

        if start_hour < 0 or start_hour > 23 or end_hour < 0 or end_hour > 23:
            return False

        if start_hour > end_hour:
            return False

        device = _find_device(device_name)
        if device is None:
            return False

        device.set_priority_window((start_datetime, end_datetime))

        # Non-negotiable: priority edits must trigger a full replan immediately.
        st.session_state[KEY_NEEDS_REPLAN] = True
        return True
    except Exception:
        return False


def getPowerGridConsumption() -> float:
    """Return current grid draw (kWh), using stored value or a live estimate."""
    try:
        stored = _safe_get(KEY_GRID_CONSUMPTION)
        if stored is not None:
            return max(_safe_float(stored), 0.0)

        required = getDevicesConsumption()
        generated = getSolarConsumption()
        accumulator = _get_accumulator_power()

        # Battery covers the deficit before any energy is pulled from the grid.
        deficit = getCurrentPowerGridConsumption(required, generated)
        return max(deficit - accumulator, 0.0)
    except Exception:
        return 0.0


def getGridCost() -> float:
    """Return current grid cost in RON at the fixed 1.2 RON/kWh rate."""
    try:
        stored = _safe_get(KEY_GRID_COST)
        if stored is not None:
            return max(_safe_float(stored), 0.0)

        return getCurrentGridCost(getPowerGridConsumption())
    except Exception:
        return 0.0


def getDailyGridConsumption() -> float:
    """Return the grid energy (kWh) the plan expects to buy for the rest of today."""
    try:
        return max(_safe_float(_safe_get(KEY_DAILY_GRID_KWH, 0.0)), 0.0)
    except Exception:
        return 0.0


def getDailyGridCost() -> float:
    """Return today's estimated grid cost (RON) from the current planner."""
    try:
        stored = _safe_get(KEY_DAILY_GRID_COST)
        if stored is not None:
            return max(_safe_float(stored), 0.0)
        return getCurrentGridCost(getDailyGridConsumption())
    except Exception:
        return 0.0


def getPlanSource() -> str:
    """Return which planner produced the active schedule (RL vs heuristic)."""
    try:
        return str(_safe_get(KEY_PLAN_SOURCE, "none"))
    except Exception:
        return "none"
