from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import streamlit as st

from accumulator import MAX_CAPACITY_KWH, MIN_CAPACITY_KWH, Accumulator
from devices import Device
from gridConsumption import (
    GRID_PRICE_RON_PER_KWH,
    getCurrentGridCost,
    getCurrentPowerGridConsumption,
)

# --- file locations ---
_BASE_DIR = Path(__file__).resolve().parent
SOLAR_MODEL_PATH = _BASE_DIR / "solar_model.joblib"
PPO_MODEL_PATH = _BASE_DIR / "ppo_smarthome.zip"

# --- session keys (kept aligned with utilities.py / dashboard.py) ---
KEY_DEVICES = "devices"
KEY_SELECTED_LOCATION = "selected_location"
KEY_CURRENT_WEATHER = "current_weather"
KEY_SOLAR_FORECAST = "solar_generation_forecast"
KEY_ACCUMULATOR = "accumulator"
KEY_ACCUMULATOR_POWER = "accumulator_power"
KEY_PLANNER_SCHEDULE = "planner_schedule"
KEY_NEEDS_REPLAN = "needs_replan"
KEY_LAST_OVERRIDE_HASH = "last_override_hash"
KEY_GRID_CONSUMPTION = "grid_consumption"
KEY_GRID_COST = "grid_cost"
# Plan-wide (whole remaining day) results produced by the optimizer.
KEY_DAILY_GRID_KWH = "daily_grid_kwh"
KEY_DAILY_GRID_COST = "daily_grid_cost"
KEY_PLAN_DEVICE_HOURS = "plan_device_hours"  # {device_name: [hours on today]}
KEY_PLAN_SOURCE = "plan_source"  # "RL agent" | "heuristic optimizer" | ...

# Accumulator block labels in the planner table.
ACC_CHARGE_LABEL = "Accumulator (Charge)"
ACC_DISCHARGE_LABEL = "Accumulator (Discharge)"

HOURS_PER_DAY = 24

# Feature order MUST match trainML.py so the saved model receives valid input.
SOLAR_FEATURE_ORDER = (
    "cloud_cover",
    "sunshine_duration",
    "temperature_2m",
    "is_day",
    "hour",
)

# Minimal location registry so a location string maps to API coordinates.
# Defaulting to Bucharest keeps the RON pricing context coherent.
LOCATIONS: Dict[str, Tuple[float, float]] = {
    "Bucharest": (44.4268, 26.1025),
    "Cluj-Napoca": (46.7712, 23.6236),
    "Timisoara": (45.7489, 21.2087),
    "Iasi": (47.1585, 27.6014),
    "Constanta": (44.1598, 28.6348),
    "California": (37.7749, -122.4194),
}
DEFAULT_COORDS = LOCATIONS["Bucharest"]

# IANA timezones aligned with each location so UI/planner hours match Open-Meteo.
LOCATION_TIMEZONES: Dict[str, str] = {
    "Bucharest": "Europe/Bucharest",
    "Cluj-Napoca": "Europe/Bucharest",
    "Timisoara": "Europe/Bucharest",
    "Iasi": "Europe/Bucharest",
    "Constanta": "Europe/Bucharest",
    "California": "America/Los_Angeles",
}
DEFAULT_TIMEZONE = "Europe/Bucharest"


#Get keys from session state defensively so a missing key never crashes a rerun.
def _safe_session_get(key: str, default: Any = None) -> Any:
    try:
        return st.session_state.get(key, default)
    except Exception:
        return default


def getLocationTimezone(location: Optional[str] = None) -> ZoneInfo:
    if location is None:
        location = _safe_session_get(KEY_SELECTED_LOCATION, "Bucharest")
    tz_name = LOCATION_TIMEZONES.get(str(location), DEFAULT_TIMEZONE)
    return ZoneInfo(tz_name)


def getLocationNow(location: Optional[str] = None) -> datetime:
    return datetime.now(getLocationTimezone(location))


def getLocationCurrentHour(location: Optional[str] = None) -> int:
    return getLocationNow(location).hour


def ensureSessionDefaults() -> None:
    try:
        defaults = {
            KEY_DEVICES: [],
            KEY_SELECTED_LOCATION: "Bucharest",
            KEY_CURRENT_WEATHER: None,
            KEY_SOLAR_FORECAST: [0.0] * HOURS_PER_DAY,
            KEY_PLANNER_SCHEDULE: [],
            KEY_NEEDS_REPLAN: True,  # plan once on first load
            KEY_LAST_OVERRIDE_HASH: None,
            KEY_GRID_CONSUMPTION: 0.0,
            KEY_GRID_COST: 0.0,
            KEY_DAILY_GRID_KWH: 0.0,
            KEY_DAILY_GRID_COST: 0.0,
            KEY_PLAN_DEVICE_HOURS: {},
            KEY_PLAN_SOURCE: "none",
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    except Exception:
        # Outside a Streamlit runtime (e.g. unit tests) this is a no-op.
        pass


# ---------------------------------------------------------------------------
# Weather + solar pipeline
# ---------------------------------------------------------------------------

def fetchWeatherForecast(location: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if location is None:
        location = _safe_session_get(KEY_SELECTED_LOCATION, "Bucharest")

    latitude, longitude = LOCATIONS.get(str(location), DEFAULT_COORDS)

    try:
        import requests

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "cloud_cover,sunshine_duration,temperature_2m,is_day",
            "forecast_days": 1,
            "timezone": "auto",
        }
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast", params=params, timeout=10
        )
        response.raise_for_status()
        hourly = response.json().get("hourly", {})

        # Validate that every required series is present and non-empty.
        required = ("cloud_cover", "sunshine_duration", "temperature_2m", "is_day")
        if not all(hourly.get(k) for k in required):
            return None
        return hourly
    except Exception as exc:
        # Network/parse failures are expected offline; report but do not crash.
        _report_error(f"Weather fetch failed ({location}): {exc}")
        return None


# Models are static during a session, so cache them after the first load to
# keep replans (which happen on every UI interaction) cheap.
_SOLAR_MODEL_CACHE: Any = None
_PPO_AGENT_CACHE: Any = None


def loadSolarModel() -> Optional[Any]:
    global _SOLAR_MODEL_CACHE
    if _SOLAR_MODEL_CACHE is not None:
        return _SOLAR_MODEL_CACHE
    try:
        import joblib

        if not SOLAR_MODEL_PATH.exists():
            _report_error("solar_model.joblib not found — run trainML.py first.")
            return None
        _SOLAR_MODEL_CACHE = joblib.load(SOLAR_MODEL_PATH)
        return _SOLAR_MODEL_CACHE
    except Exception as exc:
        _report_error(f"Could not load solar model: {exc}")
        return None


def generateSolarForecast(
    model: Optional[Any], weather: Optional[Dict[str, Any]]
) -> List[float]:
    if model is not None and weather is not None:
        try:
            import pandas as pd

            rows = []
            for hour in range(HOURS_PER_DAY):
                rows.append(
                    {
                        "cloud_cover": _series_value(weather, "cloud_cover", hour),
                        "sunshine_duration": _series_value(
                            weather, "sunshine_duration", hour
                        ),
                        "temperature_2m": _series_value(weather, "temperature_2m", hour),
                        "is_day": _series_value(weather, "is_day", hour),
                        "hour": float(hour),
                    }
                )
            features = pd.DataFrame(rows, columns=list(SOLAR_FEATURE_ORDER))
            predictions = model.predict(features)
            # Solar generation can never be negative; clamp model noise to zero.
            return [max(float(p), 0.0) for p in predictions]
        except Exception as exc:
            _report_error(f"Solar prediction failed, using fallback: {exc}")

    return _fallback_solar_curve()


def _series_value(weather: Dict[str, Any], key: str, hour: int) -> float:
    try:
        series = weather.get(key, [])
        return float(series[hour])
    except (IndexError, TypeError, ValueError, AttributeError):
        return 0.0


def _fallback_solar_curve() -> List[float]:
    import math

    forecast: List[float] = []
    for hour in range(HOURS_PER_DAY):
        if 6 <= hour <= 18:
            forecast.append(round(5.0 * math.exp(-((hour - 12) ** 2) / 8.0), 3))
        else:
            forecast.append(0.0)
    return forecast


# ---------------------------------------------------------------------------
# RL scheduling pipeline
# ---------------------------------------------------------------------------

def loadPPOAgent() -> Optional[Any]:
    global _PPO_AGENT_CACHE
    if _PPO_AGENT_CACHE is not None:
        return _PPO_AGENT_CACHE
    try:
        from stable_baselines3 import PPO

        if not PPO_MODEL_PATH.exists():
            _report_error("ppo_smarthome.zip not found — run trainRL.py first.")
            return None
        _PPO_AGENT_CACHE = PPO.load(str(PPO_MODEL_PATH))
        return _PPO_AGENT_CACHE
    except Exception as exc:
        _report_error(f"Could not load PPO agent: {exc}")
        return None


# ---------------------------------------------------------------------------
# Planning context: a snapshot of the live UI state the optimizer plans on
# ---------------------------------------------------------------------------

class _PlanContext:

    def __init__(
        self,
        devices: List[Device],
        solar_forecast: List[float],
        accumulator_level: float,
        start_hour: int,
    ) -> None:
        self.start = max(0, min(int(start_hour), HOURS_PER_DAY - 1))
        self.solar = [_solar_at(solar_forecast, h) for h in range(HOURS_PER_DAY)]
        self.battery_initial = max(MIN_CAPACITY_KWH, min(float(accumulator_level), MAX_CAPACITY_KWH))

        # Fixed background load from manual-ON, optimizer-OFF devices.
        self.background = 0.0
        # Per-smart-device data, in stable catalog order.
        self.smart_names: List[str] = []
        self.consumption: Dict[str, float] = {}
        self.forced_hours: Dict[str, set] = {}
        self.required: Dict[str, int] = {}

        for device in devices:
            try:
                name = device.get_name()
                power = max(float(device.get_hourly_power_consumption_kwh()), 0.0)
                smart = bool(device.get_smart_optimizer_enabled())
                manual_on = bool(device.get_is_on())
            except Exception:
                continue

            if not smart:
                if manual_on:
                    self.background += power
                continue

            # Priority hours still in the future (we cannot run the past).
            forced = {h for h in _priority_hours_for(device) if h >= self.start}
            target = max(int(_safe_runtime(device)), len(forced))

            self.smart_names.append(name)
            self.consumption[name] = power
            self.forced_hours[name] = forced
            self.required[name] = target

    def hours(self) -> List[int]:
        return list(range(self.start, HOURS_PER_DAY))


def _safe_runtime(device: Device) -> int:
    try:
        return int(device.get_daily_runtime_hours())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Energy accounting (single source of truth for cost + battery trajectory)
# ---------------------------------------------------------------------------

def _simulate(on_hours: Dict[str, List[int]], ctx: _PlanContext) -> Dict[str, Any]:
    on_sets = {name: set(hrs) for name, hrs in on_hours.items()}
    level = ctx.battery_initial
    grid_by_hour = [0.0] * HOURS_PER_DAY
    charge_by_hour = [0.0] * HOURS_PER_DAY
    discharge_by_hour = [0.0] * HOURS_PER_DAY
    battery_by_hour = [level] * HOURS_PER_DAY
    grid_total = 0.0

    for h in ctx.hours():
        load = ctx.background
        for name in ctx.smart_names:
            if h in on_sets.get(name, set()):
                load += ctx.consumption[name]

        solar = ctx.solar[h]
        if solar >= load:
            surplus = solar - load
            stored = min(surplus, MAX_CAPACITY_KWH - level)
            level += stored
            charge_by_hour[h] = stored
        else:
            deficit = load - solar
            released = min(deficit, level)
            level -= released
            discharge_by_hour[h] = released
            grid_by_hour[h] = deficit - released

        battery_by_hour[h] = level
        grid_total += grid_by_hour[h]

    return {
        "grid_by_hour": grid_by_hour,
        "charge_by_hour": charge_by_hour,
        "discharge_by_hour": discharge_by_hour,
        "battery_by_hour": battery_by_hour,
        "grid_total": grid_total,
        "grid_cost": grid_total * GRID_PRICE_RON_PER_KWH,
    }


def _ranked_hours(ctx: _PlanContext) -> List[int]:
    return sorted(ctx.hours(), key=lambda h: (-ctx.solar[h], h))


def _greedy_decisions(ctx: _PlanContext) -> Dict[str, List[int]]:
    on_hours: Dict[str, set] = {name: set(ctx.forced_hours[name]) for name in ctx.smart_names}
    ranked = _ranked_hours(ctx)

    for name in sorted(ctx.smart_names, key=lambda n: -ctx.consumption[n]):
        need = ctx.required[name] - len(on_hours[name])
        for h in ranked:
            if need <= 0:
                break
            if h in on_hours[name]:
                continue
            on_hours[name].add(h)
            need -= 1

    return {name: sorted(hrs) for name, hrs in on_hours.items()}


def _rl_decisions(ctx: _PlanContext, agent: Any, devices: List[Device]
) -> Optional[Dict[str, List[int]]]:
    try:
        from trainRL import SmartHomeEnv, rollout_plan
    except Exception:
        return None

    if not devices:
        return None

    # The agent was trained on the full catalog order; build matching arrays.
    consumption: List[float] = []
    smart_flags: List[bool] = []
    manual_states: List[bool] = []
    priority_hours: List[set] = []
    required: List[int] = []
    names: List[str] = []
    for device in devices:
        try:
            name = device.get_name()
            power = max(float(device.get_hourly_power_consumption_kwh()), 0.0)
            smart = bool(device.get_smart_optimizer_enabled())
        except Exception:
            continue
        names.append(name)
        consumption.append(power)
        smart_flags.append(smart)
        manual_states.append(bool(device.get_is_on()) and not smart)
        priority_hours.append(ctx.forced_hours.get(name, set()) if smart else set())
        required.append(ctx.required.get(name, 0) if smart else 0)

    try:
        env = SmartHomeEnv(
            consumption=consumption,
            smart_flags=smart_flags,
            manual_states=manual_states,
            priority_hours=priority_hours,
            required_runtime=required,
            solar_forecast=ctx.solar,
            accumulator_initial_kwh=ctx.battery_initial,
            start_hour=ctx.start,
            randomize=False,
        )
        records = rollout_plan(env, agent)
    except Exception:
        return None

    # Collect the agent's ON hours for smart devices only.
    on_hours: Dict[str, set] = {name: set() for name in ctx.smart_names}
    smart_index = {i: names[i] for i in range(len(names)) if smart_flags[i] and names[i] in on_hours}
    for rec in records:
        hour = rec.get("hour")
        states = rec.get("device_on", [])
        for i, name in smart_index.items():
            if i < len(states) and states[i]:
                on_hours[name].add(int(hour))

    # Repair: enforce priority + required runtime regardless of policy quality.
    ranked = _ranked_hours(ctx)
    for name in ctx.smart_names:
        on_hours[name] |= set(ctx.forced_hours[name])
        need = ctx.required[name] - len(on_hours[name])
        for h in ranked:
            if need <= 0:
                break
            if h in on_hours[name]:
                continue
            on_hours[name].add(h)
            need -= 1

    return {name: sorted(hrs) for name, hrs in on_hours.items()}


def buildPlannerSchedule(
    devices: List[Device],
    solar_forecast: List[float],
    agent: Optional[Any] = None,
    accumulator_level: float = MIN_CAPACITY_KWH,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ctx = _PlanContext(devices, solar_forecast, accumulator_level, getLocationCurrentHour())

    if not ctx.smart_names and ctx.background <= 0.0:
        return [], _empty_plan_data(ctx)

    greedy = _greedy_decisions(ctx)
    greedy_sim = _simulate(greedy, ctx)
    chosen, sim, source = greedy, greedy_sim, "heuristic optimizer"

    if agent is not None:
        rl = _rl_decisions(ctx, agent, devices)
        if rl is not None:
            rl_sim = _simulate(rl, ctx)
            # Prefer the RL plan whenever it is at least as cheap as greedy.
            if rl_sim["grid_total"] <= greedy_sim["grid_total"] + 1e-9:
                chosen, sim, source = rl, rl_sim, "RL agent"

    schedule = _build_blocks(chosen, ctx, sim)
    plan_data = {
        "device_hours": {name: list(hrs) for name, hrs in chosen.items()},
        "grid_by_hour": sim["grid_by_hour"],
        "battery_by_hour": sim["battery_by_hour"],
        "daily_grid_kwh": round(sim["grid_total"], 3),
        "daily_grid_cost": round(sim["grid_cost"], 2),
        "source": source,
    }
    return schedule, plan_data


def _empty_plan_data(ctx: _PlanContext) -> Dict[str, Any]:
    return {
        "device_hours": {},
        "grid_by_hour": [0.0] * HOURS_PER_DAY,
        "battery_by_hour": [ctx.battery_initial] * HOURS_PER_DAY,
        "daily_grid_kwh": 0.0,
        "daily_grid_cost": 0.0,
        "source": "none",
    }


def _build_blocks(
    on_hours: Dict[str, List[int]], ctx: _PlanContext, sim: Dict[str, Any]
) -> List[Dict[str, Any]]:
    schedule: List[Dict[str, Any]] = []

    # Device run blocks.
    for name in ctx.smart_names:
        power_per_hour = ctx.consumption[name]
        for start, end in _merge_hour_set(set(on_hours.get(name, []))):
            duration = end - start
            schedule.append(
                {
                    "start_time": f"{start:02d}:00",
                    "device": name,
                    "end_time": f"{end:02d}:00",
                    "power_estimated_kwh": round(power_per_hour * duration, 2),
                }
            )

    # Accumulator charge / discharge blocks, so the user sees when and how much
    # the battery is filled from solar and drained to avoid the grid.
    schedule.extend(_battery_blocks(sim["charge_by_hour"], ACC_CHARGE_LABEL))
    schedule.extend(_battery_blocks(sim["discharge_by_hour"], ACC_DISCHARGE_LABEL))

    schedule.sort(key=lambda block: (block["start_time"], block["device"]))
    return schedule


def _battery_blocks(per_hour: List[float], label: str) -> List[Dict[str, Any]]:
    active = {h for h in range(HOURS_PER_DAY) if per_hour[h] > 1e-6}
    blocks: List[Dict[str, Any]] = []
    for start, end in _merge_hour_set(active):
        energy = sum(per_hour[h] for h in range(start, end))
        blocks.append(
            {
                "start_time": f"{start:02d}:00",
                "device": label,
                "end_time": f"{end:02d}:00",
                "power_estimated_kwh": round(energy, 2),
            }
        )
    return blocks


def _solar_at(forecast: List[float], hour: int) -> float:
    try:
        return float(forecast[hour])
    except (IndexError, TypeError, ValueError):
        return 0.0


def _merge_hour_set(hours: set) -> List[Tuple[int, int]]:
    blocks: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for hour in range(HOURS_PER_DAY + 1):
        active = hour in hours and hour < HOURS_PER_DAY
        if active and start is None:
            start = hour
        elif not active and start is not None:
            blocks.append((start, hour))
            start = None
    return blocks




def detectStateChanges() -> bool:
    try:
        current = _compute_state_hash()
        previous = _safe_session_get(KEY_LAST_OVERRIDE_HASH)
        return current != previous
    except Exception:
        # If we cannot tell, replanning is the safe choice.
        return True


def needsReplan() -> bool:
    flagged = bool(_safe_session_get(KEY_NEEDS_REPLAN, False))
    return flagged or detectStateChanges()


def _compute_state_hash() -> str:
    devices = _safe_session_get(KEY_DEVICES, [])
    signature_parts: List[str] = []
    for device in devices if isinstance(devices, list) else []:
        try:
            smart = bool(device.get_smart_optimizer_enabled())
            signature_parts.append(
                "|".join(
                    [
                        str(device.get_name()),
                        str(smart),
                        str(device.get_priority_window()),
                        str(device.get_daily_runtime_hours()),
                        # Background state only; smart state is plan-derived.
                        str(device.get_is_on()) if not smart else "X",
                    ]
                )
            )
        except Exception:
            continue

    signature_parts.append(f"hour={getLocationCurrentHour()}")
    signature_parts.append(f"loc={_safe_session_get(KEY_SELECTED_LOCATION)}")
    signature_parts.append(f"weather={_safe_session_get(KEY_CURRENT_WEATHER)}")
    signature_parts.append(f"battery={round(_accumulator_level(), 2)}")
    signature_parts.append(f"solar_model={SOLAR_MODEL_PATH.exists()}")
    signature_parts.append(f"ppo_model={PPO_MODEL_PATH.exists()}")

    raw = "::".join(signature_parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def updateScheduleInSessionState(
    schedule: List[Dict[str, Any]], plan_data: Dict[str, Any]
) -> None:
    try:
        st.session_state[KEY_PLANNER_SCHEDULE] = schedule
        st.session_state[KEY_PLAN_DEVICE_HOURS] = plan_data.get("device_hours", {})
        st.session_state[KEY_PLAN_SOURCE] = plan_data.get("source", "none")
        st.session_state[KEY_DAILY_GRID_KWH] = plan_data.get("daily_grid_kwh", 0.0)
        st.session_state[KEY_DAILY_GRID_COST] = plan_data.get("daily_grid_cost", 0.0)

        # Make the plan real for the current hour so the live metrics agree with
        # it: every smart device is switched to its planned state right now.
        _apply_current_hour_state(plan_data.get("device_hours", {}))

        # Current-hour grid draw straight from the simulated plan.
        hour = getLocationCurrentHour()
        grid_by_hour = plan_data.get("grid_by_hour", [0.0] * HOURS_PER_DAY)
        current_grid = float(grid_by_hour[hour]) if 0 <= hour < len(grid_by_hour) else 0.0
        st.session_state[KEY_GRID_CONSUMPTION] = current_grid
        st.session_state[KEY_GRID_COST] = getCurrentGridCost(current_grid)

        # Record the state we just planned for, and clear the replan trigger.
        st.session_state[KEY_LAST_OVERRIDE_HASH] = _compute_state_hash()
        st.session_state[KEY_NEEDS_REPLAN] = False
    except Exception as exc:
        _report_error(f"Failed to update planner state: {exc}")


def _apply_current_hour_state(device_hours: Dict[str, List[int]]) -> None:
    try:
        hour = getLocationCurrentHour()
        devices = _safe_session_get(KEY_DEVICES, [])
        for device in devices if isinstance(devices, list) else []:
            try:
                if not device.get_smart_optimizer_enabled():
                    continue
                planned_hours = device_hours.get(device.get_name(), [])
                device.set_is_on(hour in set(planned_hours))
            except Exception:
                continue
    except Exception:
        pass



def orchestrate(force: bool = False) -> List[Dict[str, Any]]:
    ensureSessionDefaults()

    if not force and not needsReplan():
        return _safe_session_get(KEY_PLANNER_SCHEDULE, [])


    location = _safe_session_get(KEY_SELECTED_LOCATION, "Bucharest")
    weather = fetchWeatherForecast(location)
    if weather is not None:
        try:
            st.session_state[KEY_CURRENT_WEATHER] = _series_value(
                weather, "temperature_2m", getLocationCurrentHour(location)
            )
        except Exception:
            pass

    solar_model = loadSolarModel()
    solar_forecast = generateSolarForecast(solar_model, weather)
    try:
        st.session_state[KEY_SOLAR_FORECAST] = solar_forecast
    except Exception:
        pass


    agent = loadPPOAgent()
    devices = _safe_session_get(KEY_DEVICES, [])
    devices = devices if isinstance(devices, list) else []

    try:
        schedule, plan_data = buildPlannerSchedule(
            devices, solar_forecast, agent, _accumulator_level()
        )
    except Exception as exc:
        _report_error(f"Planner generation failed, using empty schedule: {exc}")
        schedule, plan_data = [], {
            "device_hours": {},
            "grid_by_hour": [0.0] * HOURS_PER_DAY,
            "battery_by_hour": [0.0] * HOURS_PER_DAY,
            "daily_grid_kwh": 0.0,
            "daily_grid_cost": 0.0,
            "source": "none",
        }


    updateScheduleInSessionState(schedule, plan_data)
    return schedule



def _device_is_smart(device: Device) -> bool:
    try:
        return bool(device.get_smart_optimizer_enabled())
    except Exception:
        return False


def _priority_hours_for(device: Device) -> List[int]:
    try:
        window = device.get_priority_window()
        if not window:
            return []
        start, end = window
        start_hour = max(0, min(int(getattr(start, "hour", start)), HOURS_PER_DAY - 1))
        end_hour = max(0, min(int(getattr(end, "hour", end)), HOURS_PER_DAY - 1))
        if start_hour > end_hour:
            return []
        return list(range(start_hour, end_hour + 1))
    except Exception:
        return []


def _accumulator_level() -> float:
    try:
        acc = _safe_session_get(KEY_ACCUMULATOR)
        if isinstance(acc, Accumulator):
            return max(float(acc.getAccumulatorPower()), 0.0)
        return max(float(_safe_session_get(KEY_ACCUMULATOR_POWER, 0.0)), 0.0)
    except Exception:
        return 0.0


def _report_error(message: str) -> None:
    try:
        st.error(message)
    except Exception:
        pass
