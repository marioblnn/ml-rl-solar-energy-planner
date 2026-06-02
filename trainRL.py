from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from accumulator import MAX_CAPACITY_KWH, MIN_CAPACITY_KWH

HOURS_PER_DAY = 24
MODEL_PATH = Path(__file__).resolve().parent / "ppo_smarthome"  
TRAIN_TIMESTEPS = 2_500_000
GRID_PRICE_RON_PER_KWH = 1.2  

# Reward shaping. The dominant signal is the (negative) grid cost so the agent
# learns to buy as little as possible; the two extra terms only break ties.
SELF_SUFFICIENT_BONUS = 0.2  # per hour that needs zero grid energy
UNMET_RUNTIME_PENALTY = 20.0  # per device-hour of required runtime left unmet

# Used only to normalise the observation into [0, 1]; real generation peaks far
# below this, so clipping keeps the inputs well-scaled for the MLP policy.
SOLAR_SCALE = 15.0

# Device catalog mirrors the dashboard table: (name, hourly kWh, default runtime).
_DEFAULT_DEVICE_SPECS: List[Tuple[str, float, int]] = [
    ("EV Charger", 3.60, 3),
    ("Smart Water Heater", 2.50, 3),
    ("HVAC (Air Conditioning)", 1.50, 6),
    ("Dishwasher", 1.20, 2),
    ("Washing Machine", 1.00, 2),
    ("Main Computer", 0.50, 8),
    ("Computer Room 1", 0.25, 4),
    ("Smart fridge", 0.15, 24),
    ("Main TV", 0.15, 4),
    ("TV Room 2", 0.10, 3),
]


class SmartHomeEnv(gym.Env):

    metadata = {"render_modes": []} #required by gymnasium

    def __init__(self, consumption: Sequence[float], smart_flags: Optional[Sequence[bool]] = None, manual_states: Optional[Sequence[bool]] = None, priority_hours: Optional[Sequence[Set[int]]] = None, required_runtime: Optional[Sequence[int]] = None, solar_forecast: Optional[Sequence[float]] = None, accumulator_initial_kwh: float = MIN_CAPACITY_KWH, start_hour: int = 0, randomize: bool = False, seed: Optional[int] = None) -> None:
        super().__init__()

        self.consumption = np.asarray(consumption, dtype=np.float32)
        self.num_devices = int(self.consumption.shape[0])
        if self.num_devices == 0:
            raise ValueError("SmartHomeEnv requires at least one device.")

        self.randomize = bool(randomize)
        self._rng = np.random.default_rng(seed)


        self._cfg_smart = self._as_bool_array(smart_flags, default=True)
        self._cfg_manual = self._as_bool_array(manual_states, default=False)
        self._cfg_priority = self._normalise_priority(priority_hours)
        self._cfg_required = self._as_int_array(required_runtime, default=0)
        self._cfg_solar = self._normalise_solar(solar_forecast)
        self._cfg_battery = float(np.clip(accumulator_initial_kwh, MIN_CAPACITY_KWH, MAX_CAPACITY_KWH)) #clipping to ensure the battery level is within the valid range
        self._cfg_start = int(np.clip(start_hour, 0, HOURS_PER_DAY - 1))


        self.action_space = spaces.MultiDiscrete([2] * self.num_devices) #2 options for each device: OFF/ON


        obs_dim = 5 + 2 * self.num_devices #5 base features + 2 for each device
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._apply_config()
        self.current_hour = self.start_hour
        self.level = self._cfg_battery
        self.remaining = self.required_runtime.astype(np.float32).copy() 


    def _as_bool_array(self, values: Optional[Sequence[bool]], default: bool) -> np.ndarray:
        if values is not None and len(values) == self.num_devices:
            return np.array([bool(v) for v in values], dtype=bool)
        return np.full(self.num_devices, default, dtype=bool)

    def _as_int_array(self, values: Optional[Sequence[int]], default: int) -> np.ndarray:
        if values is not None and len(values) == self.num_devices:
            return np.array([max(0, int(v)) for v in values], dtype=np.int32)
        return np.full(self.num_devices, default, dtype=np.int32)



    def _normalise_priority(
        self, priority_hours: Optional[Sequence[Set[int]]]
    ) -> List[Set[int]]:
        if priority_hours is not None and len(priority_hours) == self.num_devices:
            return [
                {int(h) for h in hours if 0 <= int(h) < HOURS_PER_DAY}
                for hours in priority_hours
            ]
        return [set() for _ in range(self.num_devices)]


    def _normalise_solar(self, solar_forecast: Optional[Sequence[float]]) -> np.ndarray:
        forecast = np.zeros(HOURS_PER_DAY, dtype=np.float32)
        if solar_forecast is not None:
            for i in range(min(HOURS_PER_DAY, len(solar_forecast))):
                forecast[i] = max(float(solar_forecast[i]), 0.0)
        return forecast


    def _apply_config(self) -> None:
        self.smart_flags = self._cfg_smart.copy()
        self.manual_states = self._cfg_manual.copy()
        self.priority_hours = [set(h) for h in self._cfg_priority]
        self.required_runtime = self._cfg_required.copy()
        self.solar_forecast = self._cfg_solar.copy()
        self.start_hour = self._cfg_start



    def _randomize_scenario(self) -> None:
        rng = self._rng

        peak_hour = rng.uniform(11.0, 14.0)
        amplitude = rng.uniform(2.0, 8.0)
        width = rng.uniform(2.5, 4.5)
        cloud = rng.uniform(0.4, 1.0)
        solar = np.zeros(HOURS_PER_DAY, dtype=np.float32)
        for h in range(HOURS_PER_DAY):
            if 5 <= h <= 19:
                bell = amplitude * np.exp(-((h - peak_hour) ** 2) / (2.0 * width ** 2))
                noise = rng.uniform(0.85, 1.15)
                solar[h] = max(0.0, bell * cloud * noise)
        self.solar_forecast = solar

        # Each device is smart with p=0.6; non-smart devices may be a manual
        # background load with p=0.4.
        self.smart_flags = rng.random(self.num_devices) < 0.6
        self.manual_states = (~self.smart_flags) & (rng.random(self.num_devices) < 0.4)

        # Priority windows on ~25% of smart devices, plus randomised runtimes.
        priority: List[Set[int]] = []
        required = np.zeros(self.num_devices, dtype=np.int32)
        for d in range(self.num_devices):
            if self.smart_flags[d]:
                required[d] = int(rng.integers(0, 7))
                if rng.random() < 0.25:
                    start = int(rng.integers(0, HOURS_PER_DAY - 2))
                    length = int(rng.integers(1, 4))
                    priority.append(set(range(start, min(start + length, HOURS_PER_DAY))))
                else:
                    priority.append(set())
            else:
                priority.append(set())
        self.priority_hours = priority
        self.required_runtime = required

        self.start_hour = int(rng.integers(0, 19))
        self._cfg_battery = float(rng.uniform(MIN_CAPACITY_KWH, MAX_CAPACITY_KWH))

    # --- core gym API ----------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if self.randomize:
            self._randomize_scenario()
        else:
            self._apply_config()

        self.current_hour = self.start_hour
        self.level = float(np.clip(self._cfg_battery, MIN_CAPACITY_KWH, MAX_CAPACITY_KWH))
        self.remaining = self.required_runtime.astype(np.float32).copy()
        return self._build_observation(), {}

    def step(
        self, action: Sequence[int]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        hour = self.current_hour

        device_on = self._resolve_device_states(action, hour)

        load = float(self.consumption[device_on].sum())
        solar = float(self.solar_forecast[hour]) if hour < HOURS_PER_DAY else 0.0
        charge, discharge, grid = self._apply_energy_balance(solar, load)

        # Flat tariff: minimising cost == minimising grid kWh.
        reward = -(grid * GRID_PRICE_RON_PER_KWH)
        if grid <= 1e-9:
            reward += SELF_SUFFICIENT_BONUS

        # Credit runtime for every smart device that actually ran this hour.
        for d in range(self.num_devices):
            if self.smart_flags[d] and device_on[d] and self.remaining[d] > 0:
                self.remaining[d] -= 1

        self.current_hour += 1
        terminated = self.current_hour >= HOURS_PER_DAY
        truncated = False

        info: Dict[str, Any] = {
            "hour": hour,
            "device_on": device_on.tolist(),
            "load_kwh": load,
            "solar_kwh": solar,
            "grid_kwh": grid,
            "charge_kwh": charge,
            "discharge_kwh": discharge,
            "accumulator_kwh": self.level,
        }

        if terminated:
            # Any required runtime the agent failed to schedule is a hard miss.
            unmet = float(
                sum(self.remaining[d] for d in range(self.num_devices) if self.smart_flags[d])
            )
            reward -= UNMET_RUNTIME_PENALTY * unmet
            info["unmet_runtime"] = unmet

        return self._build_observation(), float(reward), terminated, truncated, info

    # --- environment mechanics -------------------------------------------------

    def _resolve_device_states(self, action: np.ndarray, hour: int) -> np.ndarray:
        """Map the raw action to actual ON/OFF states, honouring hard rules."""
        device_on = np.zeros(self.num_devices, dtype=bool)
        for d in range(self.num_devices):
            if self.smart_flags[d]:
                chosen = bool(action[d] == 1) if d < len(action) else False
                # HARD CONSTRAINT: priority hours force the device ON.
                if hour in self.priority_hours[d]:
                    chosen = True
                device_on[d] = chosen
            else:
                # Optimizer OFF -> agent ignored, the user's manual state wins.
                device_on[d] = bool(self.manual_states[d])
        return device_on

    def _apply_energy_balance(
        self, solar: float, load: float
    ) -> Tuple[float, float, float]:
        """Charge surplus solar / discharge to cover deficit; return flows."""
        charge = discharge = grid = 0.0
        if solar >= load:
            surplus = solar - load
            headroom = MAX_CAPACITY_KWH - self.level
            charge = min(surplus, headroom)
            self.level += charge
        else:
            deficit = load - solar
            discharge = min(deficit, self.level)
            self.level -= discharge
            grid = deficit - discharge
        return charge, discharge, grid

    def _background_now(self) -> float:
        """Fixed load from manual-ON, optimizer-OFF devices (background)."""
        total = 0.0
        for d in range(self.num_devices):
            if not self.smart_flags[d] and self.manual_states[d]:
                total += float(self.consumption[d])
        return total

    def _build_observation(self) -> np.ndarray:
        hour = min(self.current_hour, HOURS_PER_DAY - 1)
        if self.current_hour < HOURS_PER_DAY:
            solar_remaining = float(self.solar_forecast[self.current_hour:].sum())
        else:
            solar_remaining = 0.0

        base = np.array(
            [
                hour / (HOURS_PER_DAY - 1),
                min(self.solar_forecast[hour] / SOLAR_SCALE, 1.0),
                min(solar_remaining / (SOLAR_SCALE * HOURS_PER_DAY), 1.0),
                self.level / MAX_CAPACITY_KWH,
                min(self._background_now() / SOLAR_SCALE, 1.0),
            ],
            dtype=np.float32,
        )
        priority_flags = np.array(
            [1.0 if hour in self.priority_hours[d] else 0.0 for d in range(self.num_devices)],
            dtype=np.float32,
        )
        remaining = np.array(
            [min(self.remaining[d] / HOURS_PER_DAY, 1.0) for d in range(self.num_devices)],
            dtype=np.float32,
        )
        obs = np.concatenate([base, priority_flags, remaining]).astype(np.float32)
        return np.clip(obs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Deterministic rollout (shared with papeg for inference)
# ---------------------------------------------------------------------------

def rollout_plan(env: SmartHomeEnv, agent: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Run one full deterministic episode and return the per-hour info dicts.

    When ``agent`` is None every action is OFF, so only forced priority hours
    and manual background loads run — a safe, dependency-free baseline.
    """
    obs, _ = env.reset()
    records: List[Dict[str, Any]] = []
    done = False
    while not done:
        if agent is not None:
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.num_devices, dtype=np.int64)
        obs, _reward, terminated, truncated, info = env.step(action)
        records.append(info)
        done = terminated or truncated
    return records


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

def _default_solar_forecast() -> List[float]:
    """Load a representative day from solar_data.csv, else use a bell curve."""
    csv_path = Path(__file__).resolve().parent / "solar_data.csv"
    if csv_path.is_file():
        import pandas as pd

        df = pd.read_csv(csv_path)
        values = pd.to_numeric(df["generation_kwh"], errors="coerce").dropna()
        if len(values) >= HOURS_PER_DAY:
            return [max(float(v), 0.0) for v in values.iloc[:HOURS_PER_DAY]]

    forecast: List[float] = []
    for hour in range(HOURS_PER_DAY):
        if 6 <= hour <= 18:
            forecast.append(round(5.0 * np.exp(-((hour - 12) ** 2) / 8.0), 3))
        else:
            forecast.append(0.0)
    return forecast


def make_training_env(seed: Optional[int] = None) -> SmartHomeEnv:
    """A randomising environment over the full device catalog for training."""
    consumption = [kwh for _name, kwh, _rt in _DEFAULT_DEVICE_SPECS]
    return SmartHomeEnv(
        consumption=consumption,
        solar_forecast=_default_solar_forecast(),
        randomize=True,
        seed=seed,
    )


def trainAndSavePPO(
    env: Optional[SmartHomeEnv] = None,
    total_timesteps: int = TRAIN_TIMESTEPS,
    model_path: Path = MODEL_PATH,
) -> str:
    """Train PPO for the configured timesteps and save ppo_smarthome.zip."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    if env is None:
        env = make_training_env(seed=0)

    check_env(env, warn=True)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(str(model_path))

    saved = f"{model_path}.zip"
    print(f"PPO agent saved to: {saved}")
    return saved


if __name__ == "__main__":
    trainAndSavePPO()
