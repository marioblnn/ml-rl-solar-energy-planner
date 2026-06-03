
# Smart Home Energy Optimizer — Project Specification

## Role
You are an expert Full-Stack Python Developer specializing in Streamlit, scikit-learn, Gymnasium, and Stable-Baselines3.

Your task is to build a **single-page smart energy management application** that:
- predicts solar generation,
- estimates household consumption,
- manages device schedules specifically for Smart-Optimized devices,
- respects manual user overrides and non-negotiable priority windows,
- uses an RL agent to optimize appliance operation to draw as little grid energy as possible,
- and displays a clean Streamlit dashboard.

The implementation must be robust, readable, and easy to extend.

---

## Core Product Goal
Build an application that helps a household reduce grid energy usage to the absolute minimum possible value by:
1. predicting solar generation from weather data,
2. calculating current consumption,
3. scheduling flexible appliances intelligently (only those with Smart Optimizer ON),
4. strictly enforcing user priority windows at specific hours as non-negotiable constraints,
5. storing state safely across Streamlit reruns.

The app must work as a **single-page dashboard**. Do not create multi-page navigation.

---

## Non-Negotiable Rules
- Use **Streamlit** for the UI.
- Use **`st.session_state`** for all important state.
- The app must never crash due to missing session keys.
- All external operations must be wrapped in `try/except`.
- Use clear inline comments explaining the “why”, not just the “what”.
- The code must be structured so another developer can understand it quickly.
- Do not add unsupported features that are not described here.
- Do not show any “Grid Load capacity” metric in the UI.

---

## Overall Architecture
The project is split into the following files:

1.  `dashboard.py`
2.  `devices.py`
3.  `accumulator.py`
4.  `gridConsumption.py`
5.  `utilities.py`
6.  `papeg.py`
7.  `trainML.py`
8.  `trainRL.py`

The application flow is:
-  `dashboard.py` renders the UI and reads session state.
-  `utilities.py` contains helper functions used by the dashboard.
-  `papeg.py` orchestrates model loading, weather fetching, solar prediction, RL planning, and daily schedule creation.
-  `trainML.py` trains the solar generation prediction model.
-  `trainRL.py` trains the RL scheduling agent.
-  `devices.py`, `accumulator.py`, and `gridConsumption.py` contain the core domain logic.

---

# File 1: `dashboard.py`

This is the main entry point for the Streamlit app.

## Required behavior
At the very top of the file, before any UI is rendered, initialize all required `st.session_state` keys.

### Session state must include at minimum:
- device manual on/off states
- device smart optimizer enabled states
- device priority windows
- accumulator power
- current selected location
- current weather
- solar generation forecast
- planner schedule
- replan trigger flag
- last known override hash
- grid consumption value
- grid cost value

The app must be safe if Streamlit reruns on any interaction.

## Sidebar
The sidebar must include:
- “Power grid cost: [variable] RON / kWh”,
- “Power grid consumption: [variable .2f] kWh”.

## Main area
The main layout must be single-page and organized vertically.

### Row 1
Display two metrics side by side:
- Location Name
- Current Weather in °C
- Current Time

### Row 2
Display three metrics:
- Generating: ~[variable .2f] kWh
- Consuming: ~[variable .2f] kWh
- Accumulator: [variable .2f] kWh

## Graph section
Show a section titled:

**Today’s estimated energy generation**

Render a line chart with:
- X axis: time in hours from 00:00 to 23:00 (skips = 1h)
- Y axis: generated kWh from 0 to 20

## Device control layout
Display the following devices as vertically stacked cards, one below another:

| Smart Device Name | Hourly Consumption (kWh) |
| --- | ---: |
| EV Charger (Level 2) | ~3.60 |
| Smart Water Heater | ~2.50 |
| HVAC (Air Conditioning) | ~1.50 |
| Dishwasher | ~1.20 |
| Washing Machine | ~1.00 |
| Main Computer | ~0.50 |
| Computer Room 1 | ~0.25 |
| Smart fridge | ~0.15 |
| Main TV | ~0.15 |
| TV Room 2 | ~0.10 |

Each device card must show:
- device name,
- consumption per hour,
- toggle: Turn ON/OFF,
- toggle: Smart Optimizer ON/OFF,
- button: Set Priority.

### Crucial AI interaction rules
#### Manual ON/OFF toggle
This is the user’s manual hardware state. 

#### Smart Optimizer toggle
- If ON: the RL agent parses this device and attempts to schedule it efficiently around priority windows.
- If OFF: the RL agent must completely ignore this device.
- When smart optimizer is OFF, the device state is immutable from the RL agent’s perspective.
- A disabled optimizer device becomes a background load controlled only by the user’s manual toggle.
- The RL agent must optimize the remaining devices around it.

## Today’s Planner section
Show a section titled:

**Today’s Planner**

Render a table with:
- Start Time
- Device
- End Time
- Power Estimated (kWh)

The planner table should always reflect the latest generated schedule. Only devices with Smart Optimizer ON should appear as scheduled blocks.

---

# File 2: `devices.py`

Create a `Device` class representing one smart appliance.

## Required attributes
-  `name`
-  `hourly_power_consumption_kwh`
-  `is_on` as a boolean
-  `smart_optimizer_enabled` as a boolean
-  `priority_window` as a tuple `(start_datetime, end_datetime)` or `None`

## Recommended behavior
Include simple getter and setter methods for each attribute.

## Important design note
The device object should be lightweight and should work well with `st.session_state` storage.

---

# File 3: `accumulator.py`

Create an `Accumulator` class that behaves like a simple battery model.

## Required behavior
- Starting capacity: `0 kWh`
- Maximum capacity: `12 kWh`
- Method: `getAccumulatorPower()`
  - returns current stored power

## Recommended behavior
Include helper methods for:
- charging,
- discharging,
- clamping values to the valid range,
- optionally resetting the accumulator.

The battery should never exceed 12 kWh and never go below 0 kWh.

---

# File 4: `gridConsumption.py`

This file manages grid usage calculations.

## Fixed price rule
The grid energy price is always:

**1.2 RON per kWh**

## Required functions

### `getCurrentPowerGridConsumption(required_kwh, generated_kwh)`
Returns the deficit between required energy and generated energy.

Example:
- required = 4 kWh
- generated = 2 kWh
- result = 2 kWh

If generation meets or exceeds demand, return `0`.

### `getCurrentGridCost(consumed_kwh)`
Returns the financial cost in RON for the consumed grid energy.

Formula:
`consumed_kwh * 1.2`

## Recommended behavior
Add validation so negative inputs do not produce invalid results.

---

# File 5: `utilities.py`

This file contains helper functions used by `dashboard.py`.

## Required functions

### `getWeatherBadge()`
Returns the current temperature as a float.

### `getLocationName()`
Returns the current selected location string.

### `getSolarConsumption()`
Returns the estimated solar generation for the current hour.

### `getDevicesConsumption()`
Returns the total combined consumption of all currently active household devices.

### `getTodaysGeneratedEnergyApproximations()`
Returns a list of 24 float values, one for each hour from 0 to 23, used for the dashboard line graph.

### `setPriority(device_name, start_datetime, end_datetime)`
Triggered from a native Streamlit overlay such as `st.popover` or `st.dialog` attached to the “Set Priority” button.

Rules:
- save the priority window into the device’s session state,
- only accept integer hours,
- validate that `start_hour <= end_hour`,
- **NON-NEGOTIABLE TRIGGER:** immediately flag the planner for replanning after the update. The new schedule MUST respect these hours.

### `getPowerGridConsumption()`
Wrapper function that returns current real-time grid draw.

### `getGridCost()`
Wrapper function that returns current grid cost based on the fixed 1.2 RON/kWh rate.

## Recommended behavior
These helpers should be small, predictable, and safe to call during Streamlit reruns.

---

# File 6: `papeg.py`
## Predict And Plan Energy Generation

This file is the orchestration layer for the project.

It must:
1. fetch weather data from the Open-Meteo API,
2. load `solar_model.joblib`,
3. predict hourly solar generation,
4. load `ppo_smarthome.zip`,
5. check all devices that have Smart Optimizer ON,
6. check the list of Priorities and forcefully bind them to specific hours,
7. generate the daily energy plan (triggered each time a device toggle, optimizer toggle, or priority window is changed),
8. update the dashboard schedule,
9. trigger replanning whenever the user changes a relevant control.

This file should be the main coordination point between the ML model, RL policy, and Streamlit state.

## Required responsibilities
### Weather and solar pipeline
- pull weather forecast data from Open-Meteo,
- convert the forecast into model-ready features,
- call the trained solar prediction model,
- produce 24-hour solar generation estimates from the current datetime.

### RL scheduling pipeline
- load the trained PPO agent,
- construct the environment using current UI state.
- **CRITICAL:** Scan the device list. Only pass devices to the scheduler where `smart_optimizer == True`.
- generate a schedule for the day from the current datetime.
- format the result for the planner table.

### Replanning logic
The app must instantly trigger a re-plan when:
- a user manually changes a device state,
- a user changes a smart optimizer state,
- a user edits a priority window (**Priority windows trigger a full replan considering those specific hours as non-negotiable**),
- the selected location changes,
- the weather forecast updates,
- the solar model or PPO model is reloaded.

## Required helper functions
Use clear internal helper functions such as:
-  `fetchWeatherForecast()`
-  `loadSolarModel()`
-  `loadPPOAgent()`
-  `generateSolarForecast()`
-  `buildPlannerSchedule()`
-  `detectStateChanges()`
-  `needsReplan()`
-  `updateScheduleInSessionState()`

## Required safety behavior
Every external dependency must be wrapped in `try/except`.
If a file is missing or a model cannot be loaded:
- do not crash,
- show `st.error()`,
- fall back to a safe empty or rule-based schedule if necessary.

## Schedule format
The daily schedule should be stored as a list of dictionaries, each containing:
-  `start_time`
-  `device`
-  `end_time`
-  `power_estimated_kwh`

---

# File 7: `trainML.py`

This file trains the solar generation prediction model using scikit-learn.

## Purpose
Predict solar energy generation from weather and time features.

## Required functions

### `loadAndPreprocessData()`
Load `solar_data.csv` using pandas.

### Required input columns
-  `timestamp`
-  `cloud_cover`
-  `sunshine_duration`
-  `temperature_2m`
-  `is_day`
-  `generation_kwh`

### Required preprocessing
- Parse `timestamp` as a datetime column.
- Extract the hour from the timestamp as a continuous feature.
- Remove invalid rows safely.
- Handle missing values explicitly.

### Feature set
Use: `cloud_cover`, `sunshine_duration`, `temperature_2m`, `is_day`, `hour`.
Target: `generation_kwh`.

### `trainRandomForest()`
Train a `RandomForestRegressor` on the processed data (80% train / 20% test).

### `evaluateAndSaveModel()`
- calculate MAE and R2 score, print clearly,
- save the final model as `solar_model.joblib`.

---

# File 8: `trainRL.py`

This file trains the autonomous scheduling agent using Gymnasium and Stable-Baselines3.

## Purpose
Train a PPO agent that decides how to schedule flexible household devices across a 24-hour cycle while strictly minimizing grid energy usage and completely respecting non-negotiable priority windows and manual overrides.

---

## Environment: `class SmartHomeEnv(gym.Env)`

The environment must accept the current UI/configuration state at initialization.

## Initialization inputs
The environment should accept:
- the list of devices,
- which devices are smart-optimizer enabled,
- manual on/off states,
- **priority windows (treated as absolute constraints)**,
- predicted solar generation for 24 hours,
- accumulator settings.

## Observation Space
Use `spaces.Box`.
The observation must include at least:
1. current hour, ranging from 0 to 23,
2. predicted solar generation,
3. accumulator power,
4. hours of EV charging accumulated,
5. current priority demands (boolean flags or time-remaining to help the agent).

---

## Action Space
Use `spaces.MultiDiscrete`.
Each action should represent device scheduling decisions for the current hour.

### Mandatory rule: Smart Optimizer OFF
The `step()` function must strictly ignore actions for devices where `smart_optimizer == False`.
For those devices:
- the RL agent cannot change their state,
- the device behaves as a background load,
- manual state always wins.

### Mandatory rule: Non-Negotiable Priority Windows
If a device has a priority window set for a specific hour, **the action is non-negotiable**.
- The environment MUST override the agent's action and force the device ON during these specific hours.
- Priority windows are hard constraints; the RL agent's job is to optimize *around* these fixed blocks to ensure the grid penalty remains as low as possible.

### Recommended action design
Each device may have an action such as:
- 0 = OFF / do not run this hour
- 1 = ON / run this hour

---

## Reward function: `calculateReward()`

The reward function must aggressively drive grid energy usage to zero. 

### Core formula
Let:
`Net Energy = (Solar Generation + Accumulator Power) - Total Appliance Consumption`

### Reward logic
- If `Net Energy < 0`:
  - apply a severe penalty proportional to grid cost (e.g., `-10 * Grid Deficit`).
  - **The reward should heavily penalize ANY grid usage to ensure the agent uses as little grid energy as mathematically possible.**

- If `Net Energy >= 0`:
  - give a positive reward of `+10` for maintaining self-sufficiency.

### Priority window penalty
Because priority windows are forced ON by the environment, the agent doesn't "choose" to violate them. However, if an appliance requires a certain amount of *total* runtime that wasn't covered purely by the priority window, and the agent failed to schedule the remainder:
- apply a terminal penalty of `-1000`.

---

## Episode structure
- One episode represents one full day (hour 0 to 23).
- At the end of the day, the episode should terminate cleanly.

## `step()` behavior
The `step()` function should:
1. read the action vector,
2. ignore actions for optimizer-locked devices,
3. **override actions to ON for devices currently within their Priority Window,**
4. apply device states for the current hour,
5. calculate consumption and solar contribution,
6. update accumulator logic,
7. compute reward (heavily penalizing grid usage),
8. advance time by one hour,
9. return observation, reward, done flag, truncation flag, info dict.

## `reset()` behavior
- restore the hour to 0, reset accumulator and runtime counters.

## Accumulator behavior
- store excess solar energy when available, discharge to reduce grid usage, stay between 0 and 12 kWh.

## `trainAndSavePPO()`
Train PPO for `50,000` timesteps and save as `ppo_smarthome.zip`.

---

# File 9: `requirements.txt`

When the project implementation is complete, generate a `requirements.txt` file containing every third-party dependency required by the application.

## Requirements
- Include all external packages imported (e.g., `streamlit`, `pandas`, `stable-baselines3`, `gymnasium`).
- Do not include Python standard library modules.
- Pin package versions (`package_name==version`).
- Ensure compatibility with Python 3.11+.

## Installation Instructions
```bash
pip install -r requirements.txt
