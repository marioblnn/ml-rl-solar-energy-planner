from __future__ import annotations
from datetime import datetime, time
from typing import Any, List
import pandas as pd
import streamlit as st
import papeg
import utilities as utils
from accumulator import Accumulator
from devices import Device
DEVICE_CATALOG = [('EV Charger (Level 2)', 3.6, 3), ('Smart Water Heater', 2.5, 3), ('HVAC (Air Conditioning)', 1.5, 6), ('Dishwasher', 1.2, 2), ('Washing Machine', 1.0, 2), ('Main Computer', 0.5, 8), ('Computer Room 1', 0.25, 4), ('Smart fridge', 0.15, 24), ('Main TV', 0.15, 4), ('TV Room 2', 0.1, 3)]
KEY_DEVICES = 'devices'
KEY_SELECTED_LOCATION = 'selected_location'
KEY_CURRENT_WEATHER = 'current_weather'
KEY_SOLAR_FORECAST = 'solar_generation_forecast'
KEY_ACCUMULATOR = 'accumulator'
KEY_ACCUMULATOR_POWER = 'accumulator_power'
KEY_PLANNER_SCHEDULE = 'planner_schedule'
KEY_NEEDS_REPLAN = 'needs_replan'
KEY_LAST_OVERRIDE_HASH = 'last_override_hash'
KEY_GRID_CONSUMPTION = 'grid_consumption'
KEY_GRID_COST = 'grid_cost'
KEY_DAILY_GRID_KWH = 'daily_grid_kwh'
KEY_DAILY_GRID_COST = 'daily_grid_cost'
KEY_PLAN_DEVICE_HOURS = 'plan_device_hours'
KEY_PLAN_SOURCE = 'plan_source'

def initSessionState() -> None:
    try:
        if KEY_DEVICES not in st.session_state:
            st.session_state[KEY_DEVICES] = [Device(name=name, hourly_power_consumption_kwh=kwh, is_on=False, smart_optimizer_enabled=False, priority_window=None, daily_runtime_hours=runtime) for name, kwh, runtime in DEVICE_CATALOG]
        if KEY_ACCUMULATOR not in st.session_state:
            st.session_state[KEY_ACCUMULATOR] = Accumulator()
        defaults = {KEY_SELECTED_LOCATION: 'Bucharest', KEY_CURRENT_WEATHER: None, KEY_SOLAR_FORECAST: [0.0] * 24, KEY_PLANNER_SCHEDULE: [], KEY_NEEDS_REPLAN: True, KEY_LAST_OVERRIDE_HASH: None, KEY_GRID_CONSUMPTION: 0.0, KEY_GRID_COST: 0.0, KEY_ACCUMULATOR_POWER: 0.0, KEY_DAILY_GRID_KWH: 0.0, KEY_DAILY_GRID_COST: 0.0, KEY_PLAN_DEVICE_HOURS: {}, KEY_PLAN_SOURCE: 'none'}
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    except Exception as exc:
        st.error(f'Failed to initialize session state: {exc}')

def _get_devices() -> List[Device]:
    devices = st.session_state.get(KEY_DEVICES, [])
    return devices if isinstance(devices, list) else []

def renderSidebar() -> None:
    with st.sidebar:
        st.header('Grid Status')
        try:
            grid_cost = utils.getGridCost()
            grid_consumption = utils.getPowerGridConsumption()
        except Exception:
            grid_cost, grid_consumption = (0.0, 0.0)
        st.metric('Power grid cost', f'{grid_cost:.2f} RON / kWh')
        st.metric('Power grid consumption', f'{grid_consumption:.2f} kWh')
        st.divider()
        try:
            daily_kwh = utils.getDailyGridConsumption()
            daily_cost = utils.getDailyGridCost()
        except Exception:
            daily_kwh, daily_cost = (0.0, 0.0)
        st.subheader("Today's Estimate")
        st.metric("Today's estimated cost", f'{daily_cost:.2f} RON')
        st.metric('Estimated grid energy', f'{daily_kwh:.2f} kWh')
        st.divider()
        location_options = list(papeg.LOCATIONS.keys())
        current = st.session_state.get(KEY_SELECTED_LOCATION, location_options[0])
        index = location_options.index(current) if current in location_options else 0
        selected = st.selectbox('Location', location_options, index=index)
        if selected != current:
            st.session_state[KEY_SELECTED_LOCATION] = selected
            st.session_state[KEY_NEEDS_REPLAN] = True

def renderTopMetrics() -> None:
    row1 = st.columns(3)
    row1[0].metric('Location', utils.getLocationName())
    weather = utils.getWeatherBadge()
    row1[1].metric('Current Weather', f'{weather:.1f} °C')
    row1[2].metric('Current Time', utils.getCurrentTime())
    row2 = st.columns(3)
    row2[0].metric('Generating', f'~{utils.getSolarConsumption():.2f} kWh')
    row2[1].metric('Consuming', f'~{utils.getDevicesConsumption():.2f} kWh')
    accumulator_power = _accumulator_power()
    row2[2].metric('Accumulator', f'{accumulator_power:.2f} kWh')

def _accumulator_power() -> float:
    try:
        acc = st.session_state.get(KEY_ACCUMULATOR)
        if isinstance(acc, Accumulator):
            return acc.getAccumulatorPower()
        return float(st.session_state.get(KEY_ACCUMULATOR_POWER, 0.0))
    except Exception:
        return 0.0

def renderGenerationChart() -> None:
    st.subheader("Today's estimated energy generation")
    try:
        approximations = utils.getTodaysGeneratedEnergyApproximations()
        hours = [f'{h:02d}:00' for h in range(24)]
        chart_df = pd.DataFrame({'Generated kWh': approximations}, index=hours)
        chart_df['Generated kWh'] = chart_df['Generated kWh'].clip(0, 20)
        st.line_chart(chart_df, height=280)
    except Exception as exc:
        st.error(f'Could not render generation chart: {exc}')

def _render_priority_control(container: Any, device_name: str, idx: int) -> None:
    with container.popover('Set Priority', use_container_width=True):
        st.write(f'Device: **{device_name}**')
        st.caption('Priority hours are forced ON and treated as hard constraints.')
        col_start, col_end = st.columns(2)
        start_hour = col_start.number_input('Start hour', 0, 23, 8, step=1, key=f'prio_start_{idx}')
        end_hour = col_end.number_input('End hour', 0, 23, 10, step=1, key=f'prio_end_{idx}')
        if st.button('Save priority', use_container_width=True, key=f'prio_save_{idx}'):
            if start_hour > end_hour:
                st.error('Start hour must be less than or equal to end hour.')
                return
            today = utils.getLocationNow().date()
            start_dt = datetime.combine(today, time(int(start_hour)))
            end_dt = datetime.combine(today, time(int(end_hour)))
            if utils.setPriority(device_name, start_dt, end_dt):
                st.success('Priority saved. Replanning...')
                st.rerun()
            else:
                st.error('Could not save priority window.')

def _on_manual_toggle(device: Device, widget_key: str) -> None:
    try:
        device.set_is_on(bool(st.session_state.get(widget_key, False)))
        st.session_state[KEY_NEEDS_REPLAN] = True
    except Exception:
        pass

def _on_optimizer_toggle(device: Device, widget_key: str) -> None:
    try:
        device.set_smart_optimizer_enabled(bool(st.session_state.get(widget_key, False)))
        st.session_state[KEY_NEEDS_REPLAN] = True
    except Exception:
        pass

def _on_runtime_change(device: Device, widget_key: str) -> None:
    try:
        device.set_daily_runtime_hours(int(st.session_state.get(widget_key, 0)))
        st.session_state[KEY_NEEDS_REPLAN] = True
    except Exception:
        pass

def renderDeviceCards() -> None:
    st.subheader('Device Controls')
    for idx, device in enumerate(_get_devices()):
        try:
            name = device.get_name()
            consumption = device.get_hourly_power_consumption_kwh()
        except Exception:
            continue
        manual_key = f'manual_{idx}'
        optimizer_key = f'optimizer_{idx}'
        runtime_key = f'runtime_{idx}'
        is_smart = device.get_smart_optimizer_enabled()
        with st.container(border=True):
            header = st.columns([3, 1])
            running_now = ' · running now' if device.get_is_on() else ''
            header[0].markdown(f'**{name}**{running_now}')
            header[1].markdown(f'`~{consumption:.2f} kWh/h`')
            controls = st.columns([1, 1, 1, 1])
            controls[0].toggle('Turn ON', value=device.get_is_on(), key=manual_key, on_change=_on_manual_toggle, args=(device, manual_key), disabled=is_smart, help='Driven by the optimizer while Smart Optimizer is ON.' if is_smart else None)
            controls[1].toggle('Smart Optimizer', value=is_smart, key=optimizer_key, on_change=_on_optimizer_toggle, args=(device, optimizer_key))
            controls[2].number_input('Run hours/day', min_value=0, max_value=24, value=int(device.get_daily_runtime_hours()), step=1, key=runtime_key, on_change=_on_runtime_change, args=(device, runtime_key), disabled=not is_smart, help='How many hours/day the optimizer should run this device.')
            _render_priority_control(controls[3], name, idx)
            window = device.get_priority_window()
            if window:
                try:
                    st.caption(f'Priority window: {window[0].hour:02d}:00 - {window[1].hour:02d}:00')
                except Exception:
                    pass

def renderPlanner() -> None:
    st.subheader("Today's Planner")
    try:
        schedule = st.session_state.get(KEY_PLANNER_SCHEDULE, [])
        source = utils.getPlanSource()
        daily_cost = utils.getDailyGridCost()
        if source and source != 'none':
            label = 'RL agent' if source == 'RL agent' else 'heuristic optimizer'
            st.caption(f'Plan by: {label} · estimated grid cost today: {daily_cost:.2f} RON')
        if not schedule:
            st.info('No scheduled blocks yet. Enable Smart Optimizer on a device (and give it run hours or a priority window).')
            return
        table_df = pd.DataFrame(schedule)
        column_map = {'start_time': 'Start Time', 'device': 'Device', 'end_time': 'End Time', 'power_estimated_kwh': 'Power Estimated (kWh)'}
        ordered = [c for c in column_map if c in table_df.columns]
        table_df = table_df[ordered].rename(columns=column_map)
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f'Could not render planner: {exc}')

def main() -> None:
    st.set_page_config(page_title='Smart Home Energy Optimizer', layout='wide')
    initSessionState()
    try:
        papeg.orchestrate()
    except Exception as exc:
        st.error(f'Orchestration failed: {exc}')
    st.title('Smart Home Energy Optimizer')
    renderSidebar()
    renderTopMetrics()
    st.divider()
    renderGenerationChart()
    st.divider()
    renderDeviceCards()
    st.divider()
    renderPlanner()
if __name__ == '__main__':
    main()
