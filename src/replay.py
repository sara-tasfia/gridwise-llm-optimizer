"""Independent replay of a finished response.

This mirrors what the judge does in Problem Statement section 11: walk the
returned hourly_plan hour by hour and confirm every energy, battery and
directive rule holds. Used by tests/run_public_cases.py and, when SELF_CHECK
is on, as a logged sanity check on every live response.
"""

import math

TOLERANCE = 0.01


def replay(request_payload, response_payload, directives):
    """Return a list of violation strings; empty means the plan is valid."""
    problems = []
    hours = sorted(request_payload["hours"], key=lambda h: int(h["hour"]))
    battery = request_payload["battery"]
    plan = response_payload.get("hourly_plan") or []

    if response_payload.get("scenario_id") != request_payload.get("scenario_id"):
        problems.append("scenario_id does not match the request")

    interpretation = response_payload.get("directive_interpretation")
    if not isinstance(interpretation, list) or len(interpretation) != len(request_payload["operator_notes"]):
        problems.append("directive_interpretation must have one entry per operator note")
    else:
        for i, entry in enumerate(interpretation):
            if entry.get("note_index") != i:
                problems.append(f"directive_interpretation[{i}] has note_index {entry.get('note_index')}")
            is_no_op = entry.get("directive_type") == "no_op"
            if is_no_op and (entry.get("applies") is not False or entry.get("structured_adjustment") is not None):
                problems.append(f"note {i}: no_op requires applies=false and null adjustment")
            if not is_no_op and entry.get("applies") is not True:
                problems.append(f"note {i}: non-no_op directives require applies=true")
            adjustment = entry.get("structured_adjustment")
            if isinstance(adjustment, dict):
                hours_list = adjustment.get("hours")
                if (not isinstance(hours_list, list) or not hours_list
                        or hours_list != sorted(set(hours_list))
                        or any(not isinstance(h, int) or not 0 <= h <= 23 for h in hours_list)):
                    problems.append(f"note {i}: hours must be unique ascending integers 0-23")

    if len(plan) != 24 or sorted(p.get("hour") for p in plan) != list(range(24)):
        problems.append("hourly_plan must contain exactly 24 unique hours 0-23")
        return problems

    capacity = float(battery["capacity_kwh"])
    energy = float(battery["initial_energy_kwh"])
    base_min = float(battery["minimum_energy_kwh"])
    max_charge = float(battery["max_charge_kwh_per_hour"])
    max_discharge = float(battery["max_discharge_kwh_per_hour"])

    plan = sorted(plan, key=lambda p: p["hour"])
    total_grid = 0.0
    total_cost = 0.0
    peak = 0.0

    for hour_data, row in zip(hours, plan):
        h = int(hour_data["hour"])
        grid = float(row.get("grid_kwh", 0))
        solar_used = float(row.get("solar_used_kwh", 0))
        action = row.get("battery_action")
        magnitude = float(row.get("battery_kwh", 0))
        after = float(row.get("battery_energy_after_kwh", 0))

        for name, value in (("grid_kwh", grid), ("solar_used_kwh", solar_used),
                            ("battery_kwh", magnitude), ("battery_energy_after_kwh", after)):
            if not math.isfinite(value) or value < -TOLERANCE:
                problems.append(f"hour {h}: {name} must be finite and non-negative")

        effective = float(hour_data["solar_kwh"]) * directives["solar_factor"].get(h, 1.0)
        if solar_used > effective + TOLERANCE:
            problems.append(f"hour {h}: solar_used {solar_used} exceeds effective solar {effective:.2f}")

        if action not in ("charge", "discharge", "idle"):
            problems.append(f"hour {h}: invalid battery_action {action!r}")
            continue
        if action == "idle" and abs(magnitude) > TOLERANCE:
            problems.append(f"hour {h}: idle requires battery_kwh = 0")

        charge = magnitude if action == "charge" else 0.0
        discharge = magnitude if action == "discharge" else 0.0
        if charge > max_charge + TOLERANCE:
            problems.append(f"hour {h}: charge {charge} exceeds hourly limit {max_charge}")
        if discharge > max_discharge + TOLERANCE:
            problems.append(f"hour {h}: discharge {discharge} exceeds hourly limit {max_discharge}")

        balance = grid + solar_used + discharge - (float(hour_data["demand_kwh"]) + charge)
        if abs(balance) > TOLERANCE:
            problems.append(f"hour {h}: energy balance off by {balance:.4f} kWh")

        energy = energy + charge - discharge
        if abs(energy - after) > TOLERANCE:
            problems.append(f"hour {h}: battery_energy_after {after} should be {energy:.4f}")
        energy = after

        floor = max(base_min, directives["min_reserve"].get(h, 0.0))
        if after < floor - TOLERANCE:
            problems.append(f"hour {h}: battery {after} below required minimum {floor}")
        if after > capacity + TOLERANCE:
            problems.append(f"hour {h}: battery {after} above capacity {capacity}")

        if h in directives["no_charge"] and charge > TOLERANCE:
            problems.append(f"hour {h}: charging inside a no_charge_window")
        if h in directives["no_discharge"] and discharge > TOLERANCE:
            problems.append(f"hour {h}: discharging inside a no_discharge_window")
        cap = directives["max_grid"].get(h)
        if cap is not None and grid > cap + TOLERANCE:
            problems.append(f"hour {h}: grid {grid} exceeds cap {cap}")

        total_grid += grid
        total_cost += grid * float(hour_data["tariff_bdt_per_kwh"])
        peak = max(peak, grid)

    if abs(energy - float(battery["initial_energy_kwh"])) > TOLERANCE:
        problems.append(
            f"final battery {energy:.4f} must equal initial {battery['initial_energy_kwh']}"
        )

    for field, expected in (("total_grid_kwh", total_grid),
                            ("total_cost_bdt", total_cost),
                            ("peak_grid_kwh", peak)):
        reported = response_payload.get(field)
        if not isinstance(reported, (int, float)) or abs(float(reported) - expected) > TOLERANCE:
            problems.append(f"{field} reported {reported} but recomputes to {expected:.4f}")

    return problems
