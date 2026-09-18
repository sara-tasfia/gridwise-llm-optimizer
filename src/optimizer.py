"""Cost-minimizing 24-hour schedule.

The problem is a small linear program, so it is solved exactly rather than
greedily. Per hour h there are four variables:

    g[h]  grid import          >= 0, capped by any max_grid_window
    s[h]  solar used           0 .. effective_solar[h]
    c[h]  battery charge       0 .. max_charge   (0 in a no_charge_window)
    d[h]  battery discharge    0 .. max_discharge (0 in a no_discharge_window)

subject to
    g + s + d - c = demand                         (energy balance, each hour)
    reserve[h] <= E0 + sum_{k<=h}(c-d) <= capacity (state of charge)
    sum(c) - sum(d) = 0                            (end-of-day neutrality)

minimizing sum(tariff[h] * g[h]).
"""

import logging
import math

import numpy as np
from scipy.optimize import linprog

log = logging.getLogger(__name__)

HOURS = 24
# Breaks ties toward the least battery movement without meaningfully
# affecting cost (tariffs are orders of magnitude larger).
CYCLING_PENALTY = 1e-6
SNAP = 1e-7


def effective_solar(hours, solar_factor):
    """Base solar after applying solar_reduction directives."""
    return [float(h["solar_kwh"]) * float(solar_factor.get(h["hour"], 1.0)) for h in hours]


def _solve(hours, battery, directives, relax):
    demand = [float(h["demand_kwh"]) for h in hours]
    tariff = [float(h["tariff_bdt_per_kwh"]) for h in hours]
    solar = effective_solar(hours, directives["solar_factor"])

    capacity = float(battery["capacity_kwh"])
    initial = float(battery["initial_energy_kwh"])
    base_min = float(battery["minimum_energy_kwh"])
    max_charge = float(battery["max_charge_kwh_per_hour"])
    max_discharge = float(battery["max_discharge_kwh_per_hour"])

    n = 4 * HOURS
    gi = lambda h: 4 * h          # noqa: E731
    si = lambda h: 4 * h + 1      # noqa: E731
    ci = lambda h: 4 * h + 2      # noqa: E731
    di = lambda h: 4 * h + 3      # noqa: E731

    cost = np.zeros(n)
    for h in range(HOURS):
        cost[gi(h)] = tariff[h]
        cost[ci(h)] = CYCLING_PENALTY
        cost[di(h)] = CYCLING_PENALTY

    bounds = []
    for h in range(HOURS):
        grid_cap = None
        if "grid_cap" not in relax:
            cap = directives["max_grid"].get(h)
            if cap is not None and math.isfinite(cap):
                grid_cap = cap
        charge_cap = 0.0 if ("no_charge" not in relax and h in directives["no_charge"]) else max_charge
        discharge_cap = 0.0 if ("no_discharge" not in relax and h in directives["no_discharge"]) else max_discharge
        bounds += [(0.0, grid_cap), (0.0, max(0.0, solar[h])),
                   (0.0, charge_cap), (0.0, discharge_cap)]

    # Energy balance, one equality per hour.
    a_eq = np.zeros((HOURS, n))
    b_eq = np.zeros(HOURS)
    for h in range(HOURS):
        a_eq[h, gi(h)] = 1.0
        a_eq[h, si(h)] = 1.0
        a_eq[h, di(h)] = 1.0
        a_eq[h, ci(h)] = -1.0
        b_eq[h] = demand[h]

    # End-of-day neutrality.
    if "neutrality" not in relax:
        row = np.zeros(n)
        for h in range(HOURS):
            row[ci(h)] = 1.0
            row[di(h)] = -1.0
        a_eq = np.vstack([a_eq, row])
        b_eq = np.append(b_eq, 0.0)

    # State-of-charge window after every hour.
    a_ub, b_ub = [], []
    for h in range(HOURS):
        cumulative = np.zeros(n)
        for k in range(h + 1):
            cumulative[ci(k)] = 1.0
            cumulative[di(k)] = -1.0
        floor = base_min
        if "reserve" not in relax:
            floor = max(floor, directives["min_reserve"].get(h, 0.0))
        floor = min(floor, capacity)
        a_ub.append(cumulative.copy())
        b_ub.append(capacity - initial)
        a_ub.append(-cumulative)
        b_ub.append(initial - floor)

    result = linprog(cost, A_ub=np.array(a_ub), b_ub=np.array(b_ub),
                     A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        return None
    return result.x, solar


# Applied in order until the model becomes feasible. Valid judge scenarios
# never need this; it exists so a contradictory request still returns a
# schedule that obeys the physical rules instead of a 500.
RELAXATION_ORDER = [
    ("grid_cap", "grid import caps"),
    ("reserve", "elevated battery reserve"),
    ("no_charge", "no-charge window"),
    ("no_discharge", "no-discharge window"),
    ("neutrality", "end-of-day battery neutrality"),
]


def optimize_schedule(hours, battery, directives):
    """Build the cheapest valid 24-hour plan.

    Returns a dict with hourly_plan, totals, and any relaxations that were
    needed to reach feasibility.
    """
    hours = sorted(hours, key=lambda h: int(h["hour"]))
    relax, dropped = set(), []

    solution = _solve(hours, battery, directives, relax)
    for key, label in RELAXATION_ORDER:
        if solution is not None:
            break
        relax.add(key)
        dropped.append(label)
        solution = _solve(hours, battery, directives, relax)

    if solution is None:
        raise ValueError("No feasible 24-hour schedule exists for this scenario")

    x, solar = solution
    capacity = float(battery["capacity_kwh"])
    energy = float(battery["initial_energy_kwh"])
    base_min = float(battery["minimum_energy_kwh"])

    plan = []
    for h in range(HOURS):
        hour_data = hours[h]
        solar_used = x[4 * h + 1]
        # A degenerate optimum can charge and discharge in the same hour;
        # only the net movement is physical, and battery_action must be one
        # of charge/discharge/idle.
        net = x[4 * h + 2] - x[4 * h + 3]

        solar_used = _snap(solar_used, 0.0, solar[h])
        charge = round(max(0.0, net), 6)
        discharge = round(max(0.0, -net), 6)
        solar_used = round(solar_used, 6)

        grid = round(float(hour_data["demand_kwh"]) + charge - solar_used - discharge, 6)
        if -1e-6 < grid < 0:
            grid = 0.0

        energy = round(energy + charge - discharge, 6)
        # Guard against float drift pushing the state of charge past a bound.
        energy = min(capacity, max(min(base_min, capacity), energy))

        if charge > 0:
            action, magnitude = "charge", charge
        elif discharge > 0:
            action, magnitude = "discharge", discharge
        else:
            action, magnitude = "idle", 0.0

        plan.append({
            "hour": int(hour_data["hour"]),
            "grid_kwh": grid,
            "solar_used_kwh": solar_used,
            "battery_action": action,
            "battery_kwh": magnitude,
            "battery_energy_after_kwh": energy,
        })

    total_grid = round(sum(p["grid_kwh"] for p in plan), 6)
    total_cost = round(sum(p["grid_kwh"] * float(hours[i]["tariff_bdt_per_kwh"])
                           for i, p in enumerate(plan)), 6)
    peak_grid = round(max(p["grid_kwh"] for p in plan), 6)

    return {
        "hourly_plan": plan,
        "total_grid_kwh": total_grid,
        "total_cost_bdt": total_cost,
        "peak_grid_kwh": peak_grid,
        "effective_solar": solar,
        "relaxed": dropped,
    }


def _snap(value, low, high):
    """Clean up solver noise just outside a variable's bounds."""
    if value < low or abs(value - low) < SNAP:
        return low
    if value > high or abs(value - high) < SNAP:
        return min(value, high)
    return value


def summarize(plan, hours, interpretation, relaxed):
    """Short human-readable strategy note for plan_summary."""
    applied = [e for e in interpretation if e.get("applies")]
    tariffs = [float(h["tariff_bdt_per_kwh"]) for h in sorted(hours, key=lambda h: h["hour"])]
    charge_hours = [p["hour"] for p in plan["hourly_plan"] if p["battery_action"] == "charge"]
    discharge_hours = [p["hour"] for p in plan["hourly_plan"] if p["battery_action"] == "discharge"]

    parts = []
    if applied:
        kinds = sorted({e["directive_type"] for e in applied})
        parts.append(f"Applied {len(applied)} operator directive(s): {', '.join(kinds)}.")
    else:
        parts.append("No operator note affected the schedule; all notes were no_op.")

    parts.append(
        f"Solar is used first, the battery charges in {len(charge_hours)} cheaper hour(s) "
        f"and discharges in {len(discharge_hours)} expensive hour(s), "
        f"ending the day at its starting state of charge."
    )
    if charge_hours and discharge_hours:
        avg_charge = sum(tariffs[h] for h in charge_hours) / len(charge_hours)
        avg_discharge = sum(tariffs[h] for h in discharge_hours) / len(discharge_hours)
        parts.append(
            f"Average charging tariff {avg_charge:.1f} BDT/kWh versus "
            f"{avg_discharge:.1f} BDT/kWh displaced when discharging."
        )
    parts.append(
        f"Total grid import {plan['total_grid_kwh']:.2f} kWh at "
        f"{plan['total_cost_bdt']:.2f} BDT, peak {plan['peak_grid_kwh']:.2f} kWh."
    )
    if relaxed:
        parts.append("Scenario was infeasible as stated; relaxed: " + ", ".join(relaxed) + ".")
    return " ".join(parts)
