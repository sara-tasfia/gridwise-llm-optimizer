"""Request validation for POST /optimize-energy.

Structural problems (missing fields, wrong types) return 400.
Well-formed but semantically impossible requests return 422.
"""

import math

REQUIRED_HOUR_FIELDS = ("hour", "demand_kwh", "solar_kwh", "tariff_bdt_per_kwh")
REQUIRED_BATTERY_FIELDS = (
    "capacity_kwh",
    "initial_energy_kwh",
    "minimum_energy_kwh",
    "max_charge_kwh_per_hour",
    "max_discharge_kwh_per_hour",
)


class RequestError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def _number(value, field, status=400):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(f"{field} must be a number", status)
    number = float(value)
    if not math.isfinite(number):
        raise RequestError(f"{field} must be finite", status)
    return number


def validate_request(data):
    """Return (scenario_id, operator_notes, hours, battery) or raise."""
    if not isinstance(data, dict):
        raise RequestError("Request body must be a JSON object", 400)

    scenario_id = data.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise RequestError("scenario_id must be a non-empty string", 400)

    notes = data.get("operator_notes")
    if not isinstance(notes, list) or not notes:
        raise RequestError("operator_notes must be a non-empty array of strings", 400)
    for i, note in enumerate(notes):
        if not isinstance(note, str) or not note.strip():
            raise RequestError(f"operator_notes[{i}] must be a non-empty string", 400)

    hours = data.get("hours")
    if not isinstance(hours, list):
        raise RequestError("hours must be an array", 400)
    if len(hours) != 24:
        raise RequestError("hours must contain exactly 24 entries", 422)

    seen = set()
    for i, entry in enumerate(hours):
        if not isinstance(entry, dict):
            raise RequestError(f"hours[{i}] must be an object", 400)
        for field in REQUIRED_HOUR_FIELDS:
            if field not in entry:
                raise RequestError(f"hours[{i}] is missing {field}", 400)
        hour = _number(entry["hour"], f"hours[{i}].hour")
        if hour != int(hour) or not 0 <= hour <= 23:
            raise RequestError(f"hours[{i}].hour must be an integer 0-23", 422)
        if int(hour) in seen:
            raise RequestError(f"duplicate hour {int(hour)}", 422)
        seen.add(int(hour))
        for field in ("demand_kwh", "solar_kwh", "tariff_bdt_per_kwh"):
            value = _number(entry[field], f"hours[{i}].{field}")
            if value < 0:
                raise RequestError(f"hours[{i}].{field} must be non-negative", 422)
    if seen != set(range(24)):
        raise RequestError("hours must cover every hour 0 through 23", 422)

    battery = data.get("battery")
    if not isinstance(battery, dict):
        raise RequestError("battery must be an object", 400)
    for field in REQUIRED_BATTERY_FIELDS:
        if field not in battery:
            raise RequestError(f"battery is missing {field}", 400)
        value = _number(battery[field], f"battery.{field}")
        if value < 0:
            raise RequestError(f"battery.{field} must be non-negative", 422)

    capacity = float(battery["capacity_kwh"])
    initial = float(battery["initial_energy_kwh"])
    minimum = float(battery["minimum_energy_kwh"])
    if capacity <= 0:
        raise RequestError("battery.capacity_kwh must be greater than zero", 422)
    if minimum > capacity:
        raise RequestError("battery.minimum_energy_kwh exceeds capacity_kwh", 422)
    if not minimum <= initial <= capacity:
        raise RequestError(
            "battery.initial_energy_kwh must lie between minimum_energy_kwh and capacity_kwh", 422
        )

    normalized_hours = sorted(
        ({"hour": int(h["hour"]),
          "demand_kwh": float(h["demand_kwh"]),
          "solar_kwh": float(h["solar_kwh"]),
          "tariff_bdt_per_kwh": float(h["tariff_bdt_per_kwh"])} for h in hours),
        key=lambda h: h["hour"],
    )
    normalized_battery = {f: float(battery[f]) for f in REQUIRED_BATTERY_FIELDS}
    return scenario_id, notes, normalized_hours, normalized_battery
