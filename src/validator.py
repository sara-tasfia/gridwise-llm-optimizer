"""Deterministic guardrails.

Everything the model produces passes through here before it reaches the
optimizer. The contract of `validate_interpretation` is strong: whatever the
model returns, the output is always exactly one entry per operator note, in
note_index order, with a structured_adjustment that matches the shape required
for its directive type. Anything that fails a check is demoted to no_op rather
than being applied on a guess.
"""

import logging
import math

log = logging.getLogger(__name__)

VALID_DIRECTIVE_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}

REQUIRED_KEYS = {
    "solar_reduction": {"hours", "factor"},
    "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
    "no_charge_window": {"hours"},
    "no_discharge_window": {"hours"},
    "max_grid_window": {"hours", "max_grid_kwh"},
}

NO_OP = {
    "applies": False,
    "directive_type": "no_op",
    "structured_adjustment": None,
}


def _finite_number(value):
    """Accept ints/floats and numeric strings; reject bools, NaN and inf."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _clean_hours(raw):
    """Unique integer hours 0-23 in ascending order, or (None, error)."""
    if not isinstance(raw, (list, tuple)):
        return None, "hours must be a list"
    hours = set()
    for item in raw:
        number = _finite_number(item)
        if number is None or number != int(number):
            return None, f"hour is not an integer: {item!r}"
        hour = int(number)
        if not 0 <= hour <= 23:
            return None, f"hour out of range 0-23: {hour}"
        hours.add(hour)
    if not hours:
        return None, "hours is empty"
    return sorted(hours), None


def validate_directive(entry, battery):
    """Check one interpretation entry.

    Returns {"valid", "errors", "entry"} where `entry` is the normalized,
    safe-to-apply version (a no_op entry when validation failed).
    """
    errors = []
    if not isinstance(entry, dict):
        return {"valid": False, "errors": ["entry is not an object"], "entry": None}

    directive_type = entry.get("directive_type")
    if isinstance(directive_type, str):
        directive_type = directive_type.strip().lower()
    if directive_type not in VALID_DIRECTIVE_TYPES:
        return {"valid": False,
                "errors": [f"unsupported directive_type: {entry.get('directive_type')!r}"],
                "entry": None}

    explanation = entry.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = "Interpreted from the operator note."
    explanation = explanation.strip()[:300]

    # no_op: applies must be false and the adjustment must be null.
    if directive_type == "no_op":
        return {"valid": True, "errors": [],
                "entry": {**NO_OP, "explanation": explanation}}

    adjustment = entry.get("structured_adjustment")
    if not isinstance(adjustment, dict):
        return {"valid": False,
                "errors": [f"{directive_type} requires a structured_adjustment object"],
                "entry": None}

    missing = REQUIRED_KEYS[directive_type] - set(adjustment)
    if missing:
        return {"valid": False,
                "errors": [f"{directive_type} missing {sorted(missing)}"],
                "entry": None}

    hours, hours_error = _clean_hours(adjustment.get("hours"))
    if hours_error:
        return {"valid": False, "errors": [hours_error], "entry": None}

    clean = {"hours": hours}
    capacity = _finite_number(battery.get("capacity_kwh"))

    if directive_type == "solar_reduction":
        factor = _finite_number(adjustment.get("factor"))
        if factor is None:
            errors.append("factor is not a finite number")
        elif not 0.0 <= factor <= 1.0:
            errors.append(f"factor must be between 0 and 1: {factor}")
        else:
            clean["factor"] = factor

    elif directive_type == "minimum_battery_reserve":
        reserve = _finite_number(adjustment.get("minimum_energy_kwh"))
        if reserve is None:
            errors.append("minimum_energy_kwh is not a finite number")
        elif reserve < 0:
            errors.append(f"minimum_energy_kwh must be non-negative: {reserve}")
        elif capacity is not None and reserve > capacity:
            errors.append(f"minimum_energy_kwh {reserve} exceeds capacity {capacity}")
        else:
            clean["minimum_energy_kwh"] = reserve

    elif directive_type == "max_grid_window":
        cap = _finite_number(adjustment.get("max_grid_kwh"))
        if cap is None:
            errors.append("max_grid_kwh is not a finite number")
        elif cap < 0:
            errors.append(f"max_grid_kwh must be non-negative: {cap}")
        else:
            clean["max_grid_kwh"] = cap

    if errors:
        return {"valid": False, "errors": errors, "entry": None}

    return {
        "valid": True,
        "errors": [],
        "entry": {
            "applies": True,
            "directive_type": directive_type,
            "structured_adjustment": clean,
            "explanation": explanation,
        },
    }


def validate_interpretation(raw_entries, operator_notes, battery):
    """Turn raw model output into exactly one trusted entry per note.

    Returns (interpretation, rejections). `interpretation` always has
    len(operator_notes) entries with note_index 0..N-1 in ascending order.
    """
    by_index = {}
    rejections = []

    for position, raw in enumerate(raw_entries or []):
        index = None
        if isinstance(raw, dict):
            candidate = _finite_number(raw.get("note_index"))
            if candidate is not None and candidate == int(candidate):
                index = int(candidate)
        # A model that omitted note_index still maps positionally.
        if index is None or not 0 <= index < len(operator_notes):
            index = position
        if not 0 <= index < len(operator_notes):
            rejections.append(f"entry {position}: note_index out of range")
            continue
        if index in by_index:
            rejections.append(f"entry {position}: duplicate note_index {index}")
            continue

        result = validate_directive(raw, battery)
        if result["valid"]:
            by_index[index] = result["entry"]
        else:
            rejections.append(f"note {index}: " + "; ".join(result["errors"]))

    interpretation = []
    for index in range(len(operator_notes)):
        entry = by_index.get(index)
        if entry is None:
            if index not in by_index:
                rejections.append(f"note {index}: no valid interpretation, defaulted to no_op")
            entry = {**NO_OP,
                     "explanation": "No valid directive could be extracted from this note."}
        interpretation.append({"note_index": index, **entry})

    if rejections:
        log.info("Guardrails rejected %d interpretation item(s): %s",
                 len(rejections), rejections)
    return interpretation, rejections


def collect_directives(interpretation):
    """Group applied directives into the form the optimizer consumes."""
    directives = {
        "solar_factor": {},      # hour -> remaining fraction
        "min_reserve": {},       # hour -> required kWh after the hour
        "no_charge": set(),
        "no_discharge": set(),
        "max_grid": {},          # hour -> kWh ceiling
    }
    for entry in interpretation:
        if not entry.get("applies") or entry["directive_type"] == "no_op":
            continue
        adjustment = entry["structured_adjustment"]
        hours = adjustment["hours"]
        dtype = entry["directive_type"]
        for hour in hours:
            if dtype == "solar_reduction":
                # Overlapping reductions: keep the most restrictive factor.
                current = directives["solar_factor"].get(hour, 1.0)
                directives["solar_factor"][hour] = min(current, adjustment["factor"])
            elif dtype == "minimum_battery_reserve":
                current = directives["min_reserve"].get(hour, 0.0)
                directives["min_reserve"][hour] = max(current, adjustment["minimum_energy_kwh"])
            elif dtype == "no_charge_window":
                directives["no_charge"].add(hour)
            elif dtype == "no_discharge_window":
                directives["no_discharge"].add(hour)
            elif dtype == "max_grid_window":
                current = directives["max_grid"].get(hour, math.inf)
                directives["max_grid"][hour] = min(current, adjustment["max_grid_kwh"])
    return directives
