"""Operator-note interpretation.

The language model is the interpretation path: it turns free-form operator
notes into one structured directive per note. Its output is untrusted until
`validator.validate_interpretation` has checked it.

A deterministic fallback parser exists only so the service degrades safely when
the provider is down or returns unusable JSON (Problem Statement, "Safe
Failure"). It never runs when the model answers successfully.
"""

import json
import logging
import re
import time

from .config import config

log = logging.getLogger(__name__)

DIRECTIVE_TYPES = (
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
)

SYSTEM_PROMPT = """You convert campus energy operator notes into structured directives for a 24-hour scheduling optimizer.

Return ONLY a JSON object, no prose and no markdown fences:
{"directives": [ {one entry per note, in the order the notes were given} ]}

Each entry:
{
  "note_index": <0-based index of the note>,
  "applies": true | false,
  "directive_type": "solar_reduction" | "minimum_battery_reserve" | "no_charge_window" | "no_discharge_window" | "max_grid_window" | "no_op",
  "structured_adjustment": <object described below, or null>,
  "explanation": "<one short sentence>"
}

Shapes required for structured_adjustment:
  solar_reduction          {"hours": [ints], "factor": number}
  minimum_battery_reserve  {"hours": [ints], "minimum_energy_kwh": number}
  no_charge_window         {"hours": [ints]}
  no_discharge_window      {"hours": [ints]}
  max_grid_window          {"hours": [ints], "max_grid_kwh": number}
  no_op                    null

Rules:
1. Exactly one entry per note. Never merge, skip, reorder or invent notes.
2. A note that does not change today's 24-hour electricity schedule is
   "no_op" with applies=false and structured_adjustment=null. Room bookings,
   deadlines, menus, staffing news and anything scheduled for another day are
   no_op. Every other directive uses applies=true.
3. Time windows are START-INCLUSIVE and END-EXCLUSIVE. Convert to a list of
   whole hours 0-23, unique and ascending:
     "1 PM to 3 PM"        -> [13, 14]
     "6 PM until 9 PM"     -> [18, 19, 20]
     "noon until 2 PM"     -> [12, 13]
     "between 11 AM and 2 PM" -> [11, 12, 13]
     "2 AM until 5 AM"     -> [2, 3, 4]
4. "factor" is the usable fraction of solar that REMAINS, between 0 and 1.
     "drops to 25% of forecast"   -> 0.25
     "an 80% reduction"           -> 0.20
     "about half the forecast"    -> 0.50
     "roughly one-fifth of normal"-> 0.20
5. "minimum_energy_kwh" is an absolute kWh level. If the note states a
   percentage, multiply it by the battery capacity given in the user message
   ("50% of capacity" with capacity 200 -> 100).
6. "max_grid_kwh" is the hourly ceiling on grid import, in kWh.
7. Use only the values stated in the note. Never invent demand, tariff, solar
   figures or battery limits, and never use a directive type outside the list.

Worked examples:
Note: "Panel washing from one until three will leave roughly one-fifth of normal solar output."
-> {"note_index":0,"applies":true,"directive_type":"solar_reduction","structured_adjustment":{"hours":[13,14],"factor":0.2},"explanation":"Solar is reduced to 20% during panel washing."}
Note: "The charging circuit will be unavailable from 2 PM until 4 PM."
-> {"note_index":0,"applies":true,"directive_type":"no_charge_window","structured_adjustment":{"hours":[14,15]},"explanation":"Battery charging is unavailable in that window."}
Note: "Grid intake must stay at or below 190 kWh from 7 PM until 10 PM."
-> {"note_index":0,"applies":true,"directive_type":"max_grid_window","structured_adjustment":{"hours":[19,20,21],"max_grid_kwh":190},"explanation":"Grid import is capped at 190 kWh per hour."}
Note: "The library is extending book-return hours next week."
-> {"note_index":0,"applies":false,"directive_type":"no_op","structured_adjustment":null,"explanation":"This note does not affect today's schedule."}"""


def _user_prompt(notes, battery):
    lines = [
        "Battery context for converting any percentage into kWh:",
        f"  capacity_kwh = {battery.get('capacity_kwh')}",
        f"  base minimum_energy_kwh = {battery.get('minimum_energy_kwh')}",
        "",
        f"Interpret these {len(notes)} operator notes:",
    ]
    for i, note in enumerate(notes):
        lines.append(f"[{i}] {note}")
    lines.append("")
    lines.append(f'Return {{"directives": [...]}} with exactly {len(notes)} entries, note_index 0 to {len(notes) - 1}.')
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Provider calls
# --------------------------------------------------------------------------

def _call_openai(system, user):
    """OpenAI, or any OpenAI-compatible endpoint via LLM_BASE_URL."""
    from openai import OpenAI

    client = OpenAI(
        api_key=config.LLM_API_KEY or "not-needed",
        base_url=config.LLM_BASE_URL,
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=0,
    )
    kwargs = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
    }
    try:
        resp = client.chat.completions.create(
            response_format={"type": "json_object"}, **kwargs
        )
    except Exception:
        # Not every OpenAI-compatible server supports response_format.
        resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _call_anthropic(system, user):
    import anthropic

    client = anthropic.Anthropic(
        api_key=config.LLM_API_KEY,
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=0,
    )
    resp = client.messages.create(
        model=config.LLM_MODEL,
        system=system,
        messages=[
            {"role": "user", "content": user},
            # Prefill forces the reply to start as JSON.
            {"role": "assistant", "content": "{"},
        ],
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return "{" + text


_PROVIDERS = {"openai": _call_openai, "anthropic": _call_anthropic}


def _extract_json(text: str):
    """Pull a JSON object out of a model reply, fences and preamble included."""
    if not text:
        raise ValueError("empty model response")
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("no JSON object in model response")


def _as_entry_list(payload, expected):
    """Accept {"directives": [...]}, a bare list, or a single object."""
    if isinstance(payload, dict):
        for key in ("directives", "directive_interpretation", "results", "notes"):
            if isinstance(payload.get(key), list):
                return payload[key]
        if "directive_type" in payload and expected == 1:
            return [payload]
        raise ValueError("model JSON has no directive list")
    if isinstance(payload, list):
        return payload
    raise ValueError("unexpected model JSON shape")


def parse_operator_notes(operator_notes, battery):
    """Interpret every note with the LLM.

    Returns (entries, meta). `entries` is the raw, still-untrusted list the
    validator will check. `meta` records which path produced it, for the
    plan summary and logs.
    """
    notes = list(operator_notes)
    meta = {"source": "llm", "model": config.LLM_MODEL, "errors": []}

    call = _PROVIDERS.get(config.LLM_PROVIDER)
    if call is None:
        meta["errors"].append(f"unknown LLM_PROVIDER '{config.LLM_PROVIDER}'")
    elif not config.llm_configured:
        meta["errors"].append("LLM credentials not configured")
    else:
        system, user = SYSTEM_PROMPT, _user_prompt(notes, battery)
        for attempt in range(config.LLM_MAX_RETRIES + 1):
            started = time.time()
            try:
                entries = _as_entry_list(_extract_json(call(system, user)), len(notes))
                meta["latency_ms"] = round((time.time() - started) * 1000)
                meta["attempts"] = attempt + 1
                return entries, meta
            except Exception as exc:  # provider error, timeout, or bad JSON
                meta["errors"].append(f"attempt {attempt + 1}: {type(exc).__name__}")
                log.warning("LLM interpretation attempt %d failed: %s", attempt + 1, exc)
                if attempt < config.LLM_MAX_RETRIES:
                    time.sleep(0.4)

    if not config.ENABLE_RULE_FALLBACK:
        meta["source"] = "unavailable"
        return [], meta

    log.warning("Falling back to deterministic note parsing: %s", meta["errors"])
    meta["source"] = "rule_fallback"
    return [rule_based_directive(note, i, battery) for i, note in enumerate(notes)], meta


# --------------------------------------------------------------------------
# Deterministic fallback (safe-failure path only)
# --------------------------------------------------------------------------

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_FRACTIONS = {"half": 0.5, "one-fifth": 0.2, "a fifth": 0.2, "a third": 1 / 3,
              "one-third": 1 / 3, "a quarter": 0.25, "one-quarter": 0.25}


def _to_24h(value, meridiem, text_after):
    hour = value % 12
    if meridiem == "pm":
        hour += 12
    elif meridiem is None and re.search(r"\b(pm|evening|night)\b", text_after):
        hour += 12
    return hour % 24


def _find_window(text):
    """Best-effort start-inclusive / end-exclusive hour window."""
    t = text.lower()
    t = t.replace("noon", "12 pm").replace("midnight", "12 am")
    for word, num in _WORD_NUMBERS.items():
        t = re.sub(rf"\b{word}\b", str(num), t)

    pattern = (
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*"
        r"(?:-|to|until|till|through|and)\s*"
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
    )
    match = re.search(pattern, t)
    if not match:
        return None
    tail = t[match.end():]
    start = _to_24h(int(match.group(1)), match.group(3), tail)
    end = _to_24h(int(match.group(4)), match.group(6), tail)
    if match.group(3) is None and match.group(6) == "pm" and int(match.group(1)) < 12:
        start = _to_24h(int(match.group(1)), "pm", "")
    if end <= start:
        end += 12
    hours = [h for h in range(start, min(end, 24))]
    return hours or None


def _find_factor(text):
    t = text.lower()
    for phrase, value in _FRACTIONS.items():
        if phrase in t:
            return value
    pct = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", t)
    if pct:
        value = float(pct.group(1)) / 100.0
        if re.search(r"reduc|drop by|decreas|less|cut", t) and "drop to" not in t:
            return max(0.0, 1.0 - value)
        return min(1.0, value)
    return None


def _find_kwh(text):
    match = re.search(r"(\d+(?:\.\d+)?)\s*kwh", text.lower())
    return float(match.group(1)) if match else None


def rule_based_directive(note, index, battery):
    """Keyword interpretation used only when the model path fails."""
    t = (note or "").lower()
    entry = {"note_index": index, "applies": False,
             "directive_type": "no_op", "structured_adjustment": None,
             "explanation": "Fallback parser found no schedule-affecting instruction."}
    hours = _find_window(t)

    def build(dtype, adjustment, why):
        entry.update(applies=True, directive_type=dtype,
                     structured_adjustment=adjustment, explanation=why)
        return entry

    if re.search(r"solar|pv|panel|photovoltaic|inverter", t) and hours:
        factor = _find_factor(t)
        if factor is not None:
            return build("solar_reduction", {"hours": hours, "factor": factor},
                         f"Usable solar reduced to {factor:.0%} in this window.")

    if re.search(r"(no|not|don't|do not|disabl|unavailab|isolat|prohibit|avoid|stop)", t) and hours:
        if re.search(r"discharg", t):
            return build("no_discharge_window", {"hours": hours},
                         "Battery discharging is unavailable in this window.")
        if re.search(r"charg", t):
            return build("no_charge_window", {"hours": hours},
                         "Battery charging is unavailable in this window.")

    if re.search(r"reserve|at least|minimum|keep|remain|maintain", t) and re.search(r"batter|storage|stored", t) and hours:
        level = _find_kwh(t)
        if level is None:
            pct = re.search(r"(\d{1,3})\s*(?:%|percent)", t)
            if pct and battery.get("capacity_kwh"):
                level = float(pct.group(1)) / 100.0 * float(battery["capacity_kwh"])
        if level is not None:
            return build("minimum_battery_reserve",
                         {"hours": hours, "minimum_energy_kwh": level},
                         f"Battery must hold at least {level:g} kWh in this window.")

    if re.search(r"grid|import|intake|feeder|transformer|substation", t) and hours:
        cap = _find_kwh(t)
        if cap is not None and re.search(r"not exceed|no more|at or below|cap|limit|max|under", t):
            return build("max_grid_window", {"hours": hours, "max_grid_kwh": cap},
                         f"Grid import capped at {cap:g} kWh per hour.")

    return entry
