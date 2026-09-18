# GridWise — LLM-Assisted Campus Energy Optimization

BUP CSE Fest 2026 · Online Preliminary · Smart Campus Energy Optimization Challenge

One HTTP service that reads free-form campus operator notes, converts them into
machine-checkable directives with a language model, validates those directives
deterministically, and returns the cheapest valid 24-hour energy schedule.

On the ten public sample cases the service reproduces the organizer's optimal
cost exactly (optimization score 1.0000 on all ten) with a clean judge-style
replay.

---

## 1. Quickstart from a clean machine

```bash
git clone <your-repo-url> gridwise && cd gridwise

python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then set LLM_API_KEY (and LLM_MODEL/LLM_PROVIDER)
python main.py              # serves on http://0.0.0.0:5000
```

Production start (what the deployment runs):

```bash
gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 8 --timeout 60 main:app
```

### Docker

```bash
docker build -t gridwise:latest .
docker run -p 5000:5000 \
  -e LLM_PROVIDER=openai \
  -e LLM_API_KEY=sk-... \
  -e LLM_MODEL=gpt-4o-mini \
  gridwise:latest
```

The image binds `0.0.0.0:5000`, exposes port 5000, and contains no baked-in
secrets — every credential arrives through `-e` at run time.

---

## 2. Verify it works

```bash
curl -s localhost:5000/health
# {"status":"ok"}
```

```bash
curl -s -X POST localhost:5000/optimize-energy \
  -H 'Content-Type: application/json' \
  -d '{
    "scenario_id": "GRID-101",
    "operator_notes": [
      "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
      "The sports office moved next month'"'"'s registration deadline."
    ],
    "hours": [
      {"hour":0,"demand_kwh":90,"solar_kwh":0,"tariff_bdt_per_kwh":6},
      {"hour":1,"demand_kwh":85,"solar_kwh":0,"tariff_bdt_per_kwh":6},
      {"hour":2,"demand_kwh":80,"solar_kwh":0,"tariff_bdt_per_kwh":5},
      {"hour":3,"demand_kwh":80,"solar_kwh":0,"tariff_bdt_per_kwh":5},
      {"hour":4,"demand_kwh":85,"solar_kwh":0,"tariff_bdt_per_kwh":5},
      {"hour":5,"demand_kwh":95,"solar_kwh":0,"tariff_bdt_per_kwh":6},
      {"hour":6,"demand_kwh":110,"solar_kwh":5,"tariff_bdt_per_kwh":8},
      {"hour":7,"demand_kwh":130,"solar_kwh":20,"tariff_bdt_per_kwh":10},
      {"hour":8,"demand_kwh":150,"solar_kwh":50,"tariff_bdt_per_kwh":12},
      {"hour":9,"demand_kwh":165,"solar_kwh":90,"tariff_bdt_per_kwh":14},
      {"hour":10,"demand_kwh":175,"solar_kwh":130,"tariff_bdt_per_kwh":16},
      {"hour":11,"demand_kwh":180,"solar_kwh":160,"tariff_bdt_per_kwh":16},
      {"hour":12,"demand_kwh":185,"solar_kwh":180,"tariff_bdt_per_kwh":15},
      {"hour":13,"demand_kwh":180,"solar_kwh":170,"tariff_bdt_per_kwh":14},
      {"hour":14,"demand_kwh":170,"solar_kwh":140,"tariff_bdt_per_kwh":13},
      {"hour":15,"demand_kwh":165,"solar_kwh":90,"tariff_bdt_per_kwh":14},
      {"hour":16,"demand_kwh":170,"solar_kwh":45,"tariff_bdt_per_kwh":18},
      {"hour":17,"demand_kwh":185,"solar_kwh":10,"tariff_bdt_per_kwh":22},
      {"hour":18,"demand_kwh":205,"solar_kwh":0,"tariff_bdt_per_kwh":28},
      {"hour":19,"demand_kwh":215,"solar_kwh":0,"tariff_bdt_per_kwh":30},
      {"hour":20,"demand_kwh":205,"solar_kwh":0,"tariff_bdt_per_kwh":26},
      {"hour":21,"demand_kwh":175,"solar_kwh":0,"tariff_bdt_per_kwh":18},
      {"hour":22,"demand_kwh":135,"solar_kwh":0,"tariff_bdt_per_kwh":10},
      {"hour":23,"demand_kwh":105,"solar_kwh":0,"tariff_bdt_per_kwh":7}
    ],
    "battery": {
      "capacity_kwh": 220, "initial_energy_kwh": 110, "minimum_energy_kwh": 40,
      "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50
    }
  }'
```

Abbreviated response:

```json
{
  "scenario_id": "GRID-101",
  "directive_interpretation": [
    {"note_index": 0, "applies": true, "directive_type": "solar_reduction",
     "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
     "explanation": "Solar availability is reduced to 25% during panel cleaning."},
    {"note_index": 1, "applies": false, "directive_type": "no_op",
     "structured_adjustment": null,
     "explanation": "This note does not affect today's 24-hour schedule."}
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 90, "solar_used_kwh": 0,
     "battery_action": "idle", "battery_kwh": 0, "battery_energy_after_kwh": 110}
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 187.5,
  "plan_summary": "Applied 1 operator directive(s): solar_reduction. ..."
}
```

### Run the full public sample pack

```bash
python tests/run_public_cases.py                     # in-process
python tests/run_public_cases.py --url http://localhost:5000   # against a deployment
```

Every case is replayed against the **ground-truth** directives from the pack —
energy balance, state-of-charge window, hourly rate limits, effective solar,
every operator directive, end-of-day neutrality, and the three recomputed
totals — and the cost is scored as `min(1, reference / ours)`.

```
PASS SAMPLE-01  cost   38365.00 (ref  38365.00, +0.00)  score 1.0000
...
cases 10 | fully passing 10 | valid schedules 10 | interpretation correct 10
mean optimization score 1.0000
```

---

## 3. Architecture

```
POST /optimize-energy
   │
   ├─ src/schema.py       structural + semantic request check  → 400 / 422
   ├─ src/llm_parser.py   LLM reads every note → raw directives        (untrusted)
   ├─ src/validator.py    deterministic guardrails → trusted directives
   ├─ src/optimizer.py    linear program → cheapest valid 24-hour plan
   └─ src/replay.py       judge-style replay of the finished plan      (self-check)
```

| File | Role |
|---|---|
| `main.py` | Flask app, endpoints, status codes, controlled error handling |
| `src/config.py` | Environment-driven settings; no secrets in source |
| `src/llm_parser.py` | Prompt, provider clients, JSON extraction, retry, safe fallback |
| `src/validator.py` | Section 08 guardrails; collapses directives for the optimizer |
| `src/optimizer.py` | LP formulation, solve, netting, plan assembly, summary |
| `src/replay.py` | Independent re-verification of a finished response |
| `src/schema.py` | Request contract |
| `tests/run_public_cases.py` | Scores all ten public cases like the judge |

### The LLM's role

The model **is** the interpretation path. It receives all of a scenario's notes
in one call (one round trip keeps p95 latency low) plus the battery capacity, so
it can turn "50% of the battery capacity" into an absolute kWh reserve. It
returns one entry per note: `note_index`, `applies`, `directive_type`,
`structured_adjustment`, `explanation`.

The prompt pins down the three things paraphrases usually break:

- **Windows are start-inclusive, end-exclusive** — "6 PM until 9 PM" → `[18,19,20]`.
- **`factor` is the fraction remaining** — "an 80% reduction" → `0.2`, "drops to 25%" → `0.25`.
- **Percentages of capacity become kWh** using the capacity passed in the prompt.

Provider is configurable. `LLM_PROVIDER=openai` with `LLM_BASE_URL` covers
OpenAI, Groq, OpenRouter, Together, vLLM and Ollama; `LLM_PROVIDER=anthropic`
uses the Messages API with a `{` prefill. JSON mode is requested when the
endpoint supports it and the call degrades gracefully when it doesn't.

### Guardrails (`src/validator.py`)

Model output is untrusted structured data until every one of these passes:

| Check | Behaviour on failure |
|---|---|
| `directive_type` is one of the six supported values | demote to `no_op` |
| Exactly one entry per note, `note_index` 0..N-1, no duplicates | fill gaps with `no_op` |
| `hours` are unique integers 0-23, returned ascending | demote to `no_op` |
| `structured_adjustment` matches the shape required for its type | demote to `no_op` |
| `factor` finite and within 0-1 | demote to `no_op` |
| `minimum_energy_kwh` finite, non-negative, ≤ capacity | demote to `no_op` |
| `max_grid_kwh` finite and non-negative | demote to `no_op` |
| `no_op` ⇒ `applies=false` and `structured_adjustment=null` | rewritten to match |

Demoting to `no_op` is deliberate: a directive that cannot be trusted is never
guessed at and never applied. Rejections are logged with the reason. The output
of this layer always has exactly one entry per note, so the response shape
cannot drift no matter what the model returns.

### Optimizer (`src/optimizer.py`)

A linear program solved with HiGHS through `scipy.optimize.linprog`. Four
variables per hour — grid import, solar used, battery charge, battery discharge:

```
minimise   Σ tariff[h] · grid[h]

s.t.       grid + solar_used + discharge − charge = demand         every hour
           reserve[h] ≤ E₀ + Σ(charge − discharge) ≤ capacity      every hour
           Σ charge − Σ discharge = 0                              end-of-day neutrality
           0 ≤ solar_used[h] ≤ base_solar[h] · factor[h]
           0 ≤ charge[h] ≤ max_charge        (0 in a no_charge_window)
           0 ≤ discharge[h] ≤ max_discharge  (0 in a no_discharge_window)
           0 ≤ grid[h] ≤ max_grid_kwh[h]     (when a cap applies)
```

`reserve[h]` is `max(base minimum, any directive reserve for that hour)`. This
is a true optimum, not a heuristic, which is why the public cases match the
reference cost exactly.

Three details worth knowing:

- **Netting.** A degenerate optimum can charge and discharge in the same hour.
  Only the net movement is physical and `battery_action` must be exactly one of
  `charge`/`discharge`/`idle`, so the two are netted before the plan is written.
- **Exact arithmetic.** `grid_kwh` is recomputed from the balance equation after
  rounding rather than read back from the solver, so the judge's replay of every
  hour is consistent well inside the 0.01 tolerance.
- **Tie-breaking.** A 1e-6 penalty on battery movement picks the calmest of
  several equally cheap optima. It is far too small to shift the reported cost.

### Failure behaviour

| Situation | Response |
|---|---|
| Malformed JSON, missing or wrong-typed field | `400` with a short message |
| Well-formed but impossible (23 hours, reserve above capacity, negative tariff) | `422` |
| LLM times out, errors, or returns unusable JSON | one fast retry, then the deterministic fallback parser; still `200` with a valid plan |
| LLM returns an unsupported type or out-of-range number | guardrails demote that note to `no_op`; still `200` |
| Directives are mutually contradictory | constraints relaxed in a fixed order (grid caps → elevated reserve → no-charge → no-discharge → neutrality), and `plan_summary` says what was relaxed |
| Anything unexpected | `500` with a request id only — no stack trace, no secrets |

The fallback parser is a safety net, not the interpreter: it only runs when the
model path has already failed, and its use is recorded in the logs.

---

## 4. Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` (or any OpenAI-compatible host) or `anthropic` |
| `LLM_API_KEY` | — | Provider credential. **Required.** Never committed |
| `LLM_MODEL` | `gpt-4o-mini` | Model identifier |
| `LLM_BASE_URL` | unset | Set for Groq / OpenRouter / Together / vLLM / Ollama |
| `LLM_TEMPERATURE` | `0.0` | Deterministic extraction |
| `LLM_MAX_TOKENS` | `900` | Cap on the interpretation reply |
| `LLM_TIMEOUT_SECONDS` | `12` | Per-attempt timeout, inside the 30 s judge budget |
| `LLM_MAX_RETRIES` | `1` | Extra attempts before the fallback |
| `ENABLE_RULE_FALLBACK` | `true` | Deterministic safety net on provider failure |
| `SELF_CHECK` | `true` | Replay each response and log violations |
| `API_PORT` | `5000` | Listen port |
| `LOG_LEVEL` | `INFO` | Log verbosity |

---

## 5. Security notes

- No secrets in source or in the image; everything comes from the environment,
  and `.env` is git-ignored while `.env.example` carries names only.
- API responses never include stack traces, prompts or provider errors. A `500`
  returns a request id; the detail stays in the server log.
- Only the synthetic scenario data in the request body is used. Nothing is
  persisted between requests.
- `debug=False` always — a Flask debugger on a public URL is remote code
  execution.

---

## 6. Known limitations

- **Provider dependency.** If the LLM provider is unreachable the service stays
  up and returns valid schedules, but interpretation quality then depends on the
  fallback parser, which is weaker on unusual paraphrases.
- **Overlapping same-type directives.** Two `solar_reduction` directives on the
  same hour resolve to the more restrictive factor; two reserves resolve to the
  higher one. The specification does not define this case.
- **No round-trip efficiency.** The battery model is lossless, matching the
  specification's energy rules. Real storage is not.
- **Relative times.** Notes with no absolute clock reference ("for the next few
  hours") have no defined mapping to hour indices and will usually be read as
  `no_op`.
- **Latency floor.** The LP solve is a few milliseconds; end-to-end latency is
  essentially the model round trip, so a fast model matters for the p95 ≤ 5 s
  band.
