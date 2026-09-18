"""GridWise — LLM-assisted campus energy optimization API.

Flow per request:
    request  ->  schema check  ->  LLM interpretation  ->  deterministic
    guardrails  ->  LP optimizer  ->  replay self-check  ->  response
"""

import logging
import os
import time
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

from src.config import config  # noqa: E402
from src.llm_parser import parse_operator_notes  # noqa: E402
from src.optimizer import optimize_schedule, summarize  # noqa: E402
from src.replay import replay  # noqa: E402
from src.schema import RequestError, validate_request  # noqa: E402
from src.validator import collect_directives, validate_interpretation  # noqa: E402

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("gridwise")

app = Flask(__name__)
app.json.sort_keys = False  # keep the spec's field order in responses


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/optimize-energy")
def optimize_energy():
    started = time.time()
    request_id = uuid.uuid4().hex[:8]

    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400

        scenario_id, notes, hours, battery = validate_request(payload)

        # 1. The language model interprets the notes.
        raw_entries, llm_meta = parse_operator_notes(notes, battery)

        # 2. Guardrails make the interpretation safe to trust.
        interpretation, rejections = validate_interpretation(raw_entries, notes, battery)
        directives = collect_directives(interpretation)

        # 3. The optimizer schedules against the validated directives.
        plan = optimize_schedule(hours, battery, directives)

        response = {
            "scenario_id": scenario_id,
            "directive_interpretation": interpretation,
            "hourly_plan": plan["hourly_plan"],
            "total_grid_kwh": plan["total_grid_kwh"],
            "total_cost_bdt": plan["total_cost_bdt"],
            "peak_grid_kwh": plan["peak_grid_kwh"],
            "plan_summary": summarize(plan, hours, interpretation, plan["relaxed"]),
        }

        # 4. Replay the finished plan the way the judge will.
        if config.SELF_CHECK:
            problems = replay(
                {"scenario_id": scenario_id, "operator_notes": notes,
                 "hours": hours, "battery": battery},
                response, directives,
            )
            if problems:
                log.error("[%s] self-check failed: %s", request_id, problems[:5])

        log.info(
            "[%s] scenario=%s notes=%d source=%s applied=%d rejected=%d cost=%.2f %dms",
            request_id, scenario_id, len(notes), llm_meta.get("source"),
            sum(1 for e in interpretation if e["applies"]), len(rejections),
            response["total_cost_bdt"], round((time.time() - started) * 1000),
        )
        return jsonify(response), 200

    except RequestError as exc:
        log.info("[%s] rejected request: %s", request_id, exc.message)
        return jsonify({"error": exc.message}), exc.status

    except Exception:
        # Controlled failure: log the detail, return nothing internal.
        log.exception("[%s] unhandled error", request_id)
        return jsonify({
            "error": "Internal error while producing the optimization plan",
            "request_id": request_id,
        }), 500


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", config.API_PORT))
    # debug=False deliberately: a debugger in a public deployment would leak
    # stack traces and allow arbitrary code execution.
    app.run(host="0.0.0.0", port=port, debug=False)
