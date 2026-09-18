"""Run the 10 public sample cases through the full pipeline and replay them.

Usage:
    python tests/run_public_cases.py                      # in-process
    python tests/run_public_cases.py --url http://host:5000   # against a deployment
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.replay import replay
from src.validator import collect_directives

DEFAULT_CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json")


def expected_directives(expected_output):
    return collect_directives(expected_output["directive_interpretation"])


def compare_interpretation(actual, expected):
    """Semantic comparison; free-text explanations are ignored."""
    diffs = []
    if len(actual) != len(expected):
        return [f"expected {len(expected)} entries, got {len(actual)}"]
    for i, (got, want) in enumerate(zip(actual, expected)):
        if got.get("directive_type") != want.get("directive_type"):
            diffs.append(f"note {i}: type {got.get('directive_type')} != {want.get('directive_type')}")
            continue
        if got.get("applies") != want.get("applies"):
            diffs.append(f"note {i}: applies {got.get('applies')} != {want.get('applies')}")
        ga, wa = got.get("structured_adjustment"), want.get("structured_adjustment")
        if wa is None:
            if ga is not None:
                diffs.append(f"note {i}: expected null adjustment")
            continue
        if not isinstance(ga, dict):
            diffs.append(f"note {i}: missing adjustment")
            continue
        if ga.get("hours") != wa.get("hours"):
            diffs.append(f"note {i}: hours {ga.get('hours')} != {wa.get('hours')}")
        for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
            if key in wa and abs(float(ga.get(key, -999)) - float(wa[key])) > 0.01:
                diffs.append(f"note {i}: {key} {ga.get(key)} != {wa[key]}")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="base URL of a running service; omit to test in-process")
    ap.add_argument("--cases", default=DEFAULT_CASES)
    args = ap.parse_args()

    with open(args.cases) as fh:
        pack = json.load(fh)

    if args.url:
        import requests
        def send(payload):
            r = requests.post(args.url.rstrip("/") + "/optimize-energy", json=payload, timeout=35)
            return r.status_code, r.json()
    else:
        os.environ.setdefault("SELF_CHECK", "false")
        from main import app
        client = app.test_client()
        def send(payload):
            r = client.post("/optimize-energy", json=payload)
            return r.status_code, r.get_json()

    passed = valid_count = interp_ok = 0
    ratios, latencies = [], []

    for case in pack["cases"]:
        started = time.time()
        status, body = send(case["input"])
        elapsed = time.time() - started
        latencies.append(elapsed)

        if status != 200:
            print(f"FAIL {case['id']}: HTTP {status} {body}")
            continue

        want = case["expected_output"]
        diffs = compare_interpretation(body["directive_interpretation"],
                                       want["directive_interpretation"])
        # Replay against the GROUND-TRUTH directives, exactly as the judge does.
        problems = replay(case["input"], body, expected_directives(want))

        ref, got = float(want["total_cost_bdt"]), float(body["total_cost_bdt"])
        ratio = min(1.0, ref / got) if got > 0 else 0.0
        if not problems:
            valid_count += 1
            ratios.append(ratio)
        if not diffs:
            interp_ok += 1
        if not problems and not diffs:
            passed += 1

        flag = "PASS" if (not problems and not diffs) else "FAIL"
        delta = got - ref
        print(f"{flag} {case['id']:<10} cost {got:>10.2f} (ref {ref:>9.2f}, "
              f"{delta:+.2f})  score {ratio:.4f}  {elapsed*1000:>5.0f}ms")
        for d in diffs:
            print(f"     interpretation: {d}")
        for p in problems[:6]:
            print(f"     replay: {p}")

    n = len(pack["cases"])
    latencies.sort()
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
    print("-" * 78)
    print(f"cases {n} | fully passing {passed} | valid schedules {valid_count} | "
          f"interpretation correct {interp_ok}")
    if ratios:
        print(f"mean optimization score {sum(ratios)/len(ratios):.4f} "
              f"(1.0000 = matches organizer optimum)")
    print(f"p95 latency {p95*1000:.0f}ms")
    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
