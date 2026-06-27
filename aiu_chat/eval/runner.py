"""Gold-set evaluation runner for the text-to-SQL data path.

Runs each case in tests/eval/gold.yaml end-to-end through the real pipeline and
scores the EXECUTED RESULT (deterministic), not the narration prose. This is the
objective signal for whether a prompt/schema change helped or regressed.

Usage:
    python -m aiu_chat.eval.runner                 # run against the live model
    python -m aiu_chat.eval.runner --quiet         # only print the summary

The scoring functions (score_case) are pure and unit-tested; the runner wires
them to the live model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from aiu_chat.agent.text_to_sql import DataAnswer

GOLD_PATH = Path(__file__).resolve().parent.parent.parent / "tests" / "eval" / "gold.yaml"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    reasons: list[str] = field(default_factory=list)  # failure explanations


def _scalar_values(answer: DataAnswer):
    """All scalar cell values from the result, as a flat list."""
    if answer.result is None or answer.result.dataframe.empty:
        return []
    return [v for row in answer.result.dataframe.itertuples(index=False) for v in row]


def score_case(case: dict, answer: DataAnswer, route: str | None = None) -> CaseResult:
    """Score one gold case against a produced DataAnswer. Pure / no I/O."""
    reasons: list[str] = []

    # 0. Expected route (guards the router's classification).
    if "expected_route" in case and route is not None and route != case["expected_route"]:
        reasons.append(f"expected route {case['expected_route']!r}, got {route!r}")

    # 1. Answerability expectation.
    if case.get("expect_answerable") is False:
        if answer.ok:
            reasons.append("expected the model to decline, but it answered")
        return CaseResult(case["id"], passed=not answer.ok and not reasons, reasons=reasons)

    # For answerable cases, a non-ok answer is an automatic fail.
    if not answer.ok:
        reasons.append(f"expected an answer but pipeline failed: {answer.answer[:120]}")
        return CaseResult(case["id"], passed=False, reasons=reasons)

    sql_lower = (answer.sql or "").lower()
    values = _scalar_values(answer)

    # 2. SQL must/most-not contain.
    for needle in case.get("sql_must_contain", []):
        if needle.lower() not in sql_lower:
            reasons.append(f"SQL missing required text: {needle!r}")
    for needle in case.get("sql_must_not_contain", []):
        if needle.lower() in sql_lower:
            reasons.append(f"SQL contains forbidden text: {needle!r}")

    # 3. Expected scalar value within tolerance.
    if "expected_value" in case:
        target = float(case["expected_value"])
        tol = float(case.get("tolerance", 0))
        numeric = [v for v in values if _is_number(v)]
        if not any(abs(float(v) - target) <= tol for v in numeric):
            reasons.append(
                f"expected value {target} (±{tol}) not found in result {numeric}"
            )

    # 4. Expected substring in the result rows OR the answer text (so concept
    #    answers, which have no rows, can still be checked).
    if "expected_text_in" in case:
        needle = str(case["expected_text_in"]).lower()
        haystack = " ".join(str(v).lower() for v in values) + " " + (answer.answer or "").lower()
        if needle not in haystack:
            reasons.append(f"expected text {case['expected_text_in']!r} not found")

    return CaseResult(case["id"], passed=not reasons, reasons=reasons)


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def load_cases(path: Path = GOLD_PATH) -> list[dict]:
    data = yaml.safe_load(path.read_text())
    return data["cases"]


def _run_case(question: str) -> tuple[DataAnswer, str]:
    """Run a question through the full agent and adapt the result for scoring.

    Returns (scoreable DataAnswer, route). Imported lazily so importing the
    runner (e.g. for unit tests of score_case) doesn't require the whole
    orchestrator graph. Charts are skipped for speed.
    """
    from aiu_chat.agent.orchestrator import answer

    turn = answer(question)

    # For data/both, score the underlying DataAnswer (has SQL + executed rows).
    if turn.data is not None:
        # If concept text was also produced, fold it into the answer text so
        # expected_text_in checks can see it.
        if turn.concept is not None and turn.concept.ok:
            turn.data.answer = turn.answer
        return turn.data, turn.route

    # Other paths (concept / nop / dataapp / catalog): synthesize a DataAnswer
    # carrying the combined answer text and whether any sub-answer succeeded.
    ok = bool(
        (turn.concept and turn.concept.ok)
        or (turn.nop and turn.nop.ok)
        or (turn.dataapp and turn.dataapp.ok)
        or (turn.nm_live and turn.nm_live.ok)
        or turn.route == "catalog"
    )
    return (
        DataAnswer(question=question, sql=None, result=None, answer=turn.answer, ok=ok),
        turn.route,
    )


def run(quiet: bool = False, path: Path = GOLD_PATH) -> tuple[int, int]:
    """Run all cases. Returns (passed, total)."""
    cases = load_cases(path)
    results: list[CaseResult] = []

    for case in cases:
        try:
            answer, route = _run_case(case["question"])
        except Exception as exc:  # model unreachable, etc.
            results.append(CaseResult(case["id"], passed=False, reasons=[f"error: {exc}"]))
            if not quiet:
                print(f"  ERROR {case['id']}: {exc}")
            continue

        res = score_case(case, answer, route=route)
        results.append(res)
        if not quiet:
            mark = "PASS" if res.passed else "FAIL"
            print(f"  {mark} {case['id']}")
            for r in res.reasons:
                print(f"        - {r}")

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\nEval: {passed}/{total} passed.")
    return passed, total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the gold eval set.")
    parser.add_argument("--quiet", action="store_true", help="Only print the summary line.")
    args = parser.parse_args(argv)
    passed, total = run(quiet=args.quiet)
    # Non-zero exit if any failed, so this can gate CI / scripts.
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
