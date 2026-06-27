"""Unit tests for the eval scoring logic (pure; no model needed)."""
from __future__ import annotations

import pandas as pd

from aiu_chat.agent.sql_tool import SqlResult
from aiu_chat.agent.text_to_sql import DataAnswer
from aiu_chat.eval.runner import score_case


def _answer(sql=None, df=None, ok=True, answer="x"):
    result = None
    if df is not None:
        result = SqlResult(sql=sql or "", dataframe=df, row_count=len(df), truncated=False)
    return DataAnswer(question="q", sql=sql, result=result, answer=answer, ok=ok)


def test_expected_value_within_tolerance_passes():
    case = {"id": "v", "expected_value": 100.0, "tolerance": 1.0}
    ans = _answer(sql="select 1", df=pd.DataFrame({"x": [100.4]}))
    assert score_case(case, ans).passed


def test_expected_value_outside_tolerance_fails():
    case = {"id": "v", "expected_value": 100.0, "tolerance": 1.0}
    ans = _answer(sql="select 1", df=pd.DataFrame({"x": [200.0]}))
    assert not score_case(case, ans).passed


def test_expected_text_in_passes():
    case = {"id": "t", "expected_text_in": "FRANCE"}
    ans = _answer(sql="select 1", df=pd.DataFrame({"s": ["FRANCE"], "c": [5]}))
    assert score_case(case, ans).passed


def test_sql_must_contain_and_not_contain():
    case = {"id": "s", "sql_must_contain": ["sum"], "sql_must_not_contain": ["avg"]}
    good = _answer(sql="SELECT SUM(x) FROM t", df=pd.DataFrame({"x": [1]}))
    bad = _answer(sql="SELECT AVG(x) FROM t", df=pd.DataFrame({"x": [1]}))
    assert score_case(case, good).passed
    assert not score_case(case, bad).passed


def test_unanswerable_expectation():
    case = {"id": "u", "expect_answerable": False}
    declined = _answer(ok=False, answer="cannot")
    answered = _answer(sql="select 1", df=pd.DataFrame({"x": [1]}), ok=True)
    assert score_case(case, declined).passed
    assert not score_case(case, answered).passed


def test_answerable_but_failed_pipeline_fails():
    case = {"id": "a", "expected_value": 1, "tolerance": 0}
    ans = _answer(ok=False, answer="query failed")
    res = score_case(case, ans)
    assert not res.passed
    assert "failed" in res.reasons[0].lower()
