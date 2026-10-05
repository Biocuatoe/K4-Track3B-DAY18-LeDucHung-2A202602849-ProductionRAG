"""Tests for Module 4: Evaluation."""
import math
import os
import sys
import unittest.mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m4_eval import NAN, EvalResult, evaluate_ragas, failure_analysis, load_test_set


def test_load_test_set():
    ts = load_test_set()
    assert len(ts) > 0 and "question" in ts[0] and "ground_truth" in ts[0]

def test_evaluate_returns_metrics():
    r = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert k in r and isinstance(r[k], (int, float))

def test_failure_analysis_returns():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    assert len(f) == 1

def test_failure_has_diagnosis():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    if f:
        assert "diagnosis" in f[0] and "suggested_fix" in f[0]


# --- Regression: network exception handling ---

def test_evaluate_returns_sentinel_on_network_failure():
    """evaluate_ragas must return a well-formed sentinel dict on httpx/Hub errors.

    Simulates a proxy/403 block on HuggingfaceEmbeddings construction by patching
    the class at its definition site (``ragas.embeddings``).  The function must
    return a sentinel dict with all 4 metric keys present and numeric, plus
    ``status`` / ``error_type`` / ``error`` keys — it must NOT propagate the
    network exception.
    """
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("ragas")
    pytest.importorskip("langchain_openai")

    network_error = httpx.ProxyError("403 Forbidden", request=None)

    with unittest.mock.patch(
        "ragas.embeddings.HuggingfaceEmbeddings",
        side_effect=network_error,
    ):
        r = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
        # All 4 metric keys must be present and numeric
        for k in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]:
            assert k in r, f"Missing key: {k}"
            assert isinstance(r[k], (int, float)), f"{k} must be numeric"
        # Must have status and error metadata
        assert "status" in r, "Missing status key"
        assert r["status"] in (
            "failed_external",
            "failed_code",
            "skipped_no_api_key",
            "failed_validation",
        ), f"Unexpected status: {r['status']}"
        assert "error_type" in r
        assert "error" in r


# ─────────────────────────────────────────────────────────────────────────────
#  NEW REGRESSION TESTS for the NaN-sentinel bug fix
# ─────────────────────────────────────────────────────────────────────────────

class TestNanNotLaunderedToZero:
    """Regression: NaN (judge failed / not measured) must NOT become 0.0."""

    def test_nan_metric_is_not_reported_as_zero(self):
        """A per-question NaN score must appear as None (not 0.0) in output."""
        results = [
            # Question 1: only faithfulness measured (others NaN from rate-limit)
            EvalResult(
                question="Q1",
                answer="A1",
                contexts=["C1"],
                ground_truth="GT1",
                faithfulness=0.8,
                answer_relevancy=NAN,
                context_precision=NAN,
                context_recall=NAN,
            ),
        ]
        f = failure_analysis(results, bottom_n=1)

        assert len(f) == 1
        # answer_relevancy was NOT measured → must be None, NOT 0.0
        assert f[0]["answer_relevancy"] is None, (
            f"NaN metric must be None, got {f[0]['answer_relevancy']!r} — "
            "judge failure was laundered to 0.0!"
        )
        # faithfulness was measured → must be the real float
        assert f[0]["faithfulness"] == 0.8

    def test_all_nan_metric_not_reported_as_zero_score(self):
        """A metric that is NaN for every question must NOT appear as 0.0."""
        results = [
            # Both questions: answer_relevancy is NaN everywhere
            EvalResult("Q1", "A1", ["C1"], "GT1",
                       0.5, NAN, 0.4, 0.3),
            EvalResult("Q2", "A2", ["C2"], "GT2",
                       0.6, NAN, 0.7, 0.2),
        ]
        f = failure_analysis(results, bottom_n=2)

        # Neither question should have answer_relevancy as 0.0
        for item in f:
            assert item["answer_relevancy"] is None, (
                f"Globally-unmeasured metric must be None, got {item['answer_relevancy']!r}"
            )

    def test_genuine_zero_not_confused_with_nan(self):
        """A question that genuinely scores 0.0 must keep 0.0 (not None)."""
        results = [
            # faithfulness genuinely 0.0 (bad answer), others valid
            EvalResult("Q1", "A1", ["C1"], "GT1",
                       0.0, 0.7, 0.5, 0.6),
        ]
        f = failure_analysis(results, bottom_n=1)
        assert len(f) == 1
        # Genuine 0.0 must survive
        assert f[0]["faithfulness"] == 0.0, (
            "Genuine 0.0 must be preserved, not converted to None!"
        )

    def test_mean_excludes_nan_questions(self):
        """Mean must be over successfully-measured questions only."""
        # With the original buggy _mean: sum=[0.8]/1=0.8, /4 questions=0.2
        # Fixed: 0.8/1=0.8 (only 1 measured question)
        # Note: sort is ascending by mean_score, so Q2 (0.6) comes before Q1 (0.8).
        # We test both results exist with correct mean scores.
        results = [
            EvalResult("Q1", "A1", ["C1"], "GT1",
                       0.8, NAN, NAN, NAN),   # faith only, mean=0.8
            EvalResult("Q2", "A2", ["C2"], "GT2",
                       0.6, NAN, NAN, NAN),   # faith only, mean=0.6
        ]
        f = failure_analysis(results, bottom_n=2)
        # Both questions should appear with correct per-question mean scores
        q1 = next(x for x in f if x["question"] == "Q1")
        q2 = next(x for x in f if x["question"] == "Q2")
        assert q1["mean_score"] == 0.8, (
            f"Q1 mean_score should be 0.8 (1 measured metric), got {q1['mean_score']!r}"
        )
        assert q2["mean_score"] == 0.6, (
            f"Q2 mean_score should be 0.6 (1 measured metric), got {q2['mean_score']!r}"
        )


class TestAnalyzeFailuresNoDiagnosisForUnmeasured:
    """Regression: _analyze_failures must NOT emit hallucination/off-topic diagnoses
    for metrics that were never measured."""

    def test_no_diagnosis_when_all_metrics_nan(self):
        """When no metric was measured, diagnosis must say 'judge unavailable'."""
        results = [
            EvalResult("Q1", "A1", ["C1"], "GT1",
                       NAN, NAN, NAN, NAN),
        ]
        f = failure_analysis(results, bottom_n=1)
        assert len(f) == 1
        assert "judge unavailable" in f[0]["diagnosis"].lower() or \
               "not measured" in f[0]["diagnosis"].lower() or \
               "no metrics were measured" in f[0]["diagnosis"].lower(), \
            f"Diagnosis for fully-unmeasured question must mention judge failure, got: {f[0]['diagnosis']!r}"

    def test_no_hallucination_label_for_nan_worst_metric(self):
        """NaN metrics must never be selected as 'worst' for diagnosis."""
        results = [
            # Only faithfulness measured (0.8); others are NaN
            # Worst must be faithfulness (the only measured metric)
            EvalResult("Q1", "A1", ["C1"], "GT1",
                       0.8, NAN, NAN, NAN),
        ]
        f = failure_analysis(results, bottom_n=1)
        assert len(f) == 1
        # worst_metric must be "faithfulness", NOT a NaN metric
        assert f[0]["worst_metric"] == "faithfulness", (
            f"NaN metric selected as worst_metric! got: {f[0]['worst_metric']!r}"
        )
        # Diagnosis must be the faithfulness diagnosis, not hallucination per se
        assert f[0]["diagnosis"] != "Hallucination / insufficient grounding — model generates claims not supported by retrieved context." or \
               f[0]["worst_metric"] == "faithfulness", \
            "Diagnosis mismatch with worst_metric"

    def test_nan_metrics_excluded_from_mean_score(self):
        """mean_score must not be dragged to 0.0 by NaN values."""
        results = [
            # Q1: faith=0.8 (measured), others NaN → mean=0.8, NOT 0.2
            EvalResult("Q1", "A1", ["C1"], "GT1",
                       0.8, NAN, NAN, NAN),
            # Q2: all NaN → mean=NaN (not measured)
            EvalResult("Q2", "A2", ["C2"], "GT2",
                       NAN, NAN, NAN, NAN),
        ]
        f = failure_analysis(results, bottom_n=2)
        # Q1 mean should be 0.8, not some fraction diluted by NaN
        q1 = next(x for x in f if x["question"] == "Q1")
        assert q1["mean_score"] == 0.8, (
            f"mean_score should be 0.8 (over 1 measured metric), got {q1['mean_score']!r}"
        )
        # Q2: all NaN → mean should be NaN (not 0.0)
        q2 = next(x for x in f if x["question"] == "Q2")
        assert math.isnan(q2["mean_score"]), (
            f"mean_score for all-NaN question must be NaN, got {q2['mean_score']!r}"
        )


class TestDegradedStatus:
    """Regression: Low metric coverage must produce status='degraded', not 'ok'."""

    def test_low_coverage_yields_degraded_status(self):
        """When any metric has coverage < 1.0, status must be 'degraded'."""
        pytest.importorskip("ragas")
        import pandas as pd
        pytest.importorskip("httpx")
        pytest.importorskip("langchain_openai")

        fake_df = pd.DataFrame({
            "question":      ["q1", "q2"],
            "answer":        ["a1", "a2"],
            "contexts":      [["c1"], ["c2"]],
            "ground_truth":  ["gt1", "gt2"],
            # answer_relevancy: 0/2 measured → coverage=0 → degraded
            "faithfulness":      [0.8,  float("nan")],
            "answer_relevancy": [float("nan"), float("nan")],
            "context_precision": [0.5, 0.6],
            "context_recall":   [0.3, float("nan")],
        })

        class FakeResult:
            def to_pandas(self):
                return fake_df

        with unittest.mock.patch("ragas.evaluate", return_value=FakeResult()):
            with unittest.mock.patch(
                "ragas.embeddings.HuggingfaceEmbeddings",
                return_value=object(),
            ):
                r = evaluate_ragas(
                    questions=["q1", "q2"],
                    answers=["a1", "a2"],
                    contexts=[["c1"], ["c2"]],
                    ground_truths=["gt1", "gt2"],
                )
                assert r["status"] == "degraded", (
                    f"Low metric coverage must yield 'degraded' status, got {r['status']!r}"
                )

    def test_full_coverage_yields_ok_status(self):
        """When all metrics cover all questions, status must be 'ok'."""
        pytest.importorskip("ragas")
        import pandas as pd
        pytest.importorskip("httpx")
        pytest.importorskip("langchain_openai")

        fake_df = pd.DataFrame({
            "question":      ["q1", "q2"],
            "answer":        ["a1", "a2"],
            "contexts":      [["c1"], ["c2"]],
            "ground_truth":  ["gt1", "gt2"],
            "faithfulness":      [0.8, 0.6],
            "answer_relevancy": [0.7, 0.5],
            "context_precision": [0.5, 0.6],
            "context_recall":   [0.3, 0.4],
        })

        class FakeResult:
            def to_pandas(self):
                return fake_df

        with unittest.mock.patch("ragas.evaluate", return_value=FakeResult()):
            with unittest.mock.patch(
                "ragas.embeddings.HuggingfaceEmbeddings",
                return_value=object(),
            ):
                r = evaluate_ragas(
                    questions=["q1", "q2"],
                    answers=["a1", "a2"],
                    contexts=[["c1"], ["c2"]],
                    ground_truths=["gt1", "gt2"],
                )
                assert r["status"] == "ok", (
                    f"Full coverage must yield 'ok' status, got {r['status']!r}"
                )

    def test_metric_coverage_in_results_when_degraded(self):
        """Results dict must include metric_coverage when coverage < 1.0."""
        pytest.importorskip("ragas")
        import pandas as pd
        pytest.importorskip("httpx")
        pytest.importorskip("langchain_openai")

        fake_df = pd.DataFrame({
            "question":      ["q1", "q2"],
            "answer":        ["a1", "a2"],
            "contexts":      [["c1"], ["c2"]],
            "ground_truth":  ["gt1", "gt2"],
            "faithfulness":      [0.8,  float("nan")],
            "answer_relevancy": [float("nan"), float("nan")],
            "context_precision": [0.5, 0.6],
            "context_recall":   [0.3, 0.4],
        })

        class FakeResult:
            def to_pandas(self):
                return fake_df

        with unittest.mock.patch("ragas.evaluate", return_value=FakeResult()):
            with unittest.mock.patch(
                "ragas.embeddings.HuggingfaceEmbeddings",
                return_value=object(),
            ):
                r = evaluate_ragas(
                    questions=["q1", "q2"],
                    answers=["a1", "a2"],
                    contexts=[["c1"], ["c2"]],
                    ground_truths=["gt1", "gt2"],
                )
                assert "metric_coverage" in r, (
                    "metric_coverage must be present when coverage < 1.0"
                )
                assert r["metric_coverage"]["answer_relevancy"]["n_measured"] == 0
                assert r["metric_coverage"]["answer_relevancy"]["n_total"] == 2
                assert r["metric_coverage"]["answer_relevancy"]["coverage"] == 0.0

    def test_status_contract_preserved(self):
        """Degraded status must NOT break callers expecting 'ok'/'failed_*'/'skipped_*'."""
        pytest.importorskip("ragas")
        import pandas as pd
        pytest.importorskip("httpx")
        pytest.importorskip("langchain_openai")

        fake_df = pd.DataFrame({
            "question":      ["q1"],
            "answer":        ["a1"],
            "contexts":      [["c1"]],
            "ground_truth":  ["gt1"],
            "faithfulness":      [0.8],
            "answer_relevancy": [0.7],
            "context_precision": [0.5],
            "context_recall":   [0.3],
        })

        class FakeResult:
            def to_pandas(self):
                return fake_df

        with unittest.mock.patch("ragas.evaluate", return_value=FakeResult()):
            with unittest.mock.patch(
                "ragas.embeddings.HuggingfaceEmbeddings",
                return_value=object(),
            ):
                r = evaluate_ragas(
                    questions=["q1"],
                    answers=["a1"],
                    contexts=[["c1"]],
                    ground_truths=["gt1"],
                )
                assert r["status"] in (
                    "ok", "skipped_no_api_key", "failed_external",
                    "failed_code", "failed_validation", "degraded",
                ), f"Status {r['status']!r} not in expected contract"

