from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH
from src._network_exceptions import get_network_exceptions as _get_network_exceptions

# Sentinel value used throughout to mean "judge failed / not measured".
# We deliberately use the Python float NaN as the sentinel so that
# math.isnan() / math.isclose() / etc. can distinguish it from a
# genuine 0.0 score.  This sentinel is never converted to 0.0.
NAN = float("nan")


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


# ── Helpers ──────────────────────────────────────────────────────────────

def _safe_float(val: Any, default: float = NAN) -> float:
    """Coerce a value to float.

    Returns the float value, or ``NAN`` sentinel if the value is None or
    cannot be converted.  Unlike the old version this function does NOT
    return 0.0 for NaN — callers that need a 0 fallback must pass
    ``default=0.0`` explicitly so the intent is visible at the call site.
    """
    if val is None:
        return NAN
    try:
        f = float(val)
    except (TypeError, ValueError):
        return NAN
    return f


# ── Diagnostic tree ─────────────────────────────────────────────────────

_DIAGNOSTIC_MAP = {
    "faithfulness": {
        "diagnosis": "Hallucination / insufficient grounding — model generates claims "
                     "not supported by retrieved context.",
        "fix": "Strengthen grounded prompt, lower temperature, retrieve more or "
               "higher-quality context, add citation constraints.",
    },
    "context_recall": {
        "diagnosis": "Relevant evidence not retrieved — retrieved chunks miss key "
                     "facts needed to answer the question.",
        "fix": "Improve chunking strategy, add hybrid retrieval (BM25 + dense), "
               "apply chunk enrichment.",
    },
    "context_precision": {
        "diagnosis": "Too many irrelevant retrieved passages — high noise in context "
                     "reduces signal quality.",
        "fix": "Improve reranking or relevance filtering, tune similarity "
               "threshold, remove stale chunks.",
    },
    "answer_relevancy": {
        "diagnosis": "Answer does not directly address the question — response is "
                     "off-topic or incomplete.",
        "fix": "Improve answer prompt template, add explicit output constraints, "
               "ensure retrieved context is sufficient.",
    },
}

# Tiebreak priority order (lowest priority first) — deterministic
_METRIC_PRIORITY = [
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "faithfulness",
]


def _build_error_tree(result: EvalResult, worst: str, worst_score: float) -> str:
    """Construct per-question error-tree decision string.

    Follows the assignment decision chain:
      Output sai? → Context đúng? → Query OK?
    Branch is driven by the worst metric and context/ground-truth overlap.

    Parameters
    ----------
    result : EvalResult
        The evaluation result for the question.
    worst : str
        The name of the worst (lowest) metric that was actually measured
        (NaN metrics are never passed here).
    worst_score : float
        The score for the worst metric (always a finite float).
    """
    # Check context overlap with ground truth (simple keyword heuristic)
    gt_words = set(result.ground_truth.lower().split())
    ctx_words: set[str] = set()
    for ctx in result.contexts:
        ctx_words.update(ctx.lower().split())
    overlap = bool(gt_words & ctx_words)

    # Build tree string
    tree: list[str] = []
    tree.append("[Output OK?] No")
    tree.append(f"  └─ Output faithfulness = {worst_score:.3f}")
    tree.append(f"  └─ Worst metric: {worst} ({worst_score:.3f})")

    if worst in ("faithfulness", "answer_relevancy"):
        tree.append(
            "  └─ [Context grounded?] Possibly not — model diverged from context"
        )
        if not overlap:
            tree.append("  └─ [Context overlaps GT?] No — context may be misaligned")
        else:
            tree.append("  └─ [Context overlaps GT?] Yes — hallucination likely")
    elif worst == "context_recall":
        if not overlap:
            tree.append("  └─ [Context overlaps GT?] No — retrieval gap")
        else:
            tree.append(
                "  └─ [Context overlaps GT?] Yes — chunking too coarse or granular"
            )
        tree.append("  └─ [Query OK?] Yes — search/retrieval step needs improvement")
    elif worst == "context_precision":
        tree.append("  └─ [Context overlaps GT?] Possibly — but noise is high")
        tree.append("  └─ [Query OK?] Yes — reranking/filtering step needs improvement")

    return "\n".join(tree)


# ── evaluate_ragas ──────────────────────────────────────────────────────

def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run RAGAS evaluation and return aggregate metrics + per-question results.

    Returns
    -------
    dict with top-level keys:
        - "faithfulness"       : float (mean over successfully-measured questions)
        - "answer_relevancy"   : float (mean over successfully-measured questions)
        - "context_precision"  : float (mean over successfully-measured questions)
        - "context_recall"     : float (mean over successfully-measured questions)
        - "per_question"       : list[EvalResult]
        - "status"             : "ok" | "skipped_no_api_key" | "failed_external" |
                                  "failed_code" | "failed_validation" | "degraded"
        - "error"              : str (human-readable, never includes API key)
        - "error_type"          : str (exception class name, or "NoApiKey")
        - "metric_coverage"    : dict[str, {"n_measured": int, "n_total": int,
                                             "coverage": float}]
                                  — added when any metric has coverage < 1.0
    """
    # ── Input validation ──────────────────────────────────────────────
    n = len(questions)
    if not questions or not answers or not contexts or not ground_truths:
        return _make_result(
            status="failed_validation",
            error="Empty or mismatched input lists",
            error_type="ValueError",
        )
    if not (n == len(answers) == len(contexts) == len(ground_truths)):
        return _make_result(
            status="failed_validation",
            error=(
                f"Length mismatch: {n} questions, {len(answers)} answers, "
                f"{len(contexts)} contexts, {len(ground_truths)} ground_truths"
            ),
            error_type="ValueError",
        )
    if any(not c for c in contexts):
        return _make_result(
            status="failed_validation",
            error="One or more contexts lists are empty",
            error_type="ValueError",
        )

    # ── Check Groq availability ───────────────────────────────────────
    try:

        from config import (  # pylint: disable=import-error,useless-suppression
            GROQ_API_KEY,
            GROQ_BASE_URL,
            GROQ_MODEL,
        )
        if not GROQ_API_KEY:
            print(
                "  ⚠️  evaluate_ragas: GROQ_API_KEY is not set — skipping "
                "RAGAS evaluation (set GROQ_API_KEY env var to enable).",
                flush=True,
            )
            return _make_result(
                status="skipped_no_api_key",
                error="GROQ_API_KEY environment variable not set",
                error_type="NoApiKey",
            )
    except (ImportError,) + _get_network_exceptions() as exc:
        return _make_result(
            status="failed_code",
            error=str(exc),
            error_type=type(exc).__name__,
        )

    # ── Build LLM and embedding wrappers ──────────────────────────────
    try:
        from langchain_openai import ChatOpenAI  # pylint: disable=import-error
        from ragas.embeddings import HuggingfaceEmbeddings
        from ragas.llms import LangchainLLMWrapper
    except (ImportError, OSError) as exc:
        return _make_result(
            status="failed_code",
            error=f"Could not import required packages: {exc}",
            error_type=type(exc).__name__,
        )

    try:
        langchain_llm = ChatOpenAI(
            model=GROQ_MODEL,
            base_url=GROQ_BASE_URL,
            api_key=GROQ_API_KEY,
            temperature=0,
            timeout=30,
            max_retries=2,
        )
        llm = LangchainLLMWrapper(langchain_llm)
    except _get_network_exceptions() as exc:
        return _make_result(
            status="failed_code",
            error=f"Failed to construct ChatOpenAI / LangchainLLMWrapper: {exc}",
            error_type=type(exc).__name__,
        )

    try:
        # ragas ships its own SentenceTransformer-backed adapter, so RAGAS
        # embeddings stay fully local (BAAI/bge-m3) and need no OpenAI key.
        emb_wrapper = HuggingfaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except _get_network_exceptions() as exc:
        return _make_result(
            status="failed_code",
            error=f"Failed to construct HuggingfaceEmbeddings: {exc}",
            error_type=type(exc).__name__,
        )

    # ── Build dataset ─────────────────────────────────────────────────
    try:
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts,
            "ground_truth": ground_truths,
        })
    except (TypeError, ValueError, OSError) as exc:
        return _make_result(
            status="failed_code",
            error=f"Failed to construct datasets.Dataset: {exc}",
            error_type=type(exc).__name__,
        )

    # ── Run evaluate ──────────────────────────────────────────────────
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig

        run_cfg = RunConfig(timeout=120, max_retries=2)

        result_obj = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=llm,
            embeddings=emb_wrapper,
            run_config=run_cfg,
            raise_exceptions=False,
        )
    except _get_network_exceptions() as exc:
        err_str = str(exc)
        # Distinguish external (API/network) errors from code errors
        if any(
            kw in err_str.lower()
            for kw in [
                "api",
                "auth",
                "401",
                "429",
                "timeout",
                "rate",
                "connection",
                "httpx",
                "connect",
                "unauthorized",
            ]
        ):
            status = "failed_external"
        else:
            status = "failed_code"
        # Scrub API key from error messages
        err_str = _scrub_key(err_str)
        print(
            f"  ⚠️  RAGAS evaluate failed ({status}): {err_str}",
            flush=True,
        )
        return _make_result(
            status=status,
            error=err_str,
            error_type=type(exc).__name__,
        )

    # ── Process results ───────────────────────────────────────────────
    try:
        df = result_obj.to_pandas()
    except (TypeError, AttributeError) + _get_network_exceptions() as exc:
        return _make_result(
            status="failed_code",
            error=f"Failed to convert RAGAS result to DataFrame: {exc}",
            error_type=type(exc).__name__,
        )

    per_question: list[EvalResult] = []
    faith_vals: list[float] = []
    ansr_vals: list[float] = []
    cp_vals: list[float] = []
    cr_vals: list[float] = []

    for _, row in df.iterrows():
        try:
            # _safe_float without explicit default → returns NAN for None/NaN
            faith = _safe_float(row.get("faithfulness"))
            ansr  = _safe_float(row.get("answer_relevancy"))
            cp    = _safe_float(row.get("context_precision"))
            cr    = _safe_float(row.get("context_recall"))
        except Exception:  # noqa: BLE001 — pandas cell access can raise anything
            faith, ansr, cp, cr = NAN, NAN, NAN, NAN

        faith_vals.append(faith)
        ansr_vals.append(ansr)
        cp_vals.append(cp)
        cr_vals.append(cr)

        per_question.append(
            EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=faith,
                answer_relevancy=ansr,
                context_precision=cp,
                context_recall=cr,
            )
        )

    # Compute per-metric means over ONLY successfully-measured questions.
    # NaN values are excluded — the mean represents "average score among
    # questions where the judge actually ran".
    def _mean(seq: list[float]) -> float:
        vals = [v for v in seq if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else NAN

    n_total = len(per_question)
    n_measured_faith = sum(1 for v in faith_vals if not math.isnan(v))
    n_measured_ansr  = sum(1 for v in ansr_vals  if not math.isnan(v))
    n_measured_cp    = sum(1 for v in cp_vals    if not math.isnan(v))
    n_measured_cr    = sum(1 for v in cr_vals    if not math.isnan(v))

    # Determine status: "degraded" if any metric measured fewer than all questions
    coverage_map = {
        "faithfulness":      {"n_measured": n_measured_faith, "n_total": n_total,
                              "coverage": n_measured_faith / n_total if n_total else 0.0},
        "answer_relevancy":  {"n_measured": n_measured_ansr, "n_total": n_total,
                              "coverage": n_measured_ansr / n_total if n_total else 0.0},
        "context_precision": {"n_measured": n_measured_cp, "n_total": n_total,
                              "coverage": n_measured_cp / n_total if n_total else 0.0},
        "context_recall":    {"n_measured": n_measured_cr, "n_total": n_total,
                              "coverage": n_measured_cr / n_total if n_total else 0.0},
    }
    any_low_coverage = any(
        cov["coverage"] < 1.0 for cov in coverage_map.values()
    )
    status = "degraded" if any_low_coverage else "ok"

    result_dict = {
        "faithfulness":       _mean(faith_vals),
        "answer_relevancy":  _mean(ansr_vals),
        "context_precision":  _mean(cp_vals),
        "context_recall":    _mean(cr_vals),
        "per_question":       per_question,
        "status":             status,
        "error":              None,
        "error_type":         None,
    }

    # Attach coverage map whenever any metric has coverage < 1.0 so callers
    # can see exactly which metrics were not fully measured.
    if any_low_coverage:
        result_dict["metric_coverage"] = coverage_map

    return result_dict


def _make_result(
    status: str,
    error: str,
    error_type: str,
    per_question: list | None = None,
) -> dict:
    sentinel = {
        "faithfulness":      0.0,
        "answer_relevancy":  0.0,
        "context_precision": 0.0,
        "context_recall":    0.0,
        "per_question": per_question if per_question is not None else [],
    }
    sentinel.update({"status": status, "error": error, "error_type": error_type})
    return sentinel


def _scrub_key(msg: str) -> str:
    """Remove GROQ_API_KEY value from error messages."""
    try:
        from config import GROQ_API_KEY

        if GROQ_API_KEY and GROQ_API_KEY in msg:
            msg = msg.replace(GROQ_API_KEY, "<GROQ_API_KEY>")
    except (ImportError, OSError):
        pass
    return msg


# ── failure_analysis ───────────────────────────────────────────────────

def failure_analysis(
    eval_results: list[EvalResult],
    bottom_n: int = 10,
) -> list[dict]:
    """Return the bottom-N worst questions with Diagnostic Tree analysis.

    Parameters
    ----------
    eval_results : list[EvalResult]
        Per-question evaluation results from ``evaluate_ragas``.
    bottom_n : int, default 10
        Number of worst questions to return.

    Returns
    -------
    list[dict]
        Each dict contains:
        - question, worst_metric, score, diagnosis, suggested_fix,
          mean_score, all 4 per-metric scores, ground_truth, answer,
          error_tree, root_cause

    Metrics that were never measured (NaN sentinel) are excluded from the
    worst-metric selection and from the mean score.  They are reported with
    ``None`` in the per-metric score fields of the output dict so that
    callers can clearly distinguish "not measured" from "genuinely zero".
    """
    if bottom_n <= 0 or not eval_results:
        return []

    failures: list[dict] = []
    for result in eval_results:
        metrics = {
            "faithfulness":      result.faithfulness,
            "answer_relevancy":  result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall":    result.context_recall,
        }

        # Compute mean over ONLY measured (finite) metrics for this question.
        valid = [v for v in metrics.values() if not math.isnan(v)]
        mean_score = sum(valid) / len(valid) if valid else NAN

        # Identify worst metric among ONLY measured metrics.
        # NaN metrics are excluded so we never diagnose a judge failure as
        # hallucination / off-topic / etc.
        measured_metrics = {k: v for k, v in metrics.items() if not math.isnan(v)}

        if not measured_metrics:
            # All metrics are NaN — report "not measured" for the whole question.
            failures.append({
                "question":           result.question,
                "worst_metric":       None,
                "score":              None,
                "diagnosis":         "Judge unavailable — no metrics were measured "
                                     "for this question.",
                "suggested_fix":     "Check API availability and retry.",
                "mean_score":         mean_score,
                "faithfulness":      None,
                "answer_relevancy":  None,
                "context_precision": None,
                "context_recall":    None,
                "ground_truth":       result.ground_truth,
                "answer":             result.answer,
                "error_tree":         "[Output OK?] Unknown — judge was unavailable; "
                                       "no metrics measured for this question.",
                "root_cause":         "Judge unavailable — no metrics were measured.",
            })
            continue

        worst = min(
            measured_metrics,
            key=lambda m: (measured_metrics[m], _METRIC_PRIORITY.index(m)),
        )
        worst_score = measured_metrics[worst]  # always finite

        diag = _DIAGNOSTIC_MAP.get(worst, {})
        error_tree = _build_error_tree(result, worst, worst_score)

        def _fmt(val: float | None) -> float | None:
            """Return None for NaN so callers can distinguish 'not measured'."""
            return None if (val is None or math.isnan(val)) else val

        failures.append({
            "question":           result.question,
            "worst_metric":       worst,
            "score":              worst_score,
            "diagnosis":          diag.get(
                "diagnosis", "Unknown failure mode."
            ),
            "suggested_fix":     diag.get("fix", "Investigate this failure mode."),
            "mean_score":         mean_score,
            "faithfulness":      _fmt(result.faithfulness),
            "answer_relevancy":  _fmt(result.answer_relevancy),
            "context_precision": _fmt(result.context_precision),
            "context_recall":    _fmt(result.context_recall),
            "ground_truth":       result.ground_truth,
            "answer":             result.answer,
            "error_tree":         error_tree,
            "root_cause":         diag.get("diagnosis", "Unknown"),
        })

    # Sort ascending by mean_score → take bottom_n
    # NaN mean_score sorts last (NaN > any number), which is intentional:
    # questions with no measured metrics are ranked at the bottom but still
    # included so the "judge unavailable" signal is visible.
    def _sort_key(f: dict) -> tuple:
        ms = f["mean_score"]
        return (0.0 if math.isnan(ms) else ms,)

    failures.sort(key=_sort_key)
    return failures[:bottom_n]


# ── save_report ───────────────────────────────────────────────────────

def save_report(
    results: dict,
    failures: list[dict],
    path: str = "reports/ragas_report.json",
    *,
    latency_breakdown_ms: dict | None = None,
    provider_info: dict | None = None,
) -> None:
    """Save evaluation report to JSON.

    Parameters
    ----------
    results : dict
        Return value of ``evaluate_ragas``.
    failures : list[dict]
        Return value of ``failure_analysis``.
    path : str
        Output file path.
    latency_breakdown_ms : dict | None
        Optional latency breakdown (e.g. {"indexing_ms": 120, "rerank_ms": 45}).
        Included verbatim in the report under ``latency_breakdown_ms`` if provided.
    provider_info : dict | None
        Optional provider/model metadata (from ``src.llm_client.provider_info``)
        merged into the report. Never contains the API key.
    """
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Aggregate block: top-level numeric keys only (no EvalResult objects).
    # NaN metric means are preserved as null in JSON (not laundered to 0.0).
    aggregate: dict = {}
    for k, v in results.items():
        if k == "per_question":
            continue
        if isinstance(v, (int, float)):
            # Keep NaN as null so the report is honest about unmeasured metrics.
            aggregate[k] = None if math.isnan(float(v)) else float(v)
        elif isinstance(v, str):
            aggregate[k] = v
        elif v is None:
            pass  # skip None fields in aggregate

    per_question_out: list[dict] = []
    for er in results.get("per_question", []):
        if isinstance(er, EvalResult):
            def _field(val: float) -> float | None:
                return None if math.isnan(val) else val

            per_question_out.append({
                "question":          er.question,
                "answer":            er.answer,
                "contexts":          er.contexts,
                "ground_truth":      er.ground_truth,
                "faithfulness":      _field(er.faithfulness),
                "answer_relevancy":  _field(er.answer_relevancy),
                "context_precision": _field(er.context_precision),
                "context_recall":    _field(er.context_recall),
            })
        else:
            per_question_out.append(er)

    # Provider metadata — from the caller when supplied, else from config.
    try:
        from src.llm_client import provider_info as _provider_info

        pinfo = _provider_info()
    except Exception:  # noqa: BLE001
        pinfo = {}
    if provider_info:
        pinfo.update(provider_info)

    provider_block = pinfo.get("provider", "groq")
    model_block = pinfo.get("model", "openai/gpt-oss-120b")

    report: dict = {
        "aggregate":        aggregate,
        "num_questions":   len(per_question_out),
        "failures":        list(failures),
        "per_question":    per_question_out,
        "provider":        provider_block,
        "model":           model_block,
        "embedding_model": pinfo.get("embedding_model", "BAAI/bge-m3"),
        "reranker_model":  pinfo.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
        "evaluation_status": results.get("status", "unknown"),
    }

    # Propagate metric coverage when present (only when any metric < 100% coverage)
    if "metric_coverage" in results:
        report["metric_coverage"] = results["metric_coverage"]

    if latency_breakdown_ms is not None:
        report["latency_breakdown_ms"] = latency_breakdown_ms

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


# ── load_test_set ──────────────────────────────────────────────────────

def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── __main__ demo ─────────────────────────────────────────────────────

if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print(
        "Run pipeline.py first to generate answers, then call evaluate_ragas()."
    )
