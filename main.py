"""
Lab 18: Production RAG Pipeline — Main Entry Point
==================================================
Chạy toàn bộ pipeline: naive baseline → production → so sánh → report.

Usage:
    python main.py
"""

import json
import math
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _read_report(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def _move_to_reports() -> None:
    """Đảm bảo report nằm trong reports/ (một số module ghi ở cwd)."""
    for f in ["ragas_report.json", "naive_baseline_report.json"]:
        if os.path.exists(f) and not os.path.exists(os.path.join("reports", f)):
            os.replace(f, os.path.join("reports", f))


def _is_unmeasured(v: object) -> bool:
    """True when a metric was never actually measured (judge unavailable)."""
    return v is None or (isinstance(v, float) and math.isnan(v))


def _fmt_metric(v: object, width: int = 8) -> str:
    if _is_unmeasured(v):
        return f"{'n/a':>{width}}"
    return f"{float(v):>{width}.4f}"


def main():
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.time()

    os.makedirs("reports", exist_ok=True)

    # Step 1: Basic Baseline
    print("\n📌 STEP 1: Running Basic RAG Baseline...")
    print("-" * 40)
    from naive_baseline import main as run_baseline
    run_baseline()
    _move_to_reports()

    # Step 2: Production Pipeline
    print("\n📌 STEP 2: Running Production Pipeline...")
    print("-" * 40)
    from src.pipeline import LatencyTracker, build_pipeline, evaluate_pipeline
    tracker = LatencyTracker()
    search, reranker, parents_by_id = build_pipeline(tracker)
    evaluate_pipeline(search, reranker, parents_by_id, tracker)
    _move_to_reports()

    # Step 3: Comparison
    print("\n📌 STEP 3: Comparison")
    print("-" * 40)
    naive = _read_report("reports/naive_baseline_report.json")
    prod = _read_report("reports/ragas_report.json")

    n_agg = naive.get("aggregate", {})
    p_agg = prod.get("aggregate", {})

    print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Δ':>8}")
    print("-" * 55)
    for m in METRICS:
        # None/NaN = judge unavailable (rate-limit/API error), NOT a real 0.0.
        # Never coerce: printing 0.0000 would resurrect the false-confidence bug.
        n_raw, p_raw = n_agg.get(m), p_agg.get(m)
        # Δ is only meaningful when BOTH sides were actually measured.
        if _is_unmeasured(n_raw) or _is_unmeasured(p_raw):
            delta = f"{'n/a':>8}"
        else:
            delta = f"{float(p_raw) - float(n_raw):>+8.4f}"
        print(f"  {m:<23} {_fmt_metric(n_raw)} {_fmt_metric(p_raw, 12)} {delta}")

    # Surface degraded / partial-coverage status explicitly.
    for label, agg in (("Production", prod), ("Basic", naive)):
        st = agg.get("status")
        if st == "degraded":
            print(f"\n⚠️  {label} run is DEGRADED — see metric_coverage in the report.")
            print("    Metrics marked n/a were not measured; do NOT read as a verdict.")

    # Latency breakdown
    latency = prod.get("latency_breakdown_ms", {})
    if latency:
        print("\n📊 Latency breakdown (production):")
        print(f"{'Stage':<34} {'ms':>12}")
        for k, v in sorted(latency.items(), key=lambda kv: -float(kv[1])):
            print(f"  {k:<32} {float(v):>12.1f}")

    print(f"\n⏱️  Total time: {time.time() - start:.1f}s")
    print("\n📋 Next steps:")
    print("  1. Điền analysis/failure_analysis.md")
    print("  2. Viết analysis/reflections/reflection_LeDucHung.md")
    print("  3. Chạy: python check_lab.py")


if __name__ == "__main__":
    main()
