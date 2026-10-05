"""Shared Groq LLM client for Lab 18.

The student only has a Groq API key (no OpenAI key). Groq exposes an
OpenAI-compatible endpoint, so we reuse the already-installed `openai` client and
point it at Groq's base URL. Every call must pass ``model=GROQ_MODEL``
(``openai/gpt-oss-120b``) explicitly.

This module is imported by M4 (RAGAS), M5 (enrichment), the pipeline answer
generation, and the naive baseline so that all of them share one cached client.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

_CLIENT: Any = None


def groq_available() -> bool:
    """Return True when a Groq API key is configured."""
    return bool(GROQ_API_KEY)


def get_client():
    """Return a cached OpenAI client pointed at Groq, or None without a key.

    The client is created lazily and cached so that repeated calls (one per chunk
    during enrichment, one per question during evaluation) reuse the same
    connection pool instead of re-authenticating.
    """
    global _CLIENT
    if not GROQ_API_KEY:
        return None
    if _CLIENT is None:
        try:
            from openai import OpenAI

            _CLIENT = OpenAI(
                api_key=GROQ_API_KEY,
                base_url=GROQ_BASE_URL,
                timeout=60.0,
                max_retries=3,
            )
        except Exception as exc:  # pragma: no cover - client construction failure
            print(f"  ⚠️  Groq client init failed: {exc}", flush=True)
            return None
    return _CLIENT


def chat(
    system: str,
    user: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str | None:
    """Single Groq chat completion. Returns None if unavailable or on failure.

    Never raises: callers decide how to fall back. Failures are surfaced via a
    warning so that an external outage is not silently mistaken for success.

    NOTE: ``openai/gpt-oss-120b`` is a reasoning model that emits tokens into a
    hidden reasoning field *before* producing content. Those reasoning tokens
    count against ``max_tokens``. With a small budget (e.g. 20) the model
    exhausts the budget mid-reasoning and returns empty content. We clamp to a
    safe floor and retry with a larger budget if content is absent.
    """
    client = get_client()
    if client is None:
        return None

    MIN_TOKENS = 256  # floor: enough for reasoning buffer + short content

    attempts = 0
    max_attempts = 3
    budget = max(max_tokens, MIN_TOKENS)

    while attempts < max_attempts:
        attempts += 1
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=budget,
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            result = (content or "").strip() or None
            if result is not None:
                return result
            # Content was empty (reasoning model exhausted budget mid-reasoning).
            # Retry with a larger budget — double it.
            budget = budget * 2
        except Exception as exc:
            # Genuine failure — surface warning and propagate None immediately.
            print(f"  ⚠️  Groq call failed ({GROQ_MODEL}): {exc}", flush=True)
            return None

    # All retries exhausted; return None so callers' fallback paths remain active.
    return None


# ── Robust JSON extraction ───────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(raw: str) -> dict | None:
    """Parse JSON out of an LLM response, tolerating fences and stray prose.

    Groq models (especially gpt-oss) frequently wrap JSON in markdown fences or
    prefix it with commentary. We try, in order: direct parse, fenced-block
    extraction, then a balanced-brace scan. Returns None if nothing parses.
    """
    if not raw or not raw.strip():
        return None

    # 1. Direct parse.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Markdown code fences.
    for block in _FENCE_RE.findall(raw):
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue

    # 3. Balanced-brace scan (handles leading/trailing prose).
    start = raw.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(raw[start:i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except (json.JSONDecodeError, TypeError):
                        break
        start = raw.find("{", start + 1)

    return None


def provider_info() -> dict:
    """Provider metadata for reports. Never includes the API key."""
    return {
        "provider": "groq",
        "base_url": GROQ_BASE_URL,
        "model": GROQ_MODEL,
        "api_key_configured": groq_available(),
    }
