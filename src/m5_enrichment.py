from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm_client import chat, extract_json, groq_available


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Shared helpers ──────────────────────────────────────


def _is_ascii_heavy(text: str) -> bool:
    """Return True if text is likely English (ASCII-heavy)."""
    if not text:
        return False
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    ascii_ratio = sum(1 for c in alpha if ord(c) < 128) / len(alpha)
    return ascii_ratio > 0.8


def _score_sentence(sent: str) -> float:
    """Score a sentence for extractive summarisation / HyQA selection.

    Higher = more informative. Signals: length, digits (facts), keyword
    density, and position (beginning of text is usually more topical).
    """
    if not sent or len(sent) < 10:
        return -1.0
    score = 0.0
    # Length bonus (cap so very long sentences don't dominate)
    score += min(len(sent) / 100.0, 1.5)
    # Digits → factual statements
    score += min(sum(c.isdigit() for c in sent) / 10.0, 2.0)
    # Vietnamese keyword density
    keywords = {
        "ngày", "năm", "tháng", "giờ", "phút", "lần", "người",
        "vnđ", "vnd", "đồng", "phần trăm", "%",
        "quy định", "chính sách", "yêu cầu", "phải", "được",
        "hạn", "mức", "tối thiểu", "tối đa", "tăng", "giảm",
        "công ty", "nhân viên", "phòng", "ban", "bộ phận",
    }
    lc = sent.lower()
    score += sum(1 for kw in keywords if kw in lc) * 0.4
    # Penalise stop-word-heavy sentences
    stops = {
        "và", "của", "là", "có", "để", "với", "trong", "này",
        "cho", "không", "được", "theo", "tại", "hoặc", "từ",
    }
    stop_count = sum(1 for w in stops if f" {w} " in f" {lc} ")
    score -= stop_count * 0.15
    return score


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? newlines, filtering empty/too-short."""
    raw = re.split(r'(?<=[.!?\n])\s+', text)
    sents = []
    for s in raw:
        s = s.strip()
        if len(s) >= 8:
            sents.append(s)
    return sents


# ─── Technique 1: Chunk Summarisation ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk (2–3 câu tiếng Việt).

    Groq path  : single chat() call → extractive top-2 sentences.
    Deterministic fallback: extractive scoring → best 2 sentences.
    Never returns empty for non-empty input.
    """
    if not text or not text.strip():
        return ""

    if groq_available():
        raw = chat(
            system="Bạn là trợ lý tiếng Việt. Tóm tắt đoạn văn sau trong 2–3 câu ngắn gọn, bằng tiếng Việt. Chỉ trả lời bằng bản tóm tắt, không giải thích gì thêm.",
            user=text,
            max_tokens=256,
        )
        if raw:
            return raw.strip()

    # ── Deterministic extractive fallback ──────────────────
    sents = _split_sentences(text)
    if not sents:
        return text.strip()[:200]

    # Score each sentence; keep top-2 (deduplicate by overlap)
    scored = sorted(((s, _score_sentence(s)) for s in sents), key=lambda x: -x[1])
    chosen = []
    for sent, _ in scored:
        # Avoid near-duplicates (overlap > 60 % chars)
        if not any(
            len(set(sent) & set(c)) / max(len(set(sent)), 1) > 0.6
            for c in chosen
        ):
            chosen.append(sent)
            if len(chosen) >= 2:
                break

    if not chosen:
        chosen = [sents[0]]

    summary = " ".join(chosen)
    # If somehow the result is empty, return first 200 chars
    return summary.strip() if summary.strip() else text.strip()[:200]


# ─── Technique 2: Hypothesis Questions ──────────────────


_WH_WORDS = {
    "bao nhiêu": "bao nhiêu",
    "bao lâu": "bao lâu",
    "bao giờ": "bao giờ",
    "mấy": "mấy",
    "khi nào": "khi nào",
    "ở đâu": "ở đâu",
    "tại sao": "tại sao",
    "như thế nào": "như thế nào",
    "là gì": "là gì",
}


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate N câu hỏi mà chunk có thể trả lời (HyQA).

    Groq path  : chat() → strip numbering → list of strings.
    Deterministic fallback: extractive sentences → WH-style question rewriting.
    Returns [] for empty/blank input; returns exactly n_questions for non-empty.
    """
    if not text or not text.strip():
        return []

    if groq_available():
        raw = chat(
            system=(
                f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                "Mỗi câu hỏi trên 1 dòng, không đánh số, không gạch đầu dòng, "
                "không có dấu ngoặc. Trả lời chỉ bằng các câu hỏi, không giải thích gì khác."
            ),
            user=text,
            max_tokens=800,
        )
        if raw:
            lines = raw.strip().split("\n")
            qs = []
            for line in lines:
                line = line.strip().lstrip("0123456789.-—–)·*\t ")
                line = re.sub(r"^\s*[-–—•·]\s*", "", line)
                if line and ("?" in line or len(line) > 5):
                    qs.append(line.rstrip("?") + "?")
            if qs:
                return qs[:n_questions]

    # ── Deterministic fallback ──────────────────────────────
    # Strip markdown headings and leading numbering/dates before processing
    clean_text = re.sub(r"^#+\s*", "", text)
    clean_text = re.sub(r"^[\d.,;:\-–—|]+\s*", "", clean_text).strip()

    # Extract key facts for question crafting
    facts: list[str] = []

    # Duration patterns: "X ngày", "mỗi X năm", "X+Y năm"
    for m in re.finditer(
        r"(?:mỗi\s+|ít\s+nhất\s+|tối\s+thiểu\s+)?"
        r"([\d,.\-+]+)\s*(ngày|năm|tháng|giờ|lần|phút)",
        clean_text,
    ):
        num = m.group(1).strip()
        unit = m.group(2).strip()
        if len(num) <= 10:
            facts.append(("duration", num, unit))

    # Amount patterns: digits near policy keywords
    for m in re.finditer(
        r"(?:được\s+)?(tối\s+đa|tối\s+thiểu|hạn|mức|ít\s+nhất|nhiều\s+nhất|có|áp\s+dụng)\s*"
        r"([\d,.\s]+(?:%|VNĐ|đồng|người|ngày|năm|tháng|giờ)?)",
        clean_text,
    ):
        val = m.group(2).strip() if m.group(2) else m.group(1).strip()
        if val:
            facts.append(("amount", val, ""))

    # Requirement patterns: "phải X", "được phép X", "không được X"
    for m in re.finditer(
        r"(phải|được\s+phép|không\s+được|yêu\s+cầu)\s+([^\.!?]{5,40}?)(?:\.|!|\?)",
        clean_text,
    ):
        verb = m.group(1).strip()
        rest = m.group(2).strip()
        if rest:
            facts.append(("requirement", verb, rest))

    # Build questions from extracted facts
    questions: list[str] = []
    seen_q: set[str] = set()

    def _add_q(q: str) -> bool:
        """Add question if not duplicate-like. Returns True if added."""
        if not q or len(q) < 5:
            return False
        q_key = re.sub(r"\s+", " ", q.lower()).strip()
        # Skip if nearly identical to an existing question
        for existing in seen_q:
            overlap = len(set(q_key) & set(existing)) / max(len(set(q_key)), 1)
            if overlap > 0.75:
                return False
        questions.append(q)
        seen_q.add(q_key)
        return True

    for fact_type, val1, val2 in facts:
        if len(questions) >= n_questions:
            break
        if fact_type == "duration":
            q = f"{val1} {val2} là bao lâu?"
            _add_q(q)
        elif fact_type == "amount":
            q = f"Bao nhiêu {val1} được áp dụng?"
            _add_q(q)
        elif fact_type == "requirement":
            verb_map = {"phải": "phải", "được phép": "được", "không được": "không được", "yêu cầu": "yêu cầu"}
            v = verb_map.get(val1, val1)
            q = f"{val2} có quy định gì?"
            _add_q(q)

    # Generic rewrite if we don't have enough questions yet
    if len(questions) < n_questions:
        # Get remaining sentences (skip already-used ones)
        remaining_sents = _split_sentences(clean_text)
        for sent in remaining_sents:
            if len(questions) >= n_questions:
                break
            # Strip leading digits, punctuation, and whitespace
            clean = re.sub(r"^[\d.,;:\-–—|]+\s*", "", sent).strip()
            # Chỉ bỏ tiền tố đánh số dạng "1." / "a)" / "- " chứ KHÔNG bỏ chữ cái
            # hoa đầu câu (regex cũ cắt nhầm "Chính" → "hính").
            clean = re.sub(r"^(?:\(?\d+[\.\)]|\(?[a-zA-Zàáâãèéêìíòóôõùúăđĩũơư][\.\)])\s+", "", clean)
            clean = re.sub(r"[.!?,]+$", "", clean).strip()
            if len(clean) > 10:
                if len(clean) > 50:
                    # Cắt tại ranh giới từ để không sinh câu hỏi đứt đoạn.
                    clean = clean[:50].rsplit(" ", 1)[0].rstrip(".,;:")
                q = f"{clean} như thế nào?"
                _add_q(q)

    # Guarantee at least 1 question for non-empty input
    if not questions:
        fallback = re.sub(r"^[#\d.,;:\-–—|]+\s*", "",
                         re.sub(r"[.!?,]+$", "", (clean_text or "")[:60]).strip())
        fallback = re.sub(r"^[\d.,;:\-–—|]+\s*", "", fallback).strip()
        q = f"{fallback} là gì?" if len(fallback) > 3 else "Chính sách này có nội dung gì?"
        questions.append(q)

    # Pad to n_questions with varied forms.
    # Vòng lặp CÓ GIỚI HẠN: _add_q() từ chối câu trùng, nên lặp tới khi đủ
    # n_questions sẽ treo vô hạn khi mọi biến thể đều bị trùng.
    base = questions[0] if questions else "Chính sách này có nội dung gì?"
    # Lấy 2–4 từ đầu làm chủ đề, bỏ markdown/số ở đầu, để câu hỏi đọc tự nhiên
    # (tránh hiện tượng "hính sách" do cắt chuỗi).
    _words = [w for w in re.sub(r"^[#>\d.,;:\-–—|]+\s*", "", base).split() if w]
    _topic = " ".join(_words[:4]).rstrip("?.,") or "chính sách này"
    variations = [
        base,
        f"Có quy định gì về {_topic}?",
        f"{_topic} được áp dụng như thế nào?",
    ]
    for _attempt in range(n_questions * 3 + 3):
        if len(questions) >= n_questions:
            break
        v = variations[_attempt % len(variations)]
        if not _add_q(v):
            # Biến thể bị trùng → thêm hậu tố phân biệt để vẫn đủ n_questions.
            _add_q(f"{v.rstrip('?')} (mục {len(questions) + 1})?")

    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend ────────────────────


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend ONE context sentence describing where the chunk sits in the doc.

    Groq path  : chat() → single context sentence → f"{context}\\n\\n{text}".
    Deterministic fallback: ``Trích từ tài liệu {document_title}. `` + text.
    HARD REQUIREMENT: original text MUST be present in result,
    result length >= len(original).
    """
    if not text or not text.strip():
        return text

    if groq_available():
        raw = chat(
            system=(
                "Viết ĐÚNG 1 câu ngắn (dưới 25 từ) mô tả đoạn văn nằm ở đâu trong tài liệu "
                "và nói về chủ đề gì. Chỉ trả lời bằng 1 câu, không dấu ngoặc, không giải thích."
            ),
            user=f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text[:500]}",
            max_tokens=256,
        )
        if raw:
            ctx = raw.strip().rstrip(".")
            result = f"{ctx}.\n\n{text}"
            if text in result and len(result) >= len(text):
                return result

    # ── Deterministic fallback ──────────────────────────────
    prefix = f"Trích từ tài liệu {document_title}. " if document_title.strip() else ""
    result = f"{prefix}{text}"
    # HARD requirement: original text must be present and result >= original length
    assert text in result, "Original text must be present in contextual_prepend output"
    assert len(result) >= len(text), "Result must be >= original length"
    return result


# ─── Technique 4: Auto Metadata Extraction ──────────────


# Keyword sets per category
_HR_KEYWORDS = {
    "lương", "phép", "bảo hiểm", "đào tạo", "thuế", "phụ cấp",
    "hợp đồng", "thử việc", "thâm niên", "nghỉ việc", "sa thải",
    "kỷ luật", "khen thưởng", "thưởng", "thu nhập",
    "bảo hiểm xã hội", "bhxh", "bhyt", "phép năm", "nghỉ phép",
    "tăng ca", "ca làm việc", "nhân sự", "tuyển dụng", "đánh giá",
    "hiệu suất", "người phụ thuộc", "lương cơ bản", "lương gross",
    "lương net", "đóng bảo hiểm", "mức đóng",
}
_IT_KEYWORDS = {
    "mật khẩu", "password", "vpn", "malware", "mạng", "wifi",
    "email", "mail", "máy chủ", "server", "firewall", "bảo mật",
    "phần mềm", "cài đặt", "cập nhật", "patch", "backup",
    "quyền truy cập", "account", "tài khoản", "phân quyền",
    "it", "công nghệ", "thiết bị", "laptop", "máy tính",
    "wireguard", "aes", "mã hóa", "crypt",
}
_FINANCE_KEYWORDS = {
    "mua sắm", "chi phí", "tạm ứng", "phụ cấp", "thanh toán",
    "hóa đơn", "biên lai", "hoàn ứng", "hoàn tiền", "quyết toán",
    "ngân sách", "doanh thu", "lợi nhuận", "chi tiêu", "tiền lương",
    "công tác phí", "phí", "định mức", "phê duyệt chi",
    "hạn mức", "vay", "nợ", "tài chính",
}


def _infer_category(text: str) -> str:
    """Deterministic category inference from keyword sets."""
    lc = text.lower()
    hr_score = sum(1 for kw in _HR_KEYWORDS if kw in lc)
    it_score = sum(1 for kw in _IT_KEYWORDS if kw in lc)
    fin_score = sum(1 for kw in _FINANCE_KEYWORDS if kw in lc)
    scores = {"hr": hr_score, "it": it_score, "finance": fin_score, "policy": 0.5}
    best = max(scores, key=scores.get)
    return "hr" if best == "hr" and hr_score > 0 else \
           "it" if best == "it" and it_score > 0 else \
           "finance" if best == "finance" and fin_score > 0 else \
           "policy"


def _extract_entities(text: str) -> list[str]:
    """Extract capitalised tokens and numbers with units as entities."""
    entities: list[str] = []

    # Capitalised words (potential proper nouns / named entities)
    for m in re.finditer(r"\b[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ][a-zàáâãèéêìíòóôõùúăđĩũơưA-Z0-9]{2,}(?:\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ][a-zàáâãèéêìíòóôõùúăđĩũơưA-Z0-9]{2,})*\b", text):
        entities.append(m.group(0))

    # Numbers with units / labels
    for m in re.finditer(
        r"(?:[\d,.]+(?:\s*(?:VNĐ|%|\$|€|USD|EUR))|(?:[\d,.]+\s*(?:ngày|năm|tháng|giờ|lần|đồng|người|vnđ)))",
        text,
    ):
        val = m.group(0).strip()
        if val not in entities and len(val) > 1:
            entities.append(val)

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for e in entities:
        el = e.lower()
        if el not in seen:
            seen.add(el)
            unique.append(e)

    return unique[:15]  # cap at 15


def _extract_topic(text: str) -> str:
    """Extract the first meaningful clause as topic."""
    sents = _split_sentences(text)
    if sents:
        first = sents[0]
        # Strip markdown heading markers
        first = re.sub(r"^#+\s*", "", first)
        first = re.sub(r"^\s*[-–—•]\s*", "", first)
        # Strip leading numbering/dates: "1. ", "01/01/2024 | ", "> "
        first = re.sub(r"^[\d.,;:\-–—|]+\s*", "", first).strip()
        first = re.sub(r"^>\s*", "", first).strip()
        # Cap at 100 chars
        return first[:100]
    # Fallback: first 80 chars of text
    return text.strip()[:80]


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata: topic, entities, category, language.

    Groq path  : chat() + extract_json() → validate shape → coerce types.
    Deterministic fallback: keyword sets for category, regex for entities/topic.
    Returns dict with keys: topic, entities, category, language.
    Never returns {}.
    """
    if not text or not text.strip():
        return {
            "topic": "general",
            "entities": [],
            "category": "policy",
            "language": "vi",
        }

    # ── Groq path ───────────────────────────────────────────
    if groq_available():
        raw = chat(
            system=(
                'Trích xuất metadata từ đoạn văn và trả lời CHỈ bằng JSON hợp lệ, '
                'không có giải thích: '
                '{"topic": "...", "entities": ["...", "..."], '
                '"category": "policy|hr|it|finance", "language": "vi|en"}'
            ),
            user=text[:600],
            max_tokens=800,
        )
        if raw:
            parsed = extract_json(raw)
            if parsed is not None:
                # Validate shape: coerce to expected types
                result = {}
                result["topic"] = str(parsed.get("topic", "") or "")
                result["category"] = str(parsed.get("category", "") or "").lower()
                if result["category"] not in {"policy", "hr", "it", "finance"}:
                    result["category"] = _infer_category(text)
                lang_raw = str(parsed.get("language", "vi") or "vi").lower()
                result["language"] = "en" if lang_raw.startswith("en") else "vi"

                ents = parsed.get("entities", [])
                if isinstance(ents, str):
                    ents = [e.strip() for e in re.split(r"[,;]\s*", ents) if e.strip()]
                elif not isinstance(ents, list):
                    ents = []
                result["entities"] = [str(e) for e in ents]

                # If topic is empty, fill from deterministic
                if not result["topic"]:
                    result["topic"] = _extract_topic(text)
                if not result["entities"]:
                    result["entities"] = _extract_entities(text)

                return result

    # ── Deterministic fallback ──────────────────────────────
    return {
        "topic": _extract_topic(text),
        "entities": _extract_entities(text),
        "category": _infer_category(text),
        "language": "en" if _is_ascii_heavy(text) else "vi",
    }


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call → {summary, questions, context, metadata}.

    Returns a COMPLETE dict — never {} — with correctly typed fields.
    Falls back to deterministic helpers for any missing/empty field.
    Robust against: markdown fences, prose wrappers, wrong types,
    truncated JSON, and complete garbage from the model.
    """
    # ── Build from deterministic helpers (used in fallback AND as fill) ──
    def _fallback_dict() -> dict:
        return {
            "summary": summarize_chunk(text),
            "questions": generate_hypothesis_questions(text, n_questions=3),
            "context": (
                f"Trích từ tài liệu {source}. "
                if source.strip()
                else ""
            ),
            "metadata": extract_metadata(text),
        }

    if not text or not text.strip():
        return _fallback_dict()

    if not groq_available():
        return _fallback_dict()

    # ── Groq single-call ───────────────────────────────────
    raw = chat(
        system=(
            "Phân tích đoạn văn và trả lời CHỈ bằng JSON hợp lệ, không có giải thích gì ngoài JSON. "
            "Định dạng bắt buộc:\n"
            '{\n  "summary": "tóm tắt 2-3 câu tiếng Việt",\n'
            '  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],\n'
            '  "context": "1 câu ngắn mô tả đoạn văn nằm ở đâu trong tài liệu",\n'
            '  "metadata": {\n'
            '    "topic": "...",\n'
            '    "entities": ["...", "..."],\n'
            '    "category": "policy|hr|it|finance",\n'
            '    "language": "vi|en"\n'
            "  }\n"
            "}"
        ),
        user=f"Tài liệu: {source}\n\nĐoạn văn:\n{text[:600]}",
        max_tokens=1200,
    )

    if not raw:
        return _fallback_dict()

    # ── Parse JSON ─────────────────────────────────────────
    parsed = extract_json(raw)

    # Attempt light repair if extract_json returned None
    if parsed is None:
        # Try dropping trailing incomplete object/array
        for cut in range(len(raw), 0, -1):
            trial = raw[:cut].strip()
            # balance braces
            opens = trial.count("{") - trial.count("}")
            if opens == 0:
                try:
                    parsed = __import__("json").loads(trial)
                    break
                except (ValueError, TypeError):
                    pass
        if parsed is None:
            return _fallback_dict()

    # ── Validate and coerce every field ─────────────────────
    result: dict = {}

    # summary: must be str
    raw_summary = parsed.get("summary") if isinstance(parsed, dict) else None
    if isinstance(raw_summary, str) and raw_summary.strip():
        result["summary"] = raw_summary.strip()
    else:
        result["summary"] = summarize_chunk(text)

    # questions: must be list[str]
    raw_q = parsed.get("questions") if isinstance(parsed, dict) else None
    if isinstance(raw_q, list):
        questions: list[str] = []
        for q in raw_q:
            if isinstance(q, str) and q.strip():
                questions.append(q.strip().rstrip("?") + "?")
        if questions:
            result["questions"] = questions[:3]
        else:
            result["questions"] = generate_hypothesis_questions(text, n_questions=3)
    elif isinstance(raw_q, str) and raw_q.strip():
        # Comma/semicolon/newline-separated string → list
        parts = re.split(r"[,;\n]+", raw_q)
        questions = [
            p.strip().rstrip("?") + "?" for p in parts
            if p.strip() and ("?" in p or len(p.strip()) > 5)
        ]
        result["questions"] = questions[:3] if questions else generate_hypothesis_questions(text, n_questions=3)
    else:
        result["questions"] = generate_hypothesis_questions(text, n_questions=3)

    # context: must be str
    raw_ctx = parsed.get("context") if isinstance(parsed, dict) else None
    if isinstance(raw_ctx, str) and raw_ctx.strip():
        result["context"] = raw_ctx.strip()
    else:
        result["context"] = (
            f"Trích từ tài liệu {source}."
            if source.strip()
            else ""
        )

    # metadata: must be dict with required keys
    raw_meta = parsed.get("metadata") if isinstance(parsed, dict) else None
    if isinstance(raw_meta, dict):
        meta: dict = {}
        meta["topic"] = str(raw_meta.get("topic") or "")
        if not meta.get("topic"):
            meta["topic"] = _extract_topic(text)

        ents = raw_meta.get("entities", [])
        if isinstance(ents, str):
            ents = [e.strip() for e in re.split(r"[,;]+", ents) if e.strip()]
        elif not isinstance(ents, list):
            ents = []
        meta["entities"] = [str(e) for e in ents]

        cat = str(raw_meta.get("category") or "policy").lower()
        meta["category"] = cat if cat in {"policy", "hr", "it", "finance"} else _infer_category(text)

        lang = str(raw_meta.get("language") or "vi").lower()
        meta["language"] = "en" if lang.startswith("en") else "vi"

        result["metadata"] = meta

    elif isinstance(raw_meta, str) and raw_meta.strip():
        # metadata as JSON string → re-parse
        inner = extract_json(raw_meta)
        if inner and isinstance(inner, dict):
            result["metadata"] = inner  # will be coerced below
        else:
            result["metadata"] = extract_metadata(text)
    else:
        result["metadata"] = extract_metadata(text)

    # Ensure metadata always has the 4 required keys
    for key, default_fn in [
        ("topic", lambda: _extract_topic(text)),
        ("entities", lambda: _extract_entities(text)),
        ("category", lambda: _infer_category(text)),
        ("language", lambda: "vi" if not _is_ascii_heavy(text) else "en"),
    ]:
        if not isinstance(result["metadata"], dict):
            result["metadata"] = {}
        if key not in result["metadata"] or not result["metadata"].get(key):
            result["metadata"][key] = default_fn()

    # Final type sanity-check on metadata values
    if isinstance(result["metadata"], dict):
        for k in ("topic", "category", "language"):
            result["metadata"][k] = str(result["metadata"].get(k, ""))
        if not isinstance(result["metadata"].get("entities"), list):
            result["metadata"]["entities"] = []

    return result


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    - methods=None / ["combined"]: 1 API call/chunk via _enrich_single_call (production).
    - methods=["summary"] etc.: call individual functions (debug / study).
    - All paths populate every EnrichedChunk field; never {}.
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched: list[EnrichedChunk] = []
    for i, chunk in enumerate(chunks):
        text: str = chunk.get("text", "")
        source: str = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary: str = result.get("summary", "")
            questions: list[str] = list(result.get("questions", []))
            context_line: str = result.get("context", "")
            enriched_text: str = (
                f"{context_line}\n\n{text}" if context_line.strip() else text
            )
            auto_meta: dict = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = (
                generate_hypothesis_questions(text)
                if "hyqa" in methods
                else []
            )
            enriched_text = (
                contextual_prepend(text, source)
                if "contextual" in methods
                else text
            )
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = (
        "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. "
        "Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."
    )

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary:        {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual:\n{ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}\n")

    # Demo combined
    print("=== Combined Mode ===\n")
    result = _enrich_single_call(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Combined result: {result}\n")

    # Demo enrich_chunks
    chunks = [
        {"text": sample, "metadata": {"source": "policy.md"}},
        {
            "text": "Mật khẩu phải thay đổi mỗi 90 ngày.",
            "metadata": {"source": "it.md"},
        },
    ]
    ec = enrich_chunks(chunks)
    for c in ec:
        print(f"--- Chunk ({c.method}) ---")
        print(f"  original:  {c.original_text}")
        print(f"  enriched:  {c.enriched_text}")
        print(f"  summary:  {c.summary}")
        print(f"  questions:{c.hypothesis_questions}")
        print(f"  metadata: {c.auto_metadata}\n")
