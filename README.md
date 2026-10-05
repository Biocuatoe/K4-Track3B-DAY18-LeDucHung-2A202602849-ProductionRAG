# Lab 18: Production RAG Pipeline

**K4-Track3B · Ngày 18 · Production RAG**  
**Thời gian:** 2h implement + 30 phút reflection

---

## Tổng quan

Bài tập **cá nhân** — implement toàn bộ 5 modules:

```
M1 Chunking → M5 Enrichment → M2 Hybrid Search → M3 Reranking → LLM Answer → M4 RAGAS Eval
```

Xem **ASSIGNMENT.md** để biết chi tiết từng module và timeline.

## Prerequisites

| Dependency | Bắt buộc? | Dùng cho |
|-----------|-----------|----------|
| Docker (Qdrant) | ⚠️ Nên có | M2 Dense Search (thiếu thì tự fallback sang in-memory) |
| Python 3.11+ | ✅ Có | Tất cả modules (RAGAS cần 3.11+ cho asyncio) |
| `GROQ_API_KEY` | ✅ Có | RAGAS eval (M4), Enrichment LLM (M5), sinh câu trả lời |

> **Provider: Groq (không dùng OpenAI).** Toàn bộ LLM trong bài chạy qua endpoint
> OpenAI-compatible của Groq với model `openai/gpt-oss-120b`:
> `https://api.groq.com/openai/v1`. Lấy key miễn phí tại
> [console.groq.com/keys](https://console.groq.com/keys). **Không cần OpenAI key.**

**Pre-download models** (tránh timeout trong lab):
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

> **Môi trường không có Docker/Qdrant?** `DenseSearch` tự động fallback sang
> Qdrant in-memory (`QdrantClient(":memory:")`) — dense search vẫn chạy thật,
> chỉ mất tính bền vững dữ liệu giữa các lần chạy.

## Quick Start

### 1. Clone repository & tạo môi trường ảo

**Linux / macOS / Git Bash:**
```bash
git clone <repo-url>
cd K4-Track3B-Production-RAG
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
git clone <repo-url>
cd K4-Track3B-Production-RAG
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(Nếu dùng Windows CMD: chạy `.venv\Scripts\activate.bat`)*

### 2. Cài đặt dependencies & Khởi động dịch vụ

**Linux / macOS / Git Bash:**
```bash
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
cp .env.example .env                    # Tạo file .env và điền GROQ_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```

**Windows (PowerShell):**
```powershell
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
Copy-Item .env.example .env             # Tạo file .env và điền GROQ_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```
*(Nếu dùng Windows CMD: dùng `copy .env.example .env` thay cho `Copy-Item`)*

### 3. Chạy từng module (optional)

```bash
python src/m1_chunking.py               # So sánh 4 chiến lược chunking
python src/m2_search.py                 # Kiểm tra Vietnamese segmentation
python src/m3_rerank.py                 # Rerank + benchmark latency
python src/m5_enrichment.py             # Demo enrichment
```

## Chạy toàn bộ & Kiểm tra

```bash
python main.py                          # Chạy Naive + Production + In bảng so sánh
python check_lab.py                     # Script kiểm tra hợp lệ trước khi nộp (chạy được trên mọi OS)
```

## Cấu trúc repo

```
K4-Track3B-Production-RAG/
├── README.md                   # File này
├── ASSIGNMENT.md               # ★ Đề bài + timeline + reflection
├── RUBRIC.md                   # Hệ thống chấm điểm
│
├── main.py                     # Entry point: chạy toàn bộ pipeline
├── check_lab.py                # Kiểm tra định dạng trước khi nộp
├── naive_baseline.py           # Baseline (chạy trước)
├── config.py                   # Shared config
├── requirements.txt            # Dependencies
├── docker-compose.yml          # Qdrant local
├── .env.example                # API keys template
│
├── data/                       # Corpus tiếng Việt — 25 .md files + 3 PDFs (28 files total)
│   ├── nghi_phep_nam_v2023.md  # Nghỉ phép 12 ngày (v2023, superseded)
│   ├── nghi_phep_nam_v2024.md  # Nghỉ phép 15 ngày (v2024, hiện hành)
│   ├── mat_khau_v1.md          # Password policy 90 ngày (OLD)
│   ├── mat_khau_v2.md          # Password policy 120 ngày + MFA (NEW)
│   ├── ... (28 files total)    # 8 categories: leave, salary, IT, workflow, training, admin, safety, compliance
│   ├── so_tay_an_toan.pdf      # An toàn PCCC + sơ cứu (PDF text)
│   ├── BCTC.pdf                # Báo cáo tài chính (scan, cần OCR)
│   └── Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf # Nghị định BVDL (scan, cần OCR)
├── test_set.json               # 20 Q&A pairs (6 types: lookup, version, negation, multi-hop, numeric, ambiguous)
│
├── src/                        # ★ 5 modules + pipeline (đã implement đầy đủ)
│   ├── m1_chunking.py          # Module 1: Chunking
│   ├── m2_search.py            # Module 2: Hybrid Search
│   ├── m3_rerank.py            # Module 3: Reranking
│   ├── m4_eval.py              # Module 4: Evaluation
│   ├── m5_enrichment.py        # Module 5: Enrichment Pipeline
│   ├── llm_client.py           # Groq client dùng chung (M4 + M5 + pipeline)
│   └── pipeline.py             # Ghép toàn bộ pipeline
│
├── tests/                      # Auto-grading
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   ├── test_m4.py
│   └── test_m5.py
│
├── analysis/                   # ★ Deliverable
│   ├── failure_analysis.md     # Phân tích failures (cá nhân)
│   └── reflections/            # Reflection cá nhân
│       └── reflection_TEMPLATE.md
│
├── reports/                    # ★ Auto-generated (bắt buộc: reports/ragas_report.json)
│   ├── ragas_report.json
│   └── naive_baseline_report.json
│
└── templates/                  # Templates gốc (backup)
    └── failure_analysis.md
```

## Timeline (Thời lượng ước tính)

| Thời lượng | Hoạt động |
|------------|-----------|
| 10 phút | Setup môi trường + chạy `naive_baseline.py` |
| 90 phút | Implement M1 → M2 → M3 → M4 → M5 |
| 20 phút | Chạy pipeline + RAGAS + failure analysis |
| 30 phút | Reflection: lecture mapping + project plan |

## Cấu hình LLM (Groq)

Toàn bộ bài dùng **một provider duy nhất: Groq**, qua endpoint OpenAI-compatible
và model `openai/gpt-oss-120b`.

| Biến môi trường | Giá trị mặc định | Dùng ở đâu |
|---|---|---|
| `GROQ_API_KEY` | *(bắt buộc — lấy từ [console.groq.com/keys](https://console.groq.com/keys))* | M4 (RAGAS), M5 (enrichment), sinh câu trả lời |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | `src/llm_client.py` |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | `src/llm_client.py` |

Client được tạo 1 lần và cache lại trong `src/llm_client.py`:

```python
from openai import OpenAI
client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
```

**Không cần OpenAI API key.** Nếu thiếu `GROQ_API_KEY`:
- M5 tự dùng fallback xác định (extractive summary, HyQA sinh từ câu, contextual prefix, metadata theo keyword).
- M4 đánh dấu `status = "skipped_no_api_key"` và **không** bịa số liệu.
- LLM answer generation trả về passage liên quan nhất thay vì bịa nội dung.

Các model khác dùng cho embedding/reranking (chạy local, không cần key):
`BAAI/bge-m3` (dense embedding, 1024 chiều), `all-MiniLM-L6-v2` (semantic chunking),
`BAAI/bge-reranker-v2-m3` (cross-encoder reranking).

## Quy chuẩn đặt tên Repository & Nộp bài

- **Cấu trúc đặt tên repo:**  
  `K4-Track3B-DAY18-<HoVaTen>-<MSSV>-ProductionRAG`  
  *(Ví dụ: `K4-Track3B-DAY18-NguyenVanAn-AI20K001-ProductionRAG`)*
- **Hạn chót nộp bài:** **11h59 ngày hôm sau diễn ra bài lab (GMT+7)** trên cổng VLearn LMS / Codelab.
- **Chi tiết yêu cầu:** Xem tại [ASSIGNMENT.md](ASSIGNMENT.md) và [RUBRIC.md](RUBRIC.md).
