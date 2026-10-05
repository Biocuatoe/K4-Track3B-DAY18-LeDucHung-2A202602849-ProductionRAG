"""Shared configuration for Lab 18."""

import os

from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
# Lab 18 uses Groq exclusively (the student has no OpenAI key). The Groq endpoint is
# OpenAI-compatible, so the existing `openai` client is reused with a custom
# base_url and the model id below. No OpenAI API key is read anywhere in the project.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1"
)
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
SEMANTIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
