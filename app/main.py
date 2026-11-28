"""Entrypoint utama layanan REST API FastAPI UMKM Tujuan Semata (Capstone Asah 2026 — A26-CS259)."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.intent_router import SVM_PATH, TFIDF_PATH
from app.rag.chromadb_retriever import CHROMA_DB_DIR
from app.rag.llm_generator import _get_valid_api_key
from app.routers.chat_router import router as chat_router

app = FastAPI(
    title="UMKM Tujuan Semata — Gen-AI Barista API",
    description=(
        "Layanan REST API asisten menu cerdas berbasis Intent Classification (TF-IDF + Linear SVM), "
        "ChromaDB RAG, dan Gemini 3.1 Flash-Lite untuk proyek Capstone Asah 2026 (Tim A26-CS259)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/", tags=["System"])
def root_info() -> dict[str, str]:
    """Informasi ringkas layanan API dan tautan dokumentasi Swagger UI."""
    return {
        "service": "UMKM Tujuan Semata — Gen-AI Barista API",
        "team": "A26-CS259",
        "docs": "/docs",
        "health": "/health",
        "chat_endpoint": "/ai/v1/chat",
    }


@app.get("/health", tags=["System"])
def health_check() -> dict[str, object]:
    """Memeriksa kesiapan layanan, artefak model intent, vectorstore ChromaDB, dan status API key."""
    active_key = _get_valid_api_key()
    return {
        "status": "ok",
        "service": "tujuan-semata-genai-api",
        "artifacts": {
            "intent_tfidf_ready": TFIDF_PATH.exists(),
            "intent_svm_ready": SVM_PATH.exists(),
            "chromadb_ready": (CHROMA_DB_DIR / "chroma.sqlite3").exists(),
            "gemini_api_key_ready": active_key is not None,
        },
    }
