"""Endpoint POST /ai/v1/chat yang mengintegrasikan Guardrail, Intent Router, ChromaDB, dan Gemini."""

import time

from fastapi import APIRouter

from app.models.intent_router import detect_intent
from app.prompts.barista_few_shot_prompt import POLITE_REFUSAL
from app.rag.chromadb_retriever import retrieve_menu_context
from app.rag.llm_generator import generate_barista_reply
from app.schemas.chat_schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/ai/v1", tags=["AI Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Tanya Asisten Virtual Menu UMKM Tujuan Semata",
    description=(
        "Memvalidasi pesan pelanggan, mendeteksi intent serta kata sensitif/OOC, "
        "mengambil konteks katalog dari ChromaDB, dan menghasilkan jawaban via Gemini "
        "(dengan fallback otomatis ke ringkasan ChromaDB saat gangguan/HTTP 429)."
    ),
)
def chat_with_barista(payload: ChatRequest) -> ChatResponse:
    """Menjalankan pipeline utama chatbot RAG UMKM Tujuan Semata."""
    start_time = time.perf_counter()

    # 1. Klasifikasi Intent & Guardrail Kata Sensitif / Out-of-Context
    intent_status = detect_intent(payload.message)
    if intent_status == "out_of_context":
        return ChatResponse(
            reply=POLITE_REFUSAL,
            intent_status="out_of_context",
            engine_used="guardrail_local",
            latency_s=round(time.perf_counter() - start_time, 4),
        )

    # 2. Retrieval Konteks Katalog Menu dari ChromaDB (Top-8)
    contexts = retrieve_menu_context(query=payload.message, top_k=8)

    # 3. Generasi Jawaban via Gemini 3.1 Flash-Lite atau Auto-Fallback ChromaDB
    reply, engine_used = generate_barista_reply(
        query=payload.message,
        contexts=contexts,
        simulate_429=payload.simulate_429,
    )

    return ChatResponse(
        reply=reply,
        intent_status=intent_status,
        engine_used=engine_used,
        latency_s=round(time.perf_counter() - start_time, 4),
    )
