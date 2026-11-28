"""Skema Pydantic untuk endpoint percakapan asisten menu UMKM Tujuan Semata."""

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Payload masuk dari pelanggan ke endpoint POST /ai/v1/chat."""

    message: str = Field(
        ...,
        min_length=2,
        max_length=300,
        description="Pesan atau pertanyaan pelanggan (2–300 karakter).",
        examples=["Min, rekomen kopi yg manis creamy dong, brp harganya?"],
    )
    simulate_429: bool = Field(
        default=False,
        description="Flag opsional khusus pengujian otomatis untuk menyimulasikan HTTP 429 pada LLM utama.",
    )

    @field_validator("message")
    @classmethod
    def validate_non_whitespace_message(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 2:
            raise ValueError("Pesan tidak boleh kosong atau kurang dari 2 karakter setelah di-trim.")
        return stripped


class ChatResponse(BaseModel):
    """Payload balasan standar dari asisten menu UMKM Tujuan Semata."""

    reply: str = Field(..., description="Jawaban asisten atau pesan penolakan sopan.")
    intent_status: str = Field(..., description="Label hasil klasifikasi intent atau out_of_context.")
    engine_used: str = Field(
        ...,
        description="Mesin penjawab yang digunakan (gemini-3.1-flash-lite, fallback_chromadb, atau guardrail_local).",
    )
    latency_s: float = Field(..., description="Waktu pemrosesan request dalam satuan detik.")
