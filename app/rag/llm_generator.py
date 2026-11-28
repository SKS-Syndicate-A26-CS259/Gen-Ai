"""Generator jawaban menggunakan Gemini API (gemini-3.1-flash-lite) dengan fallback ChromaDB."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from google import genai
from google.genai import types

from app.prompts.barista_few_shot_prompt import SYSTEM_PROMPT, build_rag_prompt

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

PRIMARY_MODEL = "gemini-3.1-flash-lite"
PLACEHOLDER_KEYS = {"", "isi_api_key_gemini_anda_disini", "your_api_key_here"}


def _get_valid_api_key() -> str | None:
    """Mengambil GEMINI_API_KEY aktif; memprioritaskan pembacaan terbaru dari .env."""
    # Jika unit test menyetel GEMINI_API_KEY="" secara eksplisit di os.environ
    if "GEMINI_API_KEY" in os.environ and os.environ["GEMINI_API_KEY"] == "":
        return None

    # Selalu cek file .env terbaru agar perubahan key langsung terbaca tanpa restart manual
    if ENV_PATH.exists():
        file_vals = dotenv_values(ENV_PATH)
        file_key = (file_vals.get("GEMINI_API_KEY") or "").strip().strip("\"'")
        if file_key.lower() not in PLACEHOLDER_KEYS:
            return file_key

    env_val = (os.environ.get("GEMINI_API_KEY") or "").strip().strip("\"'")
    if env_val.lower() not in PLACEHOLDER_KEYS:
        return env_val

    return None


def format_chromadb_fallback(contexts: list[str], query: str = "") -> str:
    """Menyusun ringkasan katalog langsung dari ChromaDB saat Gemini API tidak tersedia / HTTP 429."""
    if not contexts:
        return (
            "Maaf Kak, layanan asisten cerdas sedang sibuk dan kami belum menemukan menu yang "
            "cocok untuk kata kunci tersebut. Silakan tanyakan nama menu atau kategori lain di "
            "UMKM Tujuan Semata ya!"
        )

    ranked_contexts = contexts
    if query:
        stop_words = {"apa", "aja", "saja", "yang", "yg", "ada", "min", "kak", "dong", "ya", "di", "dan", "atau", "berapa", "brp", "harga", "hrg", "minuman", "makanan", "menu"}
        keywords = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) >= 3 and w not in stop_words]
        if keywords:
            ranked_contexts = sorted(
                contexts,
                key=lambda doc: sum(1 for kw in keywords if kw in doc.lower()),
                reverse=True,
            )

    top_items = "\n".join(f"- {doc}" for doc in ranked_contexts[:3])
    return (
        "Maaf Kak, layanan asisten cerdas sedang padat. Berdasarkan katalog resmi UMKM Tujuan Semata, "
        f"berikut informasi menu yang paling relevan:\n{top_items}"
    )


def _call_gemini_api(api_key: str, user_prompt: str) -> str:
    """Memanggil model gemini-3.1-flash-lite dengan memprioritaskan endpoint Free Tier Gemini Developer API."""
    fast_retry = types.HttpRetryOptions(attempts=1)

    # 1. Utamakan Gemini Developer API (generativelanguage.googleapis.com — Free Tier tanpa kartu kredit/Billing)
    dev_client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(retry_options=fast_retry),
    )
    last_err: Exception | None = None

    try:
        interaction = dev_client.interactions.create(
            model=PRIMARY_MODEL,
            system_instruction=SYSTEM_PROMPT,
            input=user_prompt,
            generation_config={"temperature": 0.2},
            store=False,
        )
        text = (interaction.output_text or "").strip()
        if text:
            return text
    except Exception as err_v1:
        last_err = err_v1
        err_upper = str(err_v1).upper()
        if "429" in err_upper or "RESOURCE_EXHAUSTED" in err_upper:
            raise

    # 2. Coba endpoint v1beta generate_content pada Gemini Developer API (identik dengan Notebook 03 & 04 Free Tier)
    beta_client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta", retry_options=fast_retry),
    )
    for candidate_model in (PRIMARY_MODEL, "gemini-3.1-flash-lite-preview"):
        try:
            response = beta_client.models.generate_content(
                model=candidate_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                ),
            )
            text = (response.text or "").strip()
            if text:
                return text
        except Exception as err_beta:
            last_err = err_beta
            err_beta_upper = str(err_beta).upper()
            if "429" in err_beta_upper or "RESOURCE_EXHAUSTED" in err_beta_upper:
                raise

    raise RuntimeError(
        f"Gemini Free Tier API menolak key ini ({last_err}). "
        "Pastikan key dibuat dari https://aistudio.google.com/apikey (diawali 'AIzaSy...'), "
        "bukan dari Google Cloud Console Vertex AI ('AQ...')."
    )


def generate_barista_reply(
    query: str,
    contexts: list[str],
    simulate_429: bool = False,
) -> tuple[str, str]:
    """Menghasilkan jawaban asisten beserta penanda engine_used.

    Mengembalikan tuple `(reply, engine_used)`:
    - `(jawaban_llm, "gemini-3.1-flash-lite")` bila pemanggilan Gemini sukses.
    - `(ringkasan_chromadb, "fallback_chromadb")` bila API key belum diisi, rate-limit HTTP 429,
      atau gangguan jaringan/server lainnya.
    """
    if simulate_429 or os.getenv("SIMULATE_GEMINI_429", "").strip() == "1":
        logger.warning("Simulasi HTTP 429 aktif; mengalihkan ke fallback_chromadb.")
        return format_chromadb_fallback(contexts, query=query), "fallback_chromadb"

    api_key = _get_valid_api_key()
    if not api_key:
        logger.info("GEMINI_API_KEY belum tersedia; menggunakan fallback_chromadb.")
        return format_chromadb_fallback(contexts, query=query), "fallback_chromadb"

    user_prompt = build_rag_prompt(query=query, contexts=contexts)
    try:
        reply_text = _call_gemini_api(api_key=api_key, user_prompt=user_prompt)
        return reply_text, PRIMARY_MODEL
    except Exception as exc:
        logger.warning("Gagal memanggil Gemini API (%s); beralih ke fallback_chromadb.", exc)
        return format_chromadb_fallback(contexts, query=query), "fallback_chromadb"
