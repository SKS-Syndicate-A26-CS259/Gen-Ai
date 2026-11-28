"""Retriever konteks katalog menu UMKM Tujuan Semata menggunakan ChromaDB."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DB_DIR = BASE_DIR / "vectorstore" / "chroma_menu_db"
COLLECTION_NAME = "tujuan_semata_menu"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def get_menu_collection() -> Collection:
    """Memuat koleksi persisten ChromaDB dengan model embedding multibahasa."""
    if not CHROMA_DB_DIR.exists():
        raise FileNotFoundError(
            f"Direktori vectorstore ChromaDB tidak ditemukan di: {CHROMA_DB_DIR}"
        )

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )


def retrieve_menu_context(query: str, top_k: int = 8) -> list[str]:
    """Mengambil top-k dokumen menu dan profil warung paling relevan dari ChromaDB."""
    collection = get_menu_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
    documents = results.get("documents") if results else None
    if not documents or not documents[0]:
        return []
    return list(documents[0])
