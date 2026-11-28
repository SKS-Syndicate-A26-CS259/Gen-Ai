"""Intent router berbasis guardrail aturan sensitif/OOC + TF-IDF & Linear SVM."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "Model"
TFIDF_PATH = MODEL_DIR / "intent_tfidf.pkl"
SVM_PATH = MODEL_DIR / "intent_svm.pkl"
SLANG_PATH = Path(__file__).resolve().parent / "kamus_slang_tujuan_semata.json"

ISTILAH_TEKNIS = {"c++": "cpp", "c#": "csharp"}

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F2FF"
    "\U0000FE0F\U0000200D"
    "]+",
    flags=re.UNICODE,
)
PUNCT_RE = re.compile(r"[^a-z0-9\s]")
REPEAT_RE = re.compile(r"(.)\1{2,}")

# Kata/frasa sensitif internal UMKM & topik Out-of-Context (OOC)
GUARDRAIL_REGEX_PATTERNS = [
    # Data internal / finansial / privasi pelanggan
    r"\bhpp\b",
    r"\bharga\s+pokok\b",
    r"\bmodal\b",
    r"\bomzet\b",
    r"\bomset\b",
    r"\buntung\s+bersih\b",
    r"\bprofit\b",
    r"\blaporan\s+keuangan\b",
    r"\bgaji\s+karyawan\b",
    r"\bnomor\s+hp\b",
    r"\bno\s+hp\b",
    r"\bnomor\s+wa\b",
    r"\bnomor\s+telepon\s+member\b",
    r"\bdata\s+member\b",
    r"\bdata\s+pribadi\b",
    # Politik, finansial umum, pemrograman, akademik, hiburan, kompetitor, dll.
    r"\bpresiden\b",
    r"\bpemilu\b",
    r"\bpartai\s+politik\b",
    r"\bsaham\b",
    r"\bbbca\b",
    r"\bcrypto\b",
    r"\bbitcoin\b",
    r"\bcoding\b",
    r"\bpython\b",
    r"\bjavascript\b",
    r"\bscraping\b",
    r"\bcpp\b",
    r"\bcsharp\b",
    r"\bmeretas\b",
    r"\bhack\s+akun\b",
    r"\bstarbucks\b",
    r"\bjanji\s+jiwa\b",
    r"\bkopi\s+kenangan\b",
    r"\bfore\s+coffee\b",
    r"\bintegral\b",
    r"\bmatematika\s+sma\b",
    r"\bpertandingan\s+bola\b",
    r"\bskor\s+pertandingan\b",
    r"\bpiala\s+dunia\b",
    r"\bobat\s+sakit\b",
    r"\bapotek\b",
    r"\bbahasa\s+jepang\b",
    r"\bterjemahkan\b",
    r"\bhotel\b",
    r"\bpantai\s+kuta\b",
    r"\bojek\s+online\b",
    r"\bavengers\b",
    r"\bsinopsis\s+lengkap\b",
]
COMPILED_GUARDRAILS = [re.compile(pat, flags=re.IGNORECASE) for pat in GUARDRAIL_REGEX_PATTERNS]

# Kata kunci domain F&B Tujuan Semata agar pertanyaan menu yang valid tidak salah dibuang ke OOC
DOMAIN_MENU_HINTS = re.compile(
    r"\b(menu|kopi|coffee|latte|espresso|americano|matcha|coklat|cokelat|teh|tea|"
    r"milk|creamy|yakult|soda|refresher|dimsum|burger|platter|spaghetti|mac|puding|"
    r"sago|es\s+oyen|es\s+teler|canai|maryam|lumpia|kentang|chili\s+oil|otak|basreng|"
    r"cireng|pangsit|indomie|rum|halal|alkohol|kafein|anak|topping|glaze|harga|berapa|"
    r"tujuan\s+semata|depok|kasir|qris|bayar|pesan|rekomendasi|terlaris|best\s+seller|"
    r"minuman|makanan|camilan|cemilan|dessert|manis|pedas|segar|murah|paket|hemat|jajanan)\b",
    flags=re.IGNORECASE,
)


@lru_cache(maxsize=1)
def load_slang_dict() -> dict[str, str]:
    """Memuat kamus normalisasi slang Bahasa Indonesia."""
    if not SLANG_PATH.exists():
        return {}
    with SLANG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_intent_artifacts() -> tuple[Any, Any]:
    """Memuat artefak TF-IDF Vectorizer dan Linear SVM Classifier."""
    if not TFIDF_PATH.exists() or not SVM_PATH.exists():
        raise FileNotFoundError(
            f"Artefak model intent tidak ditemukan di {MODEL_DIR}. "
            f"Pastikan '{TFIDF_PATH.name}' dan '{SVM_PATH.name}' tersedia."
        )
    tfidf = joblib.load(TFIDF_PATH)
    clf = joblib.load(SVM_PATH)
    return tfidf, clf


def preprocess_tokens(text: str) -> list[str]:
    """Pipeline preprocessing teks yang konsisten dengan Notebook 01."""
    t = text.lower()
    t = URL_RE.sub(" ", t)
    for k, v in ISTILAH_TEKNIS.items():
        t = t.replace(k, v)
    t = EMOJI_RE.sub(" ", t)
    t = PUNCT_RE.sub(" ", t)
    t = REPEAT_RE.sub(r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()

    slang_map = load_slang_dict()
    normalized = [slang_map.get(tok, tok) for tok in t.split()]
    return " ".join(normalized).split()


def preprocess_text(text: str) -> str:
    """Mengembalikan string bersih siap transformasi TF-IDF."""
    return " ".join(preprocess_tokens(text))


def is_sensitive_or_ooc(raw_text: str, clean_text: str) -> bool:
    """Memeriksa apakah teks melanggar guardrail kata sensitif atau jelas out-of-context."""
    combined = f"{raw_text.lower()} {clean_text}"
    return any(regex.search(combined) for regex in COMPILED_GUARDRAILS)


def detect_intent(message: str) -> str:
    """Mendeteksi intent pesan pelanggan menggunakan guardrail lokal + TF-IDF Linear SVM."""
    clean_text = preprocess_text(message)
    if not clean_text:
        return "out_of_context"

    # 1. Guardrail deterministik untuk permintaan rahasia dapur/HPP/privasi/topik OOC eksplisit
    if is_sensitive_or_ooc(message, clean_text):
        return "out_of_context"

    # 2. Klasifikasi menggunakan TF-IDF + Linear SVM hasil pelatihan Notebook 02
    tfidf, clf = load_intent_artifacts()
    features = tfidf.transform([clean_text])
    predicted_label = str(clf.predict(features)[0])

    # Jika model memprediksi out_of_context padahal jelas mengandung kata kunci katalog menu warung
    # (dan sudah lolos cek kata sensitif/OOC di atas), arahkan ke tanya_produk_dan_menu_baru.
    if predicted_label == "out_of_context" and DOMAIN_MENU_HINTS.search(clean_text):
        return "tanya_produk_dan_menu_baru"

    return predicted_label
