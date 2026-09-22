"""
app/services/llm_factory.py
============================
LLM Factory — Universal AI Engine

Returns an LLM instance based on LLM_PROVIDER in .env:
  - "groq"   → ChatGroq  (cloud, free, fast — recommended for deployment)
  - "ollama" → ChatOllama (local, free, slow without GPU)

All services use get_llm() so the provider can be switched
by changing a single line in .env without touching any code.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from dotenv import load_dotenv

# BaseChatModel hanya dipakai sebagai type hint. Import-nya dibuat type-only
# karena `langchain_core.language_models.base` menarik torch saat di-import
# (±14 detik di mesin tanpa GPU) — tidak perlu dibayar saat modul ini di-import.
if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

# ---------------------------------------------------------------------------
# Model defaults
#
# PENTING (Groq): `llama-3.1-8b-instant` dan `llama-3.3-70b-versatile` sudah
# dipindahkan Groq ke tier **Enterprise**. API key free/developer yang memanggil
# model tersebut akan menerima:
#   HTTP 404 — "The model `...` does not exist or you do not have access to it."
# Tier developer saat ini melayani model keluarga gpt-oss / qwen, sehingga
# default di bawah ini dipakai agar deployment baru langsung jalan.
#
# Selalu verifikasi model untuk key Anda:
#   GET /api/v1/llm/models      (endpoint di main.py)
#   https://api.groq.com/openai/v1/models
# ---------------------------------------------------------------------------
DEFAULT_GROQ_MODEL    = "openai/gpt-oss-120b"
DEFAULT_OLLAMA_MODEL  = "qwen2.5:1.5b"
DEFAULT_OLLAMA_BASE   = "http://localhost:11434"


def get_provider() -> str:
    """Nama provider aktif sesuai .env: 'groq' atau 'ollama'."""
    return os.getenv("LLM_PROVIDER", "groq").lower().strip()


def get_groq_model() -> str:
    """Nama model Groq yang dikonfigurasi (fallback ke DEFAULT_GROQ_MODEL)."""
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL


def get_ollama_model() -> str:
    """Nama model Ollama yang dikonfigurasi."""
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL


def get_ollama_base_url() -> str:
    """Base URL server Ollama."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE).strip() or DEFAULT_OLLAMA_BASE


def get_llm(temperature: float = 0, **kwargs) -> BaseChatModel:
    """
    Kembalikan instance LLM sesuai LLM_PROVIDER di .env.

    Parameters
    ----------
    temperature : float
        Kreativitas output (0 = deterministik, cocok untuk ekstraksi data).
    **kwargs
        Parameter tambahan yang diteruskan ke konstruktor LLM.

    Returns
    -------
    BaseChatModel
        Instance ChatGroq atau ChatOllama yang siap dipakai.

    Raises
    ------
    EnvironmentError
        Jika GROQ_API_KEY tidak diset saat menggunakan provider groq.
    ValueError
        Jika LLM_PROVIDER berisi nilai yang tidak dikenali.
    """
    provider = get_provider()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        model   = get_groq_model()

        if not api_key or api_key == "gsk_your_groq_api_key_here":
            raise EnvironmentError(
                "GROQ_API_KEY belum diset. "
                "Daftar gratis di https://console.groq.com lalu isi GROQ_API_KEY di .env"
            )

        from langchain_groq import ChatGroq
        logger.info("LLM Provider: Groq — model: %s", model)
        return ChatGroq(
            api_key=api_key,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = get_ollama_base_url()
        model    = get_ollama_model()
        logger.info("LLM Provider: Ollama — model: %s @ %s", model, base_url)
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=kwargs.pop("num_predict", 1024),
            num_ctx=kwargs.pop("num_ctx", 4096),
            **kwargs,
        )

    else:
        raise ValueError(
            f"LLM_PROVIDER '{provider}' tidak dikenali. "
            "Gunakan 'groq' atau 'ollama'."
        )


def get_provider_name() -> str:
    """Kembalikan nama provider aktif untuk logging/health check."""
    if get_provider() == "groq":
        return f"Groq ({get_groq_model()})"
    return f"Ollama ({get_ollama_model()})"


def get_active_model_name() -> str:
    """Nama model yang sedang dipakai provider aktif."""
    return get_groq_model() if get_provider() == "groq" else get_ollama_model()


# ---------------------------------------------------------------------------
# Diagnosis helpers — dipakai endpoint /api/v1/llm/models, /health, dan
# pesan error service. Semuanya hanya memakai stdlib (urllib) supaya tidak
# menambah dependency baru.
# ---------------------------------------------------------------------------

def _groq_api_key() -> str:
    """Ambil GROQ_API_KEY dan pastikan sudah diisi."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "gsk_your_groq_api_key_here":
        raise EnvironmentError(
            "GROQ_API_KEY belum diset. Daftar gratis di https://console.groq.com "
            "lalu isi GROQ_API_KEY di environment deployment."
        )
    return api_key


def _list_groq_models_urllib(api_key: str, timeout: float) -> list[str]:
    """
    Fallback tanpa SDK `groq`.

    User-Agent default urllib (`Python-urllib/3.x`) ditolak Groq/Cloudflare
    dengan HTTP 403, jadi UA di bawah ini wajib ada.
    """
    request = urllib.request.Request(
        GROQ_MODELS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
            "User-Agent":    "ai-engine-api/3.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ConnectionError(
            f"Groq menolak permintaan daftar model (HTTP {exc.code}). "
            "Periksa GROQ_API_KEY."
        ) from exc
    except Exception as exc:
        raise ConnectionError(f"Gagal menghubungi Groq: {exc}") from exc

    return sorted(item["id"] for item in payload.get("data", []) if item.get("id"))


def list_groq_models(timeout: float = 10.0) -> list[str]:
    """
    Daftar model ID yang BISA DIAKSES oleh GROQ_API_KEY saat ini.

    Endpoint ini penting untuk mendiagnosis 404 `model_not_found`: model yang
    tersedia bagi akun free/developer bisa berbeda dari yang ada di dokumentasi
    (contoh: llama-3.1-8b-instant & llama-3.3-70b-versatile sudah pindah ke
    tier Enterprise sehingga key free menerima 404).

    Implementasi: SDK `groq` dipakai lebih dulu (stack HTTP yang sama dengan
    inference), fallback ke urllib bila SDK tidak terpasang.
    """
    api_key = _groq_api_key()

    try:
        from groq import Groq
    except ImportError:
        return _list_groq_models_urllib(api_key, timeout)

    try:
        client = Groq(api_key=api_key, timeout=timeout)
        return sorted(model.id for model in client.models.list().data)
    except Exception as exc:
        raise ConnectionError(
            f"Gagal membaca daftar model Groq ({type(exc).__name__}: {exc}). "
            "Periksa GROQ_API_KEY."
        ) from exc


def list_ollama_models(timeout: float = 5.0) -> list[str]:
    """Daftar model yang sudah di-pull di server Ollama lokal."""
    url = f"{get_ollama_base_url().rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise ConnectionError(
            f"Ollama tidak dapat dihubungi di {get_ollama_base_url()}: {exc}"
        ) from exc

    return sorted(m["name"] for m in payload.get("models", []) if m.get("name"))


def list_models(timeout: float = 10.0) -> list[str]:
    """Daftar model yang tersedia untuk provider aktif."""
    if get_provider() == "groq":
        return list_groq_models(timeout=timeout)
    return list_ollama_models(timeout=timeout)


def probe_llm(timeout: float = 10.0) -> tuple[bool, str]:
    """
    Cek (non-fatal) apakah model yang dikonfigurasi benar-benar dapat dipakai.

    Returns
    -------
    tuple[bool, str]
        (True, pesan sukses) atau (False, penjelasan penyebab + model yang
        tersedia untuk key tersebut).
    """
    provider = get_provider()
    try:
        available = list_models(timeout=timeout)
    except Exception as exc:
        return False, f"Tidak dapat memeriksa daftar model {provider}: {exc}"

    model = get_active_model_name()
    if model in available or f"{model}:latest" in available:
        return True, f"Model '{model}' tersedia untuk provider {provider}."

    return False, (
        f"Model '{model}' TIDAK tersedia untuk {provider}. "
        f"Model yang tersedia: {', '.join(available) if available else '(kosong)'}. "
        f"Perbaiki {provider.upper()}_MODEL pada environment deployment."
    )


def describe_llm_error(exc: Exception) -> str:
    """
    Ubah exception LLM menjadi pesan yang langsung bisa ditindaklanjuti.

    Kasus utama: 404 `model_not_found` karena model Groq dipindah ke tier
    Enterprise / nama model salah.
    """
    text = str(exc)

    if "model_not_found" in text or "does not exist or you do not have access" in text:
        try:
            available = ", ".join(list_models())
        except Exception:
            available = "(tidak dapat dibaca — cek GET /api/v1/llm/models)"

        return (
            f"Model LLM '{os.getenv('GROQ_MODEL', '')}' tidak dapat diakses (HTTP 404 "
            f"model_not_found). Set GROQ_MODEL pada environment deployment "
            f"(Railway/Render) ke salah satu model yang tersedia, lalu redeploy. "
            f"Model yang tersedia untuk API key ini: {available}."
        )

    if "invalid_api_key" in text or "401" in text or "Unauthorized" in text:
        return (
            "GROQ_API_KEY tidak valid atau sudah dicabut. "
            "Perbarui variabel GROQ_API_KEY pada environment deployment."
        )

    if "rate_limit" in text or "429" in text:
        return (
            "Rate limit / kuota LLM terlampaui. Tunggu beberapa saat lalu coba lagi."
        )

    if "timeout" in text.lower() or "timed out" in text.lower():
        return f"LLM tidak merespons (timeout). Detail: {text}"

    return text
