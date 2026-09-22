"""
app/services/llm_factory.py
============================
LLM Factory — PT Multisari Indoprima AI Engine

Mengembalikan instance LLM yang sesuai berdasarkan LLM_PROVIDER di .env:
  - "groq"   → ChatGroq  (cloud, gratis, cepat — recommended untuk deploy)
  - "ollama" → ChatOllama (lokal, gratis, lambat tanpa GPU)

Semua service menggunakan get_llm() agar provider bisa diganti
hanya dengan mengubah satu baris di .env tanpa menyentuh kode.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()
logger = logging.getLogger(__name__)


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
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        model   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

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
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model    = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
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
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
    if provider == "groq":
        return f"Groq ({os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')})"
    return f"Ollama ({os.getenv('OLLAMA_MODEL', 'qwen2.5:1.5b')})"
