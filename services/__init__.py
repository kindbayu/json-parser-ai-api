"""
services — legacy services (RAGService, PDFService).

Import bersifat lazy: `RAGService` memuat HuggingFace embeddings + LLM saat
dikonstruksi, jadi paket ini tidak boleh menyeret beban tersebut ketika hanya
endpoint `/api/v1/*` yang dipakai.

    from services.pdf_service import PDFService   # ringan
    from services import PDFService               # lewat __getattr__ (PEP 562)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - hanya untuk IDE / type checker
    from .pdf_service import PDFService
    from .rag_service import RAGService

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "RAGService": ("services.rag_service", "RAGService"),
    "PDFService": ("services.pdf_service", "PDFService"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    """PEP 562 — import modul berat hanya saat atributnya benar-benar dipakai."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

