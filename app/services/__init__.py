"""
app/services — paket service AI Engine.

PENTING (performa): modul di dalam paket ini sengaja TIDAK di-import eager.
`msds_rag` menarik sentence-transformers + torch (ratusan MB RAM saat dimuat),
sehingga meng-import `app.services.po_parser` saja tidak boleh memuatnya —
ini yang membuat cold start / free tier (Railway/Render) lambat.

Import submodule tetap bekerja seperti biasa:

    from app.services.po_parser import POParser   # ringan, tanpa torch
    from app.services import POParser             # lewat __getattr__ (PEP 562)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - hanya untuk IDE / type checker
    from .msds_rag import MSDSRagService, RAGQueryResult, SourceDocument
    from .po_parser import POExtractResponse, POItem, POParser

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "POParser":          ("app.services.po_parser", "POParser"),
    "POItem":            ("app.services.po_parser", "POItem"),
    "POExtractResponse": ("app.services.po_parser", "POExtractResponse"),
    "MSDSRagService":    ("app.services.msds_rag", "MSDSRagService"),
    "RAGQueryResult":    ("app.services.msds_rag", "RAGQueryResult"),
    "SourceDocument":    ("app.services.msds_rag", "SourceDocument"),
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
    globals()[name] = value  # cache supaya __getattr__ tidak dipanggil lagi
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
