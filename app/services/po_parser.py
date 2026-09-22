"""
app/services/po_parser.py
=========================
Purchase Order Parser — PT Multisari Indoprima AI Engine
Stack: ChatOllama (llama3.2) — 100% Gratis, Tanpa API Key

Alur kerja:
  1. Terima bytes PDF dari caller
  2. Ekstrak seluruh teks per halaman menggunakan pypdf
  3. Kirim teks ke ChatOllama dengan prompt JSON yang ketat
  4. Parse output teks → POExtractResponse via Pydantic v2

Catatan:
  ChatOllama tidak mendukung .with_structured_output() seperti OpenAI.
  Solusi: prompt engineering + JSON extraction manual dengan fallback
  yang robust untuk menangani variasi output model lokal.

Dependensi:
  langchain-community (ChatOllama), pypdf, pydantic v2, python-dotenv
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader
from app.services.llm_factory import get_llm, get_provider_name

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic v2 Schemas
# ---------------------------------------------------------------------------

class POItem(BaseModel):
    """Satu baris item dalam Purchase Order."""

    line_no:     int   = Field(description="Nomor urut baris item")
    item_code:   str   = Field(description="Kode/SKU barang, contoh: ARO-LV-001")
    description: str   = Field(description="Nama/deskripsi lengkap barang")
    quantity:    float = Field(description="Jumlah yang dipesan")
    unit:        str   = Field(description="Satuan barang, contoh: Botol, Liter, Pcs, Kg")
    unit_price:  float = Field(description="Harga satuan dalam mata uang dokumen")
    total_price: float = Field(description="Total harga baris = quantity × unit_price")


class POExtractResponse(BaseModel):
    """Hasil ekstraksi terstruktur dari dokumen Purchase Order."""

    po_number:      str            = Field(description="Nomor PO unik, contoh: PO-2026-00123")
    client_name:    str            = Field(description="Nama perusahaan pembeli/pemesan")
    client_address: Optional[str] = Field(default=None, description="Alamat lengkap perusahaan pembeli")
    order_date:     Optional[str] = Field(default=None, description="Tanggal PO diterbitkan (YYYY-MM-DD)")
    delivery_date:  Optional[str] = Field(default=None, description="Tanggal pengiriman yang diminta")
    items:          list[POItem]   = Field(description="Daftar line item dalam PO")
    total_amount:   Optional[float] = Field(default=None, description="Total nilai PO setelah pajak")
    currency:       str            = Field(default="IDR", description="Kode mata uang")
    notes:          Optional[str] = Field(default=None, description="Catatan atau syarat khusus dalam PO")


# ---------------------------------------------------------------------------
# Prompt  (instruksi sangat eksplisit agar model lokal mengikuti format JSON)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Anda adalah sistem ekstraksi data Purchase Order (PO).
Baca teks dokumen PO dan kembalikan HANYA objek JSON berisi data NYATA dari dokumen.
Jangan gunakan placeholder. Isi setiap field dengan nilai yang ada di dokumen.

CONTOH FORMAT OUTPUT (gunakan nilai nyata dari dokumen, bukan contoh ini):
{
  "po_number": "PO-2026-00123",
  "client_name": "PT Sentosa Aromatics",
  "client_address": "Jl. Industri Raya No. 45, Tangerang",
  "order_date": "2026-09-10",
  "delivery_date": "2026-09-20",
  "items": [
    {
      "line_no": 1,
      "item_code": "ARO-LV-001",
      "description": "Lavender Essential Oil 100ml",
      "quantity": 50,
      "unit": "Botol",
      "unit_price": 85000,
      "total_price": 4250000
    }
  ],
  "total_amount": 32300000,
  "currency": "IDR",
  "notes": "Harap sertakan CoA untuk setiap item."
}

ATURAN:
1. Ganti semua nilai contoh di atas dengan nilai NYATA dari dokumen.
2. Ekstrak SEMUA baris item yang ada — jangan ada yang terlewat.
3. Harga = angka murni tanpa simbol mata uang, titik ribuan, atau koma desimal.
4. Jika suatu field tidak ada dalam dokumen, gunakan null.
5. Kembalikan HANYA JSON — tidak ada teks sebelum { atau sesudah }.\
"""

_HUMAN_TEMPLATE = """\
Dokumen Purchase Order:

{document_text}

Ekstrak semua data dari dokumen di atas dan kembalikan sebagai JSON:\
"""


# ---------------------------------------------------------------------------
# Helper: ekstrak blok JSON dari output LLM yang mungkin mengandung teks lain
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> str:
    """
    Cari dan kembalikan blok JSON pertama yang valid dari output LLM.
    Menangani kasus LLM menambahkan teks, markdown code fence, atau penjelasan.
    """
    # 1. Coba hilangkan code fence markdown (```json ... ``` atau ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # 2. Cari blok JSON paling luar { ... } (greedy dari { pertama ke } terakhir)
    start = text.find("{")
    if start == -1:
        raise ValueError("Tidak ditemukan objek JSON dalam output LLM.")

    # Lacak kedalaman kurung kurawal untuk menemukan penutup yang benar
    depth   = 0
    end_idx = -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if end_idx == -1:
        raise ValueError("JSON dalam output LLM tidak lengkap (kurung kurawal tidak seimbang).")

    return text[start : end_idx + 1]


# ---------------------------------------------------------------------------
# POParser
# ---------------------------------------------------------------------------

class POParser:
    """
    Mengekstrak data terstruktur dari PDF Purchase Order menggunakan
    ChatOllama (llama3.2) + prompt engineering + Pydantic v2 parsing.

    Tidak membutuhkan API key berbayar — berjalan sepenuhnya secara lokal.
    """

    def __init__(self) -> None:
        self._llm = get_llm(temperature=0)
        logger.info("POParser siap — provider: %s", get_provider_name())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_bytes(
        self,
        pdf_bytes: bytes,
        filename: str = "upload.pdf",
    ) -> POExtractResponse:
        """
        Terima PDF sebagai bytes, ekstrak teks, parse ke POExtractResponse.

        Parameters
        ----------
        pdf_bytes : bytes
            Konten file PDF.
        filename : str
            Nama file asli (untuk logging).

        Returns
        -------
        POExtractResponse
            Data PO yang sudah tervalidasi oleh Pydantic.

        Raises
        ------
        ValueError
            Jika PDF tidak mengandung teks atau output LLM tidak valid.
        ConnectionError
            Jika Ollama tidak dapat dihubungi.
        """
        text = self._extract_text(pdf_bytes, filename)
        if not text.strip():
            raise ValueError(
                f"File '{filename}' tidak mengandung teks yang dapat diekstrak. "
                "Pastikan PDF bukan hasil scan tanpa OCR."
            )

        # Batasi panjang teks yang dikirim ke LLM.
        # pypdf sering menghasilkan banyak whitespace dan noise layout.
        # 3000 karakter pertama cukup untuk menangkap semua data PO standar.
        MAX_CHARS = 3000
        if len(text) > MAX_CHARS:
            logger.info("Teks PDF dipotong dari %d → %d karakter untuk efisiensi LLM.",
                        len(text), MAX_CHARS)
            text = text[:MAX_CHARS]

        logger.info("Mengirim teks ke Ollama untuk ekstraksi PO (file: %s, %d karakter)",
                    filename, len(text))

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_HUMAN_TEMPLATE.format(document_text=text)),
        ]

        try:
            response = self._llm.invoke(messages)
        except Exception as exc:
            raise ConnectionError(
                f"Tidak dapat terhubung ke Ollama. "
                f"Pastikan Ollama berjalan (ollama serve) dan model '{self._llm.model}' "
                f"sudah di-pull (ollama pull {self._llm.model}). Detail: {exc}"
            ) from exc

        raw_output = response.content
        logger.info("Raw output Ollama (300 char): %.300s", raw_output)

        return self._parse_llm_output(raw_output, filename)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_llm_output(self, raw_output: str, filename: str) -> POExtractResponse:
        """
        Parse output teks LLM → POExtractResponse.
        Mencoba tiga strategi jika output tidak bersih.
        """
        # Strategi 1: parse langsung
        try:
            data = json.loads(raw_output.strip())
            result = POExtractResponse(**data)
            logger.info("Ekstraksi PO berhasil (strategi 1) — PO: %s, %d item",
                        result.po_number, len(result.items))
            return result
        except (json.JSONDecodeError, ValidationError, TypeError):
            pass

        # Strategi 2: ekstrak blok JSON dari teks yang mengandung narasi
        try:
            json_block = _extract_json_block(raw_output)
            data = json.loads(json_block)
            result = POExtractResponse(**data)
            logger.info("Ekstraksi PO berhasil (strategi 2) — PO: %s, %d item",
                        result.po_number, len(result.items))
            return result
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            pass

        # Strategi 3: bersihkan karakter noise yang umum dari model lokal
        # (trailing koma, komentar // ..., newline dalam string)
        try:
            cleaned = raw_output.strip()
            # Hapus komentar inline: // ...
            cleaned = re.sub(r"//[^\n]*", "", cleaned)
            # Hapus trailing comma sebelum } atau ]
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            # Ambil blok JSON
            json_block = _extract_json_block(cleaned)
            data = json.loads(json_block)
            result = POExtractResponse(**data)
            logger.info("Ekstraksi PO berhasil (strategi 3) — PO: %s, %d item",
                        result.po_number, len(result.items))
            return result
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            logger.error("Gagal parse output LLM dari '%s': %s\nOutput: %.500s",
                         filename, exc, raw_output)
            raise ValueError(
                f"Output LLM untuk '{filename}' tidak dapat diparsing sebagai PO JSON. "
                f"Detail: {exc}\n\nRaw output (300 char pertama): {raw_output[:300]}"
            ) from exc

    @staticmethod
    def _extract_text(pdf_bytes: bytes, filename: str) -> str:
        """Ekstrak seluruh teks dari PDF menggunakan pypdf."""
        reader     = PdfReader(io.BytesIO(pdf_bytes))
        pages_text: list[str] = []

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages_text.append(f"[Halaman {page_num}]\n{page_text}")

        full_text = "\n\n".join(pages_text)
        logger.debug("Teks terekstrak dari '%s': %d halaman, %d karakter",
                     filename, len(pages_text), len(full_text))
        return full_text
