"""
main.py — PT Multisari Indoprima AI Service API
================================================
FastAPI entry point yang mengintegrasikan:
  • PO Parser   : POST /api/v1/extract-po
  • MSDS RAG    : POST /api/v1/query-msds
  • Legacy RAG  : /rag/*
  • Legacy PDF  : /pdf/*
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("indoprima.main")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Lifespan — startup & shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("⚙️  Memulai inisialisasi services...")

    # New services (spec.md)
    from app.services.msds_rag import MSDSRagService
    from app.services.po_parser import POParser

    app.state.po_parser   = POParser()
    app.state.msds_rag    = MSDSRagService()      # auto-ingest msds_sample.pdf jika belum ada

    # Legacy services
    from services.pdf_service import PDFService
    from services.rag_service import RAGService

    app.state.rag_service = RAGService()
    app.state.pdf_service = PDFService()

    logger.info("✅ Semua services siap.")
    yield
    logger.info("🛑 AI Service API dimatikan.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PT Multisari Indoprima - AI Service API",
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="""
## AI Engine Microservice — PT Multisari Indoprima

REST API untuk dua fungsi utama operasional B2B:

### 🧾 Purchase Order Parser
Ekstrak data terstruktur (nomor PO, nama klien, line items, harga) dari file PDF Purchase Order
secara otomatis menggunakan **GPT-4o-mini** dengan _structured output_.

### 🔬 MSDS RAG Assistant
Jawab pertanyaan teknis seputar Material Safety Data Sheet (MSDS) bahan wewangian
menggunakan **Retrieval-Augmented Generation** berbasis **ChromaDB + OpenAI**.

---
**Stack:** FastAPI · LangChain · ChromaDB · OpenAI · Pydantic v2 · pypdf
""",
    contact={
        "name":  "PT Multisari Indoprima — IT Division",
        "email": "dev@multisariindoprima.co.id",
    },
    license_info={
        "name": "Proprietary — PT Multisari Indoprima",
    },
    openapi_tags=[
        {
            "name": "PO Parser",
            "description": "Ekstraksi Purchase Order dari PDF ke JSON terstruktur.",
        },
        {
            "name": "MSDS RAG",
            "description": "Tanya jawab berbasis dokumen MSDS menggunakan RAG.",
        },
        {
            "name": "RAG (Legacy)",
            "description": "General-purpose RAG — ingest & query koleksi ChromaDB.",
        },
        {
            "name": "PDF (Legacy)",
            "description": "Parse dan generate PDF generik.",
        },
        {
            "name": "Health",
            "description": "Status dan info API.",
        },
    ],
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": str(exc.body)},
    )


# ---------------------------------------------------------------------------
# Schemas — baru (spec.md)
# ---------------------------------------------------------------------------

class POItem(BaseModel):
    line_no:     int
    item_code:   str
    description: str
    quantity:    float
    unit:        str
    unit_price:  float
    total_price: float


class POExtractResponse(BaseModel):
    po_number:      str
    client_name:    str
    client_address: str | None = None
    order_date:     str | None = None
    delivery_date:  str | None = None
    items:          list[POItem]
    total_amount:   float | None = None
    currency:       str = "IDR"
    notes:          str | None = None


class MSDSQueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        description="Pertanyaan tentang dokumen MSDS",
        examples=["Apa tindakan pertolongan pertama jika terkena Linalool di mata?"],
    )
    k: int = Field(default=4, ge=1, le=10, description="Jumlah chunk konteks (1–10)")


class SourceDocumentOut(BaseModel):
    content:  str
    metadata: dict


class MSDSQueryResponse(BaseModel):
    answer:           str
    source_documents: list[SourceDocumentOut]


# ---------------------------------------------------------------------------
# Schemas — legacy
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question:        str
    collection_name: str = "default"
    k:               int = 4


class QueryResponse(BaseModel):
    answer:           str
    source_documents: list


class IngestResponse(BaseModel):
    message:         str
    collection_name: str
    chunks_stored:   int


class PDFGenerateRequest(BaseModel):
    title:   str
    content: str


class CollectionListResponse(BaseModel):
    collections: list[str]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get(
    "/",
    tags=["Health"],
    summary="Root health check",
)
async def root():
    """Cek status API dan versi yang berjalan."""
    return {
        "status":  "ok",
        "service": "PT Multisari Indoprima - AI Service API",
        "version": os.getenv("APP_VERSION", "1.0.0"),
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Detail health check",
)
async def health_check():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# PO Parser endpoint
# ---------------------------------------------------------------------------

MAX_PDF_SIZE = 10 * 1024 * 1024  # 10 MB


@app.post(
    "/api/v1/extract-po",
    response_model=POExtractResponse,
    tags=["PO Parser"],
    summary="Ekstrak data Purchase Order dari PDF",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Data PO berhasil diekstrak"},
        400: {"description": "File bukan PDF atau kosong"},
        413: {"description": "Ukuran file melebihi 10 MB"},
        422: {"description": "PDF tidak mengandung teks yang dapat diparse"},
        500: {"description": "Gagal memanggil LLM"},
    },
)
async def extract_po(
    file: UploadFile = File(
        ..., description="File PDF Purchase Order (maks. 10 MB)"
    ),
):
    """
    Upload file PDF Purchase Order dan dapatkan data terstruktur dalam JSON.

    Proses:
    1. Validasi file (PDF, tidak kosong, maks 10 MB)
    2. Simpan sementara ke folder `temp/`
    3. Ekstrak teks dengan **pypdf**
    4. Parse ke JSON dengan **GPT-4o-mini structured output**
    5. Hapus file temp
    6. Kembalikan `POExtractResponse`
    """
    # — Validasi ekstensi
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hanya file PDF yang diterima.",
        )

    pdf_bytes = await file.read()

    # — Validasi isi & ukuran
    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File yang diunggah kosong.",
        )
    if len(pdf_bytes) > MAX_PDF_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Ukuran file melebihi batas maksimum 10 MB "
                   f"({len(pdf_bytes) / 1024 / 1024:.1f} MB).",
        )

    # — Simpan ke temp/ dengan nama unik
    temp_path = TEMP_DIR / f"{uuid.uuid4().hex}_{filename}"
    try:
        temp_path.write_bytes(pdf_bytes)
        logger.info("File temp tersimpan: %s", temp_path.name)

        po_parser = app.state.po_parser
        try:
            result = po_parser.extract_from_bytes(pdf_bytes, filename=filename)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
        except Exception as exc:
            logger.exception("Gagal ekstrak PO dari '%s'", filename)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Gagal memproses dokumen: {exc}",
            )

    finally:
        # — Hapus file temp (dijamin meskipun terjadi error)
        if temp_path.exists():
            temp_path.unlink()
            logger.info("File temp dihapus: %s", temp_path.name)

    return result


# ---------------------------------------------------------------------------
# MSDS RAG endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/query-msds",
    response_model=MSDSQueryResponse,
    tags=["MSDS RAG"],
    summary="Tanya jawab berbasis dokumen MSDS",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Jawaban berhasil dihasilkan"},
        500: {"description": "Gagal query RAG atau koneksi OpenAI"},
    },
)
async def query_msds(request: MSDSQueryRequest):
    """
    Ajukan pertanyaan teknis tentang bahan kimia/wewangian berdasarkan
    dokumen **MSDS** yang sudah diingesti ke ChromaDB.

    Contoh pertanyaan:
    - *"Apa tindakan pertolongan pertama jika terkena Linalool di mata?"*
    - *"Berapa titik nyala (flash point) Linalool?"*
    - *"Bagaimana cara membuang limbah Linalool dengan benar?"*
    - *"Apa APD yang dibutuhkan saat menangani bahan ini?"*
    """
    msds_rag = app.state.msds_rag
    try:
        result = msds_rag.query(question=request.question, k=request.k)
    except Exception as exc:
        logger.exception("RAG query gagal: %s", request.question[:80])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query gagal: {exc}",
        )

    return MSDSQueryResponse(
        answer=result["answer"],
        source_documents=[
            SourceDocumentOut(
                content=doc["content"],
                metadata=doc["metadata"],
            )
            for doc in result["source_documents"]
        ],
    )


# ---------------------------------------------------------------------------
# Legacy RAG endpoints  (/rag/*)
# ---------------------------------------------------------------------------

@app.post(
    "/rag/ingest",
    response_model=IngestResponse,
    tags=["RAG (Legacy)"],
    summary="Ingest PDF ke koleksi ChromaDB",
)
async def ingest_pdf(
    file: UploadFile = File(...),
    collection_name: str = Form(default="default"),
):
    """Upload PDF dan simpan embedding-nya ke ChromaDB."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang diterima.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="File kosong.")

    pdf_service = app.state.pdf_service
    rag_service = app.state.rag_service

    documents = pdf_service.parse_pdf(pdf_bytes, source_name=file.filename)
    if not documents:
        raise HTTPException(status_code=422, detail="Tidak ada teks yang dapat diekstrak.")

    chunks_stored = rag_service.ingest_documents(documents, collection_name=collection_name)
    return IngestResponse(
        message="Dokumen berhasil diingesti.",
        collection_name=collection_name,
        chunks_stored=chunks_stored,
    )


@app.post(
    "/rag/query",
    response_model=QueryResponse,
    tags=["RAG (Legacy)"],
    summary="Query koleksi ChromaDB generik",
)
async def query_rag(request: QueryRequest):
    """Tanya jawab terhadap koleksi RAG generik."""
    rag_service = app.state.rag_service
    try:
        result = rag_service.query(
            question=request.question,
            collection_name=request.collection_name,
            k=request.k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG query gagal: {exc}")
    return QueryResponse(**result)


@app.get(
    "/rag/collections",
    response_model=CollectionListResponse,
    tags=["RAG (Legacy)"],
    summary="List semua koleksi ChromaDB",
)
async def list_collections():
    """Tampilkan semua koleksi yang tersimpan di ChromaDB."""
    return CollectionListResponse(
        collections=app.state.rag_service.list_collections()
    )


@app.delete(
    "/rag/collections/{collection_name}",
    tags=["RAG (Legacy)"],
    summary="Hapus koleksi ChromaDB",
)
async def delete_collection(collection_name: str):
    """Hapus koleksi ChromaDB berdasarkan nama."""
    try:
        app.state.rag_service.delete_collection(collection_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Koleksi tidak ditemukan: {exc}")
    return {"message": f"Koleksi '{collection_name}' berhasil dihapus."}


# ---------------------------------------------------------------------------
# Legacy PDF endpoints  (/pdf/*)
# ---------------------------------------------------------------------------

@app.post(
    "/pdf/parse",
    tags=["PDF (Legacy)"],
    summary="Ekstrak teks dari PDF",
)
async def parse_pdf(file: UploadFile = File(...)):
    """Ekstrak teks dan metadata dari file PDF yang diunggah."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang diterima.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="File kosong.")

    pdf_service = app.state.pdf_service
    metadata    = pdf_service.get_pdf_metadata(pdf_bytes)
    documents   = pdf_service.parse_pdf(pdf_bytes, source_name=file.filename)

    return {
        "filename": file.filename,
        "metadata": metadata,
        "pages": [
            {"page": doc.metadata["page"], "content": doc.page_content}
            for doc in documents
        ],
    }


@app.post(
    "/pdf/generate",
    tags=["PDF (Legacy)"],
    summary="Generate PDF dari teks",
)
async def generate_pdf(request: PDFGenerateRequest):
    """Buat file PDF dari judul dan konten teks, kembalikan sebagai file unduhan."""
    try:
        pdf_bytes = app.state.pdf_service.generate_pdf(
            title=request.title, content=request.content
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal generate PDF: {exc}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{request.title}.pdf"'
        },
    )


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("DEBUG", "False").lower() == "true",
    )
