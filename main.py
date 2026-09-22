"""
main.py — AI Engine API
=======================
Universal FastAPI entry point:
  • PO Parser   : POST /api/v1/extract-po
  • Document RAG: POST /api/v1/query-docs
  • Legacy RAG  : /rag/*
  • Legacy PDF  : /pdf/*
"""

from __future__ import annotations

import asyncio
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
logger = logging.getLogger("ai_engine.main")

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

    # Auto-generate sample PDFs jika belum ada (penting untuk Render deploy)
    from pathlib import Path
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    if not (data_dir / "sample_po.pdf").exists() or not (data_dir / "sample_docs.pdf").exists():
        logger.info("Sample PDF tidak ditemukan — generating via data_gen.py...")
        import data_gen  # noqa: F401 — side effect: generates PDFs

    # ---------- Service inti ----------
    # STARTUP HARUS TAHAN GAGAL: kalau salah satu service berat tidak bisa
    # diinisialisasi (OOM / model tidak bisa diunduh / env kurang), aplikasi
    # tetap harus hidup supaya /health, /docs, dan pesan errornya bisa dibaca.
    # Crash di sini = Railway merespons 502 "Application failed to respond".
    from app.services.llm_factory import (
        describe_llm_error,
        get_active_model_name,
        get_provider,
        get_provider_name,
        probe_llm,
    )

    # POParser ringan (hanya client LLM). Kegagalan konfigurasi (mis. API key
    # hilang) dicatat sebagai error, lalu diulang saat endpoint dipanggil.
    app.state.po_parser = None
    try:
        from app.services.po_parser import POParser

        app.state.po_parser = POParser()
    except Exception as exc:
        logger.error(
            "POParser TIDAK dapat diinisialisasi: %s | Endpoint "
            "/api/v1/extract-po akan mengembalikan status 503 dengan detail ini.",
            describe_llm_error(exc),
        )

    # MSDSRagService dan legacy services dibuat LAZY (lihat get_msds_rag() /
    # get_rag_service() / get_pdf_service()). Konstruktornya memuat
    # sentence-transformers + torch + model MiniLM (ratusan MB RAM); memuatnya
    # saat startup membuat free tier Railway/Render lambat atau kehabisan
    # memori sehingga aplikasi gagal merespons.
    app.state.msds_rag    = None
    app.state.rag_service = None
    app.state.pdf_service = None

    # Cek (non-fatal) apakah model LLM yang dikonfigurasi benar-benar ada.
    # Ini yang membuat 404 model_not_found langsung terlihat di log deploy.
    llm_ok, llm_detail = await asyncio.to_thread(probe_llm)
    if llm_ok:
        logger.info("✅ Services siap — LLM: %s (%s)", get_provider_name(), llm_detail)
    else:
        logger.warning(
            "⚠️  LLM: %s — %s | Perbaiki variabel environment %s_MODEL di "
            "dashboard deployment (Railway/Render) lalu redeploy. "
            "Daftar model yang tersedia: GET /api/v1/llm/models",
            get_provider_name(), llm_detail,
            get_provider().upper(),
        )

    logger.info(
        "🚀 Startup selesai — provider: %s, model: %s, po_parser_ready: %s "
        "(msds_rag & legacy services dimuat saat pertama dipakai)",
        get_provider(), get_active_model_name(), app.state.po_parser is not None,
    )

    yield
    logger.info("🛑 AI Service API dimatikan.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title=os.getenv("APP_TITLE", "AI Engine API"),
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="""
## AI Engine — Universal Document Intelligence API

A universal REST API for document processing and knowledge retrieval.

### 🧾 Purchase Order Parser
Extract structured data (PO number, client, line items, prices) from any PDF Purchase Order
using LLM + prompt engineering + Pydantic v2.

### 🔬 Document RAG Assistant
Answer questions from any ingested PDF document
using Retrieval-Augmented Generation with ChromaDB + HuggingFace Embeddings + LLM.

---
**Stack:** FastAPI · LangChain · ChromaDB · Groq / Ollama · HuggingFace (MiniLM-L6-v2) · Pydantic v2
""",
    contact={
        "name":  "AI Engine API",
        "email": os.getenv("CONTACT_EMAIL", "admin@example.com"),
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=[
        {
            "name": "PO Parser",
            "description": "Extract structured data from Purchase Order PDFs.",
        },
        {
            "name": "Document RAG",
            "description": "Question answering from ingested documents using RAG.",
        },
        {
            "name": "RAG (Legacy)",
            "description": "General-purpose RAG — ingest & query ChromaDB collections.",
        },
        {
            "name": "PDF (Legacy)",
            "description": "Parse and generate generic PDFs.",
        },
        {
            "name": "Health",
            "description": "API status and info.",
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
# Optional API-key gate
# Aktif HANYA jika env API_KEY diisi — jadi tidak mengubah perilaku saat ini.
# Tujuan: melindungi kuota LLM gratis dari pemakaian liar.
# ---------------------------------------------------------------------------

API_KEY = os.getenv("API_KEY", "").strip()
_API_KEY_EXEMPT_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    """
    Tolak request tanpa header `X-API-Key` yang benar.

    Endpoint health & dokumentasi selalu terbuka agar healthcheck platform
    dan Swagger tetap bisa diakses.
    """
    if API_KEY and request.url.path not in _API_KEY_EXEMPT_PATHS:
        if request.headers.get("X-API-Key") != API_KEY:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Header X-API-Key tidak ada atau tidak valid."},
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Legacy services — lazy accessors
# RAGService memuat HuggingFace embeddings + LLM, jadi dibuat saat pertama
# dipakai saja (performa cold start / RAM free tier).
# ---------------------------------------------------------------------------

def get_rag_service():
    """Lazy-init legacy RAGService."""
    if getattr(app.state, "rag_service", None) is None:
        from services.rag_service import RAGService

        logger.info("Menginisialisasi legacy RAGService (lazy)...")
        app.state.rag_service = RAGService()
    return app.state.rag_service


def get_pdf_service():
    """Lazy-init legacy PDFService (ringan: pypdf + reportlab)."""
    if getattr(app.state, "pdf_service", None) is None:
        from services.pdf_service import PDFService

        logger.info("Menginisialisasi legacy PDFService (lazy)...")
        app.state.pdf_service = PDFService()
    return app.state.pdf_service


def get_msds_rag():
    """
    Lazy-init MSDSRagService.

    Konstruktornya memuat HuggingFaceEmbeddings (torch + model MiniLM, ratusan
    MB RAM) dan meng-ingest dokumen ke ChromaDB. Itu sebabnya service ini baru
    dibuat saat endpoint /api/v1/query-msds benar-benar dipanggil — supaya
    startup di free tier tidak lambat / kehabisan memori.
    """
    if getattr(app.state, "msds_rag", None) is None:
        from app.services.msds_rag import MSDSRagService

        logger.info("Menginisialisasi MSDSRagService (lazy) — memuat embeddings...")
        app.state.msds_rag = MSDSRagService()
    return app.state.msds_rag


def get_po_parser():
    """
    Ambil POParser; bila belum/tidak bisa dibuat, kembalikan error yang jelas.

    Ini mencegah aplikasi mati total hanya karena konfigurasi LLM salah
    (mis. GROQ_API_KEY hilang) — endpoint mengembalikan 503 + pesan
    yang bisa ditindaklanjuti alih-alih 502 dari platform.
    """
    if getattr(app.state, "po_parser", None) is None:
        from app.services.llm_factory import describe_llm_error
        from app.services.po_parser import POParser

        try:
            app.state.po_parser = POParser()
        except Exception as exc:
            detail = describe_llm_error(exc)
            logger.error("POParser tidak dapat diinisialisasi: %s", detail)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LLM belum dikonfigurasi dengan benar. {detail}",
            ) from exc
    return app.state.po_parser


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
        description="Question about the ingested documents",
        examples=["What are the first aid measures for eye contact?"],
    )
    k: int = Field(default=4, ge=1, le=10, description="Number of context chunks to retrieve (1–10)")


class SourceDocumentOut(BaseModel):
    content:  str
    metadata: dict


class MSDSQueryResponse(BaseModel):
    answer:           str
    source_documents: list[SourceDocumentOut]


class LLMModelsResponse(BaseModel):
    """Diagnostik model LLM yang benar-benar dapat diakses API key."""

    provider:         str
    configured_model: str
    model_available:  bool
    available_models: list[str]
    detail:           str


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
        "service": os.getenv("APP_TITLE", "AI Engine API"),
        "version": os.getenv("APP_VERSION", "1.0.0"),
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Detail health check",
)
async def health_check():
    """
    Status API + provider/model LLM yang aktif.

    Tanpa network call dan tanpa memuat service berat — endpoint ini harus
    selalu cepat supaya healthcheck platform (Railway/Render) tidak timeout.
    """
    from app.services.llm_factory import (
        get_active_model_name,
        get_provider,
        get_provider_name,
    )

    return {
        "status":          "healthy",
        "llm":             get_provider_name(),
        "provider":        get_provider(),
        "model":           get_active_model_name(),
        "po_parser_ready": getattr(app.state, "po_parser", None) is not None,
        "rag_ready":       getattr(app.state, "msds_rag", None) is not None,
        "docs":            "/docs",
    }


@app.get(
    "/api/v1/llm/models",
    response_model=LLMModelsResponse,
    tags=["Health"],
    summary="Model LLM yang benar-benar bisa diakses API key",
    responses={
        502: {"description": "Provider LLM tidak dapat dihubungi"},
    },
)
async def llm_models():
    """
    Diagnostik model LLM.

    Pakai endpoint ini saat muncul error **404 `model_not_found`**: nilai
    `GROQ_MODEL` (atau `OLLAMA_MODEL`) pada environment deployment harus ada
    di `available_models`.

    Contoh kasus nyata: `llama-3.1-8b-instant` dan `llama-3.3-70b-versatile`
    sudah dipindah Groq ke tier Enterprise, sehingga API key free menerima 404.
    """
    from app.services.llm_factory import (
        get_active_model_name,
        get_provider,
        list_models,
    )

    provider = get_provider()
    model    = get_active_model_name()

    try:
        available = await asyncio.to_thread(list_models)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gagal membaca daftar model {provider}: {exc}",
        )

    model_available = model in available
    if model_available:
        detail = f"Model '{model}' tersedia untuk provider {provider}."
    else:
        env_var = "GROQ_MODEL" if provider == "groq" else "OLLAMA_MODEL"
        detail = (
            f"Model '{model}' TIDAK tersedia untuk provider {provider}. "
            f"Set variabel environment {env_var} ke salah satu nilai "
            f"available_models, lalu redeploy."
        )

    return LLMModelsResponse(
        provider=provider,
        configured_model=model,
        model_available=model_available,
        available_models=available,
        detail=detail,
    )


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

        po_parser = get_po_parser()
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
    tags=["Document RAG"],
    summary="Question answering from ingested documents",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Answer generated successfully"},
        500: {"description": "RAG query failed or LLM unavailable"},
    },
)
async def query_msds(request: MSDSQueryRequest):
    """
    Ask any question against documents ingested into ChromaDB.

    Examples:
    - *"What are the first aid measures for eye contact?"*
    - *"What is the flash point of this substance?"*
    - *"How should this material be stored?"*
    - *"What PPE is required when handling this material?"*
    """
    try:
        msds_rag = get_msds_rag()
        result = msds_rag.query(question=request.question, k=request.k)
    except HTTPException:
        raise
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

    pdf_service = get_pdf_service()
    rag_service = get_rag_service()

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
    rag_service = get_rag_service()
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
        collections=get_rag_service().list_collections()
    )


@app.delete(
    "/rag/collections/{collection_name}",
    tags=["RAG (Legacy)"],
    summary="Hapus koleksi ChromaDB",
)
async def delete_collection(collection_name: str):
    """Hapus koleksi ChromaDB berdasarkan nama."""
    try:
        get_rag_service().delete_collection(collection_name)
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

    pdf_service = get_pdf_service()
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
        pdf_bytes = get_pdf_service().generate_pdf(
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
