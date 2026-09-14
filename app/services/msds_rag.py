"""
app/services/msds_rag.py
========================
MSDS RAG Service — PT Multisari Indoprima AI Engine
Stack: HuggingFaceEmbeddings (all-MiniLM-L6-v2) + ChatOllama (llama3.2)
       100% Gratis, Tanpa API Key, Berjalan Lokal

Alur kerja:
  Ingest (otomatis saat startup / on-demand):
    1. Baca data/msds_sample.pdf menggunakan pypdf
    2. Pecah teks dengan RecursiveCharacterTextSplitter (chunk=1000, overlap=200)
    3. Embed dengan HuggingFaceEmbeddings all-MiniLM-L6-v2 (lokal ~90 MB)
    4. Simpan ke ChromaDB persistent (./data/chroma_db)

  Query (setiap request):
    1. Similarity search ChromaDB → top-k chunks relevan
    2. Susun konteks ke prompt RAG
    3. Generate jawaban dengan ChatOllama (llama3.2)
    4. Kembalikan answer + source_documents

Catatan:
  - Model embedding diunduh sekali ke ~/.cache/huggingface/ lalu offline.
  - Koleksi ChromaDB diberi nama "msds_hf" agar tidak konflik dengan
    koleksi lama yang dibuat dengan OpenAI embeddings (dimensi berbeda).

Dependensi:
  langchain, langchain-community, chromadb, sentence-transformers,
  pypdf, pydantic v2, python-dotenv
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import TypedDict

import chromadb
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from pypdf import PdfReader

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed return values
# ---------------------------------------------------------------------------

class SourceDocument(TypedDict):
    content:  str
    metadata: dict


class RAGQueryResult(TypedDict):
    answer:           str
    source_documents: list[SourceDocument]


# ---------------------------------------------------------------------------
# RAG Prompt  (eksplisit untuk model lokal agar tidak mengarang)
# ---------------------------------------------------------------------------

_RAG_PROMPT_TEMPLATE = """\
Anda adalah asisten teknis ahli Material Safety Data Sheet (MSDS) \
untuk PT Multisari Indoprima, perusahaan di industri parfum dan bahan kimia.

INSTRUKSI:
- Jawab HANYA berdasarkan konteks MSDS yang diberikan di bawah.
- Jika informasi tidak ada dalam konteks, katakan dengan jelas: \
"Informasi tersebut tidak tersedia dalam dokumen MSDS yang dimuat."
- Jangan mengarang atau menambahkan informasi di luar konteks.
- Jawab dalam Bahasa Indonesia yang jelas, ringkas, dan profesional.

--- KONTEKS MSDS ---
{context}
--- AKHIR KONTEKS ---

Pertanyaan: {question}

Jawaban:\
"""

_RAG_PROMPT = PromptTemplate(
    template=_RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


# ---------------------------------------------------------------------------
# MSDSRagService
# ---------------------------------------------------------------------------

class MSDSRagService:
    """
    RAG service untuk dokumen MSDS menggunakan:
    - HuggingFaceEmbeddings (all-MiniLM-L6-v2) — embedding lokal gratis
    - ChatOllama (llama3.2)                     — LLM lokal gratis
    - ChromaDB (persistent)                     — vector store lokal

    Singleton-friendly: inisialisasi sekali di startup, panggil query() per request.
    """

    DEFAULT_MSDS_PDF   = Path(__file__).resolve().parents[2] / "data" / "msds_sample.pdf"
    DEFAULT_CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma_db"
    # Nama koleksi berbeda dari versi OpenAI (dimensi embedding berbeda = 384 vs 1536)
    COLLECTION_NAME    = "msds_hf"

    def __init__(
        self,
        msds_pdf_path: Path | str | None = None,
        chroma_dir:    Path | str | None = None,
    ) -> None:
        self._msds_pdf_path = Path(msds_pdf_path or self.DEFAULT_MSDS_PDF)
        self._chroma_dir    = Path(chroma_dir or os.getenv(
            "CHROMA_PERSIST_DIRECTORY", str(self.DEFAULT_CHROMA_DIR)
        ))
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        ollama_url      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model    = os.getenv("OLLAMA_MODEL", "llama3.2")

        # HuggingFace embeddings — model diunduh sekali ke cache lokal
        logger.info("Memuat HuggingFaceEmbeddings: %s", embedding_model)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},   # gunakan CPU (aman untuk semua mesin)
            encode_kwargs={"normalize_embeddings": True},
        )

        # Ollama LLM
        self._llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_url,
            temperature=0.1,
            num_predict=2048,
        )

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Vectorstore — lazy-loaded
        self._vectorstore: Chroma | None = None

        # Auto-ingest saat startup (non-fatal jika gagal)
        try:
            self._ensure_ingested()
        except Exception as exc:
            logger.warning(
                "Auto-ingest gagal — akan dicoba ulang saat query pertama. Detail: %s", exc
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str, k: int = 4) -> RAGQueryResult:
        """
        Jawab pertanyaan berdasarkan dokumen MSDS yang sudah diingesti.

        Parameters
        ----------
        question : str
            Pertanyaan dalam Bahasa Indonesia atau Inggris.
        k : int
            Jumlah chunk konteks yang diambil dari ChromaDB (1–10).

        Returns
        -------
        RAGQueryResult
            Dict berisi 'answer' (str) dan 'source_documents' (list).

        Raises
        ------
        ConnectionError
            Jika Ollama tidak dapat dihubungi.
        RuntimeError
            Jika koleksi ChromaDB kosong dan ingest gagal.
        """
        # Retry ingest jika koleksi belum ada (startup ingest mungkin gagal)
        if not self.collection_exists():
            logger.info("Koleksi kosong — mencoba ingest ulang sebelum query...")
            self._ensure_ingested()

        if not self.collection_exists():
            raise RuntimeError(
                "Koleksi MSDS belum ada di ChromaDB. "
                "Pastikan data/msds_sample.pdf tersedia dan jalankan ingest_pdf() terlebih dahulu."
            )

        vectorstore = self._get_vectorstore()

        qa_chain = RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": _RAG_PROMPT},
        )

        logger.info("RAG query (Ollama): %s", question[:120])
        try:
            result = qa_chain.invoke({"query": question})
        except Exception as exc:
            raise ConnectionError(
                f"Gagal mendapatkan jawaban dari Ollama. "
                f"Pastikan Ollama berjalan dan model '{self._llm.model}' sudah di-pull. "
                f"Detail: {exc}"
            ) from exc

        source_docs: list[SourceDocument] = [
            SourceDocument(content=doc.page_content, metadata=doc.metadata)
            for doc in result.get("source_documents", [])
        ]

        logger.info("RAG query selesai — %d source chunks digunakan.", len(source_docs))
        return RAGQueryResult(
            answer=result["result"],
            source_documents=source_docs,
        )

    def ingest_pdf(
        self,
        pdf_bytes: bytes,
        source_name: str = "msds_sample.pdf",
    ) -> int:
        """
        Ingest PDF MSDS dari bytes ke ChromaDB menggunakan HuggingFace embeddings.

        Parameters
        ----------
        pdf_bytes : bytes
            Konten file PDF.
        source_name : str
            Nama sumber untuk metadata.

        Returns
        -------
        int
            Jumlah chunk yang berhasil disimpan.
        """
        documents = self._parse_pdf_bytes(pdf_bytes, source_name)
        if not documents:
            raise ValueError(
                f"Tidak ada teks yang dapat diekstrak dari '{source_name}'. "
                "Pastikan PDF bukan hasil scan tanpa OCR."
            )

        chunks = self._text_splitter.split_documents(documents)
        logger.info("Ingesting %d chunks dari '%s' ke koleksi '%s'...",
                    len(chunks), source_name, self.COLLECTION_NAME)

        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=str(self._chroma_dir),
            collection_name=self.COLLECTION_NAME,
        )

        logger.info("Ingest selesai — %d chunks tersimpan ke ChromaDB.", len(chunks))
        return len(chunks)

    def collection_exists(self) -> bool:
        """Cek apakah koleksi MSDS sudah ada dan tidak kosong di ChromaDB."""
        try:
            client      = chromadb.PersistentClient(path=str(self._chroma_dir))
            collections = [c.name for c in client.list_collections()]
            if self.COLLECTION_NAME not in collections:
                return False
            col = client.get_collection(self.COLLECTION_NAME)
            return col.count() > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_ingested(self) -> None:
        """Auto-ingest msds_sample.pdf jika koleksi belum ada."""
        if self.collection_exists():
            logger.info("Koleksi '%s' sudah ada (%s chunks) — skip auto-ingest.",
                        self.COLLECTION_NAME,
                        chromadb.PersistentClient(path=str(self._chroma_dir))
                               .get_collection(self.COLLECTION_NAME).count())
            return

        if not self._msds_pdf_path.exists():
            logger.warning(
                "File MSDS tidak ditemukan di '%s'. "
                "Jalankan: python data_gen.py",
                self._msds_pdf_path,
            )
            return

        logger.info("Memulai auto-ingest '%s' → ChromaDB '%s'...",
                    self._msds_pdf_path.name, self.COLLECTION_NAME)
        pdf_bytes     = self._msds_pdf_path.read_bytes()
        chunks_stored = self.ingest_pdf(pdf_bytes, source_name=self._msds_pdf_path.name)
        logger.info("Auto-ingest selesai: %d chunks.", chunks_stored)

    def _get_vectorstore(self) -> Chroma:
        """Lazy-load ChromaDB vectorstore dengan HuggingFace embeddings."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                persist_directory=str(self._chroma_dir),
                embedding_function=self._embeddings,
                collection_name=self.COLLECTION_NAME,
            )
        return self._vectorstore

    @staticmethod
    def _parse_pdf_bytes(pdf_bytes: bytes, source_name: str) -> list[Document]:
        """Ekstrak teks dari PDF bytes → list[Document] (satu Document per halaman)."""
        reader    = PdfReader(io.BytesIO(pdf_bytes))
        documents: list[Document] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": source_name, "page": page_num},
                ))
        logger.debug("Parsed '%s': %d halaman dengan teks.", source_name, len(documents))
        return documents
