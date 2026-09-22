"""
app/services/msds_rag.py
========================
Document RAG Service — Universal AI Engine
Stack: HuggingFaceEmbeddings (all-MiniLM-L6-v2) + LLM via factory (Groq or Ollama)

Workflow:
  Ingest (auto on startup / on-demand):
    1. Read PDF from data/sample_docs.pdf using pypdf
    2. Split with RecursiveCharacterTextSplitter (chunk=1000, overlap=200)
    3. Embed with HuggingFaceEmbeddings all-MiniLM-L6-v2 (local ~90 MB)
    4. Store in ChromaDB persistent (./data/chroma_db)

  Query (every request):
    1. ChromaDB similarity search → top-k relevant chunks
    2. Build context prompt
    3. Generate answer with LLM (Groq or Ollama)
    4. Return answer + source_documents

Dependencies:
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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from pypdf import PdfReader
from app.services.llm_factory import get_llm, get_provider_name

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
# RAG Prompt
# ---------------------------------------------------------------------------

_RAG_PROMPT_TEMPLATE = """\
You are a helpful assistant that answers questions strictly based on the provided document context.

INSTRUCTIONS:
- Answer ONLY based on the context provided below.
- If the information is not available in the context, clearly state:
  "This information is not available in the loaded documents."
- Do not fabricate or add information outside the context.
- Be clear, concise, and professional.

--- DOCUMENT CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:\
"""

_RAG_PROMPT = PromptTemplate(
    template=_RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


# ---------------------------------------------------------------------------
# DocumentRagService
# ---------------------------------------------------------------------------

class MSDSRagService:
    """
    Universal RAG service for any PDF document using:
    - HuggingFaceEmbeddings (all-MiniLM-L6-v2) — local, free embeddings
    - LLM via factory (Groq or Ollama)          — configurable via .env
    - ChromaDB (persistent)                     — local vector store

    Singleton-friendly: initialize once at startup, call query() per request.
    """

    DEFAULT_MSDS_PDF   = Path(__file__).resolve().parents[2] / "data" / "sample_docs.pdf"
    DEFAULT_CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma_db"
    COLLECTION_NAME    = "documents"

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

        logger.info("Loading HuggingFaceEmbeddings: %s", embedding_model)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # LLM via factory (Groq or Ollama based on LLM_PROVIDER in .env)
        self._llm = get_llm(temperature=0.1)
        logger.info("DocumentRagService ready — provider: %s", get_provider_name())

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Vectorstore — lazy-loaded
        self._vectorstore: Chroma | None = None

        # Auto-ingest on startup (non-fatal if it fails)
        try:
            self._ensure_ingested()
        except Exception as exc:
            logger.warning(
                "Auto-ingest failed — will retry on first query. Detail: %s", exc
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str, k: int = 4) -> RAGQueryResult:
        """
        Answer a question based on ingested documents.

        Parameters
        ----------
        question : str
            Question in any language.
        k : int
            Number of context chunks to retrieve from ChromaDB (1–10).

        Returns
        -------
        RAGQueryResult
            Dict with 'answer' (str) and 'source_documents' (list).
        """
        if not self.collection_exists():
            logger.info("Collection empty — retrying ingest before query...")
            self._ensure_ingested()

        if not self.collection_exists():
            raise RuntimeError(
                "Document collection not found in ChromaDB. "
                "Ensure a PDF is available in data/ and call ingest_pdf()."
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

        logger.info("RAG query: %s", question[:120])
        try:
            result = qa_chain.invoke({"query": question})
        except Exception as exc:
            raise ConnectionError(
                f"Failed to get answer from LLM. "
                f"Check LLM_PROVIDER configuration. Detail: {exc}"
            ) from exc

        source_docs: list[SourceDocument] = [
            SourceDocument(content=doc.page_content, metadata=doc.metadata)
            for doc in result.get("source_documents", [])
        ]

        logger.info("RAG query done — %d source chunks used.", len(source_docs))
        return RAGQueryResult(
            answer=result["result"],
            source_documents=source_docs,
        )

    def ingest_pdf(self, pdf_bytes: bytes, source_name: str = "document.pdf") -> int:
        """
        Ingest a PDF into ChromaDB using HuggingFace embeddings.

        Returns the number of chunks stored.
        """
        documents = self._parse_pdf_bytes(pdf_bytes, source_name)
        if not documents:
            raise ValueError(
                f"No extractable text found in '{source_name}'. "
                "Ensure the PDF is not a scanned image without OCR."
            )

        chunks = self._text_splitter.split_documents(documents)
        logger.info("Ingesting %d chunks from '%s' into collection '%s'...",
                    len(chunks), source_name, self.COLLECTION_NAME)

        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=str(self._chroma_dir),
            collection_name=self.COLLECTION_NAME,
        )

        logger.info("Ingest complete — %d chunks stored in ChromaDB.", len(chunks))
        return len(chunks)

    def collection_exists(self) -> bool:
        """Check if the document collection exists and is non-empty in ChromaDB."""
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
        """Auto-ingest sample_docs.pdf if collection doesn't exist."""
        if self.collection_exists():
            count = (chromadb.PersistentClient(path=str(self._chroma_dir))
                             .get_collection(self.COLLECTION_NAME).count())
            logger.info("Collection '%s' exists (%d chunks) — skipping auto-ingest.",
                        self.COLLECTION_NAME, count)
            return

        if not self._msds_pdf_path.exists():
            logger.warning(
                "Sample document not found at '%s'. Run: python data_gen.py",
                self._msds_pdf_path,
            )
            return

        logger.info("Auto-ingesting '%s' → ChromaDB collection '%s'...",
                    self._msds_pdf_path.name, self.COLLECTION_NAME)
        pdf_bytes     = self._msds_pdf_path.read_bytes()
        chunks_stored = self.ingest_pdf(pdf_bytes, source_name=self._msds_pdf_path.name)
        logger.info("Auto-ingest complete: %d chunks.", chunks_stored)

    def _get_vectorstore(self) -> Chroma:
        """Lazy-load ChromaDB vectorstore with HuggingFace embeddings."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                persist_directory=str(self._chroma_dir),
                embedding_function=self._embeddings,
                collection_name=self.COLLECTION_NAME,
            )
        return self._vectorstore

    @staticmethod
    def _parse_pdf_bytes(pdf_bytes: bytes, source_name: str) -> list[Document]:
        """Extract text from PDF bytes → list[Document] (one per page)."""
        reader    = PdfReader(io.BytesIO(pdf_bytes))
        documents: list[Document] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": source_name, "page": page_num},
                ))
        logger.debug("Parsed '%s': %d pages with text.", source_name, len(documents))
        return documents
