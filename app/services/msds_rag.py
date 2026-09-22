"""
app/services/msds_rag.py
========================
Document RAG Service — Universal AI Engine
Stack: ChromaDB default embedding (onnxruntime) + LLM via factory (Groq or Ollama)

Railway-compatible: uses in-memory ChromaDB so ephemeral filesystem is not an issue.
Data is re-ingested on every startup from the embedded sample document.

Workflow:
  Ingest (auto on every startup):
    1. Generate sample document text (no file dependency)
    2. Split with RecursiveCharacterTextSplitter (chunk=500, overlap=50)
    3. Embed with ChromaDB default embedding (onnxruntime, no PyTorch)
    4. Store in ChromaDB (in-memory on Railway, persistent locally)

  Query (every request):
    1. ChromaDB similarity search → top-k relevant chunks
    2. Build context prompt
    3. Generate answer with LLM (Groq or Ollama)
    4. Return answer + source_documents
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import TypedDict

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from pypdf import PdfReader
from app.services.llm_factory import get_llm, get_provider_name

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedded sample document text (fallback when no PDF is available)
# This ensures Railway startup always succeeds regardless of filesystem state
# ---------------------------------------------------------------------------

_SAMPLE_DOCUMENT_TEXT = """
MATERIAL SAFETY DATA SHEET — Linalool Fragrance Compound LZ-4412
Supplier: LUZI AG | Revision: 3.1 | Date: 01 August 2026

SECTION 1 — IDENTIFICATION
Product Name: Linalool Fragrance Compound LZ-4412
Uses: Perfume raw material, cosmetics, personal care products, aromatherapy
Emergency Contact: +41 61 486 56 00 (24 hours)

SECTION 2 — HAZARD IDENTIFICATION
GHS Classification: Skin Sensitizer Cat. 1B (H317), Aquatic Chronic Cat. 3 (H412)
H317: May cause an allergic skin reaction.
H412: Harmful to aquatic life with long lasting effects.
P261: Avoid breathing vapors or sprays.
P280: Wear protective gloves and eye protection.
P302+P352: IF ON SKIN: Wash with plenty of soap and water.
P333+P313: If skin irritation or rash occurs, seek medical advice.
P273: Avoid release to the environment.
P501: Dispose of contents and container in accordance with local regulations.

SECTION 3 — COMPOSITION
Main component: Linalool (CAS 78-70-6) >= 95%
Secondary: Linalyl Acetate (CAS 115-95-7) 2-4%
Minor: Alpha-Terpineol (CAS 98-55-5) <= 1%
Trace: Geraniol (CAS 106-24-1) <= 0.5%

SECTION 4 — FIRST AID MEASURES
Eye contact: Immediately flush eyes with running water for at least 15 minutes while
holding eyelids open. Remove contact lenses if present and easy to do.
Seek immediate medical attention if irritation persists.
Skin contact: Wash affected skin with soap and water for at least 10 minutes.
Remove contaminated clothing and shoes. Consult a doctor if irritation or rash develops.
Inhalation: Move to fresh air. Rest in a position comfortable for breathing.
If breathing difficulty occurs, give oxygen. Seek medical attention immediately.
Ingestion: Do NOT induce vomiting. Rinse mouth with water.
If conscious, give 1-2 glasses of water. Seek immediate medical attention.

SECTION 5 — FIRE FIGHTING MEASURES
Flash Point: 77 degrees Celsius (Pensky-Martens Closed Cup method)
Auto-ignition Temperature: 235 degrees Celsius
Flammable Limits: LEL 0.9%, UEL 6.5%
Suitable extinguishing media: CO2, dry chemical powder, foam, water mist
Unsuitable media: Direct water jet at high pressure
Use self-contained breathing apparatus when fighting fire.
Cool containers exposed to flames with water spray.

SECTION 6 — ACCIDENTAL RELEASE MEASURES
Personal protection: Wear full PPE including nitrile gloves, safety goggles, N95 mask.
Keep away from heat sources and open flames — product is flammable.
Ensure adequate ventilation in the spill area.
Cleanup: Absorb spill with inert material such as sand or vermiculite.
Collect absorbed material in labeled containers for disposal.
Do not discharge into drains, soil, or water bodies.

SECTION 7 — HANDLING AND STORAGE
Handling: Avoid contact with skin, eyes and clothing. Avoid inhaling vapors.
Use only in well-ventilated areas. Keep away from heat, flames and oxidizing agents.
Storage Temperature: 15-25 degrees Celsius in a cool dry place.
Container: Keep in original tightly closed container. Avoid reactive metal containers.
Shelf Life: 24 months from date of manufacture when stored as recommended.

SECTION 8 — EXPOSURE CONTROLS AND PERSONAL PROTECTION
Exposure Limit: Linalool has no national threshold value — use 10 ppm (TWA) as guidance.
Respiratory protection: Half-face mask with A2/P2 combination filter if ventilation inadequate.
Hand protection: Nitrile gloves >= 0.2mm thickness, replace every 2 hours or when damaged.
Eye protection: Splash-proof safety goggles or face shield.
Body protection: Chemical resistant work clothing.

SECTION 9 — PHYSICAL AND CHEMICAL PROPERTIES
Appearance: Clear, colorless to slightly pale yellow liquid
Odor: Fresh floral, resembling lavender with woody notes
Boiling Point: 198-199 degrees Celsius at 1 atm
Melting Point: below -20 degrees Celsius
Flash Point: 77 degrees Celsius (Closed Cup)
Water Solubility: 1.7 g/L at 20 degrees Celsius
Solubility in solvents: Completely soluble in ethanol, diethyl ether, chloroform
Density: 0.858-0.868 g/mL at 20 degrees Celsius
Refractive Index: 1.462-1.466 at 20 degrees Celsius
Vapor Pressure: 0.16 hPa at 20 degrees Celsius
Partition Coefficient Log P: 2.97 (octanol/water)

SECTION 11 — TOXICOLOGICAL INFORMATION
Acute Oral Toxicity LD50: > 2790 mg/kg (rat) — Not classified
Acute Dermal Toxicity LD50: > 5000 mg/kg (rabbit) — Not classified
Inhalation Toxicity LC50: > 5 mg/L vapor, 4h, rat — Not classified
Skin sensitization: Known skin sensitizer, LLNA test positive at 5%
Eye irritation: Mild irritant, does not cause permanent eye damage
Carcinogenicity: Not classified as carcinogen by IARC, NTP or OSHA
Reproductive effects: No evidence of significant teratogenic effects

SECTION 12 — ECOLOGICAL INFORMATION
Aquatic Toxicity (Fish): LC50 Oncorhynchus mykiss 96h = 27 mg/L
Aquatic Toxicity (Daphnia): EC50 Daphnia magna 48h = 16 mg/L
Algae Toxicity: ErC50 Pseudokirchneriella subcapitata 72h = 8.6 mg/L
Persistence: Readily biodegradable — > 70% in 28 days (OECD 301B)
Bioaccumulation Potential: Log Pow = 2.97 — Low bioaccumulation potential

SECTION 13 — DISPOSAL CONSIDERATIONS
Dispose in accordance with national and local regulations for chemical waste.
Do not dispose into drains, rivers or soil.
EU Waste Code: 14 06 03* — other solvents and solvent mixtures
Use a licensed hazardous waste disposal contractor.

SECTION 15 — REGULATORY INFORMATION
EU Regulations: REACH (EC 1907/2006), CLP (EC 1272/2008)
IFRA: Linalool listed in IFRA Transparency List — included in EU 26 allergens list
Indonesia: BPOM Regulation No. 23 Year 2019 on Technical Requirements for Cosmetic Ingredients
TSCA: Listed in TSCA Inventory (USA)
""".strip()


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
    Universal RAG service for any PDF document.

    Uses ChromaDB's built-in default embedding (onnxruntime-based, ~50 MB)
    instead of sentence-transformers (PyTorch, ~800 MB) — Railway compatible.

    Uses in-memory ChromaDB so ephemeral filesystem on Railway is not an issue.
    Falls back to embedded sample text if no PDF is available.
    """

    DEFAULT_PDF_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_docs.pdf"
    COLLECTION_NAME  = "documents"

    def __init__(
        self,
        pdf_path:  Path | str | None = None,
    ) -> None:
        self._pdf_path = Path(pdf_path or self.DEFAULT_PDF_PATH)

        # Use in-memory on Railway, persistent locally
        # Railway injects RAILWAY_ENVIRONMENT_NAME automatically
        is_railway = bool(
            os.getenv("RAILWAY_ENVIRONMENT_NAME") or
            os.getenv("RAILWAY_ENVIRONMENT") or
            os.getenv("RAILWAY_PROJECT_ID")
        )
        if is_railway:
            self._chroma_client = chromadb.EphemeralClient()
            logger.info("ChromaDB: in-memory (Railway mode)")
        else:
            chroma_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", str(self.DEFAULT_CHROMA_DIR))
            Path(chroma_dir).mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=chroma_dir)
            logger.info("ChromaDB: persistent @ %s", chroma_dir)

        # ChromaDB default embedding function (onnxruntime, no PyTorch needed)
        self._ef = embedding_functions.DefaultEmbeddingFunction()

        # LLM via factory
        self._llm = get_llm(temperature=0.1)
        logger.info("DocumentRagService ready — LLM: %s", get_provider_name())

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Ingest on startup
        try:
            self._ensure_ingested()
        except Exception as exc:
            logger.warning("Auto-ingest failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str, k: int = 4) -> RAGQueryResult:
        """Answer a question based on ingested documents."""
        if not self.collection_exists():
            self._ensure_ingested()

        if not self.collection_exists():
            raise RuntimeError(
                "Document collection is empty. "
                "POST a PDF to /rag/ingest first."
            )

        # Use chromadb native query instead of LangChain vectorstore
        # to avoid HuggingFaceEmbeddings dependency
        collection = self._chroma_client.get_collection(
            self.COLLECTION_NAME,
            embedding_function=self._ef,
        )
        results = collection.query(
            query_texts=[question],
            n_results=min(k, collection.count()),
        )

        # Build context from retrieved chunks
        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        context   = "\n\n".join(docs)

        # Build prompt and call LLM directly
        prompt = _RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        from langchain_core.messages import HumanMessage
        logger.info("RAG query: %s", question[:120])
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            answer   = response.content
        except Exception as exc:
            raise ConnectionError(
                f"LLM failed to answer. Check LLM_PROVIDER config. Detail: {exc}"
            ) from exc

        source_docs = [
            SourceDocument(content=doc, metadata=meta)
            for doc, meta in zip(docs, metadatas)
        ]

        logger.info("RAG query done — %d chunks used.", len(source_docs))
        return RAGQueryResult(answer=answer, source_documents=source_docs)

    def ingest_pdf(self, pdf_bytes: bytes, source_name: str = "document.pdf") -> int:
        """Ingest a PDF into ChromaDB. Returns number of chunks stored."""
        documents = self._parse_pdf_bytes(pdf_bytes, source_name)
        if not documents:
            raise ValueError(
                f"No extractable text found in '{source_name}'. "
                "Ensure the PDF is not a scanned image without OCR."
            )
        return self._ingest_documents(documents)

    def collection_exists(self) -> bool:
        """Check if the document collection exists and is non-empty."""
        try:
            cols = [c.name for c in self._chroma_client.list_collections()]
            if self.COLLECTION_NAME not in cols:
                return False
            col = self._chroma_client.get_collection(self.COLLECTION_NAME)
            return col.count() > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_ingested(self) -> None:
        """Auto-ingest PDF or embedded sample text if collection is empty."""
        if self.collection_exists():
            count = self._chroma_client.get_collection(self.COLLECTION_NAME).count()
            logger.info("Collection '%s' ready (%d chunks).", self.COLLECTION_NAME, count)
            return

        if self._pdf_path.exists():
            logger.info("Auto-ingesting '%s'...", self._pdf_path.name)
            pdf_bytes = self._pdf_path.read_bytes()
            n = self.ingest_pdf(pdf_bytes, source_name=self._pdf_path.name)
            logger.info("Ingested %d chunks from PDF.", n)
        else:
            # Fallback: use embedded sample text (always works on Railway)
            logger.info("PDF not found — ingesting embedded sample document...")
            n = self._ingest_text(_SAMPLE_DOCUMENT_TEXT, source="sample_docs_embedded")
            logger.info("Ingested %d chunks from embedded sample.", n)

    def _ingest_documents(self, documents: list[Document]) -> int:
        """Split LangChain Documents and store in ChromaDB."""
        chunks = self._text_splitter.split_documents(documents)
        texts     = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids       = [f"doc_{i}" for i in range(len(chunks))]

        # Delete existing collection and recreate
        try:
            self._chroma_client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        collection = self._chroma_client.create_collection(
            self.COLLECTION_NAME,
            embedding_function=self._ef,
        )
        collection.add(documents=texts, metadatas=metadatas, ids=ids)
        logger.info("Stored %d chunks in collection '%s'.", len(chunks), self.COLLECTION_NAME)
        return len(chunks)

    def _ingest_text(self, text: str, source: str = "embedded") -> int:
        """Split raw text and store in ChromaDB."""
        chunks = self._text_splitter.split_text(text)
        metadatas = [{"source": source, "page": i + 1} for i in range(len(chunks))]
        ids       = [f"doc_{i}" for i in range(len(chunks))]

        try:
            self._chroma_client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        collection = self._chroma_client.create_collection(
            self.COLLECTION_NAME,
            embedding_function=self._ef,
        )
        collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        logger.info("Stored %d chunks from text in '%s'.", len(chunks), self.COLLECTION_NAME)
        return len(chunks)

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
        logger.debug("Parsed '%s': %d pages.", source_name, len(documents))
        return documents
