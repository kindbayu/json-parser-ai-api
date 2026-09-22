"""
services/rag_service.py
=======================
Legacy General-Purpose RAG Service — Universal AI Engine
Stack v3: ChromaDB default embedding (onnxruntime) + LLM via factory (Groq or Ollama)

Retained for /rag/* endpoints (ingest, query, list/delete collections).
"""

from __future__ import annotations

import logging
import os
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from app.services.llm_factory import get_llm, get_provider_name

load_dotenv()
logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self) -> None:
        chroma_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
        logger.info("RAGService (legacy) — llm: %s", get_provider_name())

        # Use in-memory on Railway, persistent locally
        if os.getenv("RAILWAY_ENVIRONMENT"):
            self._client = chromadb.EphemeralClient()
        else:
            os.makedirs(chroma_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=chroma_dir)

        self._ef = embedding_functions.DefaultEmbeddingFunction()
        self.llm = get_llm(temperature=0.2)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

    def ingest_documents(self, documents: List[Document], collection_name: str = "default") -> int:
        chunks    = self.text_splitter.split_documents(documents)
        texts     = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids       = [f"doc_{i}" for i in range(len(chunks))]

        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass

        col = self._client.create_collection(collection_name, embedding_function=self._ef)
        col.add(documents=texts, metadatas=metadatas, ids=ids)
        return len(chunks)

    def query(self, question: str, collection_name: str = "default", k: int = 4) -> dict:
        col     = self._client.get_collection(collection_name, embedding_function=self._ef)
        results = col.query(query_texts=[question], n_results=min(k, col.count()))
        docs    = results["documents"][0]
        metas   = results["metadatas"][0]
        context = "\n\n".join(docs)

        prompt   = f"Answer based on context only:\n\n{context}\n\nQuestion: {question}\nAnswer:"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {
            "answer": response.content,
            "source_documents": [
                {"content": d, "metadata": m} for d, m in zip(docs, metas)
            ],
        }

    def list_collections(self) -> List[str]:
        return [c.name for c in self._client.list_collections()]

    def delete_collection(self, collection_name: str) -> bool:
        self._client.delete_collection(collection_name)
        return True
