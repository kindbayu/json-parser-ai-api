"""
services/rag_service.py
=======================
Legacy General-Purpose RAG Service — PT Multisari Indoprima
Stack v2: HuggingFaceEmbeddings + ChatOllama (100% Gratis, Tanpa OpenAI)

Dipertahankan untuk endpoint /rag/* (ingest, query, list/delete collections).
"""

from __future__ import annotations

import logging
import os
from typing import List

import chromadb
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self) -> None:
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        ollama_url      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model    = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")

        logger.info("RAGService (legacy) — embedding: %s, llm: %s", embedding_model, ollama_model)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_url,
            temperature=0.2,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

    def ingest_documents(self, documents: List[Document], collection_name: str = "default") -> int:
        chunks = self.text_splitter.split_documents(documents)
        Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name,
        )
        return len(chunks)

    def query(self, question: str, collection_name: str = "default", k: int = 4) -> dict:
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True,
        )
        result = qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "source_documents": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in result.get("source_documents", [])
            ],
        }

    def list_collections(self) -> List[str]:
        client = chromadb.PersistentClient(path=self.persist_directory)
        return [col.name for col in client.list_collections()]

    def delete_collection(self, collection_name: str) -> bool:
        client = chromadb.PersistentClient(path=self.persist_directory)
        client.delete_collection(collection_name)
        return True
