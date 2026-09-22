# AI Engine API

A universal REST API for intelligent document processing.

## Features

- **POST /api/v1/extract-po** — Extract structured JSON from any Purchase Order PDF
- **POST /api/v1/query-msds** — Answer questions from ingested documents via RAG
- **POST /rag/ingest** — Ingest any PDF into ChromaDB
- **POST /rag/query** — Query any ChromaDB collection

**Swagger UI:** `/docs`

## Stack

- FastAPI · LangChain · ChromaDB
- LLM: Groq (cloud) or Ollama (local)
- Embeddings: HuggingFace `all-MiniLM-L6-v2`
- Pydantic v2 · pypdf

## Quick Start

```bash
cp .env.example .env
# Fill in GROQ_API_KEY at https://console.groq.com (free)

pip install -r requirements.txt
python data_gen.py       # generate sample PDFs
uvicorn main:app --reload
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | `groq` or `ollama` | `groq` |
| `GROQ_API_KEY` | Groq API key | — |
| `GROQ_MODEL` | Groq model name | `llama-3.1-8b-instant` |
| `OLLAMA_MODEL` | Ollama model name | `qwen2.5:1.5b` |
| `EMBEDDING_MODEL` | HuggingFace model | `all-MiniLM-L6-v2` |
| `APP_TITLE` | API title shown in Swagger | `AI Engine API` |
