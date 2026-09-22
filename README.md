---
title: Indoprima AI API
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# PT Multisari Indoprima — AI Engine API

REST API microservice for B2B operations:

- **POST /api/v1/extract-po** — Extract structured data from Purchase Order PDFs
- **POST /api/v1/query-msds** — Answer MSDS safety questions via RAG

**Swagger UI:** `/docs`

**Stack:** FastAPI · LangChain · ChromaDB · Groq · HuggingFace Embeddings · Pydantic v2
