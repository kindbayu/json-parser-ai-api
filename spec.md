# Spec: AI Engine Microservice — PT Multisari Indoprima
## Versi 2.0 — Stack 100% Gratis (Lokal, Tanpa OpenAI)

---

## 1. Tujuan Sistem

REST API microservice berbasis **FastAPI** untuk mendukung operasional B2B PT Multisari Indoprima dalam dua fungsi utama:

1. **Purchase Order Parsing** — Mengekstrak data terstruktur dari file PDF Purchase Order yang dikirimkan mitra bisnis (klien B2B), menghasilkan JSON siap pakai untuk sistem ERP/WMS.
2. **MSDS RAG Assistant** — Menjawab pertanyaan teknis seputar Material Safety Data Sheet (MSDS) bahan wewangian dan kimia menggunakan pendekatan Retrieval-Augmented Generation (RAG).

### Prinsip Arsitektur v2
- **100% Gratis & Lokal** — tidak ada API berbayar, tidak ada koneksi ke layanan eksternal saat runtime.
- **LLM Engine**: [Ollama](https://ollama.com) dengan model `llama3.2` berjalan lokal di mesin.
- **Embeddings**: `HuggingFaceEmbeddings` model `all-MiniLM-L6-v2` via `sentence-transformers` — berjalan lokal, tanpa API key.
- **Vector DB**: ChromaDB (persistent, lokal).
- **Web Framework**: FastAPI + Uvicorn.

---

## 2. Perbandingan Stack v1 vs v2

| Komponen        | v1 (OpenAI — Berbayar)              | v2 (Gratis — Lokal)                     |
|-----------------|-------------------------------------|-----------------------------------------|
| LLM Engine      | `ChatOpenAI` (GPT-4o-mini)          | `ChatOllama` (llama3.2 via Ollama)      |
| Embeddings      | `OpenAIEmbeddings`                  | `HuggingFaceEmbeddings` (MiniLM-L6-v2) |
| API Key         | Wajib (`OPENAI_API_KEY`)            | Tidak diperlukan                        |
| Biaya Runtime   | Per-token (berbayar)                | Gratis (CPU/GPU lokal)                  |
| Koneksi Internet| Diperlukan saat inference           | Tidak diperlukan setelah download model |
| Kecepatan       | Cepat (cloud GPU)                   | Tergantung hardware lokal               |

---

## 3. Struktur Folder Proyek

```
indoprima/
├── app/
│   ├── __init__.py
│   └── services/
│       ├── __init__.py
│       ├── po_parser.py        # PO Extraction via Ollama LLM
│       └── msds_rag.py         # RAG Service via HuggingFace + ChromaDB + Ollama
├── data/
│   ├── sample_po.pdf           # Contoh PDF Purchase Order (B2B)
│   ├── msds_sample.pdf         # Contoh PDF MSDS bahan wewangian
│   └── chroma_db/              # ChromaDB persistent storage (auto-generated)
├── temp/                       # Folder upload sementara (auto-cleanup)
├── services/                   # Legacy services (opsional, dipertahankan)
│   ├── __init__.py
│   ├── rag_service.py
│   └── pdf_service.py
├── main.py                     # FastAPI entry point & router
├── data_gen.py                 # Script generasi PDF sample
├── requirements.txt            # Dependencies Python (v2 — tanpa OpenAI)
├── .env                        # Config (tidak butuh OPENAI_API_KEY)
└── .env.example
```

---

## 4. Spesifikasi Endpoint

### 4.1 `POST /api/v1/extract-po`

Mengekstrak data terstruktur dari file PDF Purchase Order.

**Request**
- Content-Type: `multipart/form-data`
- Field: `file` (PDF, maks. 10 MB)

**Response `200 OK`**
```json
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
      "unit_price": 85000.0,
      "total_price": 4250000.0
    }
  ],
  "total_amount": 4250000.0,
  "currency": "IDR",
  "notes": "Pengiriman ke gudang utama, harap sertakan CoA."
}
```

**Response `422`** — PDF tidak mengandung teks yang dapat diparse.  
**Response `400`** — File bukan PDF atau kosong.  
**Response `503`** — Ollama tidak berjalan atau model belum diunduh.

---

### 4.2 `POST /api/v1/query-msds`

Menjawab pertanyaan teknis berdasarkan dokumen MSDS.

**Request**
```json
{
  "question": "Apa tindakan pertolongan pertama jika terjadi kontak mata dengan Linalool?",
  "k": 4
}
```

| Field      | Tipe    | Wajib | Default | Keterangan                        |
|------------|---------|-------|---------|-----------------------------------|
| `question` | string  | ✅    | —       | Pertanyaan (min. 5 karakter)      |
| `k`        | integer | ❌    | `4`     | Jumlah chunk konteks (1–10)       |

**Response `200 OK`**
```json
{
  "answer": "Jika Linalool mengenai mata, segera bilas dengan air mengalir selama minimal 15 menit...",
  "source_documents": [
    {
      "content": "Section 4 - First Aid Measures...",
      "metadata": { "source": "msds_sample.pdf", "page": 3 }
    }
  ]
}
```

---

## 5. Alur Kerja

### PO Parsing Flow (v2)
```
Client → POST /api/v1/extract-po (PDF)
       → Simpan ke temp/ (uuid filename)
       → pypdf: ekstrak teks per halaman
       → ChatOllama (llama3.2): JSON extraction via structured prompt
       → Parse output JSON dengan Pydantic v2
       → Hapus file temp (finally block)
       → Return POExtractResponse
```

### MSDS RAG Flow (v2)
```
[Ingest — auto saat startup / on-demand]
       → pypdf: parse msds_sample.pdf per halaman
       → RecursiveCharacterTextSplitter (chunk=1000, overlap=200)
       → HuggingFaceEmbeddings (all-MiniLM-L6-v2) → ChromaDB (persistent)

[Query — setiap request]
Client → POST /api/v1/query-msds
       → ChromaDB similarity search (top-k chunks)
       → LangChain RetrievalQA dengan ChatOllama (llama3.2)
       → Return answer + source_documents
```

---

## 6. Model Pydantic v2

### POItem
```python
class POItem(BaseModel):
    line_no:     int
    item_code:   str
    description: str
    quantity:    float
    unit:        str
    unit_price:  float
    total_price: float
```

### POExtractResponse
```python
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
```

### MSDSQueryRequest
```python
class MSDSQueryRequest(BaseModel):
    question: str = Field(..., min_length=5)
    k: int = Field(default=4, ge=1, le=10)
```

---

## 7. Dependensi Python (v2)

```
fastapi==0.115.12
uvicorn[standard]==0.34.3
pydantic==2.11.5
langchain==0.3.25
langchain-community==0.3.24
chromadb==0.6.3
pypdf==5.4.0
sentence-transformers==6.0.1
python-dotenv==1.1.0
python-multipart==0.0.20
reportlab==4.4.1
```

> `langchain-openai` **tidak diperlukan** di v2.

---

## 8. Environment Variables (v2)

| Variable                   | Keterangan                             | Default / Contoh          |
|----------------------------|----------------------------------------|---------------------------|
| `OLLAMA_BASE_URL`          | URL Ollama server lokal                | `http://localhost:11434`  |
| `OLLAMA_MODEL`             | Nama model Ollama untuk LLM            | `llama3.2`                |
| `EMBEDDING_MODEL`          | Model HuggingFace untuk embeddings     | `all-MiniLM-L6-v2`        |
| `CHROMA_PERSIST_DIRECTORY` | Path penyimpanan ChromaDB              | `./data/chroma_db`        |
| `APP_TITLE`                | Nama aplikasi                          | `AI Engine API`           |
| `APP_VERSION`              | Versi API                              | `2.0.0`                   |
| `DEBUG`                    | Mode debug uvicorn                     | `False`                   |

> **Tidak ada** `OPENAI_API_KEY` di v2.

---

## 9. Prasyarat Instalasi Ollama

Ollama harus terinstall dan berjalan di mesin sebelum API dijalankan.

```powershell
# 1. Download & install Ollama dari https://ollama.com/download
# 2. Pull model llama3.2 (±2 GB, sekali saja)
ollama pull llama3.2

# 3. Verifikasi Ollama berjalan
ollama list
# Seharusnya menampilkan: llama3.2 ...

# 4. Tes langsung dari terminal (opsional)
ollama run llama3.2 "Halo, apa kabar?"
```

Ollama berjalan sebagai background service otomatis setelah install.  
API endpoint Ollama: `http://localhost:11434`

---

## 10. Catatan Teknis

- `HuggingFaceEmbeddings` mengunduh model `all-MiniLM-L6-v2` (~90 MB) sekali ke cache lokal (`~/.cache/huggingface/`). Setelah itu **offline sepenuhnya**.
- ChromaDB menggunakan `PersistentClient` — data embedding tidak hilang saat restart.
- PO parsing menggunakan **prompt engineering** karena `ChatOllama` tidak mendukung `with_structured_output()` seperti OpenAI. Output JSON di-parse manual dengan fallback error handling.
- API dirancang stateless dan siap untuk containerisasi Docker.
