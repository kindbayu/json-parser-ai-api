"""
test_api.py — Manual API Test Script
=====================================
Tests all endpoints of the AI Engine API.

Run:
    python test_api.py

Make sure the server is running on port 8000:
    venv\\Scripts\\uvicorn.exe main:app --host 127.0.0.1 --port 8000
"""

import json
import sys
import time
from pathlib import Path

import requests

BASE_URL   = "http://127.0.0.1:8000"
PO_PDF     = Path(__file__).parent / "data" / "sample_po.pdf"
MSDS_PDF   = Path(__file__).parent / "data" / "sample_docs.pdf"
PASS = "✅ PASS"
FAIL = "❌ FAIL"
SEP  = "─" * 60


def print_result(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    print(f"{status}  {name}")
    if detail:
        print(f"       {detail}")


def pretty(data: dict | list, max_items: int = 3) -> str:
    """Print JSON truncating long lists."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(v, list) and len(v) > max_items:
                out[k] = v[:max_items] + [f"... (+{len(v)-max_items} more)"]
            elif isinstance(v, str) and len(v) > 200:
                out[k] = v[:200] + "..."
            else:
                out[k] = v
        return json.dumps(out, ensure_ascii=False, indent=2)
    return json.dumps(data, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────
# 1. Health Check
# ──────────────────────────────────────────────────────────
def test_health() -> bool:
    print(f"\n{SEP}")
    print("TEST 1 — Health Check  GET /")
    print(SEP)
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        ok = r.status_code == 200 and r.json().get("status") == "ok"
        print_result("GET /", ok, pretty(r.json()))
        return ok
    except Exception as e:
        print_result("GET /", False, str(e))
        return False


# ──────────────────────────────────────────────────────────
# 2. PO Extraction
# ──────────────────────────────────────────────────────────
def test_extract_po() -> bool:
    print(f"\n{SEP}")
    print("TEST 2 — PO Extraction  POST /api/v1/extract-po")
    print(SEP)

    if not PO_PDF.exists():
        print_result("File check", False,
                     f"{PO_PDF} tidak ditemukan. Jalankan: python data_gen.py")
        return False

    print_result("File check", True, str(PO_PDF))
    print("  ⏳ Mengirim PDF ke Ollama untuk ekstraksi... (bisa 10–60 detik)")

    try:
        with open(PO_PDF, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/api/v1/extract-po",
                files={"file": ("sample_po.pdf", f, "application/pdf")},
                timeout=120,
            )
    except requests.exceptions.ConnectionError:
        print_result("POST /api/v1/extract-po", False,
                     "Server tidak dapat dihubungi. Pastikan uvicorn sudah berjalan.")
        return False
    except requests.exceptions.Timeout:
        print_result("POST /api/v1/extract-po", False,
                     "Timeout (>120s). Ollama mungkin lambat — coba lagi.")
        return False

    if r.status_code != 200:
        print_result("POST /api/v1/extract-po", False,
                     f"HTTP {r.status_code}: {r.text[:300]}")
        return False

    data = r.json()
    checks = {
        "po_number ada"   : bool(data.get("po_number")),
        "client_name ada" : bool(data.get("client_name")),
        "items tidak kosong" : len(data.get("items", [])) > 0,
    }
    all_ok = all(checks.values())
    print_result("POST /api/v1/extract-po", all_ok)

    for check, result in checks.items():
        icon = "✅" if result else "❌"
        print(f"       {icon} {check}")

    if data.get("items"):
        print(f"\n       PO Number  : {data.get('po_number')}")
        print(f"       Client     : {data.get('client_name')}")
        print(f"       Items found: {len(data['items'])}")
        print(f"       First item : {data['items'][0]}")

    return all_ok


# ──────────────────────────────────────────────────────────
# 3. MSDS RAG Query
# ──────────────────────────────────────────────────────────
def test_query_msds() -> bool:
    print(f"\n{SEP}")
    print("TEST 3 — MSDS RAG Query  POST /api/v1/query-msds")
    print(SEP)

    questions = [
        "What are the first aid measures for eye contact?",
        "What is the flash point of this substance?",
        "How should this material be stored properly?",
    ]

    all_ok = True
    for i, question in enumerate(questions, 1):
        print(f"\n  Pertanyaan {i}: {question}")
        print("  ⏳ Menunggu jawaban Ollama... (bisa 10–60 detik)")

        try:
            r = requests.post(
                f"{BASE_URL}/api/v1/query-msds",
                json={"question": question, "k": 3},
                timeout=120,
            )
        except requests.exceptions.Timeout:
            print_result(f"  Query {i}", False, "Timeout (>120s)")
            all_ok = False
            continue
        except Exception as e:
            print_result(f"  Query {i}", False, str(e))
            all_ok = False
            continue

        if r.status_code != 200:
            print_result(f"  Query {i}", False,
                         f"HTTP {r.status_code}: {r.text[:200]}")
            all_ok = False
            continue

        data    = r.json()
        answer  = data.get("answer", "")
        sources = data.get("source_documents", [])
        ok      = bool(answer) and len(sources) > 0

        print_result(f"  Query {i}", ok)
        print(f"       Jawaban   : {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"       Sources   : {len(sources)} chunk(s) dari ChromaDB")

        if not ok:
            all_ok = False

    return all_ok


# ──────────────────────────────────────────────────────────
# 4. PDF Parse (legacy endpoint)
# ──────────────────────────────────────────────────────────
def test_pdf_parse() -> bool:
    print(f"\n{SEP}")
    print("TEST 4 — PDF Parse (legacy)  POST /pdf/parse")
    print(SEP)

    if not MSDS_PDF.exists():
        print_result("File check", False,
                     f"{MSDS_PDF} tidak ditemukan. Jalankan: python data_gen.py")
        return False

    try:
        with open(MSDS_PDF, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/pdf/parse",
                files={"file": ("sample_docs.pdf", f, "application/pdf")},
                timeout=15,
            )
    except Exception as e:
        print_result("POST /pdf/parse", False, str(e))
        return False

    ok = r.status_code == 200 and len(r.json().get("pages", [])) > 0
    data = r.json()
    print_result("POST /pdf/parse", ok,
                 f"{data.get('metadata', {}).get('total_pages', '?')} halaman diekstrak")
    return ok


# ──────────────────────────────────────────────────────────
# 5. List ChromaDB Collections
# ──────────────────────────────────────────────────────────
def test_list_collections() -> bool:
    print(f"\n{SEP}")
    print("TEST 5 — List Collections  GET /rag/collections")
    print(SEP)

    try:
        r = requests.get(f"{BASE_URL}/rag/collections", timeout=5)
        ok = r.status_code == 200
        print_result("GET /rag/collections", ok,
                     f"Collections: {r.json().get('collections', [])}")
        return ok
    except Exception as e:
        print_result("GET /rag/collections", False, str(e))
        return False


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "═" * 60)
    print("  AI Engine API — Universal Document Intelligence")
    print("  Stack: Groq / Ollama + HuggingFace + ChromaDB")
    print("═" * 60)

    start = time.time()

    results = {
        "Health Check"       : test_health(),
        "PDF Parse (legacy)" : test_pdf_parse(),
        "List Collections"   : test_list_collections(),
        "PO Extraction"      : test_extract_po(),
        "MSDS RAG Query"     : test_query_msds(),
    }

    elapsed = time.time() - start

    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print("═" * 60)
    for name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {name}")

    passed = sum(results.values())
    total  = len(results)
    print(f"\n  {passed}/{total} tests passed  ({elapsed:.1f}s total)")
    print("═" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
