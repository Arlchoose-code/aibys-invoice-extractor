from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import base64
import csv
import io
import json
import os
import re

import fitz  # PyMuPDF
import requests


ROOT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT_DIR / "uploads"
DATA_DIR = ROOT_DIR / "data"
JSON_DB = DATA_DIR / "invoices.json"
CSV_DB = DATA_DIR / "invoices.csv"

for directory in ("static", "templates"):
    (ROOT_DIR / directory).mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Aibys Invoice Extractor")

app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")
templates = Jinja2Templates(directory=ROOT_DIR / "templates")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")

ALLOWED_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/webp",
}
MAX_FILE_SIZE = 20 * 1024 * 1024

EXTRACT_PROMPT = """You are an expert invoice and receipt data extractor.
Analyze the provided document image and extract ALL information into a structured JSON format.

Return ONLY valid JSON, no markdown, no explanation. Use this exact structure:

{
  "document_type": "invoice" | "receipt" | "purchase_order" | "unknown",
  "vendor": {
    "name": "",
    "address": "",
    "phone": "",
    "email": "",
    "website": ""
  },
  "customer": {
    "name": "",
    "address": "",
    "phone": "",
    "email": ""
  },
  "document_info": {
    "invoice_number": "",
    "date": "",
    "due_date": "",
    "payment_method": ""
  },
  "items": [
    {
      "description": "",
      "quantity": "",
      "unit_price": "",
      "total": ""
    }
  ],
  "summary": {
    "subtotal": "",
    "tax": "",
    "discount": "",
    "shipping": "",
    "total": "",
    "currency": ""
  },
  "notes": ""
}

If a field is not found, use empty string "". Extract all line items you can find."""


CSV_FIELDS = [
    "record_id",
    "extracted_at",
    "source_file",
    "page_number",
    "document_type",
    "vendor",
    "customer",
    "invoice_number",
    "date",
    "due_date",
    "payment_method",
    "item_description",
    "quantity",
    "unit_price",
    "line_total",
    "subtotal",
    "tax",
    "discount",
    "shipping",
    "grand_total",
    "currency",
    "notes",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename).strip("._")
    return cleaned or "upload"


def load_records() -> list[dict[str, Any]]:
    if not JSON_DB.exists():
        return []
    try:
        with JSON_DB.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def save_records(records: list[dict[str, Any]]) -> None:
    temp_path = JSON_DB.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    temp_path.replace(JSON_DB)


def pdf_to_images(pdf_bytes: bytes) -> list[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        images.append(pix.tobytes("png"))
    return images


def image_to_png(image_bytes: bytes) -> bytes:
    try:
        doc = fitz.open(stream=image_bytes, filetype="image")
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
        if max(pix.width, pix.height) > 2048:
            scale = 2048 / max(pix.width, pix.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        return pix.tobytes("png")
    except Exception:
        return image_bytes


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_with_ollama(image_b64: str) -> dict[str, Any]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": EXTRACT_PROMPT,
        "images": [image_b64],
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        raw = result.get("response", "{}").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("Expected object", raw, 0)
        return parsed
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail="Model returned invalid JSON. Try a different model or file.",
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama at {OLLAMA_URL}. Make sure Ollama is running.",
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Ollama took too long to respond.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def flatten_record_for_csv(record: dict[str, Any]) -> list[dict[str, str]]:
    data = record.get("data", {})
    items = data.get("items") or []
    vendor = data.get("vendor") or {}
    customer = data.get("customer") or {}
    doc_info = data.get("document_info") or {}
    summary = data.get("summary") or {}

    base = {
        "record_id": record.get("id", ""),
        "extracted_at": record.get("extracted_at", ""),
        "source_file": record.get("source_file", ""),
        "page_number": str(record.get("page_number", "")),
        "document_type": data.get("document_type", ""),
        "vendor": vendor.get("name", ""),
        "customer": customer.get("name", ""),
        "invoice_number": doc_info.get("invoice_number", ""),
        "date": doc_info.get("date", ""),
        "due_date": doc_info.get("due_date", ""),
        "payment_method": doc_info.get("payment_method", ""),
        "subtotal": summary.get("subtotal", ""),
        "tax": summary.get("tax", ""),
        "discount": summary.get("discount", ""),
        "shipping": summary.get("shipping", ""),
        "grand_total": summary.get("total", ""),
        "currency": summary.get("currency", ""),
        "notes": data.get("notes", ""),
    }

    rows = []
    for item in items:
        rows.append(
            {
                **base,
                "item_description": item.get("description", ""),
                "quantity": item.get("quantity", ""),
                "unit_price": item.get("unit_price", ""),
                "line_total": item.get("total", ""),
            }
        )

    if not rows:
        rows.append(
            {
                **base,
                "item_description": "",
                "quantity": "",
                "unit_price": "",
                "line_total": "",
            }
        )

    return [{field: str(row.get(field, "")) for field in CSV_FIELDS} for row in rows]


def append_records_to_csv(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    write_header = not CSV_DB.exists() or CSV_DB.stat().st_size == 0
    with CSV_DB.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerows(flatten_record_for_csv(record))


def csv_bytes_for_records(records: list[dict[str, Any]]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for record in records:
        writer.writerows(flatten_record_for_csv(record))
    return output.getvalue().encode("utf-8-sig")


async def extract_upload(file: UploadFile) -> list[dict[str, Any]]:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"{file.filename}: only PDF, JPG, PNG, and WEBP files are supported.",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"{file.filename}: file too large. Max 20MB.")

    stored_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}_{safe_filename(file.filename or 'upload')}"
    (UPLOAD_DIR / stored_name).write_bytes(content)

    if file.content_type == "application/pdf":
        images = pdf_to_images(content)
        if not images:
            raise HTTPException(status_code=422, detail=f"{file.filename}: could not extract PDF pages.")
    else:
        images = [image_to_png(content)]

    records = []
    for index, image_bytes in enumerate(images, start=1):
        extracted = extract_with_ollama(image_to_base64(image_bytes))
        records.append(
            {
                "id": uuid4().hex,
                "source_file": file.filename or stored_name,
                "stored_file": stored_name,
                "page_number": index,
                "page_count": len(images),
                "extracted_at": now_iso(),
                "data": extracted,
            }
        )
    return records


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [model["name"] for model in response.json().get("models", [])]
        return {"status": "ok", "ollama": "connected", "models": models}
    except Exception:
        return {"status": "ok", "ollama": "disconnected", "models": []}


@app.get("/records")
async def records():
    return {"success": True, "records": load_records()}


@app.post("/extract")
async def extract(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one file.")

    new_records: list[dict[str, Any]] = []
    for file in files:
        new_records.extend(await extract_upload(file))

    all_records = load_records()
    all_records.extend(new_records)
    save_records(all_records)
    append_records_to_csv(new_records)

    return JSONResponse(content={"success": True, "records": new_records, "total_records": len(all_records)})


@app.post("/export-csv")
async def export_csv(request: Request):
    body = await request.json()
    if "data" in body:
        record = {
            "id": body.get("id") or uuid4().hex,
            "source_file": body.get("source_file", "manual_export"),
            "page_number": body.get("page_number", 1),
            "extracted_at": body.get("extracted_at") or now_iso(),
            "data": body.get("data") or {},
        }
        csv_bytes = csv_bytes_for_records([record])
        filename = f"invoice_{record['id']}.csv"
    else:
        records = load_records()
        csv_bytes = csv_bytes_for_records(records)
        filename = "aibys_invoices.csv"

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/export-csv")
async def export_saved_csv():
    if CSV_DB.exists() and CSV_DB.stat().st_size > 0:
        return FileResponse(CSV_DB, media_type="text/csv", filename="aibys_invoices.csv")

    records = load_records()
    return StreamingResponse(
        io.BytesIO(csv_bytes_for_records(records)),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=aibys_invoices.csv"},
    )


@app.get("/export-json")
async def export_json():
    if not JSON_DB.exists():
        save_records([])
    return FileResponse(JSON_DB, media_type="application/json", filename="aibys_invoices.json")
