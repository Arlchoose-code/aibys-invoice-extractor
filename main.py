from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse, StreamingResponse
import requests
import json
import fitz  # PyMuPDF
import base64
import io
import csv
import os

# Create required dirs before app mounts
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

app = FastAPI(title="Aibys Invoice Extractor")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")

UPLOAD_DIR = "uploads"

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


def pdf_to_images(pdf_bytes: bytes) -> list:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        images.append(img_bytes)
    return images


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_with_ollama(image_b64: str) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": EXTRACT_PROMPT,
        "images": [image_b64],
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        raw = result.get("response", "{}")
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Model returned invalid JSON. Try a different model or file.")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama at {OLLAMA_URL}. Make sure Ollama is running.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def flatten_for_csv(data: dict) -> list:
    rows = []
    items = data.get("items", [])
    vendor = data.get("vendor", {})
    doc_info = data.get("document_info", {})
    summary = data.get("summary", {})

    for item in items:
        rows.append({
            "Vendor": vendor.get("name", ""),
            "Invoice Number": doc_info.get("invoice_number", ""),
            "Date": doc_info.get("date", ""),
            "Item Description": item.get("description", ""),
            "Quantity": item.get("quantity", ""),
            "Unit Price": item.get("unit_price", ""),
            "Total": item.get("total", ""),
            "Currency": summary.get("currency", ""),
            "Grand Total": summary.get("total", ""),
            "Tax": summary.get("tax", ""),
        })

    if not rows:
        rows.append({
            "Vendor": vendor.get("name", ""),
            "Invoice Number": doc_info.get("invoice_number", ""),
            "Date": doc_info.get("date", ""),
            "Grand Total": summary.get("total", ""),
            "Currency": summary.get("currency", ""),
            "Tax": summary.get("tax", ""),
        })

    return rows


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return {"status": "ok", "ollama": "connected", "models": models}
    except:
        return {"status": "ok", "ollama": "disconnected", "models": []}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    allowed = ["application/pdf", "image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Only PDF, JPG, PNG, and WEBP files are supported.")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 20MB.")

    if file.content_type == "application/pdf":
        images = pdf_to_images(content)
        if not images:
            raise HTTPException(status_code=422, detail="Could not extract pages from PDF.")
        image_b64 = image_to_base64(images[0])
    else:
        try:
            doc = fitz.open(stream=content, filetype="image")
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            if max(pix.width, pix.height) > 2048:
                scale = 2048 / max(pix.width, pix.height)
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            image_b64 = image_to_base64(pix.tobytes("png"))
        except Exception:
            image_b64 = image_to_base64(content)

    extracted = extract_with_ollama(image_b64)
    return JSONResponse(content={"success": True, "data": extracted})


@app.post("/export-csv")
async def export_csv(request: Request):
    body = await request.json()
    data = body.get("data", {})

    rows = flatten_for_csv(data)

    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    output.seek(0)
    filename = f"invoice_{data.get('document_info', {}).get('invoice_number', 'export') or 'export'}.csv"

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
