# Aibys Invoice Extractor

AI-powered invoice and receipt data extractor. Upload PDFs or images, extract structured data with Ollama, then keep the results in local JSON and CSV files for ongoing use.

Part of the **Aibys Document Intelligence** series.

## Features

- Upload multiple PDF, JPG, PNG, or WEBP files in one batch
- PDF pages are converted to images and extracted page by page
- AI extracts invoice data such as vendor, customer, line items, totals, dates, and payment method
- Structured result view with saved extraction history
- Raw JSON output for each record
- Persistent local storage using plain files, no SQL database
- CSV is appended to the same `data/invoices.csv` file across sessions and days
- Download saved CSV or JSON database from the UI
- Fully local when using a local Ollama vision model
- Responsive UI with no frontend build step

## How It Works

1. Upload one or many invoices or receipts.
2. FastAPI validates each file and stores the original upload in `uploads/`.
3. PDF files are rendered into page images with PyMuPDF.
4. Each image is sent to Ollama with a structured extraction prompt.
5. Extracted records are saved into `data/invoices.json`.
6. Flattened rows are appended into `data/invoices.csv`.
7. The UI renders the latest batch and the saved history.

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- A vision-capable model, for example:

```bash
ollama pull gemma4:31b-cloud
```

### Install & Run

```bash
git clone https://github.com/Arlchoose-code/aibys-invoice-extractor.git
cd aibys-invoice-extractor
pip install -r requirements.txt
uvicorn main:app --reload
```

Open:

```text
http://localhost:8000
```

## Configuration

Set environment variables as needed:

```bash
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:31b-cloud
```

Other possible vision models: `llava`, `moondream`, `minicpm-v`.

## Local Data Files

The app creates these files and folders automatically:

| Path | Purpose |
|---|---|
| `uploads/` | Original uploaded files |
| `data/invoices.json` | Full structured extraction history |
| `data/invoices.csv` | Append-only CSV export for spreadsheet use |

This project intentionally uses JSON and CSV files instead of SQL so it stays simple and portable.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Ollama connection status |
| `POST` | `/extract` | Extract one or more files using multipart field `files` |
| `GET` | `/records` | Return saved JSON records |
| `GET` | `/export-csv` | Download saved CSV |
| `POST` | `/export-csv` | Export posted extraction data as CSV |
| `GET` | `/export-json` | Download saved JSON database |

## Tech Stack

- **Backend:** FastAPI + Python
- **AI:** Ollama
- **PDF/Image Processing:** PyMuPDF
- **Storage:** JSON + CSV files
- **Frontend:** Vanilla HTML/CSS/JS

## Extracted Fields

| Category | Fields |
|---|---|
| Document Info | Invoice number, date, due date, payment method |
| Vendor | Name, address, phone, email, website |
| Customer | Name, address, phone, email |
| Line Items | Description, quantity, unit price, total |
| Summary | Subtotal, tax, discount, shipping, total, currency |

## Aibys Document Intelligence Series

| Repo | Description |
|---|---|
| **Aibys Invoice Extractor** | Extract invoice and receipt data |
| Aibys Legal Analyzer | Highlight risky clauses in contracts |
| Aibys Medical Explainer | Explain medical reports in plain language |
| Aibys Research Summarizer | Summarize academic papers |

## Author

**Syahril Haryono** - [github.com/Arlchoose-code](https://github.com/Arlchoose-code)

## License

MIT License
