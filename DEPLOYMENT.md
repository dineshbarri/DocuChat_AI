# DocuChat_AI Deployment Guide

This guide covers deploying DocuChat_AI as a Streamlit app on Hugging Face Spaces, with optional private access, OCR support, and evaluation workflows.

## 1. Recommended Hugging Face Setup

For the standard app without scanned PDF OCR:

```text
SDK: Streamlit
Python: 3.10
Hardware: CPU Basic or higher
```

Required files:

```text
app.py
README.md
requirements.txt
runtime.txt
.gitattributes
assets/
```

Required secret:

```text
GROQ_API_KEY=your_groq_key_here
```

Optional private access secret:

```text
APP_PASSWORD=your_private_app_password
```

If `APP_PASSWORD` is set, the app shows a password screen before users can access the workspace. If it is not set, the app runs in public demo mode.

## 2. OCR Deployment Notes

DocuChat_AI includes an OCR fallback for scanned PDFs using:

```text
pypdfium2
pytesseract
Pillow
Tesseract OCR binary
```

Python packages are installed from `requirements.txt`, but the Tesseract system binary is usually not available in a normal Streamlit SDK Space.

For best OCR support, deploy using the included `Dockerfile`, which installs:

```text
tesseract-ocr
poppler-utils
```

Recommended OCR setup:

```text
SDK: Docker
Hardware: CPU Upgrade if processing larger scanned PDFs
```

If OCR dependencies are missing, the app fails gracefully and still works for normal text-based PDFs, DOCX, and TXT files.

Important:

```text
If your scanned PDF returns "No readable text found", switch the Space SDK to Docker or create a new Docker Space using this repository.
```

## 3. Evaluation Dashboard

The Evaluation tab supports labeled RAG testing with CSV files.

Required CSV columns:

```text
question,expected_answer
```

Optional column:

```text
expected_source
```

Example:

```csv
question,expected_answer,expected_source
What is the contract deadline?,The deadline is 15 June 2026.,contract.pdf page 3
Who is the policy owner?,The policy owner is the HR department.,policy.pdf
```

Metrics shown:

```text
Correctness score
Faithfulness score
Confidence score
Citation coverage
Retrieved chunks
```

The correctness and faithfulness scores are LLM-judged and should be treated as practical evaluation signals, not formal benchmark results.

## 4. Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes | API key for Groq LLM inference |
| `APP_PASSWORD` | No | Enables private password-protected app access |

## 5. Production Recommendations

- Use Hugging Face Secrets for all keys.
- Use Docker mode if OCR is important.
- Keep uploads temporary unless you add explicit user accounts and storage consent.
- Add `APP_PASSWORD` for portfolio demos shared with recruiters.
- Use the Evaluation tab with a small labeled test set before demos.
- Use CPU Upgrade for larger PDFs or frequent OCR use.
