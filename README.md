---
title: DocuChat_AI
emoji: 💻
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
python_version: '3.10'
pinned: false
license: mit
---

# Document Intelligence RAG Assistant

Upload documents, generate summaries, and ask grounded questions using Retrieval-Augmented Generation.

## Demo

[Live Demo](YOUR_HUGGING_FACE_SPACE_URL)

Screenshots:

- `assets/app_home.png`
- `assets/document_processing.png`
- `assets/chat_with_sources.png`

## Problem Statement

Professionals often need to quickly understand long PDFs, policies, resumes, contracts, research papers, reports, meeting notes, and business documents. Reading every page manually is slow, and generic chatbots can hallucinate when they are not grounded in the source material.

Document Intelligence RAG Assistant helps users upload documents, generate structured summaries, ask questions, inspect source citations, and export useful outputs from a clean Streamlit interface.

## Key Features

- PDF, TXT, and DOCX upload
- Document summarization with multiple modes
- Question answering over uploaded files
- FAISS vector search
- HuggingFace sentence-transformer embeddings
- Groq / LLM-powered generation
- Source citations with file, page, chunk, and preview text
- Chat history within the current session
- Export summary and chat history as Markdown
- Streamlit user interface
- Hugging Face Spaces deployment support

## Architecture

The app follows a lightweight RAG pipeline:

Document Upload -> Text Extraction -> Chunking -> Embeddings -> FAISS Vector Store -> Retriever -> LLM -> Grounded Answer with Sources

```mermaid
flowchart LR
    A[Upload Documents] --> B[Extract Text]
    B --> C[Split into Chunks]
    C --> D[Generate Embeddings]
    D --> E[FAISS Vector Store]
    E --> F[Retriever]
    F --> G[LLM]
    G --> H[Answer with Citations]
```

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace sentence-transformers
- Groq API / LLM provider
- PyPDF and document loaders
- Docker / Hugging Face Spaces

## Local Setup

1. Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

5. Run the app:

```bash
streamlit run app.py
```

## Hugging Face Spaces Deployment

1. Create a new Hugging Face Space.
2. Choose Streamlit as the Space SDK.
3. Upload `app.py`, `README.md`, `requirements.txt`, `runtime.txt`, `.gitattributes`, and the `assets/` folder. `Dockerfile` is included for GitHub/Docker users, but a normal Streamlit Space does not require it.
4. Add `GROQ_API_KEY` under Space Settings -> Secrets.
5. Confirm `app.py` is the entry point.
6. Restart the Space if dependencies are updated.

The app is designed to run on Hugging Face Spaces CPU. It uses CPU-compatible embeddings and avoids GPU-only dependencies.

## Usage Guide

1. Upload one or more PDF, TXT, or DOCX files.
2. Click **Process documents** to extract text, chunk content, generate embeddings, and build a FAISS index.
3. Choose a summary mode and click **Generate summary**.
4. Ask document-specific questions in the chat tab.
5. Open source citations under each answer to inspect retrieved evidence.
6. Export the summary or chat history as Markdown.

## Example Questions

- "Summarize this document in 5 bullet points."
- "What are the key risks mentioned?"
- "Extract all dates, people, and organizations."
- "What are the main recommendations?"
- "Explain the document for a non-technical audience."
- "What evidence supports this answer?"

## Project Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
├── Dockerfile
├── .gitattributes
└── assets/
    └── .gitkeep
```

## Privacy and Security

- API keys are read from Hugging Face Space secrets, environment variables, or a password input field.
- Uploaded documents are processed through temporary files and are not intentionally stored by the app.
- The FAISS index and chat history live only in the active Streamlit session.
- Users should avoid uploading highly sensitive documents to public demo deployments.

## Limitations

- Quality depends on uploaded document text extraction.
- Scanned PDFs may require OCR, which is not included in this version.
- LLM answers are grounded in retrieved chunks, but users should verify important outputs.
- Free Hugging Face Spaces may have CPU and memory limits.
- Very large documents may need smaller files or reduced chunk settings.

## Future Improvements

- OCR for scanned PDFs
- Multi-document comparison
- Persistent vector database option
- User authentication
- Retrieval quality evaluation metrics
- CSV, Excel, and HTML support
- Better citation highlighting inside source documents

## Why This Project Matters

This project demonstrates practical skills across RAG architecture, document NLP, vector search, LLM integration, Streamlit product UI, Hugging Face deployment, and AI-assisted workflow automation. It is relevant for Data Analyst, Data Scientist, AI Engineer, and RAG Engineer portfolios because it turns unstructured documents into searchable, explainable, and exportable insights.

## License

MIT License.