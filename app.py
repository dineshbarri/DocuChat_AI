import os
import time
import json
import hashlib
import tempfile
import csv
import io
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# PAGE CONFIG 
# ─────────────────────────────────────────────
load_dotenv()
st.set_page_config(
    page_title="DocuChat_AT",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 6rem;
    }

    .docuchat-hero {
        background:
            linear-gradient(135deg, rgba(255,255,255,0.98), rgba(239,246,255,0.96)),
            linear-gradient(90deg, #2563eb, #14b8a6);
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 1.8rem 2rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.12);
    }

    .hero-layout {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 290px;
        gap: 1.2rem;
        align-items: start;
    }

    .docuchat-hero h1 {
        color: #111827 !important;
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 0 0 0.75rem;
    }

    .docuchat-hero p {
        color: #334155 !important;
        font-size: 1.08rem;
        line-height: 1.55;
        margin: 0;
    }

    .hero-kicker {
        color: #2563eb;
        font-weight: 800;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }

    .signature-card {
        background: #0f172a;
        border: 1px solid rgba(148, 163, 184, 0.32);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.18);
    }

    .signature-card small {
        color: #93c5fd;
        display: block;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .signature-card strong {
        color: #ffffff;
        display: block;
        font-size: 1.05rem;
        margin-bottom: 0.25rem;
    }

    .signature-card p {
        color: #cbd5e1 !important;
        font-size: 0.84rem;
        line-height: 1.45;
        margin: 0 0 0.75rem;
    }

    .signature-links {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.45rem;
    }

    .signature-links a {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        color: #f8fafc !important;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.5rem 0.55rem;
        text-align: center;
        text-decoration: none !important;
    }

    .signature-links a:hover {
        background: #2563eb;
        border-color: #60a5fa;
        color: #ffffff !important;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 1rem 0 1.15rem;
    }

    .feature-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        min-height: 116px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }

    .feature-card strong {
        display: block;
        color: #111827;
        font-size: 0.98rem;
        margin-bottom: 0.35rem;
    }

    .feature-card span {
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .status-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.6rem 0 1.2rem;
    }

    .status-tile {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1rem 1.1rem;
    }

    .status-tile small {
        display: block;
        color: #94a3b8;
        font-size: 0.78rem;
        margin-bottom: 0.3rem;
    }

    .status-tile b {
        color: #f8fafc;
        font-size: 1.15rem;
    }

    .intel-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        min-height: 125px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }

    .intel-card small {
        display: block;
        color: #64748b;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
    }

    .intel-card b {
        color: #111827;
        display: block;
        font-size: 1.45rem;
        line-height: 1.2;
        margin-bottom: 0.35rem;
    }

    .intel-card span {
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .section-title {
        color: #e5e7eb;
        font-size: 1.12rem;
        font-weight: 800;
        margin: 1rem 0 0.4rem;
    }

    .section-copy {
        color: #94a3b8;
        margin: 0 0 0.85rem;
    }

    div[data-testid="stTabs"] button {
        font-weight: 700;
    }

    div.stButton > button {
        border-radius: 9px;
        min-height: 2.65rem;
        font-weight: 700;
    }

    @media (max-width: 900px) {
        .hero-layout {
            grid-template-columns: 1fr;
        }

        .feature-grid,
        .status-strip {
            grid-template-columns: 1fr;
        }

        .docuchat-hero h1 {
            font-size: 1.75rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MAX_PAGES = 1000       
CHUNK_SIZE = 500       # Changed to tokens (using tiktoken)
CHUNK_OVERLAP = 100
MAX_CONTEXT_CHARS = 12000

MODELS = {
    "⚡ Llama 3.1 8B (Fastest)":   "llama-3.1-8b-instant",
    "🧠 Llama 3.3 70B (Smartest)": "llama-3.3-70b-versatile",
    "🌀 Mixtral 8x7B (Balanced)":  "mixtral-8x7b-32768",
    "💎 Gemma2 9B":                "gemma2-9b-it",
}

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "messages": [],
    "vectors": None,
    "doc_stats": {},
    "doc_intelligence": {},
    "rag_metrics": {},
    "eval_results": [],
    "eval_summary": {},
    "auth_ok": False,
    "last_file_hash": "",
    "full_raw_text": "", # BUG FIX: Stores text so summary can use it without reloading
    "pending_query": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

SAMPLE_QUESTIONS = [
    "Summarize this document in 6 crisp bullet points.",
    "What are the most important facts, dates, names, and numbers?",
    "What questions would a reviewer ask about this document?",
    "Explain the document like I am new to the topic.",
    "Find risks, warnings, limitations, or missing information.",
    "Create an action-item checklist from this document.",
]

TASK_PROMPTS = {
    "executive_summary": "Create an executive summary with key points, purpose, conclusions, and recommended next steps.",
    "key_takeaways": "Extract the top 10 key takeaways from the document and group them by theme.",
    "important_terms": "List important terms, names, dates, numbers, and definitions from the document.",
    "action_items": "Find every action item, task, owner, deadline, and dependency mentioned in the document.",
    "risks": "Analyze risks, gaps, contradictions, assumptions, and possible red flags in the document.",
    "decisions": "Identify decisions made or decisions needed, then explain the evidence for each.",
    "study_notes": "Turn this document into study notes with sections, bullet points, and likely exam/interview questions.",
    "email_brief": "Write a professional email brief summarizing this document for a busy stakeholder.",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Cache embeddings model — loads ONCE for the whole session."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
    )

def compute_files_hash(files) -> str:
    h = hashlib.md5()
    for f in files:
        h.update(f.name.encode())
        h.update(str(f.size).encode())
    return h.hexdigest()

def require_authentication():
    app_password = os.getenv("APP_PASSWORD", "").strip()
    if not app_password:
        st.session_state.auth_ok = True
        return
    if st.session_state.auth_ok:
        return

    st.markdown("### 🔐 Private Workspace")
    st.caption("This deployment is protected. Enter the app password to continue.")
    password = st.text_input("App Password", type="password", placeholder="Enter workspace password")
    if st.button("Unlock Workspace", type="primary"):
        if password == app_password:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

def ocr_pdf_pages(pdf_path: str, source_name: str, max_pages: int = 8) -> list:
    """Optional OCR for scanned PDFs. Requires pypdfium2, pytesseract, Pillow, and Tesseract binary."""
    try:
        import pypdfium2 as pdfium
        import pytesseract
    except Exception as e:
        raise RuntimeError("OCR packages are not installed. Install pypdfium2, pytesseract, and Pillow.") from e

    docs = []
    pdf = pdfium.PdfDocument(pdf_path)
    page_count = min(len(pdf), max_pages)
    for page_index in range(page_count):
        page = pdf[page_index]
        bitmap = page.render(scale=2.0)
        image = bitmap.to_pil()
        text = pytesseract.image_to_string(image)
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": page_index,
                        "extraction": "ocr",
                    },
                )
            )
    return docs

def has_readable_text(docs: list) -> bool:
    return any(doc.page_content and doc.page_content.strip() for doc in docs)

def normalize_source_metadata(docs: list, source_name: str, file_type: str, extraction: str = "text") -> list:
    for doc in docs:
        doc.metadata["source"] = source_name
        doc.metadata["file_type"] = file_type
        doc.metadata.setdefault("extraction", extraction)
    return docs

def load_documents(files, use_ocr: bool = False, ocr_page_limit: int = 8) -> list:
    """Safely load documents using tempfile (No local folder clutter)."""
    docs = []
    for file in files:
        ext = f".{file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(file.getbuffer())
            temp_path = temp_file.name
            
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(temp_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(temp_path)
            else:
                loader = TextLoader(temp_path, encoding="utf-8")
            
            loaded = loader.load()
            loaded = normalize_source_metadata(loaded, file.name, ext, "text")
            if ext == ".pdf" and use_ocr:
                extracted_chars = sum(len(doc.page_content.strip()) for doc in loaded)
                if extracted_chars < 250:
                    st.write(f"🔍 Running OCR for scanned PDF: {file.name}")
                    loaded = ocr_pdf_pages(temp_path, file.name, max_pages=ocr_page_limit)
            loaded = [doc for doc in loaded if doc.page_content and doc.page_content.strip()]
            docs.extend(loaded[:MAX_PAGES]) 
        except Exception as e:
            st.error(f"⚠️ Error loading `{file.name}`: {e}")
        finally:
            os.remove(temp_path) # Auto-cleanup immediately after reading
            
    return docs

def export_chat() -> str:
    lines = [f"# DocuChat_AT Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    for m in st.session_state.messages:
        role = "👤 User" if m["role"] == "user" else "🤖 Assistant"
        lines.append(f"**{role}:** {m['content']}\n")
    return "\n".join(lines)

def safe_json_loads(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

def ensure_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []

def build_document_intelligence(api_key: str, model_name: str, text: str) -> dict:
    sample = text[:10000]
    if not sample.strip():
        return {}

    llm_intel = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0,
        max_tokens=1800,
    )
    intel_prompt = ChatPromptTemplate.from_template(
        """
        You are a senior document intelligence system.
        Analyze the document sample and return ONLY valid JSON.

        Allowed document_type values:
        Research Paper, Contract, CV, Invoice, Policy, Report, Other

        JSON schema:
        {{
          "document_type": "Research Paper | Contract | CV | Invoice | Policy | Report | Other",
          "classification_confidence": 0-100,
          "classification_reason": "short reason",
          "entities": {{
            "people": [],
            "organizations": [],
            "dates": [],
            "money": [],
            "locations": []
          }},
          "risks": {{
            "legal_risks": [],
            "missing_information": [],
            "deadlines": []
          }},
          "action_items": []
        }}

        Rules:
        - Extract only information visible in the document.
        - Keep lists concise and high-signal.
        - If nothing is found for a field, return an empty list.

        Document sample:
        {context}
        """
    )
    chain = intel_prompt | llm_intel | StrOutputParser()
    parsed = safe_json_loads(chain.invoke({"context": sample}))
    entities = parsed.get("entities", {}) if isinstance(parsed.get("entities"), dict) else {}
    risks = parsed.get("risks", {}) if isinstance(parsed.get("risks"), dict) else {}

    return {
        "document_type": parsed.get("document_type", "Other"),
        "classification_confidence": int(parsed.get("classification_confidence", 0) or 0),
        "classification_reason": parsed.get("classification_reason", "Classified from visible document content."),
        "entities": {
            "people": ensure_list(entities.get("people")),
            "organizations": ensure_list(entities.get("organizations")),
            "dates": ensure_list(entities.get("dates")),
            "money": ensure_list(entities.get("money")),
            "locations": ensure_list(entities.get("locations")),
        },
        "risks": {
            "legal_risks": ensure_list(risks.get("legal_risks")),
            "missing_information": ensure_list(risks.get("missing_information")),
            "deadlines": ensure_list(risks.get("deadlines")),
        },
        "action_items": ensure_list(parsed.get("action_items")),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

def calculate_rag_metrics(retrieved_docs, top_k: int, context_chars: int, answer: str) -> dict:
    retrieved_count = len(retrieved_docs)
    citation_coverage = round(min(100, (retrieved_count / max(top_k, 1)) * 100))
    context_utilization = round(min(100, (context_chars / MAX_CONTEXT_CHARS) * 100))
    answer_lower = answer.lower()
    grounded_penalty = 25 if "isn't in the context" in answer_lower or "not in the context" in answer_lower else 0
    confidence_score = round(max(20, min(96, 45 + (citation_coverage * 0.32) + (context_utilization * 0.18) - grounded_penalty)))
    unique_sources = {
        os.path.basename(doc.metadata.get("source", "Unknown"))
        for doc in retrieved_docs
    }
    return {
        "retrieved_chunks": retrieved_count,
        "confidence_score": confidence_score,
        "citation_coverage": citation_coverage,
        "context_utilization": context_utilization,
        "unique_sources": len(unique_sources),
        "generated_at": datetime.now().strftime("%H:%M:%S"),
    }

def build_retriever(llm, top_k: int):
    retriever = st.session_state.vectors.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 3},
    )
    ctx_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, rephrase it as a standalone search query. Return ONLY the reformulated query."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return create_history_aware_retriever(llm, retriever, ctx_prompt)

def format_retrieved_context(retrieved_docs: list) -> tuple[str, int]:
    context_parts = []
    total_chars = 0
    for doc in retrieved_docs:
        if total_chars + len(doc.page_content) <= MAX_CONTEXT_CHARS:
            context_parts.append(doc.page_content)
            total_chars += len(doc.page_content)
        else:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                context_parts.append(doc.page_content[:remaining])
                total_chars += remaining
            break
    return "\n\n---\n\n".join(context_parts), total_chars

def answer_from_documents(llm, user_query: str, top_k: int, chat_history=None) -> dict:
    history = chat_history if chat_history is not None else st.session_state.chat_history
    history_aware_retriever = build_retriever(llm, top_k)
    retrieved_docs = history_aware_retriever.invoke({
        "input": user_query,
        "chat_history": history,
    })
    formatted_context, total_chars = format_retrieved_context(retrieved_docs)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert assistant. Answer using ONLY the provided context. If the answer isn't in the context, say so clearly.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = qa_prompt | llm | StrOutputParser()
    answer = qa_chain.invoke({
        "input": user_query,
        "chat_history": history,
        "context": formatted_context,
    })
    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "context": formatted_context,
        "context_chars": total_chars,
        "metrics": calculate_rag_metrics(retrieved_docs, top_k, total_chars, answer),
    }

def parse_eval_csv(uploaded_file) -> list:
    raw = uploaded_file.getvalue().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(raw))
    rows = []
    for row in reader:
        question = (row.get("question") or row.get("Question") or "").strip()
        expected = (row.get("expected_answer") or row.get("Expected Answer") or row.get("answer") or "").strip()
        expected_source = (row.get("expected_source") or row.get("source") or "").strip()
        if question and expected:
            rows.append({
                "question": question,
                "expected_answer": expected,
                "expected_source": expected_source,
            })
    return rows

def judge_eval_answer(llm, question: str, expected_answer: str, actual_answer: str, context: str) -> dict:
    judge_prompt = ChatPromptTemplate.from_template(
        """
        You are evaluating a RAG answer. Return ONLY valid JSON.

        Score:
        - correctness_score: 0-100, how well the actual answer matches the expected answer.
        - faithfulness_score: 0-100, whether the actual answer is supported by the retrieved context.
        - notes: one short sentence.

        JSON schema:
        {{
          "correctness_score": 0-100,
          "faithfulness_score": 0-100,
          "notes": "short note"
        }}

        Question: {question}
        Expected answer: {expected_answer}
        Actual answer: {actual_answer}
        Retrieved context: {context}
        """
    )
    parsed = safe_json_loads((judge_prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "expected_answer": expected_answer,
        "actual_answer": actual_answer,
        "context": context[:6000],
    }))
    return {
        "correctness_score": int(parsed.get("correctness_score", 0) or 0),
        "faithfulness_score": int(parsed.get("faithfulness_score", 0) or 0),
        "notes": parsed.get("notes", "Evaluation completed."),
    }

def run_eval_suite(eval_rows: list, llm, top_k: int) -> tuple[list, dict]:
    results = []
    for index, row in enumerate(eval_rows, start=1):
        rag = answer_from_documents(llm, row["question"], top_k, chat_history=[])
        judge = judge_eval_answer(
            llm,
            row["question"],
            row["expected_answer"],
            rag["answer"],
            rag["context"],
        )
        results.append({
            "test_id": index,
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "actual_answer": rag["answer"],
            "retrieved_chunks": rag["metrics"]["retrieved_chunks"],
            "citation_coverage": rag["metrics"]["citation_coverage"],
            "confidence_score": rag["metrics"]["confidence_score"],
            "correctness_score": judge["correctness_score"],
            "faithfulness_score": judge["faithfulness_score"],
            "notes": judge["notes"],
        })

    if not results:
        return [], {}

    summary = {
        "tests": len(results),
        "avg_correctness": round(sum(r["correctness_score"] for r in results) / len(results), 1),
        "avg_faithfulness": round(sum(r["faithfulness_score"] for r in results) / len(results), 1),
        "avg_confidence": round(sum(r["confidence_score"] for r in results) / len(results), 1),
        "avg_citation_coverage": round(sum(r["citation_coverage"] for r in results) / len(results), 1),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return results, summary

def queue_prompt(prompt: str):
    st.session_state.pending_query = prompt

def render_prompt_button(label: str, prompt: str, key: str, help_text: str | None = None):
    st.button(label, key=key, use_container_width=True, help=help_text, on_click=queue_prompt, args=(prompt,))

def render_hero():
    st.markdown(
        """
        <section class="docuchat-hero">
            <div class="hero-layout">
                <div>
                    <div class="hero-kicker">Document intelligence workspace</div>
                    <h1>DocuChat_AT (Document Intelligence RAG Assistant)</h1>
                    <p>Upload documents, generate summaries, extract insights, and ask grounded questions with source citations.</p>
                </div>
                <aside class="signature-card">
                    <small>Built by</small>
                    <strong>Dinesh Barri</strong>
                    <p>AI document assistant built for fast research, review, and knowledge extraction.</p>
                    <div class="signature-links">
                        <a href="https://github.com/dineshbarri" target="_blank">GitHub</a>
                        <a href="https://www.linkedin.com/in/dinesh-barri-7654b010b/" target="_blank">LinkedIn</a>
                        <a href="https://dineshbarri.dev" target="_blank">Portfolio</a>
                        <a href="mailto:dineshbarri1997@gmail.com">Email</a>
                    </div>
                </aside>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

def render_status_panel():
    stats = st.session_state.doc_stats or {}
    ready_label = "Ready" if st.session_state.vectors else "Upload and process"
    files_label = str(stats.get("files", 0))
    chunks_label = str(stats.get("chunks", 0))
    st.markdown(
        f"""
        <div class="status-strip">
            <div class="status-tile"><small>Knowledge base</small><b>{ready_label}</b></div>
            <div class="status-tile"><small>Documents loaded</small><b>{files_label}</b></div>
            <div class="status-tile"><small>Search chunks</small><b>{chunks_label}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_capability_cards():
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card"><strong>Ask Anything</strong><span>Chat with PDFs, Word files, and text documents using grounded answers.</span></div>
            <div class="feature-card"><strong>AI Intelligence</strong><span>Classify documents, extract entities, detect risks, and identify actions.</span></div>
            <div class="feature-card"><strong>Eval Dashboard</strong><span>Run labeled test questions and score correctness, faithfulness, and citations.</span></div>
            <div class="feature-card"><strong>OCR Ready</strong><span>Optional scanned PDF OCR with graceful fallback for Streamlit deployments.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_chip_list(items, empty_text="No items detected yet."):
    if not items:
        st.caption(empty_text)
        return
    for item in items[:12]:
        st.markdown(f"- {item}")

def render_intelligence_panel():
    intel = st.session_state.get("doc_intelligence", {})
    metrics = st.session_state.get("rag_metrics", {})

    if not intel:
        st.info("Process a document to generate classification, entities, risks, and action items.")
    else:
        entities = intel.get("entities", {})
        risks = intel.get("risks", {})
        total_entities = sum(len(entities.get(key, [])) for key in entities)
        total_risks = sum(len(risks.get(key, [])) for key in risks)
        confidence = intel.get("classification_confidence", 0)
        doc_type = intel.get("document_type", "Other")

        st.markdown(
            f"""
            <div class="status-strip">
                <div class="intel-card"><small>Document Type</small><b>{doc_type}</b><span>{intel.get("classification_reason", "")}</span></div>
                <div class="intel-card"><small>Classification Confidence</small><b>{confidence}%</b><span>LLM-based classification, CPU-friendly for Spaces.</span></div>
                <div class="intel-card"><small>Signals Detected</small><b>{total_entities + total_risks}</b><span>{total_entities} entities and {total_risks} risk signals found.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        entity_tab, risk_tab, action_tab, quality_tab = st.tabs([
            "Entities",
            "Risks",
            "Action Items",
            "RAG Quality",
        ])

        with entity_tab:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("##### People")
                render_chip_list(entities.get("people", []), "No people found.")
                st.markdown("##### Money")
                render_chip_list(entities.get("money", []), "No money values found.")
            with c2:
                st.markdown("##### Organizations")
                render_chip_list(entities.get("organizations", []), "No organizations found.")
                st.markdown("##### Locations")
                render_chip_list(entities.get("locations", []), "No locations found.")
            with c3:
                st.markdown("##### Dates")
                render_chip_list(entities.get("dates", []), "No dates found.")

        with risk_tab:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("##### Legal Risks")
                render_chip_list(risks.get("legal_risks", []), "No legal risks detected.")
            with c2:
                st.markdown("##### Missing Information")
                render_chip_list(risks.get("missing_information", []), "No missing information detected.")
            with c3:
                st.markdown("##### Deadlines")
                render_chip_list(risks.get("deadlines", []), "No deadlines detected.")

        with action_tab:
            st.markdown("##### Extracted Action Items")
            render_chip_list(intel.get("action_items", []), "No action items detected.")

        with quality_tab:
            if not metrics:
                st.info("Ask a question after processing documents to generate RAG quality metrics.")
            else:
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Retrieved Chunks", metrics.get("retrieved_chunks", 0))
                q2.metric("Confidence", f"{metrics.get('confidence_score', 0)}%")
                q3.metric("Citation Coverage", f"{metrics.get('citation_coverage', 0)}%")
                q4.metric("Context Used", f"{metrics.get('context_utilization', 0)}%")
                st.caption(f"Last evaluated at {metrics.get('generated_at', 'N/A')}. These are lightweight heuristic metrics for visibility, not formal benchmark scores.")

def render_evaluation_panel():
    if not st.session_state.vectors:
        st.info("Process documents before running an evaluation suite.")
        return

    st.markdown("##### Upload labeled test questions")
    st.caption("CSV columns required: question, expected_answer. Optional: expected_source.")
    eval_file = st.file_uploader("Evaluation CSV", type=["csv"], key="eval_csv")

    c1, c2 = st.columns([1, 1])
    with c1:
        run_eval = st.button("🧪 Run Evaluation", type="primary", use_container_width=True)
    with c2:
        clear_eval = st.button("Clear Evaluation", use_container_width=True)

    if clear_eval:
        st.session_state.eval_results = []
        st.session_state.eval_summary = {}
        st.rerun()

    if run_eval:
        if not eval_file:
            st.warning("Upload a labeled CSV first.")
        else:
            rows = parse_eval_csv(eval_file)
            if not rows:
                st.error("No valid rows found. Use columns: question, expected_answer.")
            else:
                with st.spinner(f"Running {len(rows)} RAG evaluation tests..."):
                    st.session_state.eval_results, st.session_state.eval_summary = run_eval_suite(rows, llm, top_k)
                st.success("Evaluation complete.")

    summary = st.session_state.get("eval_summary", {})
    results = st.session_state.get("eval_results", [])
    if summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tests", summary.get("tests", 0))
        m2.metric("Correctness", f"{summary.get('avg_correctness', 0)}%")
        m3.metric("Faithfulness", f"{summary.get('avg_faithfulness', 0)}%")
        m4.metric("Citation Coverage", f"{summary.get('avg_citation_coverage', 0)}%")
        st.caption(f"Generated at {summary.get('generated_at')}. Scores are LLM-judged and should be reviewed for critical use cases.")

    if results:
        st.markdown("##### Test Results")
        st.dataframe(results, use_container_width=True, hide_index=True)
        export = json.dumps({"summary": summary, "results": results}, indent=2)
        st.download_button(
            "⬇️ Download Evaluation JSON",
            data=export,
            file_name=f"docuchat_eval_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )

def render_workspace():
    render_hero()
    render_status_panel()
    render_capability_cards()

    st.markdown('<div class="section-title">Document command center</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Choose a workflow or start with a suggested prompt. Each button sends a ready-made instruction to the assistant.</p>', unsafe_allow_html=True)

    chat_tab, summary_tab, extract_tab, analyze_tab, intel_tab, eval_tab, deliver_tab = st.tabs([
        "Chat",
        "Summaries",
        "Extract",
        "Analyze",
        "Intelligence",
        "Evaluation",
        "Deliverables",
    ])

    with chat_tab:
        st.markdown("#### Sample questions")
        cols = st.columns(2)
        for index, prompt in enumerate(SAMPLE_QUESTIONS):
            with cols[index % 2]:
                render_prompt_button(prompt, prompt, f"sample_{index}")

    with summary_tab:
        st.markdown("#### Summary workflows")
        c1, c2 = st.columns(2)
        with c1:
            render_prompt_button("Executive summary", TASK_PROMPTS["executive_summary"], "task_exec")
            render_prompt_button("Top 10 takeaways", TASK_PROMPTS["key_takeaways"], "task_takeaways")
        with c2:
            render_prompt_button("Study notes", TASK_PROMPTS["study_notes"], "task_study")
            render_prompt_button("Email brief", TASK_PROMPTS["email_brief"], "task_email")

    with extract_tab:
        st.markdown("#### Extraction tools")
        c1, c2 = st.columns(2)
        with c1:
            render_prompt_button("Terms, dates, names", TASK_PROMPTS["important_terms"], "task_terms")
        with c2:
            render_prompt_button("Action items", TASK_PROMPTS["action_items"], "task_actions")

    with analyze_tab:
        st.markdown("#### Critical analysis")
        c1, c2 = st.columns(2)
        with c1:
            render_prompt_button("Risks and gaps", TASK_PROMPTS["risks"], "task_risks")
        with c2:
            render_prompt_button("Decisions needed", TASK_PROMPTS["decisions"], "task_decisions")

    with intel_tab:
        st.markdown("#### AI document intelligence")
        render_intelligence_panel()

    with eval_tab:
        st.markdown("#### RAG evaluation dashboard")
        render_evaluation_panel()

    with deliver_tab:
        st.markdown("#### Ready-to-use outputs")
        c1, c2, c3 = st.columns(3)
        with c1:
            render_prompt_button("Meeting notes", "Create concise meeting notes from this document with agenda, key discussion points, decisions, and follow-ups.", "task_meeting")
        with c2:
            render_prompt_button("Presentation outline", "Create a polished presentation outline from this document with slide titles and bullet points.", "task_presentation")
        with c3:
            render_prompt_button("FAQ", "Create a useful FAQ from this document with clear answers grounded in the content.", "task_faq")

# ─────────────────────────────────────────────
# SIDEBAR UI
# ─────────────────────────────────────────────
require_authentication()

with st.sidebar:
    st.title("📄 DocuChat_AT")
    if os.getenv("APP_PASSWORD", "").strip():
        st.success("🔐 Private mode enabled")
    else:
        st.caption("Public demo mode")
    
    if st.session_state.vectors:
        st.success("✅ Vector DB Ready", icon="🟢")
    else:
        st.info("ℹ️ No documents loaded", icon="🟡")
        
    st.divider()

    st.header("🔑 Configuration")
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    model_label = st.selectbox("Model", list(MODELS.keys()), index=0)
    selected_model = MODELS[model_label]

    with st.expander("⚙️ Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
        top_k = st.slider("Retrieved Chunks (Top-K)", 2, 10, 4)

    st.divider()

    st.header("📂 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    st.subheader("🔍 Scanned PDF OCR")
    use_ocr = st.toggle("Enable OCR for scanned PDFs", value=True, help="Use this for image-based PDFs that have no selectable text.")
    ocr_page_limit = st.slider("OCR page limit", 1, 25, 8, help="Higher values are slower on CPU Spaces.")
    st.caption("OCR needs Tesseract. Use Docker Space for the most reliable scanned PDF support.")

    col1, col2 = st.columns(2)
    process_btn   = col1.button("🔄 Process", type="primary", use_container_width=True)
    summarize_btn = col2.button("📜 Summary", use_container_width=True)

    # ── Processing Logic ──
    if process_btn or summarize_btn:
        if not api_key:
            st.error("❌ API Key missing!")
        elif not uploaded_files:
            st.warning("⚠️ Upload files first.")
        else:
            file_hash = f"{compute_files_hash(uploaded_files)}-ocr-{use_ocr}-{ocr_page_limit}"
            force_reprocess = (file_hash != st.session_state.last_file_hash)

            if not force_reprocess and st.session_state.vectors:
                st.toast("✅ Already processed (using cached vectors)!")
            else:
                # Using Streamlit's native status container for cool processing UI
                with st.status("Processing Documents...", expanded=True) as status:
                    t0 = time.time()
                    
                    st.write("📥 Loading files into memory...")
                    raw_docs = load_documents(uploaded_files, use_ocr=use_ocr, ocr_page_limit=ocr_page_limit)
                    
                    if not raw_docs or not has_readable_text(raw_docs):
                        status.update(label="No content found!", state="error")
                        st.error(
                            "No readable text was found. If this is a scanned PDF, keep OCR enabled and deploy with Docker so Tesseract OCR is installed."
                        )
                        st.stop()

                    # BUG FIX: Save the full raw text into session state for the summary function
                    st.session_state.full_raw_text = " ".join([d.page_content for d in raw_docs])

                    st.write("✂️ Splitting into chunks...")
                    # Improved Text Splitter based on Tokens
                    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )
                    chunks = splitter.split_documents(raw_docs)

                    if not chunks:
                        status.update(label="No searchable chunks created!", state="error")
                        st.error(
                            "The document loaded, but no searchable text chunks were created. For scanned PDFs, enable OCR and use Docker deployment with Tesseract."
                        )
                        st.stop()

                    st.write("🧠 Generating embeddings...")
                    embeddings = get_embeddings()
                    
                    st.write("📦 Building FAISS index...")
                    st.session_state.vectors = FAISS.from_documents(chunks, embeddings)
                    st.session_state.last_file_hash = file_hash
                    elapsed = round(time.time() - t0, 1)

                    st.session_state.doc_stats = {
                        "files": len(uploaded_files),
                        "pages": len(raw_docs),
                        "chunks": len(chunks),
                        "time": elapsed,
                    }
                    st.session_state.rag_metrics = {}

                    st.write("🧾 Classifying document and extracting intelligence...")
                    try:
                        st.session_state.doc_intelligence = build_document_intelligence(
                            api_key,
                            selected_model,
                            st.session_state.full_raw_text,
                        )
                    except Exception as e:
                        st.session_state.doc_intelligence = {}
                        st.warning(f"Document intelligence could not be generated: {e}")

                    status.update(label=f"Done in {elapsed}s!", state="complete", expanded=False)

            if summarize_btn:
                with st.spinner("Generating Summary..."):
                    llm_sum = ChatGroq(groq_api_key=api_key, model_name=selected_model, temperature=0.2)
                    
                    # BUG FIX: Use the saved text instead of raw_docs
                    if not st.session_state.full_raw_text:
                        temp_docs = load_documents(uploaded_files, use_ocr=use_ocr, ocr_page_limit=ocr_page_limit)
                        st.session_state.full_raw_text = " ".join([d.page_content for d in temp_docs])
                        
                    full_text = st.session_state.full_raw_text[:6000] 
                    
                    sum_prompt = ChatPromptTemplate.from_template(
                        "Summarize the following document in exactly 6 clear bullet points:\n\n{context}"
                    )
                    chain = sum_prompt | llm_sum | StrOutputParser()
                    summary = chain.invoke({"context": full_text})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"📋 **Document Summary**\n\n{summary}"
                    })
                    st.rerun()

    # ── Doc Stats Panel (Native Streamlit Metrics) ──
    if st.session_state.doc_stats:
        st.divider()
        st.subheader("📊 Index Stats")
        s = st.session_state.doc_stats
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        m1.metric("Files", s['files'])
        m2.metric("Pages", s['pages'])
        m3.metric("Chunks", s['chunks'])
        m4.metric("Time", f"{s['time']}s")

    if st.session_state.vectors:
        st.divider()
        st.subheader("🧠 Intelligence")
        if st.session_state.doc_intelligence:
            st.success(f"Type: {st.session_state.doc_intelligence.get('document_type', 'Other')}")
        else:
            st.info("Not generated yet.")
        if st.button("🔎 Refresh Intelligence", use_container_width=True):
            if not st.session_state.full_raw_text:
                st.warning("Process documents first.")
            else:
                with st.spinner("Classifying and extracting entities..."):
                    st.session_state.doc_intelligence = build_document_intelligence(
                        api_key,
                        selected_model,
                        st.session_state.full_raw_text,
                    )
                st.rerun()

    # ── Actions Panel ──
    st.divider()
    st.subheader("🛠 Actions")
    if st.button("💬 New Chat", use_container_width=True, help="Start a fresh conversation while keeping processed documents ready."):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.pending_query = ""
        st.rerun()

    if st.button("🔁 Reset Workspace", use_container_width=True, help="Clear chat, documents, vector index, and app state."):
        st.session_state.clear()
        st.rerun()

    if st.session_state.messages:
        st.download_button(
            "⬇️ Export Chat",
            data=export_chat(),
            file_name=f"rag_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ─────────────────────────────────────────────
# MAIN CHAT UI
# ─────────────────────────────────────────────
if not api_key:
    st.warning("👈 Please enter your Groq API key in the sidebar to start.")
    st.stop()

# Initialize LLM
llm = ChatGroq(
    groq_api_key=api_key,
    model_name=selected_model,
    temperature=temperature,
    max_tokens=2048,
    max_retries=3 # Handles API rate limits automatically
)

render_workspace()

# Render Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
pending_query = st.session_state.get("pending_query", "")
if pending_query:
    user_query = pending_query
    st.session_state.pending_query = ""
else:
    user_query = st.chat_input("Ask about your documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if not st.session_state.vectors:
        with st.chat_message("assistant"):
            st.warning("⚠️ No documents processed yet. Upload files and click **Process**.")
    else:
        retriever = st.session_state.vectors.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": top_k, "fetch_k": top_k * 3},
        )

        ctx_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and the latest user question, rephrase it as a standalone search query. Return ONLY the reformulated query."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, ctx_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant. Answer using ONLY the provided context. If the answer isn't in the context, say so clearly.\n\nContext:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_chain = qa_prompt | llm | StrOutputParser()

        with st.chat_message("assistant"):
            start_time = time.time()
            
            try:
                # 1. Retrieve Docs
                retrieved_docs = history_aware_retriever.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history,
                })

                # Trim context
                context_parts = []
                total_chars = 0
                for doc in retrieved_docs:
                    if total_chars + len(doc.page_content) <= MAX_CONTEXT_CHARS:
                        context_parts.append(doc.page_content)
                        total_chars += len(doc.page_content)
                    else:
                        remaining = MAX_CONTEXT_CHARS - total_chars
                        if remaining > 200:
                            context_parts.append(doc.page_content[:remaining])
                        break

                formatted_context = "\n\n---\n\n".join(context_parts)

                # 2. Native Streamlit Streaming (Replaces the custom for-loop)
                response_stream = qa_chain.stream({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history,
                    "context": formatted_context,
                })
                
                full_response = st.write_stream(response_stream)
                elapsed = round(time.time() - start_time, 2)
                st.session_state.rag_metrics = calculate_rag_metrics(
                    retrieved_docs,
                    top_k,
                    total_chars,
                    full_response,
                )

                # Native UI Details
                col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
                col1.caption(f"⏱ {elapsed}s")
                col2.caption(f"📚 {len(retrieved_docs)} chunks used")
                col3.caption(f"🎯 {st.session_state.rag_metrics['confidence_score']}% confidence")
                col4.caption(f"🔗 {st.session_state.rag_metrics['citation_coverage']}% citation coverage")

                with st.expander("View Source Citations"):
                    for i, doc in enumerate(retrieved_docs):
                        page  = doc.metadata.get("page", "N/A")
                        src   = os.path.basename(doc.metadata.get("source", "Unknown"))
                        preview = doc.page_content[:200].replace("\n", " ")
                        st.info(f"**{src} (Page {page})**\n\n{preview}...", icon="📄")

                # Update State
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.chat_history.extend([
                    HumanMessage(content=user_query),
                    AIMessage(content=full_response),
                ])

                # Keep history bounded
                if len(st.session_state.chat_history) > 40:
                    st.session_state.chat_history = st.session_state.chat_history[-40:]

            except Exception as e:
                st.error(f"❌ Error: {e}")
