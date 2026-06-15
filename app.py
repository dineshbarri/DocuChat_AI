import os
import time
import json
import hashlib
import tempfile
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

def load_documents(files) -> list:
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

def queue_prompt(prompt: str):
    st.session_state.pending_query = prompt

def render_prompt_button(label: str, prompt: str, key: str, help_text: str | None = None):
    st.button(label, key=key, use_container_width=True, help=help_text, on_click=queue_prompt, args=(prompt,))

def render_hero():
    st.markdown(
        """
        <section class="docuchat-hero">
            <div class="hero-kicker">Document intelligence workspace</div>
            <h1>DocuChat_AT (Document Intelligence RAG Assistant)</h1>
            <p>Upload documents, generate summaries, extract insights, and ask grounded questions with source citations.</p>
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
            <div class="feature-card"><strong>Instant Briefs</strong><span>Generate executive summaries, study notes, and stakeholder-ready updates.</span></div>
            <div class="feature-card"><strong>Deep Extraction</strong><span>Pull out action items, risks, decisions, dates, names, and definitions.</span></div>
            <div class="feature-card"><strong>Source Citations</strong><span>Inspect the retrieved document chunks used to answer each question.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_workspace():
    render_hero()
    render_status_panel()
    render_capability_cards()

    st.markdown('<div class="section-title">Document command center</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Choose a workflow or start with a suggested prompt. Each button sends a ready-made instruction to the assistant.</p>', unsafe_allow_html=True)

    chat_tab, summary_tab, extract_tab, analyze_tab, deliver_tab = st.tabs([
        "Chat",
        "Summaries",
        "Extract",
        "Analyze",
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
with st.sidebar:
    st.title("📄 DocuChat_AT")
    
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
            file_hash = compute_files_hash(uploaded_files)
            force_reprocess = (file_hash != st.session_state.last_file_hash)

            if not force_reprocess and st.session_state.vectors:
                st.toast("✅ Already processed (using cached vectors)!")
            else:
                # Using Streamlit's native status container for cool processing UI
                with st.status("Processing Documents...", expanded=True) as status:
                    t0 = time.time()
                    
                    st.write("📥 Loading files into memory...")
                    raw_docs = load_documents(uploaded_files)
                    
                    if not raw_docs:
                        status.update(label="No content found!", state="error")
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
                    status.update(label=f"Done in {elapsed}s!", state="complete", expanded=False)

            if summarize_btn:
                with st.spinner("Generating Summary..."):
                    llm_sum = ChatGroq(groq_api_key=api_key, model_name=selected_model, temperature=0.2)
                    
                    # BUG FIX: Use the saved text instead of raw_docs
                    if not st.session_state.full_raw_text:
                        temp_docs = load_documents(uploaded_files)
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

    # ── Actions Panel ──
    st.divider()
    st.subheader("🛠 Actions")
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

    if st.button("🔁 Reset Everything", use_container_width=True):
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

                # Native UI Details
                col1, col2 = st.columns([1, 4])
                col1.caption(f"⏱ {elapsed}s")
                col2.caption(f"📚 {len(retrieved_docs)} chunks used")

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
