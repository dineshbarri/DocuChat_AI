import hashlib
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(
    page_title="Document Intelligence RAG Assistant",
    page_icon="DI",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Document Intelligence RAG Assistant"
SUPPORTED_EXTENSIONS = {"pdf", "txt", "docx"}
MAX_FILE_SIZE_MB = 25
MAX_CONTEXT_CHARS = 14_000
MAX_SUMMARY_CHARS = 28_000
MAX_CHAT_TURNS = 16

MODEL_OPTIONS = {
    "Llama 3.1 8B - fast": "llama-3.1-8b-instant",
    "Llama 3.3 70B - stronger reasoning": "llama-3.3-70b-versatile",
    "Gemma2 9B - efficient": "gemma2-9b-it",
}

SUMMARY_MODES = {
    "Executive summary": "Write a concise executive summary for a busy professional. Include the core message, major conclusions, and practical implications.",
    "Bullet-point summary": "Summarize the document as clear bullet points. Prioritize facts, decisions, risks, and recommendations.",
    "Detailed study notes": "Create structured study notes with headings, definitions, important details, and takeaways.",
    "Key facts / entities / dates": "Extract important facts, people, organizations, locations, dates, metrics, obligations, and deadlines.",
}

EXAMPLE_QUESTIONS = [
    "Summarize this document in 5 bullet points.",
    "What are the key risks mentioned?",
    "Extract action items and owners.",
    "List important dates, people, and organizations.",
    "Explain this for a non-technical audience.",
    "What evidence supports the main recommendation?",
]


def init_state() -> None:
    defaults = {
        "vectorstore": None,
        "documents": [],
        "chunks": [],
        "raw_text": "",
        "file_hash": "",
        "doc_stats": {},
        "messages": [],
        "chat_history": [],
        "latest_sources": [],
        "summary": "",
        "summary_mode": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


st.markdown(
    """
    <style>
        .block-container {padding-top: 1.5rem; max-width: 1240px;}
        .app-hero {
            border: 1px solid #e6e8ef;
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            background: #ffffff;
            margin-bottom: 1rem;
        }
        .app-hero h1 {font-size: 2.05rem; margin-bottom: .2rem;}
        .app-hero p {color: #4b5563; margin-bottom: 0;}
        .small-muted {color: #6b7280; font-size: .92rem;}
        .source-box {
            border-left: 3px solid #2563eb;
            padding: .65rem .85rem;
            background: #f8fafc;
            border-radius: 6px;
            margin: .4rem 0;
        }
        .chip-row {display: flex; flex-wrap: wrap; gap: .45rem; margin: .5rem 0 1rem 0;}
        .chip {
            border: 1px solid #d8dee9;
            border-radius: 999px;
            padding: .35rem .7rem;
            background: #f9fafb;
            color: #374151;
            font-size: .88rem;
        }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def get_llm(api_key: str, model: str, temperature: float, max_tokens: int = 2048) -> ChatGroq:
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
    )


def file_fingerprint(files: Iterable) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(str(file.size).encode("utf-8"))
        digest.update(file.getvalue()[:4096])
    return digest.hexdigest()


def validate_files(files: list) -> list[str]:
    errors = []
    for file in files:
        suffix = Path(file.name).suffix.lower().lstrip(".")
        if suffix not in SUPPORTED_EXTENSIONS:
            errors.append(f"{file.name}: unsupported file type.")
        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            errors.append(f"{file.name}: file is larger than {MAX_FILE_SIZE_MB} MB.")
        if file.size == 0:
            errors.append(f"{file.name}: file is empty.")
    return errors


def load_uploaded_documents(files: list) -> list[Document]:
    loaded_docs: list[Document] = []

    for file in files:
        suffix = Path(file.name).suffix.lower()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file.getbuffer())
                temp_path = temp_file.name

            if suffix == ".pdf":
                loader = PyPDFLoader(temp_path)
            elif suffix == ".docx":
                loader = Docx2txtLoader(temp_path)
            else:
                loader = TextLoader(temp_path, encoding="utf-8", autodetect_encoding=True)

            docs = loader.load()
            for index, doc in enumerate(docs):
                doc.metadata["source"] = file.name
                doc.metadata["file_type"] = suffix.lstrip(".")
                doc.metadata["page"] = doc.metadata.get("page", index)
            loaded_docs.extend(docs)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return [doc for doc in loaded_docs if doc.page_content and doc.page_content.strip()]


def build_vectorstore(docs: list[Document], chunk_size: int, chunk_overlap: int) -> tuple[FAISS, list[Document]]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = index
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings), chunks


def format_source_label(doc: Document) -> str:
    source = doc.metadata.get("source", "Uploaded document")
    page = doc.metadata.get("page")
    page_text = f", page {int(page) + 1}" if isinstance(page, int) else ""
    chunk_id = doc.metadata.get("chunk_id")
    chunk_text = f", chunk {chunk_id}" if chunk_id else ""
    return f"{source}{page_text}{chunk_text}"


def preview_text(text: str, limit: int = 380) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def format_docs_for_context(docs: list[Document]) -> str:
    parts = []
    total = 0
    for doc in docs:
        label = format_source_label(doc)
        content = doc.page_content.strip()
        block = f"Source: {label}\n{content}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total
            if remaining > 500:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def make_chat_export() -> str:
    lines = [f"# {APP_TITLE} - Chat Export", "", f"Exported: {datetime.now():%Y-%m-%d %H:%M}", ""]
    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.extend([f"## {role}", message["content"], ""])
        sources = message.get("sources") or []
        if sources:
            lines.append("### Sources")
            for source in sources:
                lines.append(f"- {source['label']}: {source['preview']}")
            lines.append("")
    return "\n".join(lines)


def make_summary_export() -> str:
    return "\n".join(
        [
            f"# {APP_TITLE} - Summary",
            "",
            f"Mode: {st.session_state.summary_mode or 'Summary'}",
            f"Exported: {datetime.now():%Y-%m-%d %H:%M}",
            "",
            st.session_state.summary or "",
        ]
    )


def reset_documents() -> None:
    for key in ["vectorstore", "documents", "chunks", "raw_text", "file_hash", "doc_stats", "latest_sources", "summary", "summary_mode"]:
        st.session_state[key] = "" if key in {"raw_text", "file_hash", "summary", "summary_mode"} else [] if key in {"documents", "chunks", "latest_sources"} else {} if key == "doc_stats" else None
    st.session_state.messages = []
    st.session_state.chat_history = []


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("No source chunks were returned.")
        return
    for source in sources:
        st.markdown(
            f"""
            <div class="source-box">
                <strong>{source["label"]}</strong><br/>
                <span class="small-muted">{source["preview"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.header("Configuration")
    env_api_key = os.getenv("GROQ_API_KEY", "")
    api_key = env_api_key or st.text_input("Groq API key", type="password", placeholder="gsk_...")

    selected_model = MODEL_OPTIONS[
        st.selectbox("LLM model", options=list(MODEL_OPTIONS.keys()), index=0)
    ]
    temperature = st.slider("Answer creativity", 0.0, 0.8, 0.1, 0.05)

    with st.expander("Retrieval settings", expanded=False):
        chunk_size = st.slider("Chunk size", 350, 1200, 700, 50)
        chunk_overlap = st.slider("Chunk overlap", 50, 300, 120, 10)
        top_k = st.slider("Source chunks", 2, 8, 4, 1)
        retrieval_mode = st.radio("Retrieval mode", ["MMR", "Similarity"], horizontal=True)

    st.divider()
    st.header("Upload")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or DOCX files",
        type=sorted(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
    )

    process_clicked = st.button("Process documents", type="primary", use_container_width=True)

    summary_mode = st.selectbox("Summary mode", list(SUMMARY_MODES.keys()))
    summary_clicked = st.button("Generate summary", use_container_width=True)

    st.divider()
    col_a, col_b = st.columns(2)
    if col_a.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    if col_b.button("Reset app", use_container_width=True):
        reset_documents()
        st.rerun()

    if st.session_state.summary:
        st.download_button(
            "Download summary",
            data=make_summary_export(),
            file_name=f"document_summary_{datetime.now():%Y%m%d_%H%M}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.session_state.messages:
        st.download_button(
            "Download chat",
            data=make_chat_export(),
            file_name=f"document_chat_{datetime.now():%Y%m%d_%H%M}.md",
            mime="text/markdown",
            use_container_width=True,
        )


st.markdown(
    f"""
    <div class="app-hero">
        <h1>{APP_TITLE}</h1>
        <p>Upload documents, generate summaries, and ask grounded questions with source citations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_col, stats_col = st.columns([2, 1])
with overview_col:
    st.markdown(
        "This assistant extracts text from uploaded documents, splits it into searchable chunks, builds a local FAISS index with HuggingFace embeddings, and uses a Groq-hosted LLM to answer only from retrieved context."
    )
with stats_col:
    if st.session_state.doc_stats:
        stats = st.session_state.doc_stats
        st.metric("Files", stats["files"])
        st.metric("Chunks", stats["chunks"])
    else:
        st.info("Upload and process documents to start.")


if process_clicked:
    if not uploaded_files:
        st.warning("Upload at least one PDF, TXT, or DOCX file first.")
    else:
        validation_errors = validate_files(uploaded_files)
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        else:
            current_hash = file_fingerprint(uploaded_files)
            if current_hash == st.session_state.file_hash and st.session_state.vectorstore:
                st.success("These files are already processed. The current index is ready.")
            else:
                with st.status("Processing documents", expanded=True) as status:
                    try:
                        started = time.time()
                        st.write("Reading uploaded files from temporary storage...")
                        docs = load_uploaded_documents(uploaded_files)
                        if not docs:
                            status.update(label="No readable text found.", state="error")
                            st.stop()

                        st.write("Splitting document text into retrieval chunks...")
                        vectorstore, chunks = build_vectorstore(docs, chunk_size, chunk_overlap)

                        raw_text = "\n\n".join(doc.page_content for doc in docs).strip()
                        elapsed = round(time.time() - started, 2)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.documents = docs
                        st.session_state.chunks = chunks
                        st.session_state.raw_text = raw_text
                        st.session_state.file_hash = current_hash
                        st.session_state.summary = ""
                        st.session_state.summary_mode = ""
                        st.session_state.doc_stats = {
                            "files": len(uploaded_files),
                            "pages": len(docs),
                            "chunks": len(chunks),
                            "characters": len(raw_text),
                            "seconds": elapsed,
                        }
                        status.update(label=f"Index ready in {elapsed}s.", state="complete", expanded=False)
                    except Exception as exc:
                        status.update(label="Processing failed.", state="error")
                        st.error(f"Could not process the uploaded files: {exc}")


if summary_clicked:
    if not api_key:
        st.error("Add a Groq API key in Space secrets or the sidebar before generating a summary.")
    elif not st.session_state.raw_text:
        st.warning("Process documents before generating a summary.")
    else:
        try:
            with st.spinner("Generating grounded summary..."):
                llm = get_llm(api_key, selected_model, temperature=0.1, max_tokens=2200)
                text_for_summary = st.session_state.raw_text[:MAX_SUMMARY_CHARS]
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You summarize uploaded documents for professional analysis. Use only the supplied document text. If the text is insufficient, say so.",
                        ),
                        (
                            "human",
                            "Summary mode: {mode}\n\nInstructions: {instructions}\n\nDocument text:\n{document_text}",
                        ),
                    ]
                )
                chain = prompt | llm | StrOutputParser()
                st.session_state.summary = chain.invoke(
                    {
                        "mode": summary_mode,
                        "instructions": SUMMARY_MODES[summary_mode],
                        "document_text": text_for_summary,
                    }
                )
                st.session_state.summary_mode = summary_mode
        except Exception as exc:
            st.error(f"Summary generation failed: {exc}")


tabs = st.tabs(["Upload", "Summary", "Chat", "Sources", "Export"])

with tabs[0]:
    st.subheader("Upload and Process")
    if st.session_state.doc_stats:
        stats = st.session_state.doc_stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Files", stats["files"])
        c2.metric("Pages / sections", stats["pages"])
        c3.metric("Chunks", stats["chunks"])
        c4.metric("Processing time", f"{stats['seconds']}s")
        st.caption("Uploaded files are read through temporary files and are not stored permanently by the app.")
        st.markdown('<div class="chip-row">' + "".join(f'<span class="chip">{q}</span>' for q in EXAMPLE_QUESTIONS) + "</div>", unsafe_allow_html=True)
    else:
        st.info("No document index yet. Upload files in the sidebar and select Process documents.")
        st.markdown("Accepted formats: PDF, TXT, DOCX. Maximum file size per upload: 25 MB.")

with tabs[1]:
    st.subheader("Summary")
    if st.session_state.summary:
        with st.expander(st.session_state.summary_mode, expanded=True):
            st.markdown(st.session_state.summary)
    else:
        st.info("Choose a summary mode in the sidebar after processing documents.")

with tabs[2]:
    st.subheader("Chat")
    if not api_key:
        st.warning("Add a Groq API key in Space secrets or the sidebar to enable chat.")
    elif not st.session_state.vectorstore:
        st.info("Process documents first. Answers are grounded only in your uploaded files.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Source citations"):
                    render_sources(message["sources"])
            if message.get("meta"):
                st.caption(message["meta"])

    user_query = st.chat_input("Ask a question about the processed documents")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        if not api_key:
            st.error("A Groq API key is required for answers.")
        elif not st.session_state.vectorstore:
            st.warning("No document index is available. Upload and process documents first.")
        else:
            with st.chat_message("assistant"):
                try:
                    started = time.time()
                    llm = get_llm(api_key, selected_model, temperature=temperature)
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr" if retrieval_mode == "MMR" else "similarity",
                        search_kwargs={"k": top_k, "fetch_k": max(top_k * 4, 12)}
                        if retrieval_mode == "MMR"
                        else {"k": top_k},
                    )

                    rewrite_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "Rewrite the latest user question as a standalone retrieval query. Return only the query.",
                            ),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{question}"),
                        ]
                    )
                    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
                    retrieval_query = rewrite_chain.invoke(
                        {
                            "question": user_query,
                            "chat_history": st.session_state.chat_history[-MAX_CHAT_TURNS:],
                        }
                    )
                    retrieved_docs = retriever.invoke(retrieval_query)
                    context = format_docs_for_context(retrieved_docs)

                    qa_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are a document intelligence assistant. Answer using only the provided context. "
                                "If the answer is not supported by the context, say: 'The uploaded document does not contain enough information to answer that.' "
                                "Cite source labels naturally when useful. Do not invent facts.\n\nContext:\n{context}",
                            ),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{question}"),
                        ]
                    )
                    qa_chain = qa_prompt | llm | StrOutputParser()
                    stream = qa_chain.stream(
                        {
                            "question": user_query,
                            "chat_history": st.session_state.chat_history[-MAX_CHAT_TURNS:],
                            "context": context,
                        }
                    )
                    answer = st.write_stream(stream)
                    elapsed = round(time.time() - started, 2)

                    sources = [
                        {"label": format_source_label(doc), "preview": preview_text(doc.page_content)}
                        for doc in retrieved_docs
                    ]
                    meta = f"Response time: {elapsed}s | Source chunks used: {len(retrieved_docs)}"
                    st.caption(meta)
                    with st.expander("Source citations"):
                        render_sources(sources)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources, "meta": meta}
                    )
                    st.session_state.chat_history.extend([HumanMessage(content=user_query), AIMessage(content=answer)])
                    st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_TURNS:]
                    st.session_state.latest_sources = sources
                except Exception as exc:
                    st.error(f"Answer generation failed: {exc}")

with tabs[3]:
    st.subheader("Sources")
    if st.session_state.latest_sources:
        render_sources(st.session_state.latest_sources)
    elif st.session_state.chunks:
        st.caption("Ask a question to see the source chunks used for the answer.")
        sample_sources = [
            {"label": format_source_label(doc), "preview": preview_text(doc.page_content)}
            for doc in st.session_state.chunks[: min(3, len(st.session_state.chunks))]
        ]
        render_sources(sample_sources)
    else:
        st.info("Source citations will appear after documents are processed and queried.")

with tabs[4]:
    st.subheader("Export")
    st.markdown("Download summaries and chat history as Markdown for reports, notes, or portfolio demos.")
    if st.session_state.summary:
        st.download_button(
            "Download summary as Markdown",
            data=make_summary_export(),
            file_name=f"document_summary_{datetime.now():%Y%m%d_%H%M}.md",
            mime="text/markdown",
        )
    if st.session_state.messages:
        st.download_button(
            "Download chat as Markdown",
            data=make_chat_export(),
            file_name=f"document_chat_{datetime.now():%Y%m%d_%H%M}.md",
            mime="text/markdown",
        )
    if not st.session_state.summary and not st.session_state.messages:
        st.info("Generate a summary or chat with the document to enable exports.")

st.divider()
st.caption(
    "Privacy note: documents are processed in the current Streamlit session through temporary files. Answers are grounded only in retrieved chunks from uploaded documents."
)
