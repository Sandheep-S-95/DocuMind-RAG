import os
import time
import datetime

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env FIRST so all env vars are in place before any rag_core import
# ---------------------------------------------------------------------------
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    override=True
)

import rag_core

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Documind — MedStudy RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached heavyweight resources (initialised once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource
def warm_up_resources():
    """
    Pre-warm the LLM, embedding model, and Tavily client by triggering
    their lru_cached constructors.
    """
    try:
        rag_core._get_embeddings()
        rag_core._get_llm()
        rag_core._get_web_search()
        return True
    except Exception as exc:
        return str(exc)


warm_result = warm_up_resources()

# ---------------------------------------------------------------------------
# One-time bootstrap: reconnect to cloud Chroma if docs already exist
# ---------------------------------------------------------------------------
if "bootstrapped" not in st.session_state:
    rag_core.bootstrap_from_disk()
    st.session_state.bootstrapped = True

# ---------------------------------------------------------------------------
# Sidebar — Drive sync & status
# ---------------------------------------------------------------------------
DRIVE_URL = os.environ.get("KNOWLEDGE_BASE_DRIVE_LINK", "")

with st.sidebar:
    st.title("🩺 Documind")
    st.caption("MedStudy RAG · Powered by Gemini + Chroma Cloud")
    st.divider()

    # ---------- Google Drive Sync ----------
    st.subheader("📂 Knowledge Source")
    if DRIVE_URL:
        st.caption(f"Google Drive folder linked")
        if st.button("🔄 Sync from Google Drive", use_container_width=True):
            if not rag_core.GLOBAL_STATE["is_indexing"]:
                rag_core.start_observer_sync(DRIVE_URL)
                st.rerun()
    else:
        st.warning("KNOWLEDGE_BASE_DRIVE_LINK not set in .env")

    st.divider()

    # ---------- Stats ----------
    st.subheader("📊 Index Stats")
    doc_count = rag_core.GLOBAL_STATE["doc_count"]
    st.metric("Documents indexed", doc_count)
    st.caption(f"Collection: `{rag_core.CHROMA_COLLECTION}`")
    st.caption(f"Model: `{rag_core.LLM_MODEL}`")

    # ---------- Web search indicator ----------
    st.divider()
    st.subheader("🌐 Web Search")
    tavily_ok = bool(os.environ.get("TAVILY_API_KEY"))
    if tavily_ok:
        st.success("Tavily enabled — answers supplement local PDFs with live web results.")
    else:
        st.warning("TAVILY_API_KEY not set — web search disabled.")

# ---------------------------------------------------------------------------
# One-time auto-sync on startup if the database is empty
# ---------------------------------------------------------------------------
if "first_sync_attempted" not in st.session_state:
    st.session_state.first_sync_attempted = True
    if rag_core.GLOBAL_STATE["retriever"] is None and not rag_core.GLOBAL_STATE["is_indexing"] and DRIVE_URL:
        rag_core.start_observer_sync(DRIVE_URL)

# ---------------------------------------------------------------------------
# Indexing gate: block the UI while background thread is working
# ---------------------------------------------------------------------------
if rag_core.GLOBAL_STATE["is_indexing"]:
    with st.status("🔄 Updating knowledge base …", expanded=True):
        st.info(rag_core.GLOBAL_STATE["status_message"])
        time.sleep(0.5)
        st.rerun()

# ---------------------------------------------------------------------------
# Empty-KB notice
# ---------------------------------------------------------------------------
if rag_core.GLOBAL_STATE["retriever"] is None:
    st.info(rag_core.GLOBAL_STATE["status_message"])
    st.caption("Click 'Sync from Google Drive' in the sidebar to download your documents and go online.")
    st.stop()

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("🩺 MedStudy RAG")

col_kb, col_chat = st.columns([1, 2])

# ---------- Left: KB status ----------
with col_kb:
    st.header("📈 Knowledge Base")
    st.success(rag_core.GLOBAL_STATE["status_message"])

    with st.expander("📄 Indexed documents", expanded=True):
        files = rag_core.GLOBAL_STATE.get("indexed_files", [])
        if files:
            for fname in sorted(files):
                st.text(f"• {fname}")
        else:
            st.caption("No documents indexed yet.")

# ---------- Right: Chat (fragment — only this reruns on chat submit) ----------
with col_chat:

    @st.fragment
    def chat_section():
        st.header("💬 Ask a question")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Render existing messages
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["message"])
                if message["role"] == "assistant":
                    if message.get("reasoning"):
                        with st.expander("🧠 Show reasoning"):
                            st.markdown(message["reasoning"])
                    if message.get("local_results"):
                        with st.expander("📄 Show Local Context Chunks"):
                            for chunk in message["local_results"]:
                                st.markdown(f"**Source**: `{chunk['source']}`")
                                st.markdown(chunk['content'])
                                st.divider()
                    if message.get("web_used") and message.get("web_results"):
                        with st.expander("🌐 Show Web Search Results"):
                            for res in message["web_results"]:
                                st.markdown(f"**Source**: [{res['url']}]({res['url']})")
                                st.markdown(res['content'])
                                st.divider()

        # Handle new input
        user_input = st.chat_input("Ask a question about your study material …")
        if user_input:
            st.session_state.chat_history.append(
                {"role": "user", "message": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            history_buffer = "\n".join(
                f"{m['role']}: {m['message']}"
                for m in st.session_state.chat_history[:-1]
            )

            with st.chat_message("assistant"):
                with st.spinner("Thinking …"):
                    try:
                        result = rag_core.get_answer(user_input, history_buffer)
                    except Exception as exc:
                        result = {
                            "answer": f"Sorry, something went wrong: {exc}",
                            "reasoning": "",
                            "web_used": False,
                            "web_results": [],
                            "local_results": [],
                        }

                st.markdown(result["answer"])
                if result.get("reasoning"):
                    with st.expander("🧠 Show reasoning"):
                        st.markdown(result["reasoning"])
                if result.get("local_results"):
                    with st.expander("📄 Show Local Context Chunks"):
                        for chunk in result["local_results"]:
                            st.markdown(f"**Source**: `{chunk['source']}`")
                            st.markdown(chunk['content'])
                            st.divider()
                if result.get("web_used") and result.get("web_results"):
                    with st.expander("🌐 Show Web Search Results"):
                        for res in result["web_results"]:
                            st.markdown(f"**Source**: [{res['url']}]({res['url']})")
                            st.markdown(res['content'])
                            st.divider()

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "message": result["answer"],
                    "reasoning": result.get("reasoning", ""),
                    "web_used": result.get("web_used", False),
                    "web_results": result.get("web_results", []),
                    "local_results": result.get("local_results", []),
                }
            )

    chat_section()
