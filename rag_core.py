"""
rag_core.py
-----------
All RAG/indexing logic with ZERO Streamlit imports.

Architecture changes vs original:
  - ChromaDB CLOUD (chromadb.CloudClient) instead of local SQLite embed.
  - Tavily web search is run alongside vector retrieval in get_answer().
  - .env is loaded via python-dotenv so secrets never need to be typed by hand.
  - LLM, Embeddings, and the Tavily client are each memoized with
    functools.lru_cache so they are initialised exactly once per process.
  - bootstrap_from_disk() now probes the cloud collection doc-count instead
    of checking for a local chroma_db/ directory.
  - Everything else (locking, incremental indexing, GLOBAL_STATE) is unchanged.
"""

import functools
import glob
import json
import logging
import os
import threading

from dotenv import load_dotenv

# Load .env from the project root before anything touches os.environ
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    override=True
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base_dir")
MANIFEST_FILE = os.path.join(BASE_DIR, "kb_manifest.json")

EMBEDDING_MODEL = "models/gemini-embedding-2"
LLM_MODEL = "gemini-2.5-flash"
CHROMA_COLLECTION = "documind_kb"

os.makedirs(KB_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared cross-thread state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
GLOBAL_STATE: dict = {
    "retriever": None,
    "is_indexing": False,
    "status_message": "Initializing …",
    "doc_count": 0,
}


# ---------------------------------------------------------------------------
# Cached, lazily-initialised heavy objects (one per process)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


@functools.lru_cache(maxsize=1)
def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)


@functools.lru_cache(maxsize=1)
def _get_chroma_client():
    """Return an authenticated ChromaDB Cloud client."""
    import chromadb

    api_key = os.environ.get("CHROMA_CLOUD_KEY", "")
    db_name = os.environ.get("CHROMA_CLOUD_DB_NAME", "")
    tenant = os.environ.get("CHROMA_TENANT", "")

    if not api_key:
        raise RuntimeError("CHROMA_CLOUD_KEY is not set in .env")

    kwargs = {"api_key": api_key}
    if db_name:
        kwargs["database"] = db_name
    if tenant:
        kwargs["tenant"] = tenant

    return chromadb.CloudClient(**kwargs)


@functools.lru_cache(maxsize=1)
def _get_web_search():
    """Return a Tavily search tool; cached for the life of the process."""
    from langchain_community.tools.tavily_search import TavilySearchResults

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set in .env")
    return TavilySearchResults(max_results=3, tavily_api_key=api_key)


# ---------------------------------------------------------------------------
# Vector store helper
# ---------------------------------------------------------------------------

def _open_vectorstore() -> Chroma:
    """Open (or reuse) the cloud-backed Chroma vector store."""
    return Chroma(
        client=_get_chroma_client(),
        collection_name=CHROMA_COLLECTION,
        embedding_function=_get_embeddings(),
    )


# ---------------------------------------------------------------------------
# Manifest helpers (tracks what's been indexed locally)
# ---------------------------------------------------------------------------

def _load_manifest() -> dict:
    if not os.path.exists(MANIFEST_FILE):
        return {}
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_manifest(manifest: dict) -> None:
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _current_kb_files() -> dict:
    """Return {absolute_path: mtime} for every PDF currently on disk."""
    files = glob.glob(os.path.join(KB_DIR, "*.pdf"))
    return {os.path.abspath(p): os.path.getmtime(p) for p in files}


def detect_directory_mutation() -> bool:
    """Cheap check: does the disk state differ from the saved manifest?"""
    return _current_kb_files() != _load_manifest()


# ---------------------------------------------------------------------------
# Indexing (runs in a background thread)
# ---------------------------------------------------------------------------

def run_incremental_indexing() -> None:
    """
    Diffs knowledge_base_dir against the manifest and applies only the
    deltas (add / modify / delete) to the cloud Chroma collection.
    """
    with _lock:
        GLOBAL_STATE["status_message"] = "Scanning knowledge base for changes …"

    print("\n[INDEXING] Scanning local directory for changes ...", flush=True)
    try:
        current = _current_kb_files()
        manifest = _load_manifest()

        deleted = [p for p in manifest if p not in current]
        added_or_modified = [p for p, m in current.items() if manifest.get(p) != m]

        print(f"[INDEXING] Scanning complete. Local folder contains {len(current)} PDF(s).", flush=True)
        if deleted:
            print(f"[INDEXING] Detected deleted PDFs: {[os.path.basename(p) for p in deleted]}", flush=True)
        if added_or_modified:
            print(f"[INDEXING] Detected new/modified PDFs: {[os.path.basename(p) for p in added_or_modified]}", flush=True)

        if not current:
            _save_manifest({})
            with _lock:
                GLOBAL_STATE["retriever"] = None
                GLOBAL_STATE["doc_count"] = 0
                GLOBAL_STATE["status_message"] = "Knowledge base is empty. Sync from Google Drive to begin."
            print("[INDEXING] Local knowledge base folder is empty. Cleared manifest.\n", flush=True)
            return

        vectorstore = _open_vectorstore()

        # Purge stale vectors first
        for path in deleted + added_or_modified:
            fname = os.path.basename(path)
            try:
                print(f"[INDEXING] [CHROMA] Purging existing vectors for source file: {fname} ...", flush=True)
                vectorstore._collection.delete(where={"source": path})
            except Exception as e:
                print(f"[INDEXING] [CHROMA] [WARNING] Failed to delete existing vectors for {fname}: {e}", flush=True)
                pass
            if path in deleted:
                manifest.pop(path, None)

        # Re-embed new / changed files
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        failures: list[str] = []

        for path in added_or_modified:
            fname = os.path.basename(path)
            try:
                with _lock:
                    GLOBAL_STATE["status_message"] = (
                        f"Indexing {fname} …"
                    )
                print(f"[INDEXING] [PDF] Loading {fname} ...", flush=True)
                loader = PyPDFLoader(path)
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = path
                print(f"[INDEXING] [PDF] Loaded {len(docs)} pages from {fname}.", flush=True)
                
                chunks = splitter.split_documents(docs)
                print(f"[INDEXING] [SPLIT] Split {fname} into {len(chunks)} text chunks.", flush=True)
                
                if chunks:
                    print(f"[INDEXING] [CHROMA] Embedding and uploading {len(chunks)} chunks to Chroma Cloud ...", flush=True)
                    vectorstore.add_documents(chunks)
                    print(f"[INDEXING] [CHROMA] Successfully uploaded chunks for {fname}.", flush=True)
                manifest[path] = current[path]
            except Exception as exc:
                print(f"[INDEXING] [ERROR] Failed to process {fname}: {exc}", flush=True)
                failures.append(f"{fname} ({exc})")

        _save_manifest(manifest)

        with _lock:
            GLOBAL_STATE["doc_count"] = len(manifest)
            if manifest:
                GLOBAL_STATE["retriever"] = vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )
                msg = f"Online. {len(manifest)} document(s) indexed."
                if failures:
                    msg += f" Skipped (will retry): {', '.join(failures)}"
                GLOBAL_STATE["status_message"] = msg
            else:
                GLOBAL_STATE["retriever"] = None
                GLOBAL_STATE["status_message"] = "No documents could be indexed."

        print(f"[INDEXING] Finished incremental run. Total indexed docs: {len(manifest)}.\n", flush=True)

    except Exception as exc:
        print(f"[INDEXING] [FATAL ERROR] {exc}", flush=True)
        import traceback
        traceback.print_exc()
        with _lock:
            GLOBAL_STATE["status_message"] = f"Indexing error: {exc}"
    finally:
        with _lock:
            GLOBAL_STATE["is_indexing"] = False


def start_observer_sync() -> None:
    """
    Atomically check-and-set is_indexing under the lock BEFORE spawning the
    thread, preventing duplicate concurrent indexing runs.
    """
    with _lock:
        if GLOBAL_STATE["is_indexing"]:
            return
        GLOBAL_STATE["is_indexing"] = True

    thread = threading.Thread(target=run_incremental_indexing, daemon=True)
    thread.start()


def bootstrap_from_disk() -> None:
    """
    Called once at process startup.
    Probes the cloud Chroma collection: if documents already exist, reconnect
    the retriever immediately so the first user request doesn't have to wait
    for a full re-index.
    """
    print("\n[BOOTSTRAP] Connecting to cloud Chroma and verifying local manifest ...", flush=True)
    manifest = _load_manifest()
    if not manifest:
        with _lock:
            GLOBAL_STATE["status_message"] = "Knowledge base is empty. Sync from Google Drive to begin."
        print("[BOOTSTRAP] Local manifest is empty. No documents to load.\n", flush=True)
        return

    try:
        vectorstore = _open_vectorstore()
        count = vectorstore._collection.count()
        print(f"[BOOTSTRAP] Chroma Cloud database contains {count} vector records.", flush=True)
        if count > 0:
            with _lock:
                GLOBAL_STATE["retriever"] = vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )
                GLOBAL_STATE["doc_count"] = len(manifest)
                GLOBAL_STATE["status_message"] = (
                    f"Online. {len(manifest)} document(s) indexed."
                )
            print(f"[BOOTSTRAP] Successfully initialized retriever from cloud with {len(manifest)} local files recorded.\n", flush=True)
        else:
            with _lock:
                GLOBAL_STATE["status_message"] = (
                    "Connected to cloud. Knowledge base not indexed yet."
                )
            print("[BOOTSTRAP] Connected to cloud. Cloud database is empty.\n", flush=True)
    except Exception as exc:
        with _lock:
            GLOBAL_STATE["status_message"] = (
                f"Could not connect to Chroma Cloud: {exc}"
            )
        print(f"[BOOTSTRAP] [ERROR] Failed to bootstrap connection to cloud: {exc}\n", flush=True)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are MedStudy Assistant, a precise study companion for medical students "
    "preparing for exams. You have access to two sources of information:\n"
    "1. LOCAL STUDY MATERIALS: Excerpts from the student's own uploaded PDFs.\n"
    "2. WEB SEARCH RESULTS: Live results from the internet.\n\n"
    "Always prioritise LOCAL STUDY MATERIALS. Only draw on WEB SEARCH RESULTS "
    "when the local materials are insufficient or silent on the topic — and when "
    "you do, clearly prefix those points with '(Web)' so the student knows the "
    "source. If neither source contains enough information, say so plainly."
)

FEW_SHOT_EXAMPLES = """\
Example 1 — answered from local materials:
Context: "The sinoatrial (SA) node initiates electrical impulses at 60-100 bpm."
Question: What structure initiates the heartbeat?
Reasoning: The local context names the SA node as the pacemaker.
Final Answer: The sinoatrial (SA) node, located in the right atrium, initiates the heartbeat at 60-100 beats per minute.

Example 2 — supplemented with web:
Local context: (no relevant content)
Web context: "Beta-blockers reduce heart rate by blocking β1-adrenergic receptors."
Question: How do beta-blockers slow the heart?
Reasoning: The local materials don't cover this. Using web search result.
Final Answer: (Web) Beta-blockers competitively block β1-adrenergic receptors in the SA node, reducing its firing rate and slowing the heart.\
"""

RAG_TEMPLATE = """\
{system_prompt}

Examples of the reasoning style to follow:
{few_shot}

Answer the new question using the sources below.
First write brief step-by-step reasoning grounded in the sources, then give \
the final answer after the exact marker "Final Answer:".

Conversation History:
{history}

━━━ LOCAL STUDY MATERIALS ━━━
{local_context}

━━━ WEB SEARCH RESULTS ━━━
{web_context}

Question: {question}
"""


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def get_answer(question: str, history_buffer: str = "") -> dict:
    """
    Runs retrieval (local PDFs + web search) then the CoT/few-shot chain.

    Returns a dict:
        answer        : str  – clean final answer shown in the chat bubble
        reasoning     : str  – CoT trace shown in the 'Show reasoning' expander
        raw           : str  – full model output (for debugging)
        web_used      : bool – True if web search returned any results
        web_results   : list – list of {"url": url, "content": content} search results
        local_results : list – list of {"source": source_file, "content": page_content} local matches
    """
    print(f"\n[QUERY] New user query received: '{question}'", flush=True)
    with _lock:
        retriever = GLOBAL_STATE.get("retriever")

    # --- 1. Local vector retrieval ---
    local_results_list = []
    if retriever is not None:
        print("[QUERY] [CHROMA] Querying local document vectors ...", flush=True)
        docs = retriever.invoke(question)
        print(f"[QUERY] [CHROMA] Retrieved {len(docs)} relevant text chunks.", flush=True)
        for i, d in enumerate(docs):
            src = os.path.basename(d.metadata.get("source", "unknown"))
            print(f"  -> Chunk {i+1} source: {src}", flush=True)
            local_results_list.append({
                "source": src,
                "content": d.page_content
            })
        local_context = "\n\n".join(d.page_content for d in docs)
    else:
        print("[QUERY] [CHROMA] Retriever not available (empty/indexing DB). Skipping local retrieval.", flush=True)
        local_context = "(Knowledge base is empty or still indexing.)"

    # --- 2. Tavily web search ---
    web_context = "(Web search unavailable.)"
    web_used = False
    web_results_list = []
    print("[QUERY] [TAVILY] Triggering Tavily web search ...", flush=True)
    try:
        results = _get_web_search().invoke(question)
        if results:
            print(f"[QUERY] [TAVILY] Retrieved {len(results)} search results.", flush=True)
            snippets = []
            for r in results:
                if isinstance(r, dict):
                    url = r.get('url', '')
                    content = r.get('content', '')
                    print(f"  -> Search hit: {url}", flush=True)
                    web_results_list.append({
                        "url": url,
                        "content": content
                    })
                    snippets.append(f"[{url}]\n{content}")
                else:
                    web_results_list.append({
                        "url": "Unknown",
                        "content": str(r)
                    })
                    snippets.append(str(r))
            web_context = "\n\n".join(snippets)
            web_used = True
        else:
            print("[QUERY] [TAVILY] Tavily web search returned 0 results.", flush=True)
    except Exception as exc:
        print(f"[QUERY] [TAVILY] [ERROR] Tavily search failed: {exc}", flush=True)
        web_context = f"(Web search failed: {exc})"

    # --- 3. LLM chain ---
    print("[QUERY] [LLM] Generating response using Gemini CoT model ...", flush=True)
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    chain = prompt | _get_llm() | StrOutputParser()

    raw = chain.invoke(
        {
            "system_prompt": SYSTEM_PROMPT,
            "few_shot": FEW_SHOT_EXAMPLES,
            "history": history_buffer or "(none yet)",
            "local_context": local_context.strip() or "(no relevant local context found)",
            "web_context": web_context,
            "question": question,
        }
    )

    if "Final Answer:" in raw:
        reasoning, final = raw.split("Final Answer:", 1)
        print("[QUERY] [LLM] Response successfully generated with CoT reasoning.\n", flush=True)
    else:
        reasoning, final = "", raw
        print("[QUERY] [LLM] Response generated without 'Final Answer:' delimiter.\n", flush=True)

    return {
        "answer": final.strip(),
        "reasoning": reasoning.strip(),
        "raw": raw,
        "web_used": web_used,
        "web_results": web_results_list,
        "local_results": local_results_list,
    }
