"""
rag_core.py
-----------
Stateless cloud-native RAG core.
- Connects directly to Chroma Cloud (no local database).
- Syncs PDFs from Google Drive into a temporary directory using gdown, 
  indexes them, and cleans up immediately (no local files are kept).
- Stores index metadata (filenames and sizes) directly in Chroma Cloud's 
  chunk metadata, eliminating the need for local manifest files.
"""

import functools
import glob
import json
import logging
import os
import re
import tempfile
import threading

from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    override=True
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "models/gemini-embedding-2"
LLM_MODEL = "gemini-2.5-flash"
CHROMA_COLLECTION = "documind_kb"

# ---------------------------------------------------------------------------
# Shared cross-thread state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
GLOBAL_STATE: dict = {
    "retriever": None,
    "is_indexing": False,
    "status_message": "Initializing …",
    "doc_count": 0,
    "indexed_files": [],
}


# ---------------------------------------------------------------------------
# Cached, lazily-initialised heavy objects
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
    """Return a Tavily search tool."""
    from langchain_community.tools.tavily_search import TavilySearchResults

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set in .env")
    return TavilySearchResults(max_results=3, tavily_api_key=api_key)


# ---------------------------------------------------------------------------
# Vector store helper
# ---------------------------------------------------------------------------

def _open_vectorstore() -> Chroma:
    """Open the cloud-backed Chroma vector store."""
    return Chroma(
        client=_get_chroma_client(),
        collection_name=CHROMA_COLLECTION,
        embedding_function=_get_embeddings(),
    )


# ---------------------------------------------------------------------------
# Cloud metadata state helpers (Chroma Cloud is the single source of truth)
# ---------------------------------------------------------------------------

def _calculate_file_hash(filepath: str) -> str:
    """Calculate SHA-256 hash of a file's content to detect updates reliably."""
    import hashlib
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def get_indexed_files_from_cloud() -> dict[str, str]:
    """Query Chroma Cloud to get {filename: file_hash} of currently indexed documents."""
    try:
        vectorstore = _open_vectorstore()
        results = vectorstore._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        
        files = {}
        for meta in metadatas:
            if meta and "source" in meta:
                fname = os.path.basename(meta["source"])
                fhash = meta.get("file_hash", "")
                files[fname] = fhash
        return files
    except Exception as exc:
        print(f"[METADATA] [ERROR] Failed to fetch indexed files from cloud: {exc}", flush=True)
        return {}


def _extract_folder_id(drive_url: str) -> str:
    """Extract the raw folder ID from a Google Drive URL."""
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", drive_url)
    if match:
        return match.group(1)
    return drive_url.split("?")[0].strip("/").split("/")[-1]


# ---------------------------------------------------------------------------
# Indexing (runs in a background thread)
# ---------------------------------------------------------------------------

def run_incremental_indexing(drive_url: str) -> None:
    """
    Downloads files from Google Drive to a temporary directory, checks for 
    changes against current Chroma Cloud metadata, and updates the database.
    """
    import gdown

    with _lock:
        GLOBAL_STATE["status_message"] = "Connecting to Google Drive …"

    folder_id = _extract_folder_id(drive_url)
    print(f"\n[INDEXING] Starting sync for Google Drive folder ID: {folder_id} ...", flush=True)

    try:
        # Create a temporary directory that cleans up automatically
        with tempfile.TemporaryDirectory() as tmp_dir:
            print("[INDEXING] Downloading folder contents using gdown to temp directory...", flush=True)
            downloaded = gdown.download_folder(
                id=folder_id,
                output=tmp_dir,
                quiet=True,
                use_cookies=False,
            )
            if not downloaded:
                downloaded = []

            # Filter PDFs
            pdf_paths = [
                p for p in downloaded
                if p and p.lower().endswith(".pdf") and os.path.exists(p)
            ]

            temp_files = {os.path.basename(p): _calculate_file_hash(p) for p in pdf_paths}
            print(f"[INDEXING] Google Drive folder contains {len(temp_files)} PDF(s).", flush=True)

            # Get database status directly from Chroma Cloud
            print("[INDEXING] Fetching current database status from Chroma Cloud ...", flush=True)
            cloud_files = get_indexed_files_from_cloud()
            print(f"[INDEXING] Chroma Cloud contains {len(cloud_files)} indexed document(s).", flush=True)

            deleted = [name for name in cloud_files if name not in temp_files]
            added_or_modified = [name for name, fhash in temp_files.items() if cloud_files.get(name) != fhash]

            if deleted:
                print(f"[INDEXING] Detected deleted PDFs: {deleted}", flush=True)
            if added_or_modified:
                print(f"[INDEXING] Detected new/modified PDFs: {added_or_modified}", flush=True)

            if not temp_files and not cloud_files:
                with _lock:
                    GLOBAL_STATE["retriever"] = None
                    GLOBAL_STATE["doc_count"] = 0
                    GLOBAL_STATE["status_message"] = "Knowledge base is empty. Sync from Google Drive to begin."
                print("[INDEXING] Google Drive is empty and database is clean.\n", flush=True)
                return

            vectorstore = _open_vectorstore()

            # 1. Purge deleted/modified documents by matching ID metadata
            if deleted or added_or_modified:
                print("[INDEXING] [CHROMA] Checking for stale database records to purge ...", flush=True)
                results = vectorstore._collection.get(include=["metadatas"])
                metadatas = results.get("metadatas", [])
                ids = results.get("ids", [])
                
                ids_to_delete = []
                for i, meta in enumerate(metadatas):
                    if meta and "source" in meta:
                        fname = os.path.basename(meta["source"])
                        if fname in deleted or fname in added_or_modified:
                            ids_to_delete.append(ids[i])
                
                if ids_to_delete:
                    print(f"[INDEXING] [CHROMA] Deleting {len(ids_to_delete)} stale vector chunks ...", flush=True)
                    vectorstore._collection.delete(ids=ids_to_delete)
                    print("[INDEXING] [CHROMA] Purge complete.", flush=True)

            # 2. Embed new / changed files
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            failures: list[str] = []

            for fname in added_or_modified:
                path = None
                # Locate path in temp dir
                for p in pdf_paths:
                    if os.path.basename(p) == fname:
                        path = p
                        break

                if not path:
                    continue

                fhash = temp_files[fname]
                try:
                    with _lock:
                        GLOBAL_STATE["status_message"] = f"Indexing {fname} …"
                    print(f"[INDEXING] [PDF] Loading {fname} ...", flush=True)
                    loader = PyPDFLoader(path)
                    docs = loader.load()
                    for d in docs:
                        # Save source name and file hash in chunk metadata
                        d.metadata["source"] = fname
                        d.metadata["file_hash"] = fhash
                    print(f"[INDEXING] [PDF] Loaded {len(docs)} pages from {fname}.", flush=True)
                    
                    chunks = splitter.split_documents(docs)
                    print(f"[INDEXING] [SPLIT] Split {fname} into {len(chunks)} text chunks.", flush=True)
                    
                    if chunks:
                        print(f"[INDEXING] [CHROMA] Embedding and uploading {len(chunks)} chunks to Chroma Cloud ...", flush=True)
                        vectorstore.add_documents(chunks)
                        print(f"[INDEXING] [CHROMA] Successfully uploaded chunks for {fname}.", flush=True)
                except Exception as exc:
                    print(f"[INDEXING] [ERROR] Failed to process {fname}: {exc}", flush=True)
                    import traceback
                    traceback.print_exc()
                    failures.append(f"{fname} ({exc})")

            # Final status update
            final_files = get_indexed_files_from_cloud()
            with _lock:
                GLOBAL_STATE["doc_count"] = len(final_files)
                GLOBAL_STATE["indexed_files"] = list(final_files.keys())
                if final_files:
                    GLOBAL_STATE["retriever"] = vectorstore.as_retriever(
                        search_kwargs={"k": 4}
                    )
                    msg = f"Online. {len(final_files)} document(s) indexed."
                    if failures:
                        msg += f" Skipped (will retry): {', '.join(failures)}"
                    GLOBAL_STATE["status_message"] = msg
                else:
                    GLOBAL_STATE["retriever"] = None
                    GLOBAL_STATE["status_message"] = "No documents could be indexed."

            print(f"[INDEXING] Finished incremental run. Total indexed docs: {len(final_files)}.\n", flush=True)

    except Exception as exc:
        print(f"[INDEXING] [FATAL ERROR] {exc}", flush=True)
        import traceback
        traceback.print_exc()
        with _lock:
            GLOBAL_STATE["status_message"] = f"Indexing error: {exc}"
    finally:
        with _lock:
            GLOBAL_STATE["is_indexing"] = False


def start_observer_sync(drive_url: str) -> None:
    """Atomically check-and-set is_indexing and spawn background thread."""
    with _lock:
        if GLOBAL_STATE["is_indexing"]:
            return
        GLOBAL_STATE["is_indexing"] = True

    thread = threading.Thread(target=run_incremental_indexing, args=(drive_url,), daemon=True)
    thread.start()


def bootstrap_from_disk() -> None:
    """
    Called once at process startup.
    Probes the cloud Chroma collection: if documents already exist, reconnect
    the retriever immediately so the first user request doesn't have to wait
    for a full re-index.
    """
    print("\n[BOOTSTRAP] Connecting to cloud Chroma and verifying database status ...", flush=True)
    try:
        vectorstore = _open_vectorstore()
        count = vectorstore._collection.count()
        print(f"[BOOTSTRAP] Chroma Cloud database contains {count} vector records.", flush=True)
        if count > 0:
            cloud_files = get_indexed_files_from_cloud()
            with _lock:
                GLOBAL_STATE["retriever"] = vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )
                GLOBAL_STATE["doc_count"] = len(cloud_files)
                GLOBAL_STATE["indexed_files"] = list(cloud_files.keys())
                GLOBAL_STATE["status_message"] = (
                    f"Online. {len(cloud_files)} document(s) indexed."
                )
            print(f"[BOOTSTRAP] Successfully initialized retriever from cloud with {len(cloud_files)} files: {list(cloud_files.keys())}\n", flush=True)
        else:
            with _lock:
                GLOBAL_STATE["status_message"] = (
                    "Connected to cloud. Knowledge base is empty. Sync from Google Drive to begin."
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

    # --- Parallel Retrieval with LangChain ---
    print("[QUERY] [RETRIEVAL] Triggering parallel retrieval from Chroma and Tavily ...", flush=True)
    
    def safe_web_search(q):
        try:
            return _get_web_search().invoke(q)
        except Exception as exc:
            print(f"[QUERY] [TAVILY] [ERROR] Tavily search failed: {exc}", flush=True)
            return []

    parallel_dict = {"web": RunnableLambda(safe_web_search)}
    if retriever is not None:
        parallel_dict["local"] = retriever
        
    parallel_chain = RunnableParallel(parallel_dict)
    parallel_results = parallel_chain.invoke(question)

    # --- 1. Process Local vector retrieval ---
    local_results_list = []
    if retriever is not None and "local" in parallel_results:
        docs = parallel_results["local"]
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

    # --- 2. Process Tavily web search ---
    web_context = "(Web search unavailable.)"
    web_used = False
    web_results_list = []
    if "web" in parallel_results:
        results = parallel_results["web"]
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
