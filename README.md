# Documind: MedStudy RAG Assistant

Documind is a high-performance, cloud-backed Retrieval-Augmented Generation (RAG) study companion designed for medical students. It leverages Google Gemini, Chroma Cloud, and Tavily Web Search to answer complex medical questions. It prioritizes local PDF study materials and falls back to live web searches when necessary, complete with step-by-step chain-of-thought reasoning.

---

## 1. Architectural Diagram

Below is the high-level architecture showing how the Streamlit frontend, background threads, Chroma Cloud database, Tavily search, and the Gemini LLM interact:

```mermaid
flowchart TD
    subgraph Frontend [Streamlit UI]
        UI_Sync[Sync from Google Drive Button]
        UI_Query[User asks a question]
        UI_Status[Reads GLOBAL_STATE for Status]
    end

    subgraph Threading [State Management]
        Lock((Threading Lock))
        State[(GLOBAL_STATE Dict)]
    end

    subgraph Background [Background Indexing Thread]
        GDrive[gdown sync]
        TempDir[tempfile.TemporaryDirectory]
        Embedder[Google Generative AI Embeddings]
    end

    subgraph Database [Chroma Cloud]
        Chroma[(Chroma Cloud Collection)]
    end

    subgraph WebSearch [Web Search Integration]
        Tavily[Tavily Search API]
    end

    subgraph Generation [LLM Pipeline]
        Retriever[Chroma Cloud Retriever k=4]
        LLM[Gemini 2.5 Flash]
    end

    %% Flow for Indexing
    UI_Sync --> GDrive
    GDrive -->|PDF Downloads| TempDir
    TempDir <-->|Diff filenames & sizes directly| Chroma
    TempDir -->|Delta PDF files| Embedder
    Embedder -->|Vector Chunks + Metadata| Chroma
    TempDir -->|Automatic Cleanup| Clean[Folder Deleted]
    TempDir -->|Update Status| Lock
    Lock --- State

    %% Flow for Querying
    UI_Query --> Retriever
    Retriever -->|Fetch relevant chunks| Chroma
    Chroma -->|Return Top-K chunks| Retriever
    UI_Query --> Tavily
    Tavily -->|Fetch Web Snippets| LLM
    Retriever --> LLM
    LLM -->|Clean Answer + Reasoning| UI_Query
    UI_Status --> State
```

---

## 2. Important Features & Background Flows

While the user interacts with a clean chat interface, several critical flows occur silently behind the scenes:

### Feature A: Stateless Incremental Indexing
Most basic RAG apps re-embed the entire directory every time indexing is triggered, wasting API tokens and execution time.
*   **Behind the scenes:** The database itself serves as the single source of truth. Document metadata (filenames and file sizes) are stored directly inside Chroma Cloud's chunk metadatas.
*   **The Flow:** When indexing runs, the PDFs are downloaded to a temporary directory. The app queries Chroma Cloud to get the currently indexed files and their sizes. If a file is deleted from Google Drive, it purges its vectors from Chroma Cloud. If a file is added or modified, it embeds and uploads *only* that file, keeping the system clean and local-storage-free.

### Feature B: Thread-Safe State Management (Decoupled from Streamlit)
Streamlit's `st.session_state` is tied to specific browser session context (`ScriptRunContext`). A background thread attempting to update UI state would normally crash Streamlit.
*   **Behind the scenes:** The background indexing task never touches `st.session_state` directly.
*   **The Flow:** Instead, a module-level Python dictionary (`GLOBAL_STATE` in `rag_core.py`) is used, protected by a `threading.Lock()`. The background thread safely writes to this dictionary, and the Streamlit UI reads from it on every reload, providing live progress updates safely.

### Feature C: Context-Grounded Prompting with Web Fallback
Medical RAG systems must not guess answers if the information is missing from the uploaded guides.
*   **Behind the scenes:** The pipeline uses a strict Chain-of-Thought (CoT) and Few-Shot system prompt that merges local vector database results and live web search.
*   **The Flow:** Before outputting the final answer, the LLM is forced to output a `Reasoning:` block. This grounds the model in the context chunks retrieved by Chroma Cloud. It prioritizes local PDF materials; if information is only found via web search, it prefixes those points with `(Web)`. If neither contains the answer, the LLM outputs a clean refusal rather than fabricating medical advice.

---

## 3. Chroma Cloud and Google Drive Integration

*   **Google Drive Syncing**: Uses `gdown` to connect to a publicly shared Google Drive folder link without requiring OAuth configurations. The files are downloaded directly into a `tempfile.TemporaryDirectory()`, indexed, and automatically deleted, keeping local storage completely clean.
*   **Chroma Cloud (`CloudClient`)**: Connects to the serverless Chroma Cloud hosting using your account credentials (`CHROMA_TENANT` and `CHROMA_CLOUD_KEY`), avoiding local database storage overhead.

---

## 4. UI Performance Enhancements

*   **@st.fragment Chat Section**: The entire chat UI is wrapped in a Streamlit fragment. When a user submits a message, only the chat section reruns instead of reloading the entire page, index status, or sidebar metrics, improving response times.
*   **Resource Pre-Warming**: LangChain embeddings, the Gemini LLM, and the Tavily client are pre-warmed on server startup using `@st.cache_resource`. This eliminates the "first-query delay" for users.

---

## 5. Testing & Quality Evaluation

The pipeline features two comprehensive test scripts to verify core system integrity:

### ROUGE-Score Evaluation (`test_app_rouge.py`)
To avoid heavier enterprise evaluation frameworks (like RAGAS or LangSmith), the project uses a lightweight functional test with the `rouge-score` library.
*   **Textual Overlap**: Measures ROUGE-1 and ROUGE-L (Longest Common Subsequence) recall and F1 overlap against a suite of hand-written reference answers mapping to the 5 medical PDF subjects.
*   **Refusal Validation**: Intentionally includes out-of-scope questions (e.g., *"What is a food allergy and how is it treated?"*) to ensure that the system refuses to answer and does not hallucinate, checking for phrases like *"insufficient"* or *"no relevant"*.
*   **Lenient Smoke Test**: Runs with a ROUGE-L F1 threshold of `0.28` to verify indexing, retrieval, and prompts are operational.

### End-to-End Deletion & Query Lifecycle Test (`test_app_rogue2.py`)
Validates the dynamic lifecycle of the document sync and database state:
1.  **Generation**: Programmatically generates a neuroscience test PDF (`Test_Neuroscience.pdf`) in the folder.
2.  **Creation Indexing**: Indexes the new file and confirms the database connection.
3.  **Positive Verification**: Queries a highly specific question (*"What neurotransmitter does the lateral habenula use..."*) and asserts that it retrieves the answer from the document.
4.  **Negative & Fallback Validation**: Queries out-of-scope/unrelated questions to confirm web search fallbacks and refusals are grounded.
5.  **Deletion Incremental Indexing**: Moves the PDF to simulate deletion and triggers incremental indexing.
6.  **Purge Verification**: Re-queries the specific neuroscience question to confirm that its vectors were successfully purged from the database and the model no longer answers from the document.

### Verification Results & Test Outputs
The indexing pipelines and document lifecycles were successfully run and verified. Below are the actual execution logs showing the add-query-delete-verify cycle working against the live Chroma Cloud instance:

#### 1. Sync & Indexing Trial Run Output (`test_run.py`)
```text
[GDRIVE SYNC] Starting sync for folder ID: 1k9XaWHSBNbHbIXyZrstssumE1l9OG7Ap ...
[GDRIVE SYNC] Found 5 PDF files in Google Drive folder.
[GDRIVE SYNC] Completed: 5 added/updated, 0 unchanged, 0 deleted.

[INDEXING] Scanning local directory for changes ...
[INDEXING] Scanning complete. Local folder contains 5 PDF(s).
[INDEXING] [CHROMA] Purging existing vectors for source file: Cardiovascular_Physiology.pdf ...
[INDEXING] [PDF] Loaded 2 pages from Cardiovascular_Physiology.pdf.
[INDEXING] [CHROMA] Embedding and uploading 6 chunks to Chroma Cloud ...
[INDEXING] Finished incremental run. Total indexed docs: 5.
```

#### 2. Lifecycle and Purge Verification Output (`test_app_rogue2.py`)
```text
[Step 1] Generating test PDF: Test_Neuroscience.pdf...
[Step 2] Triggering incremental indexing to add the document...
[INDEXING] Detected new/modified PDFs: ['Test_Neuroscience.pdf']
[INDEXING] [CHROMA] Embedding and uploading 1 chunks to Chroma Cloud ...

[Step 3] Running Phase 1 QA Tests (Document Present)...
[Category A - Specific to Document] Query: What neurotransmitter does the lateral habenula use...
--- ANSWER ---
The lateral habenula (LHb) uses glutamate as its primary neurotransmitter...

[Step 4] Moving test PDF to backup folder to simulate deletion...
[Step 5] Triggering incremental indexing to process document deletion...
[INDEXING] Detected deleted PDFs: ['Test_Neuroscience.pdf']
[INDEXING] [CHROMA] Purging existing vectors for source file: Test_Neuroscience.pdf ...

[Step 6] Running Phase 2 QA Tests (Document Deleted)...
[Category A - Specific to Document (Verify Deletion)] Query: What neurotransmitter does the lateral habenula use...
--- ANSWER ---
Neither the local study materials nor the web search results contain enough information...
```

---

## 6. Local Development Setup

### Installation
1.  Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration
Create a `.env` file in the root folder with the following:
```env
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
CHROMA_CLOUD_KEY=your_chroma_cloud_api_key
CHROMA_TENANT=your_chroma_cloud_tenant_id
CHROMA_CLOUD_DB_NAME=Documind
KNOWLEDGE_BASE_DRIVE_LINK=your_public_google_drive_folder_url
```

### Running the App
Start the Streamlit server:
```bash
.\venv\Scripts\python.exe -m streamlit run app.py
```

### Running Tests
Execute the ROUGE score or lifecycle evaluation test scripts:
```bash
# Verify textual overlap & refusals
.\venv\Scripts\python.exe test_app_rouge.py

# Verify indexing addition, query, deletion, and vector purging
.\venv\Scripts\python.exe test_app_rogue2.py
```
