import os
import shutil
import time
import sys

# Ensure reportlab is available
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
except ImportError:
    print("Error: reportlab is required to generate the test PDF. Run 'pip install reportlab' first.")
    sys.exit(1)

import rag_core

# ---------------------------------------------------------------------------
# Test configuration and paths
# ---------------------------------------------------------------------------
TEST_PDF_NAME = "Test_Neuroscience.pdf"
TEST_PDF_PATH = os.path.join(rag_core.KB_DIR, TEST_PDF_NAME)
BACKUP_DIR = os.path.join(rag_core.BASE_DIR, "test_backup")
BACKUP_PDF_PATH = os.path.join(BACKUP_DIR, TEST_PDF_NAME)

os.makedirs(BACKUP_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: Create test PDF
# ---------------------------------------------------------------------------
def generate_test_pdf():
    print(f"\n[Step 1] Generating test PDF: {TEST_PDF_PATH}...")
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("DocTitle", parent=styles["Title"], spaceAfter=18)
    h_style = ParagraphStyle("H", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=8)

    doc = SimpleDocTemplate(TEST_PDF_PATH, pagesize=letter, topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    story = [
        Paragraph("Neurobiology of the Habenula", title_style),
        Spacer(1, 6),
        Paragraph("Functional Role of the Lateral Habenula", h_style),
        Paragraph(
            "The lateral habenula (LHb) acts as a key node in the anti-reward system, firing when expected rewards are omitted. "
            "It projects signals to the rostromedial tegmental nucleus (RMTg) using glutamate as its primary neurotransmitter. "
            "In patients with severe refractory depression, deep brain stimulation (DBS) targeting the lateral habenula has been "
            "shown to reduce depressive symptoms by inhibiting LHb hyperactivity.",
            body_style
        )
    ]
    doc.build(story)
    print("Test PDF successfully generated.")

# ---------------------------------------------------------------------------
# Helper: Print Query and Response
# ---------------------------------------------------------------------------
def ask_question(question: str, category_name: str):
    print("\n" + "=" * 60)
    print(f"[{category_name}] Query: {question}")
    print("=" * 60)
    
    result = rag_core.get_answer(question)
    
    print("\n--- REASONING ---")
    print(result.get("reasoning", "(No reasoning provided)"))
    print("\n--- ANSWER ---")
    print(result.get("answer", "(No answer returned)"))
    print("=" * 60)
    return result

# ---------------------------------------------------------------------------
# Main test execution sequence
def main():
    # Attempt to load GOOGLE_API_KEY from venv/.env
    env_file = os.path.join(rag_core.BASE_DIR, "venv", ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip("'\"")
                            if key == "GOOGLE_API_KEY":
                                os.environ["GOOGLE_API_KEY"] = val
                                print(f"Loaded {key} from venv/.env")
        except Exception as e:
            print(f"Warning: could not read venv/.env: {e}")

    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable is not set and was not found in venv/.env.")
        sys.exit(1)

    # 1. Generate the neuroscience PDF in the knowledge base directory
    generate_test_pdf()

    # 2. Run incremental indexing
    print("\n[Step 2] Triggering incremental indexing to add the document...")
    rag_core.run_incremental_indexing()
    print(f"Indexing Status: {rag_core.GLOBAL_STATE['status_message']}")

    # 3. Test Phase 1: Ask the three categories of questions
    print("\n[Step 3] Running Phase 1 QA Tests (Document Present)...")
    
    # Category A: Very specific to the document
    q_a = "What neurotransmitter does the lateral habenula use to project signals to the rostromedial tegmental nucleus?"
    ask_question(q_a, "Category A - Specific to Document")
    
    # Category B: Specific to document topic but outside the document (testing web search/outside knowledge refusal)
    q_b = "What is the typical voltage and frequency used in deep brain stimulation of the lateral habenula for depression?"
    ask_question(q_b, "Category B - Specific but Outside Document")

    # Category C: Unrelated question
    q_c = "What is the capital of France?"
    ask_question(q_c, "Category C - Unrelated Question")

    # 4. Remove the document (move it to another folder)
    print("\n[Step 4] Moving test PDF to backup folder to simulate deletion...")
    if os.path.exists(TEST_PDF_PATH):
        shutil.move(TEST_PDF_PATH, BACKUP_PDF_PATH)
        print(f"Moved {TEST_PDF_NAME} to {BACKUP_PDF_PATH}")
    else:
        print("Error: Test PDF not found at path.")
        sys.exit(1)

    # 5. Run incremental indexing again
    print("\n[Step 5] Triggering incremental indexing to process document deletion...")
    rag_core.run_incremental_indexing()
    print(f"Indexing Status: {rag_core.GLOBAL_STATE['status_message']}")

    # 6. Test Phase 2: Ask the Category A question again to verify deletion
    print("\n[Step 6] Running Phase 2 QA Tests (Document Deleted)...")
    ask_question(q_a, "Category A - Specific to Document (Verify Deletion)")

    # Removed cleanup for .env as it should be persistent

    print("\nTesting complete.")

if __name__ == "__main__":
    main()
