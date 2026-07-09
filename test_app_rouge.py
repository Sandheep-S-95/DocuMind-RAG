"""
test_app_rouge.py
------------------
Lightweight functional test for the RAG pipeline, using the `rouge-score`
package instead of a heavier eval framework (RAGAS, LangSmith, etc). This is
NOT a rigorous correctness proof -- ROUGE just measures textual overlap
between the model's answer and a hand-written reference answer. It's meant
to catch outright breakage (indexing failing, retrieval returning nothing,
prompt formatting errors) before a hackathon demo, not to grade quality.

Usage:
    export GOOGLE_API_KEY="your-key"
    python test_app_rouge.py

Make sure knowledge_base_dir already contains the generated study PDFs
(run generate_kb_docs.py first) before running this.
"""

import os
import sys
import time

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

from rouge_score import rouge_scorer

import rag_core

# ---------------------------------------------------------------------------
# QA pairs matched to the content of the 5 generated study PDFs.
# Reference answers are intentionally short and factual -- ROUGE-L rewards
# overlap in key terms/phrases, which is what we care about here.
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "question": "What structure is the heart's natural pacemaker?",
        "reference": "The sinoatrial (SA) node, located in the right atrium, is the heart's natural pacemaker, firing at 60-100 beats per minute.",
    },
    {
        "question": "Why does the AV node delay the electrical impulse?",
        "reference": "The AV node delays the impulse by about 0.1 seconds so the atria can finish contracting and the ventricles can fill before ventricular contraction begins.",
    },
    {
        "question": "What is functional residual capacity and how is it calculated?",
        "reference": "Functional residual capacity (FRC) is the volume of air remaining in the lungs after a normal tidal expiration, calculated as expiratory reserve volume plus residual volume (FRC = ERV + RV).",
    },
    {
        "question": "What causes a rightward shift of the oxygen-hemoglobin dissociation curve?",
        "reference": "A rightward shift is caused by increased temperature, increased PCO2, decreased pH (Bohr effect), or increased 2,3-BPG, and it decreases hemoglobin's affinity for oxygen, promoting oxygen unloading at tissues.",
    },
    {
        "question": "What is the difference between a high anion gap and a normal anion gap metabolic acidosis?",
        "reference": "A high anion gap metabolic acidosis occurs when an unmeasured acid accumulates, such as in lactic acidosis or diabetic ketoacidosis, while a normal anion gap metabolic acidosis occurs when bicarbonate is lost directly and chloride rises to compensate, such as in severe diarrhea.",
    },
    {
        "question": "Why is blood glucose high in Type 2 diabetics even though they still produce insulin?",
        "reference": "Because peripheral tissues become resistant to insulin's effects, glucose uptake decreases despite normal or elevated insulin levels, so the problem is impaired response to insulin rather than an absolute lack of it.",
    },
    {
        "question": "How do beta-lactam antibiotics kill bacteria?",
        "reference": "Beta-lactam antibiotics bind and inhibit penicillin-binding proteins (transpeptidases), blocking peptidoglycan cross-linking in the cell wall, which causes osmotic lysis of the bacterium.",
    },
    {
        "question": "What is a food allergy and how is it treated?",
        "reference": "NOT_IN_KNOWLEDGE_BASE",  # intentionally out-of-scope, to test that the model doesn't hallucinate
    },
]

PASS_THRESHOLD = 0.28  # ROUGE-L F1 threshold; deliberately lenient, this is a smoke test


def run():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: set GOOGLE_API_KEY before running this test.")
        sys.exit(1)

    drive_url = os.environ.get("KNOWLEDGE_BASE_DRIVE_LINK", "")
    if not drive_url:
        print("ERROR: KNOWLEDGE_BASE_DRIVE_LINK not set in .env")
        sys.exit(1)

    print("Running indexing synchronously (no Streamlit / no background thread needed for testing)...")
    rag_core.run_incremental_indexing(drive_url)
    print(rag_core.GLOBAL_STATE["status_message"])

    if rag_core.GLOBAL_STATE["retriever"] is None:
        print("ERROR: no retriever available after indexing. Did you run generate_kb_docs.py first?")
        sys.exit(1)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    results = []
    for case in TEST_CASES:
        question = case["question"]
        reference = case["reference"]
        print(f"\nQ: {question}")

        result = rag_core.get_answer(question)
        answer = result["answer"]
        print(f"A: {answer[:300]}{'...' if len(answer) > 300 else ''}")

        if reference == "NOT_IN_KNOWLEDGE_BASE":
            # For out-of-scope questions we just check the model admits it
            # doesn't know, rather than scoring ROUGE overlap.
            refusal_signals = ["don't", "does not contain", "cannot", "insufficient", "no relevant", "not contain"]
            passed = any(sig in answer.lower() for sig in refusal_signals)
            results.append((question, None, passed))
            print(f"  -> Out-of-scope check: {'PASS' if passed else 'FAIL (model may have hallucinated)'}")
            continue

        scores = scorer.score(reference, answer)
        rouge_l_f1 = scores["rougeL"].fmeasure
        passed = rouge_l_f1 >= PASS_THRESHOLD
        results.append((question, rouge_l_f1, passed))
        print(f"  -> ROUGE-L F1: {rouge_l_f1:.3f} {'PASS' if passed else 'FAIL'}")

        time.sleep(1)  # be gentle on free-tier rate limits

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for _, _, p in results if p)
    for question, score, passed in results:
        score_str = f"{score:.3f}" if score is not None else "n/a"
        print(f"[{'PASS' if passed else 'FAIL'}] (score={score_str}) {question}")
    print(f"\n{n_pass}/{len(results)} checks passed.")

    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    run()
