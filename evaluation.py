import os
import re
import json
import time
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from retrieval import ask

# =========================================================
# CONFIG
# =========================================================
QUESTIONS_FILE = "questions.txt"
DOCS_PATH = "docs"
RESULTS_CSV = "ragas_results.csv"
DETAILS_JSON = "ragas_details.json"

MAX_QUESTIONS = 20
OLLAMA_MODEL = "llama3.2"


# =========================================================
# HELPERS
# =========================================================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def clean_answer_text(answer: str) -> str:
    answer = re.sub(r"\*\*Section:\*\*.*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\*\*Source:\*\*.*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


# =========================================================
# BLEU SCORE
# =========================================================
def compute_bleu(reference: str, candidate: str) -> float:
    ref_tokens = normalize_text(reference).split()
    cand_tokens = normalize_text(candidate).split()

    if not cand_tokens or not ref_tokens:
        return 0.0

    from collections import Counter
    clipped = 0
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)

    for token, count in cand_counts.items():
        clipped += min(count, ref_counts.get(token, 0))

    precision = clipped / len(cand_tokens) if cand_tokens else 0
    brevity = min(1.0, len(cand_tokens) / len(ref_tokens)) if ref_tokens else 0

    return precision * brevity


# =========================================================
# LLM-BASED EVALUATION (using local Ollama)
# =========================================================
def evaluate_faithfulness(llm: ChatOllama, response: str, contexts: List[str]) -> float:
    if not contexts or not response.strip():
        return 0.0

    context_text = "\n\n".join(contexts[:3])
    prompt = f"""You are an evaluation judge. Rate how faithful the following response is to the provided context.
A faithful response only contains information that can be found in or inferred from the context.

Context:
{context_text}

Response:
{response}

Rate faithfulness as a score from 0.0 to 1.0 where:
- 1.0 = completely faithful, all claims supported by context
- 0.0 = completely unfaithful, no claims supported

Reply with ONLY a number between 0.0 and 1.0, nothing else."""

    try:
        result = llm.invoke(prompt)
        score = float(re.search(r"([01]\.?\d*)", result.content).group(1))
        return min(max(score, 0.0), 1.0)
    except Exception:
        return 0.0


def evaluate_context_recall(llm: ChatOllama, response: str, contexts: List[str], reference: str) -> float:
    if not contexts or not reference.strip():
        return 0.0

    context_text = "\n\n".join(contexts[:3])
    prompt = f"""You are an evaluation judge. Rate how well the retrieved context covers the information needed to answer correctly.

Reference answer (ground truth):
{reference}

Retrieved context:
{context_text}

Rate context recall as a score from 0.0 to 1.0 where:
- 1.0 = context contains all information from the reference
- 0.0 = context contains none of the reference information

Reply with ONLY a number between 0.0 and 1.0, nothing else."""

    try:
        result = llm.invoke(prompt)
        score = float(re.search(r"([01]\.?\d*)", result.content).group(1))
        return min(max(score, 0.0), 1.0)
    except Exception:
        return 0.0


# =========================================================
# LOAD QUESTIONS
# =========================================================
def load_questions(file_path: str = QUESTIONS_FILE, limit: int = MAX_QUESTIONS) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Questions file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    if not questions:
        raise ValueError("No questions found in question.txt")

    return questions[:limit]


# =========================================================
# PARSE MARKDOWN FILES
# =========================================================
def extract_qa_pairs_from_markdown(content: str) -> List[Tuple[str, str]]:
    pattern = re.compile(
        r"##\s*Question\s*\n(.*?)\n+###\s*Answer\s*\n(.*?)(?=\n+---|\n+##\s*Question|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(content)
    return [(q.strip(), clean_answer_text(a.strip())) for q, a in matches if q.strip() and a.strip()]


def build_reference_map(docs_path: str = DOCS_PATH) -> Dict[str, str]:
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Docs folder not found: {docs_path}")

    reference_map: Dict[str, str] = {}
    for filename in os.listdir(docs_path):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(docs_path, filename), "r", encoding="utf-8") as f:
            content = f.read()
        for question, answer in extract_qa_pairs_from_markdown(content):
            normalized_q = normalize_text(question)
            if normalized_q not in reference_map:
                reference_map[normalized_q] = answer

    if not reference_map:
        raise ValueError("No question-answer pairs found in markdown files.")
    return reference_map


# =========================================================
# FALLBACK REFERENCE
# =========================================================
def build_reference_from_sources(sources: List[Dict]) -> str:
    if not sources:
        return "No reliable reference available."
    top_content = sources[0].get("content", "").strip()
    return top_content[:3000] if top_content else "No reliable reference available."


# =========================================================
# RUN RAG
# =========================================================
def run_rag_on_questions(questions: List[str], reference_map: Dict[str, str]) -> List[Dict]:
    rows = []
    for idx, question in enumerate(questions, start=1):
        print(f"[{idx}/{len(questions)}] {question}")
        result = ask(question)

        answer = result.get("answer", "").strip()
        sources = result.get("sources", [])
        retrieved_contexts = [
            src.get("content", "").strip()
            for src in sources if src.get("content", "").strip()
        ]

        normalized_q = normalize_text(question)
        reference = reference_map.get(normalized_q)
        if not reference:
            reference = build_reference_from_sources(sources)

        rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": retrieved_contexts,
            "reference": reference,
            "detected_disease": result.get("detected_disease"),
            "detected_section": result.get("detected_section"),
        })
    return rows


# =========================================================
# EVALUATION
# =========================================================
def run_evaluation(rows: List[Dict]) -> Dict:
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)

    faithfulness_scores = []
    context_recall_scores = []
    bleu_scores = []

    for idx, row in enumerate(rows, start=1):
        print(f"  Evaluating [{idx}/{len(rows)}] {row['user_input'][:60]}...")

        f_score = evaluate_faithfulness(llm, row["response"], row["retrieved_contexts"])
        faithfulness_scores.append(f_score)

        cr_score = evaluate_context_recall(llm, row["response"], row["retrieved_contexts"], row["reference"])
        context_recall_scores.append(cr_score)

        bleu = compute_bleu(row["reference"], row["response"])
        bleu_scores.append(bleu)

    avg = lambda s: sum(s) / len(s) if s else 0
    return {
        "faithfulness": avg(faithfulness_scores),
        "context_recall": avg(context_recall_scores),
        "bleu": avg(bleu_scores),
        "details": [
            {"faithfulness": f, "context_recall": c, "bleu": b}
            for f, c, b in zip(faithfulness_scores, context_recall_scores, bleu_scores)
        ],
    }


# =========================================================
# SAVE OUTPUTS
# =========================================================
def save_json(rows: List[Dict], file_path: str = DETAILS_JSON) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    print("Loading questions from question.txt...")
    questions = load_questions(QUESTIONS_FILE, MAX_QUESTIONS)
    print(f"Loaded {len(questions)} questions.\n")

    print("Building reference map from markdown files...")
    reference_map = build_reference_map(DOCS_PATH)
    print(f"Loaded {len(reference_map)} question-answer pairs from docs.\n")

    print("Running RAG pipeline...")
    rows = run_rag_on_questions(questions, reference_map)

    print("\nRunning evaluation (using local Ollama + BLEU)...")
    results = run_evaluation(rows)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (%)")
    print("=" * 60)
    print(f"  Faithfulness:    {results['faithfulness'] * 100:.1f}%")
    print(f"  Context Recall:  {results['context_recall'] * 100:.1f}%")
    print(f"  BLEU Score:      {results['bleu'] * 100:.1f}%")
    print("=" * 60)

    import pandas as pd
    details = results["details"]
    for i, row in enumerate(rows):
        details[i]["user_input"] = row["user_input"]
        details[i]["response"] = row["response"]
    results_df = pd.DataFrame(details)
    results_df.to_csv(RESULTS_CSV, index=False)
    save_json(rows, DETAILS_JSON)

    print(f"\nSaved results to: {RESULTS_CSV}")
    print(f"Saved details to: {DETAILS_JSON}")


if __name__ == "__main__":
    main()
