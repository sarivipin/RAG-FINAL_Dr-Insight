import os
import json
import pandas as pd
from retrieval import ask

QUESTIONS_FILE = "questions.txt"


# =========================
# METRICS
# =========================
def tokenize(text):
    return text.lower().split()


def context_overlap(answer, context):
    """Token-overlap faithfulness: what fraction of answer tokens appear in context."""
    a = set(tokenize(answer))
    c = set(tokenize(context))
    return len(a & c) / len(a) if a else 0


def context_coverage(answer, context):
    """What fraction of context tokens appear in the answer."""
    a = set(tokenize(answer))
    c = set(tokenize(context))
    return len(a & c) / len(c) if c else 0


def relevance_score(query, answer):
    """What fraction of query tokens appear in the answer."""
    q = set(tokenize(query))
    a = set(tokenize(answer))
    return len(q & a) / len(q) if q else 0


def build_context(sources):
    parts = []
    for src in sources:
        content = src.get("content", "")
        if content:
            parts.append(content)
    return " ".join(parts)


# =========================
# LOAD QUESTIONS
# =========================
def load_questions():
    if not os.path.exists(QUESTIONS_FILE):
        raise FileNotFoundError("questions.txt not found")
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return [q.strip() for q in f.readlines() if q.strip()]


# =========================
# MAIN LOOP
# =========================
def run_batch():
    questions = load_questions()
    results = []

    print("\n=== BATCH EVALUATION STARTED ===\n")

    for i, query in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {query}")

        result = ask(query)
        answer = result.get("answer", "").strip()
        sources = result.get("sources", [])
        context = build_context(sources)

        faith = context_overlap(answer, context)
        coverage = context_coverage(answer, context)
        relevance = relevance_score(query, answer)

        print(f"  Faith: {faith:.2f} | Coverage: {coverage:.2f} | Relevance: {relevance:.2f}")

        results.append({
            "query": query,
            "answer": answer,
            "faithfulness": faith,
            "coverage": coverage,
            "relevance": relevance,
        })

    return results


# =========================
# SAVE RESULTS
# =========================
def save_results(results):
    df = pd.DataFrame(results)
    df.to_csv("batch_results.csv", index=False)

    with open("batch_details.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== AVERAGE SCORES ===")
    print(df[["faithfulness", "coverage", "relevance"]].mean())


# =========================
# RUN
# =========================
if __name__ == "__main__":
    results = run_batch()
    save_results(results)
