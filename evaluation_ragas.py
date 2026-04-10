"""
RAGAS Library Evaluation — Uses the actual RAGAS framework with Groq API.
Evaluates the RAG pipeline using RAGAS metrics: Faithfulness, Context Recall, Context Precision.
"""
import os
import re
import json
import time
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from ragas.metrics import faithfulness, context_recall, context_precision
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from retrieval import ask

# =========================================================
# CONFIG
# =========================================================
QUESTIONS_FILE = "questions.txt"
DOCS_PATH = "docs"
RESULTS_CSV = "ragas_lib_results.csv"
DETAILS_JSON = "ragas_lib_details.json"

MAX_QUESTIONS = 20
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_RETRIES = 8
INITIAL_BACKOFF = 30


# =========================================================
# SETUP LLM & EMBEDDINGS FOR RAGAS
# =========================================================
def make_groq_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
    )


def make_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# Patch retrieval to use Groq instead of Ollama
import retrieval

class _RetryLLMWrapper:
    def __init__(self, llm):
        self._llm = llm

    def invoke(self, prompt):
        for attempt in range(MAX_RETRIES):
            try:
                return self._llm.invoke(prompt)
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = INITIAL_BACKOFF * (2 ** attempt)
                    print(f"    [RAG] Rate limited. Waiting {wait}s ({attempt+1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("RAG LLM failed after retries.")

    def __getattr__(self, name):
        return getattr(self._llm, name)


# Use a smaller model for generation to avoid rate limits
# (RAGAS judging with 70B needs the quota)
GROQ_GEN_MODEL = "llama-3.1-8b-instant"

retrieval.load_llm = lambda: _RetryLLMWrapper(
    ChatGroq(model=GROQ_GEN_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=0.2)
)


# =========================================================
# HELPERS
# =========================================================
def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def clean_answer_text(answer):
    answer = re.sub(r"\*\*Section:\*\*.*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\*\*Source:\*\*.*", "", answer, flags=re.IGNORECASE)
    return answer.strip()


def load_questions():
    with open(QUESTIONS_FILE, "r") as f:
        return [l.strip() for l in f if l.strip()][:MAX_QUESTIONS]


def build_reference_map():
    ref = {}
    for fn in os.listdir(DOCS_PATH):
        if not fn.endswith(".md"):
            continue
        with open(os.path.join(DOCS_PATH, fn), "r", encoding="utf-8") as f:
            content = f.read()
        pattern = re.compile(
            r"##\s*Question\s*\n(.*?)\n+###\s*Answer\s*\n(.*?)(?=\n+---|\n+##\s*Question|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        for q, a in pattern.findall(content):
            q, a = q.strip(), clean_answer_text(a.strip())
            if q and a:
                ref[normalize_text(q)] = a
    return ref


# =========================================================
# MAIN
# =========================================================
def main():
    print("=== RAGAS Library Evaluation (with Groq API) ===\n")

    questions = load_questions()
    print(f"Loaded {len(questions)} questions.")

    reference_map = build_reference_map()
    print(f"Loaded {len(reference_map)} reference answers.\n")

    # Step 1: Run RAG pipeline on all questions
    print("Running RAG pipeline...")
    samples = []
    raw_results = []

    for idx, q in enumerate(questions, 1):
        print(f"  [{idx}/{len(questions)}] {q}")
        result = ask(q)
        time.sleep(5)  # Pace API calls

        answer = result.get("answer", "").strip()
        sources = result.get("sources", [])
        contexts = [
            src.get("content", "").strip()
            for src in sources if src.get("content", "").strip()
        ]

        norm_q = normalize_text(q)
        reference = reference_map.get(norm_q, "No reference available.")

        # Create RAGAS SingleTurnSample
        sample = SingleTurnSample(
            user_input=q,
            response=answer,
            retrieved_contexts=contexts if contexts else ["No context retrieved."],
            reference=reference,
        )
        samples.append(sample)

        raw_results.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "reference": reference,
        })

    # Step 2: Create RAGAS EvaluationDataset
    print("\nCreating RAGAS evaluation dataset...")
    eval_dataset = EvaluationDataset(samples=samples)

    # Step 3: Setup RAGAS with Groq LLM
    print("Setting up RAGAS with Groq LLM judge...")
    llm = LangchainLLMWrapper(make_groq_llm())
    emb = LangchainEmbeddingsWrapper(make_embeddings())

    metrics = [faithfulness, context_recall, context_precision]

    # Step 4: Run RAGAS evaluation
    print("Running RAGAS evaluation (this may take a minute)...\n")
    try:
        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=emb,
        )
    except Exception as e:
        print(f"RAGAS evaluation error: {e}")
        print("Trying with individual retries...")
        # Fallback: evaluate one at a time
        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=emb,
        )

    # Step 5: Print results
    print("=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    results_df = results.to_pandas()
    for col in ["faithfulness", "context_recall", "context_precision"]:
        if col in results_df.columns:
            avg_score = results_df[col].mean()
            print(f"  {col}: {avg_score*100:.1f}%")

    print("=" * 60)

    # Step 6: Save results
    import pandas as pd

    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved: {RESULTS_CSV}")

    # Save detailed JSON
    detail_output = {
        "overall": {},
        "per_question": [],
    }
    for col in ["faithfulness", "context_recall", "context_precision"]:
        if col in results_df.columns:
            detail_output["overall"][col] = float(results_df[col].mean())

    for i, row in results_df.iterrows():
        detail_output["per_question"].append({
            "question": raw_results[i]["question"],
            "answer": raw_results[i]["answer"],
            "reference": raw_results[i]["reference"],
            "faithfulness": float(row.get("faithfulness", 0)) if not pd.isna(row.get("faithfulness", 0)) else 0.0,
            "context_recall": float(row.get("context_recall", 0)) if not pd.isna(row.get("context_recall", 0)) else 0.0,
            "context_precision": float(row.get("context_precision", 0)) if not pd.isna(row.get("context_precision", 0)) else 0.0,
        })

    with open(DETAILS_JSON, "w", encoding="utf-8") as f:
        json.dump(detail_output, f, indent=2, ensure_ascii=False)
    print(f"Saved: {DETAILS_JSON}")


if __name__ == "__main__":
    main()
