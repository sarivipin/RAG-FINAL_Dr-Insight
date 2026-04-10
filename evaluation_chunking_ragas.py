"""
RAGAS evaluation of all 3 chunking methods.
Each method uses its own vector store. Groq 8B for generation, Groq 70B for RAGAS judging.
"""
import os
import re
import json
import time
import sys
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from ragas.metrics import faithfulness, context_recall, context_precision
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# =========================================================
# CONFIG
# =========================================================
QUESTIONS_FILE = "questions.txt"
DOCS_PATH = "docs"
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MAX_QUESTIONS = 20
MAX_RETRIES = 8
INITIAL_BACKOFF = 10

CHUNKING_CONFIGS = {
    "qa_pair": {
        "persist_dir": "db/chroma_db",
        "collection": "medical_markdown_docs",
        "label": "QA-Pair (Original)",
    },
    "recursive": {
        "persist_dir": "db/chroma_db_recursive",
        "collection": "medical_recursive_chunks",
        "label": "Recursive Character",
    },
    "semantic": {
        "persist_dir": "db/chroma_db_semantic",
        "collection": "medical_semantic_chunks",
        "label": "Semantic Chunking",
    },
}


# =========================================================
# SHARED UTILITIES
# =========================================================
def make_llm(temp=0.2):
    return ChatGroq(model=GROQ_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=temp)

def invoke_retry(llm, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = INITIAL_BACKOFF * (2 ** attempt)
                print(f"      Rate limited. Waiting {wait}s ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Failed after retries.")

_embeddings = None
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True})
    return _embeddings

_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

def normalize_text(t):
    return re.sub(r"\s+", " ", t.strip().lower())

def clean_answer_text(a):
    a = re.sub(r"\*\*Section:\*\*.*", "", a, flags=re.IGNORECASE)
    a = re.sub(r"\*\*Source:\*\*.*", "", a, flags=re.IGNORECASE)
    return a.strip()

def load_questions():
    with open(QUESTIONS_FILE) as f:
        return [l.strip() for l in f if l.strip()][:MAX_QUESTIONS]

def build_reference_map():
    ref = {}
    for fn in os.listdir(DOCS_PATH):
        if not fn.endswith(".md"): continue
        with open(os.path.join(DOCS_PATH, fn), "r", encoding="utf-8") as f:
            content = f.read()
        for q, a in re.findall(
            r"##\s*Question\s*\n(.*?)\n+###\s*Answer\s*\n(.*?)(?=\n+---|\n+##\s*Question|\Z)",
            content, re.DOTALL | re.IGNORECASE):
            q, a = q.strip(), clean_answer_text(a.strip())
            if q and a:
                ref[normalize_text(q)] = a
    return ref


# =========================================================
# RETRIEVAL (standalone per chunking method)
# =========================================================
def retrieve_and_generate(query, db, llm):
    docs = db.similarity_search(query, k=10)
    reranker = get_reranker()
    if docs:
        pairs = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        docs = [d for d, _ in scored[:5]]

    if not docs:
        return {"answer": "No relevant documents found.", "contexts": []}

    context = "\n\n---\n\n".join([d.page_content for d in docs[:4]])
    prompt = f"""You are a helpful medical information assistant. Answer based ONLY on the context.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

    answer = invoke_retry(llm, prompt)
    contexts = [d.page_content for d in docs[:4]]
    return {"answer": answer, "contexts": contexts}


# =========================================================
# EVALUATE ONE CHUNKING METHOD WITH RAGAS
# =========================================================
def evaluate_method(method_key, config, questions, reference_map, ragas_llm, ragas_emb):
    label = config["label"]
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    db = Chroma(persist_directory=config["persist_dir"],
                collection_name=config["collection"],
                embedding_function=get_embeddings())
    gen_llm = make_llm(0.2)

    # Generate answers
    samples = []
    raw = []
    for idx, q in enumerate(questions, 1):
        print(f"  [{idx}/{len(questions)}] {q}")
        result = retrieve_and_generate(q, db, gen_llm)
        norm_q = normalize_text(q)
        reference = reference_map.get(norm_q, "No reference available.")

        sample = SingleTurnSample(
            user_input=q,
            response=result["answer"],
            retrieved_contexts=result["contexts"] if result["contexts"] else ["No context."],
            reference=reference,
        )
        samples.append(sample)
        raw.append({"question": q, "answer": result["answer"], "contexts": result["contexts"], "reference": reference})
        time.sleep(3)  # Pace API calls

    # Run RAGAS
    print(f"  Running RAGAS evaluation...")
    eval_dataset = EvaluationDataset(samples=samples)
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, context_recall, context_precision],
        llm=ragas_llm, embeddings=ragas_emb,
    )
    results_df = results.to_pandas()

    # Extract scores
    per_question = []
    for i in range(len(questions)):
        pq = {
            "question": questions[i],
            "faithfulness": float(results_df.iloc[i].get("faithfulness", 0)) if not __import__("pandas").isna(results_df.iloc[i].get("faithfulness", 0)) else 0.0,
            "context_recall": float(results_df.iloc[i].get("context_recall", 0)) if not __import__("pandas").isna(results_df.iloc[i].get("context_recall", 0)) else 0.0,
            "context_precision": float(results_df.iloc[i].get("context_precision", 0)) if not __import__("pandas").isna(results_df.iloc[i].get("context_precision", 0)) else 0.0,
        }
        per_question.append(pq)

    avg_f = sum(p["faithfulness"] for p in per_question) / len(per_question)
    avg_r = sum(p["context_recall"] for p in per_question) / len(per_question)
    avg_p = sum(p["context_precision"] for p in per_question) / len(per_question)

    print(f"  Faithfulness:      {avg_f*100:.1f}%")
    print(f"  Context Recall:    {avg_r*100:.1f}%")
    print(f"  Context Precision: {avg_p*100:.1f}%")

    return {
        "method": method_key,
        "label": label,
        "avg_faithfulness": avg_f,
        "avg_context_recall": avg_r,
        "avg_context_precision": avg_p,
        "per_question": per_question,
    }


# =========================================================
# MAIN
# =========================================================
def main():
    print("=== Chunking Method Comparison with RAGAS ===\n")

    questions = load_questions()
    reference_map = build_reference_map()
    print(f"Loaded {len(questions)} questions, {len(reference_map)} references.\n")

    # Setup RAGAS judge (once, shared across all methods)
    ragas_llm = LangchainLLMWrapper(make_llm(0.0))
    ragas_emb = LangchainEmbeddingsWrapper(get_embeddings())

    all_results = {}
    for key, config in CHUNKING_CONFIGS.items():
        all_results[key] = evaluate_method(key, config, questions, reference_map, ragas_llm, ragas_emb)

    # Save
    with open("chunking_ragas_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 70)
    print("CHUNKING COMPARISON (RAGAS EVALUATION)")
    print("=" * 70)
    print(f"{'Method':<25} {'Faithfulness':>14} {'Context Recall':>16} {'Ctx Precision':>15}")
    print("-" * 70)
    for key in CHUNKING_CONFIGS:
        r = all_results[key]
        print(f"{r['label']:<25} {r['avg_faithfulness']*100:>13.1f}% {r['avg_context_recall']*100:>15.1f}% {r['avg_context_precision']*100:>14.1f}%")
    print("=" * 70)
    print("\nSaved: chunking_ragas_results.json")


if __name__ == "__main__":
    main()
