"""
RAGAS vs Custom Comparison — Evaluates the SAME Groq-generated answers
using both RAGAS library and our custom metrics side by side.
Loads saved answers from ragas_details_groq.json (no regeneration).
"""
import os
import re
import json
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from ragas.metrics import faithfulness, context_recall, context_precision
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

# =========================================================
# CONFIG
# =========================================================
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INPUT_FILE = "ragas_details.json"
RESULTS_CSV = "ragas_vs_custom_results.csv"
DETAILS_JSON = "ragas_vs_custom_details.json"


# =========================================================
# CUSTOM METRICS (same as evaluation_groq.py)
# =========================================================
MAX_RETRIES = 8
INITIAL_BACKOFF = 10

def make_llm(temp=0.0):
    return ChatGroq(model=GROQ_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=temp)

def invoke_retry(llm, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = INITIAL_BACKOFF * (2 ** attempt)
                print(f"    Rate limited. Waiting {wait}s ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Failed after retries.")

def normalize_text(t):
    return re.sub(r"\s+", " ", t.strip().lower())

def compute_bleu(reference, candidate):
    rt, ct = normalize_text(reference).split(), normalize_text(candidate).split()
    if not ct or not rt: return 0.0
    from collections import Counter
    rc, cc = Counter(rt), Counter(ct)
    cl = sum(min(c, rc.get(t,0)) for t,c in cc.items())
    return (cl/len(ct)) * min(1.0, len(ct)/len(rt))

def custom_faithfulness(llm, response, contexts):
    if not contexts or not response.strip(): return 0.0
    prompt = f"""You are an evaluation judge. Rate how faithful the following response is to the provided context.
A faithful response only contains information that can be found in or inferred from the context.

Context:
{chr(10).join(contexts[:3])}

Response:
{response}

Rate faithfulness as a score from 0.0 to 1.0 where:
- 1.0 = completely faithful, all claims supported by context
- 0.0 = completely unfaithful, no claims supported

Reply with ONLY a number between 0.0 and 1.0, nothing else."""
    try:
        score = float(re.search(r"([01]\.?\d*)", invoke_retry(llm, prompt)).group(1))
        return min(max(score, 0.0), 1.0)
    except: return 0.0

def custom_context_recall(llm, contexts, reference):
    if not contexts or not reference.strip(): return 0.0
    prompt = f"""You are an evaluation judge. Rate how well the retrieved context covers the information needed to answer correctly.

Reference answer (ground truth):
{reference}

Retrieved context:
{chr(10).join(contexts[:3])}

Rate context recall as a score from 0.0 to 1.0 where:
- 1.0 = context contains all information from the reference
- 0.0 = context contains none of the reference information

Reply with ONLY a number between 0.0 and 1.0, nothing else."""
    try:
        score = float(re.search(r"([01]\.?\d*)", invoke_retry(llm, prompt)).group(1))
        return min(max(score, 0.0), 1.0)
    except: return 0.0


# =========================================================
# MAIN
# =========================================================
def main():
    print("=== RAGAS vs Custom (Ollama) Evaluation Comparison ===")
    print(f"Using saved Ollama-generated answers from: {INPUT_FILE}")
    print(f"Custom scores from original evaluation.py run\n")

    # Load saved Ollama answers
    with open(INPUT_FILE, "r") as f:
        saved_data = json.load(f)
    print(f"Loaded {len(saved_data)} saved Ollama answers.\n")

    # Load original custom evaluation scores (from evaluation.py)
    original_csv = pd.read_csv("ragas_results.csv")
    print(f"Loaded original custom scores from ragas_results.csv\n")

    # ---- STEP 1: Run RAGAS evaluation on Ollama answers ----
    print("Step 1: Running RAGAS library evaluation on Ollama answers...")
    samples = []
    for item in saved_data:
        sample = SingleTurnSample(
            user_input=item["user_input"],
            response=item["response"],
            retrieved_contexts=item["retrieved_contexts"] if item["retrieved_contexts"] else ["No context."],
            reference=item["reference"],
        )
        samples.append(sample)

    eval_dataset = EvaluationDataset(samples=samples)
    llm = LangchainLLMWrapper(make_llm(0.0))
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device":"cpu"},
                              encode_kwargs={"normalize_embeddings":True})
    )

    ragas_results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, context_recall, context_precision],
        llm=llm, embeddings=emb,
    )
    ragas_df = ragas_results.to_pandas()
    print("  RAGAS evaluation complete.\n")

    # ---- STEP 2: Compare with original custom scores ----
    print("=" * 70)
    print("COMPARISON: RAGAS Library (Groq 70B judge)")
    print("       vs   Custom Evaluation (Ollama 3B judge)")
    print("Same Ollama-generated answers, different evaluation methods")
    print("=" * 70)

    questions = [item["user_input"] for item in saved_data]
    comparison = []

    for i in range(len(saved_data)):
        row = {
            "question": questions[i],
            "ragas_faithfulness": float(ragas_df.iloc[i].get("faithfulness", 0)) if not pd.isna(ragas_df.iloc[i].get("faithfulness", 0)) else 0.0,
            "custom_faithfulness": float(original_csv.iloc[i]["faithfulness"]),
            "ragas_context_recall": float(ragas_df.iloc[i].get("context_recall", 0)) if not pd.isna(ragas_df.iloc[i].get("context_recall", 0)) else 0.0,
            "custom_context_recall": float(original_csv.iloc[i]["context_recall"]),
            "ragas_context_precision": float(ragas_df.iloc[i].get("context_precision", 0)) if not pd.isna(ragas_df.iloc[i].get("context_precision", 0)) else 0.0,
            "custom_bleu": float(original_csv.iloc[i]["bleu"]),
        }
        comparison.append(row)

    # Print per-question
    for row in comparison:
        q = row["question"][:50]
        print(f"\n  {q}...")
        print(f"    Faithfulness:      RAGAS={row['ragas_faithfulness']:.2f}  Custom(Ollama)={row['custom_faithfulness']:.2f}")
        print(f"    Context Recall:    RAGAS={row['ragas_context_recall']:.2f}  Custom(Ollama)={row['custom_context_recall']:.2f}")
        print(f"    Context Precision: RAGAS={row['ragas_context_precision']:.2f}")
        print(f"    BLEU (custom):     {row['custom_bleu']:.2f}")

    # Print averages
    avg = lambda lst, k: sum(r[k] for r in lst) / len(lst)
    print(f"\n{'='*70}")
    print(f"{'Metric':<25} {'RAGAS (Groq judge)':>20} {'Custom (Ollama judge)':>22}")
    print(f"{'-'*67}")
    print(f"{'Faithfulness':<25} {avg(comparison,'ragas_faithfulness')*100:>19.1f}% {avg(comparison,'custom_faithfulness')*100:>21.1f}%")
    print(f"{'Context Recall':<25} {avg(comparison,'ragas_context_recall')*100:>19.1f}% {avg(comparison,'custom_context_recall')*100:>21.1f}%")
    print(f"{'Context Precision':<25} {avg(comparison,'ragas_context_precision')*100:>19.1f}% {'N/A':>22}")
    print(f"{'BLEU Score':<25} {'N/A':>20} {avg(comparison,'custom_bleu')*100:>21.1f}%")
    print(f"{'='*70}")

    # Save
    pd.DataFrame(comparison).to_csv(RESULTS_CSV, index=False)
    with open(DETAILS_JSON, "w") as f:
        json.dump({"comparison": comparison}, f, indent=2)
    print(f"\nSaved: {RESULTS_CSV}")
    print(f"Saved: {DETAILS_JSON}")


if __name__ == "__main__":
    main()
