"""
Run all 4 improvement experiments in subprocesses, then evaluate with RAGAS.
Baseline (ChromaDB), FAISS, FAISS+MPNet, Hybrid (BM25+Vector)
"""
import subprocess, json, sys, os, time
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics import faithfulness, context_recall, context_precision
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

EXPERIMENTS = ["baseline", "faiss", "mpnet", "hybrid"]
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print("=== RAG Improvements — RAGAS Evaluation ===\n")

    # Step 1: Run each experiment in subprocess
    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, "run_single_experiment_ragas.py", exp],
            capture_output=True, text=True, timeout=600)
        print(result.stdout[-500:] if result.stdout else "")
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[-300:]}")
            continue
        print(f"  Done.")

    # Step 2: Load all results and evaluate with RAGAS
    print(f"\n{'='*60}")
    print("Running RAGAS evaluation on all experiments...")
    print(f"{'='*60}\n")

    ragas_llm = LangchainLLMWrapper(
        ChatGroq(model=GROQ_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=0.0))
    ragas_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device":"cpu"},
                              encode_kwargs={"normalize_embeddings":True}))

    all_results = {}
    for exp in EXPERIMENTS:
        fpath = f"ragas_imp_{exp}.json"
        if not os.path.exists(fpath):
            print(f"  Skipping {exp} — no results file.")
            continue

        with open(fpath) as f:
            exp_data = json.load(f)

        print(f"  Evaluating {exp_data['label']}...")

        samples = []
        for r in exp_data["results"]:
            samples.append(SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"] if r["contexts"] else ["No context."],
                reference=r["reference"],
            ))

        eval_dataset = EvaluationDataset(samples=samples)
        results = evaluate(
            dataset=eval_dataset,
            metrics=[faithfulness, context_recall, context_precision],
            llm=ragas_llm, embeddings=ragas_emb,
        )
        results_df = results.to_pandas()

        per_question = []
        for i in range(len(exp_data["results"])):
            pq = {
                "question": exp_data["results"][i]["question"],
                "faithfulness": float(results_df.iloc[i].get("faithfulness", 0)) if not pd.isna(results_df.iloc[i].get("faithfulness", 0)) else 0.0,
                "context_recall": float(results_df.iloc[i].get("context_recall", 0)) if not pd.isna(results_df.iloc[i].get("context_recall", 0)) else 0.0,
                "context_precision": float(results_df.iloc[i].get("context_precision", 0)) if not pd.isna(results_df.iloc[i].get("context_precision", 0)) else 0.0,
                "retrieve_ms": exp_data["results"][i]["retrieve_ms"],
            }
            per_question.append(pq)

        avg = lambda k: sum(p[k] for p in per_question) / len(per_question)
        all_results[exp] = {
            "key": exp,
            "label": exp_data["label"],
            "avg_faithfulness": avg("faithfulness"),
            "avg_context_recall": avg("context_recall"),
            "avg_context_precision": avg("context_precision"),
            "avg_retrieve_ms": exp_data["avg_retrieve_ms"],
            "per_question": per_question,
        }

        print(f"    Faith: {avg('faithfulness')*100:.1f}% | Recall: {avg('context_recall')*100:.1f}% | "
              f"Precision: {avg('context_precision')*100:.1f}% | Speed: {exp_data['avg_retrieve_ms']:.0f}ms")

    # Save
    with open("improvements_ragas_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*75}")
    print("IMPROVEMENT EXPERIMENTS — RAGAS EVALUATION")
    print(f"{'='*75}")
    print(f"{'Method':<28} {'Faith':>8} {'Recall':>8} {'Precision':>10} {'Speed':>10}")
    print("-" * 65)
    for exp in EXPERIMENTS:
        if exp in all_results:
            r = all_results[exp]
            print(f"{r['label']:<28} {r['avg_faithfulness']*100:>7.1f}% {r['avg_context_recall']*100:>7.1f}% "
                  f"{r['avg_context_precision']*100:>9.1f}% {r['avg_retrieve_ms']:>8.0f}ms")
    print(f"{'='*75}")
    print("\nSaved: improvements_ragas_results.json")


if __name__ == "__main__":
    main()
