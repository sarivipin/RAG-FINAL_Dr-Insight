"""Run a single improvement experiment and save answers for RAGAS evaluation."""
import os, re, json, time, sys, pickle
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from sentence_transformers import CrossEncoder

DOCS_PATH = "docs"
QUESTIONS_FILE = "questions.txt"
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 8
INITIAL_BACKOFF = 10
EMBEDDING_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MPNET = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

def make_llm(temp=0.2):
    return ChatGroq(model=GROQ_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=temp)

def invoke_retry(llm, prompt):
    for attempt in range(MAX_RETRIES):
        try: return llm.invoke(prompt).content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = INITIAL_BACKOFF * (2**attempt)
                print(f"    Rate limited. Waiting {wait}s ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else: raise
    raise RuntimeError("Failed")

def get_emb(name):
    return HuggingFaceEmbeddings(model_name=name, model_kwargs={"device":"cpu"},
                                  encode_kwargs={"normalize_embeddings":True})

def normalize_text(t): return re.sub(r"\s+", " ", t.strip().lower())

def load_questions():
    with open(QUESTIONS_FILE) as f: return [l.strip() for l in f if l.strip()][:20]

def build_ref_map():
    ref = {}
    for fn in os.listdir(DOCS_PATH):
        if not fn.endswith(".md"): continue
        with open(os.path.join(DOCS_PATH, fn), encoding="utf-8") as f: content = f.read()
        for q, a in re.findall(r"##\s*Question\s*\n(.*?)\n+###\s*Answer\s*\n(.*?)(?=\n+---|\n+##\s*Question|\Z)", content, re.DOTALL|re.IGNORECASE):
            q = q.strip()
            a = re.sub(r"\*\*(Section|Source):\*\*.*", "", a, flags=re.IGNORECASE).strip()
            if q and a: ref[normalize_text(q)] = a
    return ref

def load_doc_chunks():
    loader = DirectoryLoader(DOCS_PATH, glob="*.md", loader_cls=TextLoader,
                             loader_kwargs={"encoding":"utf-8"}, show_progress=False)
    docs = []
    for doc in loader.load():
        blocks = re.split(r"(?=^\s*##\s+Question\s*$)", doc.page_content, flags=re.MULTILINE)
        for i, b in enumerate(blocks):
            b = b.strip()
            if b.startswith("## Question") and len(b) > 20:
                docs.append(Document(page_content=b, metadata={"source": doc.metadata.get("source",""), "idx": i}))
    return docs

def rerank_docs(query, docs, top_n=5):
    if not docs: return docs
    rr = CrossEncoder(RERANKER_MODEL)
    scores = rr.predict([(query, d.page_content) for d in docs])
    return [d for d,_ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_n]]

def gen_answer(llm, query, ctxs):
    context = "\n\n---\n\n".join(ctxs[:4])
    return invoke_retry(llm, f"Answer based ONLY on context. If insufficient, say so.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:")


# =========================================================
# EXPERIMENT IMPLEMENTATIONS
# =========================================================
def run_baseline(questions, ref_map):
    from langchain_chroma import Chroma
    emb = get_emb(EMBEDDING_MINILM)
    db = Chroma(persist_directory="db/chroma_db", collection_name="medical_markdown_docs", embedding_function=emb)
    llm = make_llm(0.2)
    results = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q}")
        t0 = time.time()
        docs = rerank_docs(q, db.similarity_search(q, k=10))
        ms = (time.time()-t0)*1000
        ctxs = [d.page_content for d in docs[:4]]
        ans = gen_answer(llm, q, ctxs)
        ref = ref_map.get(normalize_text(q), "No ref.")
        results.append({"question":q, "answer":ans, "contexts":ctxs, "reference":ref, "retrieve_ms":ms})
        time.sleep(3)
    return results

def run_faiss(questions, ref_map):
    import faiss as faiss_lib
    emb = get_emb(EMBEDDING_MINILM)
    docs = load_doc_chunks()
    print(f"  Building FAISS index ({len(docs)} chunks)...")
    embeddings = np.array(emb.embed_documents([d.page_content for d in docs]), dtype="float32")
    index = faiss_lib.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    llm = make_llm(0.2)
    results = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q}")
        t0 = time.time()
        qe = np.array([emb.embed_query(q)], dtype="float32")
        _, ids = index.search(qe, 10)
        retrieved = [docs[j] for j in ids[0] if j < len(docs)]
        ranked = rerank_docs(q, retrieved)
        ms = (time.time()-t0)*1000
        ctxs = [d.page_content for d in ranked[:4]]
        ans = gen_answer(llm, q, ctxs)
        ref = ref_map.get(normalize_text(q), "No ref.")
        results.append({"question":q, "answer":ans, "contexts":ctxs, "reference":ref, "retrieve_ms":ms})
        time.sleep(3)
    return results

def run_mpnet(questions, ref_map):
    import faiss as faiss_lib
    emb = get_emb(EMBEDDING_MPNET)
    docs = load_doc_chunks()
    print(f"  Building FAISS+MPNet index ({len(docs)} chunks)...")
    embeddings = np.array(emb.embed_documents([d.page_content for d in docs]), dtype="float32")
    index = faiss_lib.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    llm = make_llm(0.2)
    results = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q}")
        t0 = time.time()
        qe = np.array([emb.embed_query(q)], dtype="float32")
        _, ids = index.search(qe, 10)
        retrieved = [docs[j] for j in ids[0] if j < len(docs)]
        ranked = rerank_docs(q, retrieved)
        ms = (time.time()-t0)*1000
        ctxs = [d.page_content for d in ranked[:4]]
        ans = gen_answer(llm, q, ctxs)
        ref = ref_map.get(normalize_text(q), "No ref.")
        results.append({"question":q, "answer":ans, "contexts":ctxs, "reference":ref, "retrieve_ms":ms})
        time.sleep(3)
    return results

def run_hybrid(questions, ref_map):
    from langchain_chroma import Chroma
    from rank_bm25 import BM25Okapi
    emb = get_emb(EMBEDDING_MINILM)
    db = Chroma(persist_directory="db/chroma_db", collection_name="medical_markdown_docs", embedding_function=emb)
    docs = load_doc_chunks()
    print(f"  Building BM25 index ({len(docs)} chunks)...")
    bm25 = BM25Okapi([d.page_content.lower().split() for d in docs])
    llm = make_llm(0.2)
    results = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q}")
        t0 = time.time()
        bm25_scores = bm25.get_scores(q.lower().split())
        bm25_top = [docs[j] for j in np.argsort(bm25_scores)[-10:][::-1]]
        vec_docs = db.similarity_search(q, k=10)
        seen = set(); merged = []
        for d in vec_docs + bm25_top:
            k = d.page_content[:200]
            if k not in seen: seen.add(k); merged.append(d)
        ranked = rerank_docs(q, merged)
        ms = (time.time()-t0)*1000
        ctxs = [d.page_content for d in ranked[:4]]
        ans = gen_answer(llm, q, ctxs)
        ref = ref_map.get(normalize_text(q), "No ref.")
        results.append({"question":q, "answer":ans, "contexts":ctxs, "reference":ref, "retrieve_ms":ms})
        time.sleep(3)
    return results

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    exp = sys.argv[1]
    questions = load_questions()
    ref_map = build_ref_map()

    runners = {"baseline": run_baseline, "faiss": run_faiss, "mpnet": run_mpnet, "hybrid": run_hybrid}
    LABELS = {"baseline": "Baseline (ChromaDB+MiniLM)", "faiss": "FAISS+MiniLM",
              "mpnet": "FAISS+MPNet", "hybrid": "Hybrid (BM25+Vector)"}

    results = runners[exp](questions, ref_map)

    output = {"key": exp, "label": LABELS[exp], "results": results,
              "avg_retrieve_ms": sum(r["retrieve_ms"] for r in results)/len(results)}

    with open(f"ragas_imp_{exp}.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: ragas_imp_{exp}.json")
