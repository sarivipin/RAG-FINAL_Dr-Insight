import os
import re
import json
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder


# =========================================================
# CONFIG
# =========================================================
PERSIST_DIRECTORY = "db/chroma_db"
COLLECTION_NAME = "medical_markdown_docs"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"

SIMILARITY_K = 10
MMR_K = 10
MMR_FETCH_K = 30
FINAL_TOP_N_QUESTIONS = 4

OLLAMA_TEMPERATURE = 0.2

# Confidence / fallback
MIN_RERANK_SCORE_TO_ANSWER = -2.0
MIN_CONTEXT_DOCS = 1

# Logging
ENABLE_JSON_LOG = True
JSON_LOG_PATH = "retrieval_debug.json"


# =========================================================
# LOAD COMPONENTS
# =========================================================
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vector_store() -> Chroma:
    embeddings = load_embeddings()
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL_NAME)


def load_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
    )


# =========================================================
# TEXT HELPERS
# =========================================================
def clean_query(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


# =========================================================
# QUERY INTELLIGENCE
# =========================================================
def detect_section(query: str) -> Optional[str]:
    q = query.lower()

    if "symptom" in q or "sign" in q:
        return "symptoms"
    if "cause" in q or "causes" in q:
        return "causes"
    if "prevent" in q or "prevention" in q:
        return "prevention"
    if "treat" in q or "treatment" in q or "therapy" in q or "manage" in q:
        return "treatment"
    if "diagnos" in q or "test" in q or "scan" in q:
        return "diagnosis"
    if "risk factor" in q or "risk factors" in q or "at risk" in q:
        return "risk_factors"
    if "when to see" in q or "see a gp" in q or "doctor" in q or "gp" in q:
        return "when_to_see_your_gp"

    return None


def extract_disease_from_query(query: str, db: Chroma, max_scan: int = 5000) -> Optional[str]:
    """
    Try to detect disease name by matching query against disease metadata values already in the DB.
    This is a lightweight approach and works well for your disease-based dataset.
    """
    try:
        result = db.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])[:max_scan]
    except Exception:
        return None

    diseases = set()
    for meta in metadatas:
        disease = meta.get("disease")
        if disease:
            diseases.add(disease)

    q = normalize_text(query)

    # longest match first so "abdominal aortic aneurysm" wins over smaller words
    sorted_diseases = sorted(diseases, key=lambda x: len(x), reverse=True)

    for disease in sorted_diseases:
        d = normalize_text(disease)
        if d and d in q:
            return disease

    return None


# =========================================================
# FIRST-STAGE RETRIEVAL
# =========================================================
def similarity_retrieve(
    db: Chroma,
    query: str,
    k: int,
    section: Optional[str] = None,
    disease: Optional[str] = None,
) -> List[Document]:
    filters = {}
    if section:
        filters["section"] = section
    if disease:
        filters["disease"] = disease

    if filters:
        try:
            docs = db.similarity_search(query, k=k, filter=filters)
            if docs:
                return docs
        except Exception:
            pass

    # fallback disease-only
    if disease:
        try:
            docs = db.similarity_search(query, k=k, filter={"disease": disease})
            if docs:
                return docs
        except Exception:
            pass

    # fallback section-only
    if section:
        try:
            docs = db.similarity_search(query, k=k, filter={"section": section})
            if docs:
                return docs
        except Exception:
            pass

    return db.similarity_search(query, k=k)


def mmr_retrieve(
    db: Chroma,
    query: str,
    k: int,
    fetch_k: int,
    section: Optional[str] = None,
    disease: Optional[str] = None,
) -> List[Document]:
    filters = {}
    if section:
        filters["section"] = section
    if disease:
        filters["disease"] = disease

    if filters:
        try:
            docs = db.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                filter=filters,
            )
            if docs:
                return docs
        except Exception:
            pass

    if disease:
        try:
            docs = db.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                filter={"disease": disease},
            )
            if docs:
                return docs
        except Exception:
            pass

    if section:
        try:
            docs = db.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                filter={"section": section},
            )
            if docs:
                return docs
        except Exception:
            pass

    return db.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
    )


def dedupe_docs_by_chunk_id(docs: List[Document]) -> List[Document]:
    seen = set()
    unique_docs = []

    for doc in docs:
        cid = doc.metadata.get("chunk_id")
        if not cid:
            continue
        if cid not in seen:
            seen.add(cid)
            unique_docs.append(doc)

    return unique_docs


def apply_metadata_boosts(
    docs: List[Document],
    query: str,
    section: Optional[str],
    disease: Optional[str],
) -> List[Document]:
    """
    Add lightweight heuristic boosts before reranking.
    """
    q = normalize_text(query)

    for doc in docs:
        score = 0.0
        meta = dict(doc.metadata)

        doc_disease = normalize_text(meta.get("disease", ""))
        doc_section = normalize_text(meta.get("section", ""))
        doc_question = normalize_text(meta.get("question", ""))

        if disease and doc_disease == normalize_text(disease):
            score += 2.0

        if section and doc_section == normalize_text(section):
            score += 1.5

        if doc_question and doc_question in q:
            score += 1.0

        if q in doc_question:
            score += 1.0

        meta["pre_rerank_boost"] = score
        doc.metadata = meta

    return docs


def retrieve_candidates(db: Chroma, query: str) -> Dict:
    section = detect_section(query)
    disease = extract_disease_from_query(query, db)

    sim_docs = similarity_retrieve(
        db=db,
        query=query,
        k=SIMILARITY_K,
        section=section,
        disease=disease,
    )

    mmr_docs = mmr_retrieve(
        db=db,
        query=query,
        k=MMR_K,
        fetch_k=MMR_FETCH_K,
        section=section,
        disease=disease,
    )

    combined = sim_docs + mmr_docs
    combined = dedupe_docs_by_chunk_id(combined)
    combined = apply_metadata_boosts(combined, query, section, disease)

    return {
        "docs": combined,
        "section": section,
        "disease": disease,
    }


# =========================================================
# RERANKING
# =========================================================
def rerank_documents(query: str, docs: List[Document], reranker: CrossEncoder) -> List[Document]:
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    cross_scores = reranker.predict(pairs)

    reranked = []
    for doc, ce_score in zip(docs, cross_scores):
        meta = dict(doc.metadata)
        boost = float(meta.get("pre_rerank_boost", 0.0))
        final_score = float(ce_score) + boost

        meta["cross_encoder_score"] = float(ce_score)
        meta["rerank_score"] = final_score
        doc.metadata = meta
        reranked.append(doc)

    reranked.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    return reranked


# =========================================================
# QUESTION-LEVEL MERGING
# =========================================================
def fetch_docs_by_question_hash(db: Chroma, question_hash: str) -> List[Document]:
    try:
        result = db.get(where={"question_hash": question_hash})

        docs = []
        texts = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        for text, meta in zip(texts, metadatas):
            docs.append(Document(page_content=text, metadata=meta))

        docs.sort(key=lambda d: d.metadata.get("answer_part", 1))
        return docs
    except Exception:
        return []


def merge_question_group(group_docs: List[Document]) -> Dict:
    first = group_docs[0]
    merged_text = "\n\n".join(doc.page_content for doc in group_docs)

    return {
        "question_hash": first.metadata.get("question_hash", ""),
        "question": first.metadata.get("question", ""),
        "disease": first.metadata.get("disease", ""),
        "section": first.metadata.get("section", ""),
        "source_url": first.metadata.get("source_url", ""),
        "rerank_score": max(float(doc.metadata.get("rerank_score", 0.0)) for doc in group_docs),
        "content": merged_text,
    }


def select_top_question_groups(db: Chroma, reranked_docs: List[Document], top_n: int) -> List[Dict]:
    selected_hashes = []
    seen = set()

    for doc in reranked_docs:
        qhash = doc.metadata.get("question_hash")
        if not qhash or qhash in seen:
            continue

        seen.add(qhash)
        selected_hashes.append((qhash, float(doc.metadata.get("rerank_score", 0.0))))

        if len(selected_hashes) >= top_n:
            break

    merged_results = []

    for qhash, score in selected_hashes:
        group_docs = fetch_docs_by_question_hash(db, qhash)
        if not group_docs:
            continue

        for d in group_docs:
            meta = dict(d.metadata)
            meta["rerank_score"] = score
            d.metadata = meta

        merged_results.append(merge_question_group(group_docs))

    merged_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return merged_results


# =========================================================
# CONFIDENCE / FALLBACK
# =========================================================
def should_answer(question_groups: List[Dict]) -> bool:
    if len(question_groups) < MIN_CONTEXT_DOCS:
        return False

    best_score = question_groups[0]["rerank_score"]
    return best_score >= MIN_RERANK_SCORE_TO_ANSWER


def build_fallback_answer() -> str:
    return (
        "I could not find strong enough evidence in the retrieved documents to answer that reliably. "
        "Try asking with the disease name and what you want to know, such as symptoms, causes, diagnosis, or treatment."
    )


# =========================================================
# CONTEXT + PROMPT
# =========================================================
def build_context(question_groups: List[Dict]) -> str:
    blocks = []

    for idx, item in enumerate(question_groups, start=1):
        block = f"""Document {idx}
Disease: {item['disease']}
Question: {item['question']}
Section: {item['section']}
Source: {item['source_url']}

{item['content']}"""
        blocks.append(block.strip())

    return "\n\n" + ("\n\n" + "-" * 80 + "\n\n").join(blocks)


def build_prompt(query: str, context: str) -> str:
    return f"""
You are a medical information assistant for an educational RAG system.

Answer using ONLY the provided context.
Do not invent facts.
If the answer is incomplete in the context, say that clearly.
Keep the answer direct and readable.
At the end, include the source links you used if available.

User question:
{query}

Retrieved context:
{context}

Final answer:
""".strip()


# =========================================================
# LOGGING
# =========================================================
def document_to_log_dict(doc: Document) -> Dict:
    return {
        "chunk_id": doc.metadata.get("chunk_id"),
        "question_hash": doc.metadata.get("question_hash"),
        "question": doc.metadata.get("question"),
        "disease": doc.metadata.get("disease"),
        "section": doc.metadata.get("section"),
        "source_url": doc.metadata.get("source_url"),
        "pre_rerank_boost": doc.metadata.get("pre_rerank_boost"),
        "cross_encoder_score": doc.metadata.get("cross_encoder_score"),
        "rerank_score": doc.metadata.get("rerank_score"),
        "preview": doc.page_content[:400],
    }


def save_debug_log(payload: Dict) -> None:
    if not ENABLE_JSON_LOG:
        return

    with open(JSON_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# =========================================================
# MAIN QA PIPELINE
# =========================================================
def ask(query: str) -> Dict:
    query = clean_query(query)

    db = load_vector_store()
    reranker = load_reranker()
    llm = load_llm()

    retrieval_info = retrieve_candidates(db, query)
    candidates = retrieval_info["docs"]
    section = retrieval_info["section"]
    disease = retrieval_info["disease"]

    reranked = rerank_documents(query, candidates, reranker)
    top_groups = select_top_question_groups(db, reranked, top_n=FINAL_TOP_N_QUESTIONS)

    can_answer = should_answer(top_groups)

    if can_answer:
        context = build_context(top_groups)
        prompt = build_prompt(query, context)
        response = llm.invoke(prompt)
        final_answer = response.content
    else:
        context = ""
        final_answer = build_fallback_answer()

    log_payload = {
        "query": query,
        "detected_section": section,
        "detected_disease": disease,
        "num_candidates": len(candidates),
        "candidates": [document_to_log_dict(doc) for doc in reranked[:12]],
        "selected_question_groups": top_groups,
        "can_answer": can_answer,
        "final_answer": final_answer,
    }
    save_debug_log(log_payload)

    return {
        "query": query,
        "answer": final_answer,
        "sources": top_groups,
        "detected_section": section,
        "detected_disease": disease,
    }


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    print("=== Advanced Medical RAG Retrieval ===")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        result = ask(query)

        print("\nDetected disease:", result["detected_disease"])
        print("Detected section:", result["detected_section"])

        print("\nFinal Answer:\n")
        print(result["answer"])

        print("\nSources:\n")
        if not result["sources"]:
            print("No strong sources selected.\n")
            continue

        for idx, src in enumerate(result["sources"], start=1):
            print(f"{idx}. {src['question']}")
            print(f"   Disease: {src['disease']}")
            print(f"   Section: {src['section']}")
            print(f"   Rerank Score: {src['rerank_score']:.4f}")
            print(f"   Source: {src['source_url']}\n")