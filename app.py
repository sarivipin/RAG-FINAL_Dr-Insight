import os
import re
import streamlit as st

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
MIN_RERANK_SCORE = -2.0


# =========================================================
# STREAMLIT PAGE
# =========================================================
st.set_page_config(
    page_title="DR.Insight - Medical RAG Chatbot",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 DR.Insight - Medical RAG Chatbot")
st.caption("Disease-aware retrieval, reranking, and Ollama generation from your markdown dataset")


# =========================================================
# LOAD COMPONENTS
# =========================================================
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def load_vector_store():
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(
            f"Vector store not found at '{PERSIST_DIRECTORY}'. Run ingest.py first."
        )

    embeddings = get_embedding_model()

    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return db


@st.cache_resource
def get_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME)


@st.cache_resource
def get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.2,
    )


# =========================================================
# HELPERS
# =========================================================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_section(query: str):
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


def unique_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


# =========================================================
# DISEASE / QUESTION CATALOG
# =========================================================
@st.cache_data
def load_disease_question_catalog():
    db = load_vector_store()
    result = db.get(include=["metadatas"])

    disease_to_questions = {}

    for meta in result.get("metadatas", []):
        disease = meta.get("disease", "").strip()
        question = meta.get("question", "").strip()

        if not disease or not question:
            continue

        disease_to_questions.setdefault(disease, []).append(question)

    for disease in disease_to_questions:
        disease_to_questions[disease] = sorted(unique_preserve_order(disease_to_questions[disease]))

    diseases = sorted(disease_to_questions.keys())
    return diseases, disease_to_questions


# =========================================================
# RETRIEVAL
# =========================================================
def similarity_retrieve(db, query, k=SIMILARITY_K, disease=None, section=None):
    filters = {}
    if disease:
        filters["disease"] = disease
    if section:
        filters["section"] = section

    if filters:
        try:
            docs = db.similarity_search(query, k=k, filter=filters)
            if docs:
                return docs
        except Exception:
            pass

    if disease:
        try:
            docs = db.similarity_search(query, k=k, filter={"disease": disease})
            if docs:
                return docs
        except Exception:
            pass

    if section:
        try:
            docs = db.similarity_search(query, k=k, filter={"section": section})
            if docs:
                return docs
        except Exception:
            pass

    return db.similarity_search(query, k=k)


def mmr_retrieve(db, query, k=MMR_K, fetch_k=MMR_FETCH_K, disease=None, section=None):
    filters = {}
    if disease:
        filters["disease"] = disease
    if section:
        filters["section"] = section

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

    return db.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)


def dedupe_docs_by_chunk_id(docs):
    seen = set()
    unique_docs = []

    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id")
        if not chunk_id:
            continue
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique_docs.append(doc)

    return unique_docs


def apply_metadata_boosts(docs, query, disease=None, section=None):
    q = normalize_text(query)

    for doc in docs:
        meta = dict(doc.metadata)
        score = 0.0

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


def retrieve_candidates(query, retrieval_method="hybrid", disease=None):
    db = load_vector_store()
    section = detect_section(query)

    if retrieval_method == "similarity":
        docs = similarity_retrieve(db, query, k=SIMILARITY_K, disease=disease, section=section)
    elif retrieval_method == "mmr":
        docs = mmr_retrieve(db, query, k=MMR_K, fetch_k=MMR_FETCH_K, disease=disease, section=section)
    else:
        sim_docs = similarity_retrieve(db, query, k=SIMILARITY_K, disease=disease, section=section)
        mmr_docs = mmr_retrieve(db, query, k=MMR_K, fetch_k=MMR_FETCH_K, disease=disease, section=section)
        docs = sim_docs + mmr_docs

    docs = dedupe_docs_by_chunk_id(docs)
    docs = apply_metadata_boosts(docs, query=query, disease=disease, section=section)

    return docs, section


# =========================================================
# RERANKING + MERGING
# =========================================================
def rerank_documents(query, docs, top_n=FINAL_TOP_N_QUESTIONS * 3):
    if not docs:
        return []

    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    reranked = []
    for doc, ce_score in zip(docs, scores):
        meta = dict(doc.metadata)
        boost = float(meta.get("pre_rerank_boost", 0.0))
        final_score = float(ce_score) + boost

        meta["cross_encoder_score"] = float(ce_score)
        meta["rerank_score"] = final_score
        doc.metadata = meta
        reranked.append(doc)

    reranked.sort(key=lambda d: d.metadata.get("rerank_score", 0.0), reverse=True)
    return reranked[:top_n]


def fetch_docs_by_question_hash(db, question_hash):
    try:
        result = db.get(where={"question_hash": question_hash})

        docs = []
        for text, meta in zip(result.get("documents", []), result.get("metadatas", [])):
            docs.append(Document(page_content=text, metadata=meta))

        docs.sort(key=lambda d: d.metadata.get("answer_part", 1))
        return docs
    except Exception:
        return []


def merge_question_group(group_docs, group_score):
    first = group_docs[0]
    merged_text = "\n\n".join(doc.page_content for doc in group_docs)

    return {
        "question_hash": first.metadata.get("question_hash", ""),
        "question": first.metadata.get("question", ""),
        "disease": first.metadata.get("disease", ""),
        "section": first.metadata.get("section", ""),
        "source_url": first.metadata.get("source_url", ""),
        "rerank_score": float(group_score),
        "content": merged_text,
    }


def select_top_question_groups(reranked_docs, top_n=FINAL_TOP_N_QUESTIONS):
    db = load_vector_store()

    chosen = []
    seen = set()

    for doc in reranked_docs:
        qhash = doc.metadata.get("question_hash")
        if not qhash or qhash in seen:
            continue

        seen.add(qhash)
        score = doc.metadata.get("rerank_score", 0.0)

        group_docs = fetch_docs_by_question_hash(db, qhash)
        if not group_docs:
            continue

        chosen.append(merge_question_group(group_docs, score))

        if len(chosen) >= top_n:
            break

    chosen.sort(key=lambda x: x["rerank_score"], reverse=True)
    return chosen


# =========================================================
# GENERATION
# =========================================================
def build_context(question_groups):
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


def build_prompt(query, context):
    return f"""
You are a medical information assistant for an educational RAG system.

Answer using ONLY the provided context.
Do not invent facts.
Do not mention retrieval, chunks, or documents.
If the answer is incomplete in the context, say that clearly.
Keep the answer direct, helpful, and readable.

User question:
{query}

Retrieved context:
{context}

Return in this format:

Summary:
1-2 sentences directly answering the question.

Answer:
- Use short bullet points if needed
- Only include information relevant to the question

Source Links:
- Include the relevant source URLs used
""".strip()


def generate_answer(query, question_groups):
    if not question_groups:
        return "No reliable answer found in the retrieved dataset."

    if question_groups[0]["rerank_score"] < MIN_RERANK_SCORE:
        return (
            "I could not find strong enough evidence in the retrieved documents to answer that reliably. "
            "Try selecting the disease and choosing one of the listed questions."
        )

    llm = get_llm()
    context = build_context(question_groups)
    prompt = build_prompt(query, context)
    response = llm.invoke(prompt)
    return response.content


# =========================================================
# UI HELPERS
# =========================================================
def format_source_links(question_groups):
    links = []
    for item in question_groups:
        url = item.get("source_url", "").strip()
        if url and url not in links:
            links.append(url)
    return links


def calculate_confidence(question_groups):
    if not question_groups:
        return "Low"

    best = question_groups[0]["rerank_score"]

    if best >= 8:
        return f"High ({best:.2f})"
    if best >= 4:
        return f"Medium ({best:.2f})"
    return f"Low ({best:.2f})"


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Settings")
    retrieval_method = st.selectbox("Retrieval method", ["hybrid", "similarity", "mmr"], index=0)
    final_top_n = st.slider("Final question groups", 1, 8, FINAL_TOP_N_QUESTIONS)
    show_debug = st.checkbox("Show debug info", value=False)


# =========================================================
# MAIN UI
# =========================================================
try:
    diseases, disease_to_questions = load_disease_question_catalog()
except Exception as e:
    st.error(f"Failed to load disease catalog: {e}")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    selected_disease = st.selectbox(
        "Select a disease",
        options=["-- Select a disease --"] + diseases,
    )

with col2:
    available_questions = []
    if selected_disease != "-- Select a disease --":
        available_questions = disease_to_questions.get(selected_disease, [])

    selected_question = st.selectbox(
        "Questions available for this disease",
        options=["-- Select a question --"] + available_questions if available_questions else ["-- Select a question --"],
    )

custom_query = st.text_input(
    "Or type your own question",
    value="",
    placeholder="Example: What are the symptoms of abscess?",
)

query_to_use = ""

if custom_query.strip():
    query_to_use = custom_query.strip()
elif selected_question != "-- Select a question --":
    query_to_use = selected_question

if st.button("Generate Answer", use_container_width=True):
    if not query_to_use:
        st.warning("Please select a disease and a question, or type your own question.")
    else:
        try:
            with st.spinner("Retrieving and reranking relevant information..."):
                candidate_docs, detected_section = retrieve_candidates(
                    query=query_to_use,
                    retrieval_method=retrieval_method,
                    disease=selected_disease if selected_disease != "-- Select a disease --" else None,
                )

                reranked_docs = rerank_documents(
                    query=query_to_use,
                    docs=candidate_docs,
                    top_n=max(final_top_n * 3, 8),
                )

                top_question_groups = select_top_question_groups(
                    reranked_docs,
                    top_n=final_top_n,
                )

            with st.spinner("Generating answer with Ollama..."):
                answer = generate_answer(query_to_use, top_question_groups)

            confidence = calculate_confidence(top_question_groups)
            source_links = format_source_links(top_question_groups)

            st.subheader("Final Answer")
            st.markdown(answer)

            st.subheader("Detected Section")
            st.write(detected_section or "None")

            st.subheader("Confidence")
            st.write(confidence)

            st.subheader("Source Links")
            if source_links:
                for link in source_links:
                    st.markdown(f"- {link}")
            else:
                st.write("No source links available.")

            if show_debug:
                st.subheader("Top Retrieved Question Groups")
                for i, item in enumerate(top_question_groups, start=1):
                    with st.expander(f"{i}. {item['question']} | Score: {item['rerank_score']:.2f}"):
                        st.write("**Disease:**", item["disease"])
                        st.write("**Section:**", item["section"])
                        st.write("**Source:**", item["source_url"])
                        st.write(item["content"][:3000])

                st.subheader("Reranked Candidate Chunks")
                for i, doc in enumerate(reranked_docs[:10], start=1):
                    with st.expander(f"Chunk {i} | Score: {doc.metadata.get('rerank_score', 0.0):.2f}"):
                        st.write("**Disease:**", doc.metadata.get("disease", ""))
                        st.write("**Question:**", doc.metadata.get("question", ""))
                        st.write("**Section:**", doc.metadata.get("section", ""))
                        st.write("**Source:**", doc.metadata.get("source_url", ""))
                        st.write(doc.page_content[:2500])

        except Exception as e:
            st.error(f"Error: {e}")