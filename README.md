# Dr. Insight — Medical RAG System

> An end-to-end Retrieval-Augmented Generation (RAG) system for intelligent healthcare Q&A, built on HSE (Health Service Executive) medical data.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB%20%7C%20FAISS-orange)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203-purple)
![RAGAS](https://img.shields.io/badge/Evaluation-RAGAS-red)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What It Does

Most LLMs hallucinate when asked medical questions because they rely on training memory. **Dr. Insight** fixes this by retrieving real, verified medical content from the HSE website and grounding every answer in that evidence before generating a response.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────┐
│           Query Intelligence                │
│  • Detects disease name from query          │
│  • Detects section (symptoms/causes/etc.)   │
└─────────────────┬───────────────────────────┘
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
 Similarity Search      MMR Search
 (ChromaDB / FAISS)   (diversity-aware)
       └──────────┬──────────┘
                  ▼
      Merge + Deduplicate (~20 docs)
                  │
                  ▼
      Metadata Boost (disease/section match)
                  │
                  ▼
      CrossEncoder Reranker
      (ms-marco-MiniLM-L6-v2)
                  │
                  ▼
      Top 4 Question Groups
                  │
                  ▼
      Groq LLaMA 3 (Answer Generation)
                  │
                  ▼
        Final Answer + Sources
```

---

## Key Features

- **Smart retrieval** — Combines similarity search + MMR + metadata filtering + CrossEncoder reranking
- **Two vector DB experiments** — ChromaDB (persistent) vs FAISS (in-memory) comparison
- **Three chunking strategies** — QA-pair, Recursive Character, Semantic chunking
- **Fair LLM evaluation** — Both Ollama and Groq answers judged by the same Groq 70B model
- **RAGAS evaluation** — Faithfulness, Context Recall, Context Precision scored automatically
- **Streamlit UI** — Interactive disease selector with suggested questions
- **20 medical questions** — Covering appendicitis, diabetes, MRSA, migraine, stroke, and more

---

## Project Structure

```
├── app.py                          # Streamlit UI
├── ingest.py                       # Build ChromaDB vector store (QA-pair chunks)
├── ingest_recursive.py             # Build ChromaDB with recursive chunking
├── ingest_semantic.py              # Build ChromaDB with semantic chunking
├── retrieval.py                    # Core RAG pipeline (query → retrieve → rerank → answer)
├── generation.py                   # LLM generation helpers
│
├── evaluation_ragas.py             # RAGAS evaluation on base RAG (ChromaDB)
├── evaluation_groq.py              # Groq LLM evaluation
├── evaluation.py                   # Ollama-based evaluation
├── evaluation_fair_comparison.py   # Same judge (Groq 70B) for both Ollama & Groq answers
├── evaluation_chunking_ragas.py    # Compare all 3 chunking methods with RAGAS
├── evaluation_FAISS_ChromaDB.py    # ChromaDB vs FAISS comparison with RAGAS
├── run_single_experiment_ragas.py  # Run one experiment (baseline/faiss) and save results
│
├── questions.txt                   # 20 evaluation questions
├── docs/                           # Medical knowledge base (HSE markdown files)
├── db/                             # Vector store (not tracked in git)
└── requirements.txt
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/sarivipin/RAG-FINAL_Dr-Insight.git
cd RAG-FINAL_Dr-Insight
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
pip install faiss-cpu langchain-groq python-dotenv
```

### 2. Configure API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com)

### 3. Build the vector store

```bash
python ingest.py
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## Evaluation

Run evaluations in this order (each script depends on the previous output):

```bash
# 1. Base RAG evaluation with RAGAS
python evaluation_ragas.py

# 2. Groq-based generation evaluation
python evaluation_groq.py

# 3. Fair comparison (same judge for both Ollama & Groq)
python evaluation_fair_comparison.py

# 4. Chunking method comparison
python evaluation_chunking_ragas.py

# 5. ChromaDB vs FAISS comparison
python evaluation_FAISS_ChromaDB.py
```

### Metrics Explained

| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer only use information from the retrieved context? |
| **Context Recall** | Did retrieval find all the information needed to answer correctly? |
| **Context Precision** | Was the retrieved context relevant, or full of noise? |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM (generation) | Groq `llama-3.1-8b-instant` |
| LLM (evaluation judge) | Groq `llama-3.3-70b-versatile` |
| Vector DB | ChromaDB, FAISS |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| RAG Framework | LangChain |
| Evaluation | RAGAS |
| UI | Streamlit |
| Data Source | [HSE Ireland](https://www.hse.ie/eng/health/az/) |

---

## Data Source

All medical content is sourced from the **HSE (Health Service Executive) Ireland** public website and converted into structured markdown files covering diseases such as appendicitis, type 1 & 2 diabetes, MRSA, migraine, hypertension, stroke, kidney stones, and more.

---

## License

MIT License — free to use, modify, and distribute.
