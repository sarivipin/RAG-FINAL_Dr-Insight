/import os
import re
import shutil
import hashlib
from typing import List, Dict, Optional, Tuple

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================================================
# CONFIG
# =========================================================
DOCS_PATH = "docs"
PERSIST_DIRECTORY = "db/chroma_db"
COLLECTION_NAME = "medical_markdown_docs"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MIN_QUESTION_CHARS = 5
MIN_ANSWER_CHARS = 10
MAX_ANSWER_CHARS = 800

RESET_DB_ON_RUN = True


# =========================================================
# TEXT HELPERS
# =========================================================
def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_newlines(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_inline_text(text: str) -> str:
    text = clean_text(text)
    return re.sub(r"\s+", " ", text).strip()


def safe_slug(text: str) -> str:
    text = clean_inline_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "general"


# =========================================================
# FILE LOADING
# =========================================================
def load_documents(docs_path: str = DOCS_PATH) -> List[Document]:
    print(f"Loading markdown files from: {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"Directory '{docs_path}' does not exist. Create it and add your .md files."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(f"No .md files found in '{docs_path}'.")

    print(f"Loaded {len(documents)} markdown files.\n")
    return documents


# =========================================================
# TITLE EXTRACTION
# =========================================================
def get_title_from_content_or_filename(content: str, source_path: str) -> str:
    content = clean_text(content)

    match = re.search(r"^\s*#\s+(.+?)\s*$", content, flags=re.MULTILINE)
    if match:
        return clean_inline_text(match.group(1))

    filename = os.path.basename(source_path)
    name = os.path.splitext(filename)[0]
    name = name.replace("_", " ").replace("-", " ")
    return clean_inline_text(name).title()


# =========================================================
# MARKDOWN BLOCK PARSING
# =========================================================
def split_into_question_blocks(content: str) -> List[str]:
    """
    Split by each '## Question' block.
    Robust to blank lines, indentation, and missing final '---'.
    """
    content = clean_text(content)

    parts = re.split(
        r"(?=^\s*##\s+Question\s*$)",
        content,
        flags=re.MULTILINE,
    )

    blocks = []
    for part in parts:
        part = part.strip()
        if part.startswith("## Question"):
            blocks.append(part)

    return blocks


def extract_between(
    text: str,
    start_pattern: str,
    end_patterns: Optional[List[str]] = None,
    flags: int = re.MULTILINE | re.DOTALL,
) -> Optional[str]:
    start_match = re.search(start_pattern, text, flags=flags)
    if not start_match:
        return None

    remaining = text[start_match.end():]

    if not end_patterns:
        return remaining.strip()

    end_positions = []
    for pattern in end_patterns:
        m = re.search(pattern, remaining, flags=flags)
        if m:
            end_positions.append(m.start())

    if end_positions:
        return remaining[:min(end_positions)].strip()

    return remaining.strip()


def parse_question_block(block: str, disease_title: str) -> Optional[Dict]:
    """
    Expected format:

    ## Question
    What are the symptoms of abscess?

    ### Answer
    A skin abscess often appears ...

    **Section:** symptoms
    **Source:** https://...
    """
    block = clean_text(block)

    if not block.startswith("## Question"):
        return None

    question = extract_between(
        block,
        r"^\s*##\s+Question\s*$",
        end_patterns=[r"^\s*###\s+Answer\s*$"],
    )

    answer = extract_between(
        block,
        r"^\s*###\s+Answer\s*$",
        end_patterns=[
            r"^\s*\*\*Section:\*\*",
            r"^\s*\*\*Source:\*\*",
            r"^\s*---\s*$",
        ],
    )

    section_match = re.search(
        r"^\s*\*\*Section:\*\*\s*(.+?)\s*$",
        block,
        flags=re.MULTILINE,
    )
    source_match = re.search(
        r"^\s*\*\*Source:\*\*\s*(.+?)\s*$",
        block,
        flags=re.MULTILINE,
    )

    question = clean_text(question or "")
    answer = clean_text(answer or "")

    # Safety cleanup if metadata leaked into answer
    answer = re.sub(
        r"\n?\*\*Section:\*\*.*$",
        "",
        answer,
        flags=re.MULTILINE | re.DOTALL,
    ).strip()
    answer = re.sub(
        r"\n?\*\*Source:\*\*.*$",
        "",
        answer,
        flags=re.MULTILINE | re.DOTALL,
    ).strip()
    answer = re.sub(
        r"\n?---\s*$",
        "",
        answer,
        flags=re.MULTILINE | re.DOTALL,
    ).strip()

    section = clean_inline_text(section_match.group(1)) if section_match else "general"
    source_url = clean_inline_text(source_match.group(1)) if source_match else ""

    if len(question) < MIN_QUESTION_CHARS:
        return None

    if len(answer) < MIN_ANSWER_CHARS:
        return None

    return {
        "disease": disease_title,
        "question": question,
        "answer": answer,
        "section": safe_slug(section),
        "source_url": source_url,
    }


def parse_markdown_file(content: str, source_path: str) -> List[Dict]:
    disease_title = get_title_from_content_or_filename(content, source_path)
    blocks = split_into_question_blocks(content)

    items = []
    for block in blocks:
        parsed = parse_question_block(block, disease_title)
        if parsed:
            items.append(parsed)

    return items


# =========================================================
# DEDUPLICATION
# =========================================================
def make_qa_key(disease: str, question: str, answer: str) -> str:
    normalized = "||".join([
        clean_inline_text(disease).lower(),
        clean_inline_text(question).lower(),
        clean_inline_text(answer).lower(),
    ])
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def make_question_hash(file_source: str, item: Dict) -> str:
    unique_base = "||".join([
        clean_inline_text(file_source).lower(),
        clean_inline_text(item["disease"]).lower(),
        clean_inline_text(item["question"]).lower(),
        clean_inline_text(item["answer"]).lower(),
    ])
    return hashlib.md5(unique_base.encode("utf-8")).hexdigest()[:16]


# =========================================================
# ANSWER CHUNKING
# =========================================================
def split_long_answer(answer: str, max_chars: int = MAX_ANSWER_CHARS) -> List[str]:
    answer = clean_text(answer)

    if len(answer) <= max_chars:
        return [answer]

    paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(para) <= max_chars:
            if not current:
                current = para
            elif len(current) + 2 + len(para) <= max_chars:
                current += "\n\n" + para
            else:
                chunks.append(current.strip())
                current = para
            continue

        sentences = re.split(r"(?<=[.!?])\s+", para)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    piece = sentence[i:i + max_chars].strip()
                    if not piece:
                        continue
                    if current:
                        chunks.append(current.strip())
                        current = ""
                    chunks.append(piece)
                continue

            if not current:
                current = sentence
            elif len(current) + 1 + len(sentence) <= max_chars:
                current += " " + sentence
            else:
                chunks.append(current.strip())
                current = sentence

    if current.strip():
        chunks.append(current.strip())

    return [clean_text(c) for c in chunks if clean_text(c)]


# =========================================================
# DOCUMENT BUILDING
# =========================================================
def build_chunk_text(
    disease: str,
    question: str,
    answer_part: str,
    section: str,
    source_url: str,
) -> str:
    parts = [
        f"# {disease}",
        "",
        "## Question",
        question,
        "",
        "### Answer",
        answer_part,
        "",
        f"**Section:** {section}",
    ]

    if source_url:
        parts.extend(["", f"**Source:** {source_url}"])

    return "\n".join(parts).strip()


def create_documents_from_item(item: Dict, file_source: str, question_index: int) -> List[Document]:
    documents = []

    answer_parts = split_long_answer(item["answer"], max_chars=MAX_ANSWER_CHARS)
    total_parts = len(answer_parts)
    question_hash = make_question_hash(file_source, item)

    for part_no, answer_part in enumerate(answer_parts, start=1):
        chunk_id = f"{question_hash}_p{part_no}"

        chunk_text = build_chunk_text(
            disease=item["disease"],
            question=item["question"],
            answer_part=answer_part,
            section=item["section"],
            source_url=item["source_url"],
        )

        metadata = {
            "chunk_id": chunk_id,
            "source": file_source,
            "file_source": file_source,
            "disease": item["disease"],
            "question": item["question"],
            "section": item["section"],
            "source_url": item["source_url"],
            "chunk_type": "qa_pair",
            "question_index": question_index,
            "answer_part": part_no,
            "total_answer_parts": total_parts,
            "question_hash": question_hash,
        }

        documents.append(Document(page_content=chunk_text, metadata=metadata))

    return documents


def create_documents(markdown_docs: List[Document]) -> List[Document]:
    print("Parsing markdown into QA chunks...\n")

    raw_items: List[Tuple[Dict, str]] = []

    for doc in markdown_docs:
        content = clean_text(doc.page_content)
        source_path = doc.metadata.get("source", "")

        parsed_items = parse_markdown_file(content, source_path)
        print(f"{source_path} -> {len(parsed_items)} parsed items")

        for item in parsed_items:
            raw_items.append((item, source_path))

    print(f"\nRaw items found: {len(raw_items)}")

    if not raw_items:
        raise ValueError(
            "No valid QA pairs were found. Check whether your markdown files follow the expected format."
        )

    deduped_items: List[Tuple[Dict, str]] = []
    seen_qa_keys = set()

    for item, source_path in raw_items:
        qa_key = make_qa_key(item["disease"], item["question"], item["answer"])
        if qa_key not in seen_qa_keys:
            seen_qa_keys.add(qa_key)
            deduped_items.append((item, source_path))

    final_docs: List[Document] = []
    seen_chunk_ids = set()

    for idx, (item, source_path) in enumerate(deduped_items, start=1):
        docs = create_documents_from_item(
            item=item,
            file_source=source_path,
            question_index=idx,
        )

        for d in docs:
            cid = d.metadata["chunk_id"]
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                final_docs.append(d)

    print(f"After QA deduplication: {len(deduped_items)}")
    print(f"Final vector chunks: {len(final_docs)}\n")

    if final_docs:
        for i, chunk in enumerate(final_docs[:3], start=1):
            print(f"Chunk {i}")
            print(f"Metadata: {chunk.metadata}")
            print("-" * 80)
            print(chunk.page_content[:700])
            print("=" * 100)

    return final_docs


# =========================================================
# EMBEDDINGS
# =========================================================
def get_embedding_model() -> HuggingFaceEmbeddings:
    print("Loading Sentence Transformer embedding model...")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# =========================================================
# VECTOR STORE
# =========================================================
def reset_vector_store(persist_directory: str = PERSIST_DIRECTORY) -> None:
    if os.path.exists(persist_directory):
        print(f"Deleting existing vector store at: {persist_directory}")
        shutil.rmtree(persist_directory)


def create_vector_store(chunks: List[Document], persist_directory: str = PERSIST_DIRECTORY) -> Chroma:
    print("Creating vector store with Sentence Transformers embeddings...")

    os.makedirs(os.path.dirname(persist_directory), exist_ok=True)

    embeddings = get_embedding_model()
    ids = [doc.metadata["chunk_id"] for doc in chunks]

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=ids,
        persist_directory=persist_directory,
        collection_name=COLLECTION_NAME,
    )

    print(f"Vector store saved to: {persist_directory}")
    return vectorstore


# =========================================================
# MAIN
# =========================================================
def main():
    print("=== Medical Markdown RAG Ingestion with QA Pair Chunking ===\n")

    documents = load_documents()
    chunks = create_documents(documents)

    if RESET_DB_ON_RUN:
        reset_vector_store(PERSIST_DIRECTORY)

    vectorstore = create_vector_store(chunks)

    print("\nIngestion complete.")
    print(f"Stored {vectorstore._collection.count()} chunks.")
    print(f"Persist directory: {PERSIST_DIRECTORY}")


if __name__ == "__main__":
    main()