"""
Ingestion using Recursive Character Text Splitting.
Splits markdown files into overlapping chunks based on natural text boundaries
(paragraphs → sentences → words) with configurable chunk size and overlap.
"""
import os
import re
import shutil
import hashlib
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================================================
# CONFIG
# =========================================================
DOCS_PATH = "docs"
PERSIST_DIRECTORY = "db/chroma_db_recursive"
COLLECTION_NAME = "medical_recursive_chunks"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

RESET_DB_ON_RUN = True


# =========================================================
# HELPERS
# =========================================================
def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_title_from_content(content: str, source_path: str) -> str:
    match = re.search(r"^\s*#\s+(.+?)\s*$", content, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    name = os.path.splitext(os.path.basename(source_path))[0]
    return name.replace("_", " ").replace("-", " ").title()


def extract_section(text: str) -> str:
    match = re.search(r"\*\*Section:\*\*\s*(.+)", text)
    return match.group(1).strip() if match else "general"


def extract_source_url(text: str) -> str:
    match = re.search(r"\*\*Source:\*\*\s*(.+)", text)
    return match.group(1).strip() if match else ""


# =========================================================
# LOADING & CHUNKING
# =========================================================
def load_documents(docs_path: str = DOCS_PATH) -> List[Document]:
    print(f"Loading markdown files from: {docs_path}")
    loader = DirectoryLoader(
        path=docs_path, glob="*.md",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    if not documents:
        raise FileNotFoundError(f"No .md files found in '{docs_path}'.")
    print(f"Loaded {len(documents)} markdown files.\n")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    print(f"Splitting with RecursiveCharacterTextSplitter (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        content = clean_text(doc.page_content)
        source_path = doc.metadata.get("source", "")
        disease = get_title_from_content(content, source_path)

        splits = splitter.split_text(content)

        for i, split_text in enumerate(splits):
            chunk_id = hashlib.md5(f"{source_path}_{i}".encode()).hexdigest()[:16]
            section = extract_section(split_text)
            source_url = extract_source_url(split_text)

            chunks.append(Document(
                page_content=split_text,
                metadata={
                    "chunk_id": f"{chunk_id}_r{i}",
                    "source": source_path,
                    "disease": disease,
                    "section": section,
                    "source_url": source_url,
                    "chunk_type": "recursive",
                    "chunk_index": i,
                },
            ))

    print(f"Created {len(chunks)} chunks.\n")
    return chunks


# =========================================================
# VECTOR STORE
# =========================================================
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(chunks: List[Document]) -> Chroma:
    print("Creating vector store...")
    if RESET_DB_ON_RUN and os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    os.makedirs(os.path.dirname(PERSIST_DIRECTORY), exist_ok=True)
    embeddings = get_embedding_model()
    ids = [doc.metadata["chunk_id"] for doc in chunks]

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, ids=ids,
        persist_directory=PERSIST_DIRECTORY, collection_name=COLLECTION_NAME,
    )
    print(f"Vector store saved to: {PERSIST_DIRECTORY}")
    return vectorstore


# =========================================================
# MAIN
# =========================================================
def main():
    print("=== Ingestion: Recursive Character Splitting ===\n")
    documents = load_documents()
    chunks = chunk_documents(documents)
    vectorstore = create_vector_store(chunks)
    print(f"\nIngestion complete. Stored {vectorstore._collection.count()} chunks.")


if __name__ == "__main__":
    main()
