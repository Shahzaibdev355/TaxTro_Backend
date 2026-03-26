from dotenv import load_dotenv
import os
import json
import hashlib
import time
from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
    JSONLoader,
)
from langchain_core.documents import Document

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()


# -------------------------------
# Initialize Embeddings
# -------------------------------
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},  # change to "cpu" if no GPU
)



# -------------------------------
# Initialize Pinecone
# -------------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
index_name = os.getenv("PINECONE_INDEX")

vectorstore = PineconeVectorStore(index=index, embedding=embedding, text_key="text")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# def read_doc(directory):
#     file_loader=PyPDFDirectoryLoader(directory)
#     documents=file_loader.load()
#     return documents
    

# doc=read_doc('/home/ammar/Documents/Learning/FYP/Data/blank.pdf')

def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return doc  # Fixed: was returning 'docs' instead of 'doc'

# Configuration
DOC_MAP_PATH = Path("doc_map.json")

def compute_file_hash(path: Path, chunk_size: int = 8192) -> str:
    """SHA1 hash of file contents (fast, reliable for change detection)."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:10]  # short hash

def sanitize_namespace(file_path: Path, file_hash: str) -> str:
    """
    Create a safe namespace string from filename and short hash.
    Example: "my_report.pdf__a1b2c3d4e5"
    """
    name = file_path.name.lower().replace(" ", "_")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
    sanitized = "".join(ch for ch in name if ch in allowed)
    return f"{sanitized}__{file_hash}"

def load_doc_map() -> dict:
    if DOC_MAP_PATH.exists():
        return json.loads(DOC_MAP_PATH.read_text(encoding="utf-8"))
    return {}

def save_doc_map(m: dict):
    DOC_MAP_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")

def read_document(path: Path):
    """
    Load document based on file extension.
    Supports: .pdf, .txt, .docx, .json
    """
    suffix = path.suffix.lower()
    
    if suffix == '.pdf':
        loader = PyMuPDFLoader(str(path))
    elif suffix == '.txt':
        loader = TextLoader(str(path), encoding='utf-8')
    elif suffix == '.docx':
        loader = Docx2txtLoader(str(path))
    elif suffix == '.json':
        try:
            loader = JSONLoader(str(path), jq_schema='.', text_content=False)
        except:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return [Document(page_content=content, metadata={'source': str(path)})]
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt, .docx, .json")
    
    return loader.load()


def ingest_file(path: Path, force_reingest: bool = False):
    """
    Ingest a single file with namespace tracking:
    - Computes hash to detect changes
    - Generates unique namespace
    - Skips if already ingested and unchanged (unless force_reingest=True)
    - Deletes old namespace if file changed
    
    Returns: namespace string
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    
    if not path.is_file():
        raise ValueError(f"{path} is not a file")

    file_hash = compute_file_hash(path)
    namespace = sanitize_namespace(path, file_hash)
    doc_map = load_doc_map()

    # Check if path already in mapping
    existing_entry = None
    for ns, meta in list(doc_map.items()):
        if meta.get("path") == str(path):
            existing_entry = (ns, meta)
            break
    
    if existing_entry:
        old_ns, old_meta = existing_entry
        if old_meta.get("hash") != file_hash or force_reingest:
            print(f"File changed or reingest requested: {path}  -> removing old namespace {old_ns}")
            try:
                # pc.Index(index_name).delete(namespace=old_ns, delete_all=True)
                index.delete(namespace=old_ns, delete_all=True)
                print(f"Deleted old namespace {old_ns}")
            except Exception as e:
                print(f"Warning: failed to delete old namespace {old_ns}: {e}")
            doc_map.pop(old_ns, None)
        else:
            print(f"No changes for {path}; already ingested under namespace {old_ns}")
            return old_ns

    # Load & chunk
    print(f"Loading {path} ...")
    docs = read_document(path)
    print(f"Chunking {len(docs)} raw doc(s) ...")
    chunks = chunk_data(docs)

    # Ingest to Pinecone with namespace
    print(f"Upserting {len(chunks)} chunks to Pinecone namespace: {namespace} ...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        index_name=index_name,
        namespace=namespace
    )

    # Update mapping
    doc_map[namespace] = {
        "path": str(path),
        "hash": file_hash,
        "ingested_at": int(time.time())
    }
    save_doc_map(doc_map)
    print(f"Ingest completed for {path} -> namespace: {namespace}")
    return namespace


# ========== USAGE EXAMPLES ==========

# IMPORTANT: Re-ingest with force_reingest=True because chunk_data was fixed
print("Re-ingesting file with fixed chunk_data function...\n")
ingest_file(
    Path("test.pdf"),
    force_reingest=True  # Force re-ingest to use fixed chunking
)

print("\n" + "="*80)
print("Ingestion complete! Now check testing_db.ipynb to query the data.")
print("="*80)

# View all ingested files
print("\n=== Currently Ingested Files ===")
doc_map = load_doc_map()
for namespace, meta in doc_map.items():
    print(f"\nNamespace: {namespace}")
    print(f"  File: {meta['path']}")
    print(f"  Hash: {meta['hash']}")
    print(f"  Ingested: {meta['ingested_at']}")