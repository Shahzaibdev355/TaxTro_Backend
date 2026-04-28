from dotenv import load_dotenv
import os
import json
import hashlib
import time
import urllib.request
from pathlib import Path
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
    JSONLoader,
)
from langchain_core.documents import Document
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# ── Staging dir for downloaded files ──────────────────────────────────────────
UPLOAD_DIR = Path("uploads_staging")
UPLOAD_DIR.mkdir(exist_ok=True)

DOC_MAP_PATH = Path("doc_map.json")

load_dotenv()

# ── Embeddings ─────────────────────────────────────────────────────────────────
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
)

# ── Pinecone ───────────────────────────────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
index_name = os.getenv("PINECONE_INDEX")


# ── Helpers ────────────────────────────────────────────────────────────────────
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def compute_file_hash(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()[:10]


def sanitize_namespace(file_path: Path, file_hash: str) -> str:
    name = file_path.name.lower().replace(" ", "_")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
    sanitized = "".join(ch for ch in name if ch in allowed)
    return f"{sanitized}__{file_hash}"


def load_doc_map() -> dict:
    if DOC_MAP_PATH.exists():
        text = DOC_MAP_PATH.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        return json.loads(text)
    return {}


def save_doc_map(m: dict):
    DOC_MAP_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")


def read_document(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyMuPDFLoader(str(path))
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix == ".json":
        try:
            loader = JSONLoader(str(path), jq_schema=".", text_content=False)
        except Exception:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": str(path)})]
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader.load()


# ── Core functions ─────────────────────────────────────────────────────────────
def ingest_from_url(url: str, force_reingest: bool = False) -> str:
    """Download file from Vercel Blob URL, ingest into Pinecone, return namespace."""
    filename = url.split("/")[-1].split("?")[0]
    local_path = UPLOAD_DIR / filename

    print(f"Downloading: {url} -> {local_path}")
    urllib.request.urlretrieve(url, str(local_path))

    file_hash = compute_file_hash(local_path)
    namespace = sanitize_namespace(Path(filename), file_hash)
    doc_map = load_doc_map()

    # Check if already ingested
    existing_entry = None
    for ns, meta in list(doc_map.items()):
        if meta.get("path") == url:
            existing_entry = (ns, meta)
            break

    if existing_entry:
        old_ns, old_meta = existing_entry
        if old_meta.get("hash") == file_hash and not force_reingest:
            print(f"Already ingested: {url} -> namespace {old_ns}")
            local_path.unlink(missing_ok=True)
            return old_ns
        try:
            index.delete(namespace=old_ns, delete_all=True)
            print(f"Deleted old namespace: {old_ns}")
        except Exception as e:
            print(f"Warning: could not delete {old_ns}: {e}")
        doc_map.pop(old_ns, None)

    docs = read_document(local_path)
    chunks = chunk_data(docs)

    print(f"Upserting {len(chunks)} chunks -> namespace: {namespace}")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        index_name=index_name,
        namespace=namespace,
    )

    doc_map[namespace] = {
        "path": url,
        "hash": file_hash,
        "ingested_at": int(time.time()),
    }
    save_doc_map(doc_map)

    # Delete local staging file after ingestion
    local_path.unlink(missing_ok=True)
    print(f"Ingestion done -> namespace: {namespace}")
    return namespace



def delete_from_pinecone(url: str) -> bool:
    """Remove a file's Pinecone namespace using its Vercel Blob URL."""
    doc_map = load_doc_map()
    for ns, meta in list(doc_map.items()):
        if meta.get("path") == url:
            try:
                index.delete(namespace=ns, delete_all=True)
                print(f"Deleted Pinecone namespace: {ns}")
            except Exception as e:
                print(f"Warning: could not delete namespace {ns}: {e}")
            doc_map.pop(ns)
            save_doc_map(doc_map)
            return True
    print(f"No namespace found for URL: {url}")
    return False