import os
import json
import time
import uuid
import shutil
import torch

from dotenv import load_dotenv
from collections import defaultdict
from typing import Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever

from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from vercel_blob import put

app = FastAPI(title="Tax Audit AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
       "http://localhost:8080",
       "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------- CONFIG ---------------- #

load_dotenv()

MODEL_POOL = [
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "llama3-8b-8192",
]
current_model_idx = 0

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)


def get_llm():
    return ChatGroq(model=MODEL_POOL[current_model_idx], temperature=0.1)


llm = get_llm()

# ---------------- MEMORY ---------------- #

session_dbs = {}
session_histories = {}
session_flags = {}
flag_counts = defaultdict(lambda: {"Red": 0, "Yellow": 0, "Green": 0})

# ---------------- FASTAPI ---------------- #




# ---------------- MODELS ---------------- #


class ChatRequest(BaseModel):
    question: str


# ---------------- HELPERS ---------------- #


def rotate_model():
    global llm, current_model_idx
    current_model_idx = (current_model_idx + 1) % len(MODEL_POOL)
    llm = get_llm()


def fallback_result(chunk):

    txt = chunk.lower()

    if any(
        w in txt
        for w in [
            "late",
            "missing withholding",
            "irregular",
            "violation",
            "unauthorized",
            "non-compliance",
        ]
    ):
        return {
            "flag": "Red",
            "text": chunk,
            "summary": "Possible compliance violation detected.",
            "reason": "Violation indicators found.",
            "recommendation": "Review and remediate immediately.",
        }

    elif any(w in txt for w in ["timely", "compliant", "proper", "approved"]):
        return {
            "flag": "Green",
            "text": chunk,
            "summary": "Appears compliant.",
            "reason": "Positive compliance indicators found.",
            "recommendation": "No action needed.",
        }

    return {
        "flag": "Yellow",
        "text": chunk,
        "summary": "Needs manual review.",
        "reason": "Insufficient or ambiguous evidence.",
        "recommendation": "Validate supporting documentation.",
    }


def classify_batch(chunks):

    chunk_block = ""

    for i, c in enumerate(chunks, 1):
        chunk_block += f"[{i}] {c[:350]}\n\n"

    prompt = f"""
You are a tax audit risk classifier.

Classify each chunk:
Red = clear violation
Yellow = ambiguous or needs review
Green = compliant

IMPORTANT:
- summary MAX 20 words
- reason MAX 15 words
- recommendation MAX 15 words
- concise professional wording
- do not write paragraphs

Return JSON only:

[
{{
"id":1,
"flag":"Red",
"summary":"Missing withholding identified.",
"reason":"Statutory deduction absent.",
"recommendation":"Correct and refile."
}}
]

CHUNKS:
{chunk_block}
"""

    for _ in range(len(MODEL_POOL)):
        try:
            response = llm.invoke(prompt)

            raw = response.content.strip()

            start = raw.find("[")
            end = raw.rfind("]") + 1

            parsed = json.loads(raw[start:end])

            results = []

            for i, item in enumerate(parsed):

                results.append(
                    {
                        "flag": item["flag"],
                        "text": chunks[i],
                        "summary": item["summary"],
                        "reason": item["reason"],
                        "recommendation": item["recommendation"],
                    }
                )

            while len(results) < len(chunks):
                results.append(fallback_result(chunks[len(results)]))

            return results

        except Exception as e:

            if "429" in str(e):
                rotate_model()
                time.sleep(1)
            else:
                break

    return [fallback_result(c) for c in chunks]


def process_pdf(path, session_id, batch_size=8):

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    db = FAISS.from_documents(chunks, embeddings)

    session_dbs[session_id] = db
    session_histories[session_id] = ChatMessageHistory()

    all_results = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i : i + batch_size]

        results = classify_batch(batch)

        all_results.extend(results)

        for r in results:
            flag_counts[session_id][r["flag"]] += 1

    session_flags[session_id] = all_results

    total = len(all_results)

    return {
        "session_id": session_id,
        "stats": {
            "total_chunks": total,
            "red": flag_counts[session_id]["Red"],
            "yellow": flag_counts[session_id]["Yellow"],
            "green": flag_counts[session_id]["Green"],
        },
        "flags": all_results,
    }


def build_rag(session_id):

    db = session_dbs.get(session_id)

    if not db:
        raise ValueError("Session not found")

    retriever = db.as_retriever(search_kwargs={"k": 5})

    contextual_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Convert user question to standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    hist_retriever = create_history_aware_retriever(llm, retriever, contextual_prompt)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are Pakistani tax audit assistant.
Answer using context only.
Be concise but accurate.

Context:
{context}
""",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = create_retrieval_chain(
        hist_retriever, create_stuff_documents_chain(llm, qa_prompt)
    )

    return RunnableWithMessageHistory(
        chain,
        lambda _: session_histories[session_id],
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# ---------------- ROUTES ---------------- #


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/audit/upload")
async def upload_audit(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Upload PDF only")

    session_id = str(uuid.uuid4())

    # Read uploaded bytes once
    pdf_bytes = await file.read()

    # save temp for LangChain processing
    path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")

    with open(path, "wb") as f:
        f.write(pdf_bytes)

    # Upload to Vercel Blob
    blob = put(
        f"audits/{session_id}.pdf",
        pdf_bytes,
        {
            "access": "public",
            "token": os.getenv("BLOB_READ_WRITE_TOKEN"),
        },
    )

    # run analysis
    result = process_pdf(path, session_id)

    result["pdf_url"] = blob["url"]

    return result


@app.get("/audit/pdf/{session_id}")
def serve_pdf(session_id: str):
    path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")
    if not os.path.exists(path):
        raise HTTPException(404, "PDF not found")
    return FileResponse(path, media_type="application/pdf")

@app.get("/audit/report/{session_id}")
def get_report(session_id: str):

    if session_id not in session_flags:
        raise HTTPException(404, "Session not found")

    return {
        "session_id": session_id,
        "stats": flag_counts[session_id],
        "flags": session_flags[session_id],
    }


@app.post("/audit/chat/{session_id}")
def chat_with_audit(session_id: str, body: ChatRequest):
    if session_id not in session_dbs:
        raise HTTPException(404, "Session not found")

    chain = build_rag(session_id)

    result = chain.invoke(
        {"input": body.question}, config={"configurable": {"session_id": session_id}}
    )

    return {"question": body.question, "answer": result["answer"]}


# uvicorn main3:app --reload
