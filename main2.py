from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import re
import uuid
from fastapi import Request, Response

# LangChain & Vector DB
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


# for chat history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder


# Load environment variables
load_dotenv()

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(
    title="TaxTro API",
    description="AI Tax Advisor for Pakistan Tax Laws",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Request / Response Models
# -------------------------------
class AskRequest(BaseModel):
    question: str


class SourceDocument(BaseModel):
    content_snippet: str
    metadata: dict


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


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

vectorstore = PineconeVectorStore(index=index, embedding=embedding, text_key="text")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -------------------------------
# Initialize LLM
# -------------------------------


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.5
)

# -------------------------------
# Prompt Template (STRICT)
# -------------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are **TaxGPT**, an AI Tax Advisor specialized exclusively in **Pakistan Tax Laws**
(Income Tax Ordinance, Sales Tax Act, Federal Excise Act, SROs, Rules, Notifications).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 STRICT OPERATING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. You MUST primarily use the retrieved **Context** to answer.
2. You may ONLY use your general Pakistan tax knowledge if the context is incomplete.
   If the user asks a comparative or follow-up question based on prior discussion,
   you may rely on the chat history even if the retrieved context does not explicitly
   contain a direct comparison.
   For comparative questions, provide a high-level legal position even if exact tax
   rates or figures are not available, without inventing specific percentages or provisions.
3. DO NOT invent SRO numbers, sections, rules, summaries, case law, or PDF URLs.
4. If the query is NOT related to Pakistan tax law → politely refuse.
5. If NO relevant legal reference exists → clearly state that.
6. Output MUST follow the EXACT structure below.
7. Recommendation section MUST appear **ONLY IF**:
   - The user describes a personal, business, or practical scenario.
   - If the question is purely informational → OMIT recommendation.
8. In the “Source Documents (PDF)” section:
   - If no PDF is explicitly mentioned in context, write:
     “(No PDF URL available, as the context is a text snippet)”

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 CONTEXT (Retrieved Documents)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬 USER QUESTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{input}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧾 REQUIRED OUTPUT FORMAT (NO DEVIATION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### ✅ Main Answer
Provide a clear, legally accurate response to the user's question in professional advisory tone.

### 📜 Applicable Legal References
⚠️ Strict Rules (Must Follow):

Do NOT repeat the same legal reference.

Uniqueness Rules (MANDATORY):

- Treat a parent Section as ONE legal reference.
- If multiple sub-sections, clauses, or provisos belong to the SAME Section:
  → MERGE them into a SINGLE reference.
- Do NOT list sub-sections (e.g., 80C(4), 80C(5), 80C(6)) separately.
- Use ONLY the parent section number (e.g., Section 80C).
- The Short Description may summarize multiple sub-sections together.
- If a reference has already been listed once, do NOT include it again, even if it appears in multiple contexts.
- Only include distinct and relevant legal references.

For EACH UNIQUE reference, list them separately as clickable items in UI:

- **Title:** (e.g., Income Tax Ordinance, 2001 / SRO 123(I)/2023)
- **Provision Type:** (SRO / Rule / Section / Sub-Section / Clause, etc.)
- **Provision Number:** (e.g., Rule 45A, Section 80C, Sub-section (5))
- **Short Description:** (7–8 lines, concise, extracted from legal context — no repetition)

---

### 📝 Summary
Provide a concise summary (7–8 bullet points max) explaining the legal position in simple language.

### ⚠️ Recommendation
❗ Include this section ONLY IF the user describes a specific situation.
- Provide cautious, compliance-focused guidance.
- Use wording like “may consider”, “it is advisable”, “subject to FBR interpretation”.
- Do NOT provide aggressive tax planning or evasion advice.

If the user did NOT describe a case → OMIT THIS SECTION COMPLETELY.


### 📎 Source Documents (PDF)
List ONLY PDFs that were actually used from the context:

- **Document Title**
- **PDF URL**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚫 PROHIBITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- No assumptions
- No foreign tax laws
- No legal advice disclaimer text
- No markdown outside the defined sections
"""
)


# -------------------------------
# Create RAG Chain
# -------------------------------
document_chain = create_stuff_documents_chain(llm, prompt)
# retrieval_chain = create_retrieval_chain(retriever, document_chain)




# -------------------------------
# Session Store (IN-MEMORY)
# -------------------------------
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]




# -------------------------------
# History-aware retriever
# -------------------------------
contextualize_q_system_prompt = """
Given a chat history and the latest user question,
which might reference context in the chat history,
formulate a standalone question that can be understood
without the chat history. Do NOT answer the question.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

retrieval_chain = create_retrieval_chain(
    history_aware_retriever,
    document_chain
)

conversational_rag_chain = RunnableWithMessageHistory(
    retrieval_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)






def parse_llm_response(text: str):
    def extract(section):
        pattern = rf"{section}\n(.*?)(?=\n### |\Z)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    return {
        "main_answer": extract("### ✅ Main Answer"),
        "legal_references": extract("### 📜 Applicable Legal References"),
        "summary": extract("### 📝 Summary"),
        "recommendation": extract("### ⚠️ Recommendation"),
        "source_documents_text": extract("### 📎 Source Documents \\(PDF\\)"),
    }


# -------------------------------
# API Endpoint
# -------------------------------
@app.get('/test')
async def home():
    return 'Backend running!'



@app.post("/ask")
async def ask_taxgpt(
    request: AskRequest,
    req: Request,
    res: Response
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:

         # 1️⃣ Get session ID ONLY from cookies
        session_id = req.cookies.get("session_id")

        # 2️⃣ Create new session if missing
        if not session_id:
            session_id = str(uuid.uuid4())

         # 3️⃣ Invoke history-aware chain
        response = conversational_rag_chain.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": session_id}}
        )

        parsed = parse_llm_response(response["answer"])

        source_docs = []
        seen = set()

        for i, doc in enumerate(response.get("context", []), start=1):
            meta = doc.metadata or {}

            if meta.get("pdf_url") and meta["pdf_url"] not in seen:
                seen.add(meta["pdf_url"])
                source_docs.append(
                    {
                        "index": i,
                        "title": meta.get("title", "Unknown"),
                        "file": meta.get("source"),
                        "pdf_url": meta.get("pdf_url"),
                    }
                )


         # 4️⃣ Set cookie ONLY if new session
        if not req.cookies.get("session_id"):
            res.set_cookie(
                key="session_id",
                value=session_id,
                max_age=30 * 24 * 60 * 60,
                httponly=True,
                secure= False,
                samesite="lax",
            )

        # return {
        #     "answer": response["answer"],
        #     "source_documents": source_docs
        # }

        # 4️⃣ Return clean JSON
        return {
            "Answer": parsed["main_answer"],
            "Applicable_Legal_References": parsed["legal_references"],
            "Summary": parsed["summary"],
            "Recommendation": parsed["recommendation"],
            "Source_Documents": [
                {"Title": doc["title"], "File": doc["file"], "URL": doc["pdf_url"]}
                for doc in source_docs
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/clear_history")
async def clear_history(req: Request, res: Response):
    session_id = req.cookies.get("session_id")

    if session_id and session_id in store:
        del store[session_id]
        res.delete_cookie("session_id")
        return {"status": "history cleared"}

    return {"status": "no active session"}


# uvicorn main:app --reload
