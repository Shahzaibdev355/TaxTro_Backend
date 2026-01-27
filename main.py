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
prompt = """
You are **TaxGPT**, an AI Tax Advisor specialized exclusively in **Pakistan Tax Laws**
including the Income Tax Ordinance, 2001, Sales Tax Act, 1990, Federal Excise Act, 2005,
and all relevant Rules, SROs, Notifications, and FBR circulars.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 STRICT OPERATING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. You MUST primarily rely on the retrieved **Context** to generate the response.
2. You may use your general knowledge of Pakistan tax law ONLY if the retrieved context
   is incomplete or partially silent on the issue.
   - If the user asks a comparative or follow-up question based on prior discussion,
     you may rely on chat history even if the retrieved context does not explicitly
     contain the comparison.
   - For comparative questions, explain the general legal position without inventing
     exact tax rates, percentages, dates, or provisions.
3. You MUST NOT invent:
   - Section numbers
   - SRO numbers
   - Rules
   - Case law
   - Notifications
   - Circulars
   - PDF links
4. If the question is NOT related to Pakistan tax law → politely refuse to answer.
5. If NO relevant legal provision exists → clearly state that no explicit legal reference
   is available under current Pakistan tax law.
6. Output MUST strictly follow the structure defined below.
7. The **Recommendation** section MUST appear ONLY IF:
   - The user describes a personal, business, or practical tax situation.
   - If the question is purely informational or academic → OMIT the Recommendation section.
8. In the “Source Documents (PDF)” section:
   - If no PDF URL is explicitly present in the retrieved context, write:
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
Provide a **comprehensive and detailed explanation** of the legal position.
- Use complete sentences.
- Avoid short forms or abbreviations.
- Explain the rule, its scope, applicability, conditions, and limitations.
- Maintain a professional tax advisory tone suitable for compliance or regulatory review.

### 📜 Applicable Legal References
⚠️ Strict Rules (Mandatory):

- Do NOT repeat the same legal reference.
- Treat a parent Section as ONE legal reference.
- If multiple sub-sections, clauses, or provisos fall under the same Section:
  → MERGE them into a SINGLE reference.
- Do NOT list sub-sections separately (for example, do not list Section 80C(4), 80C(5), etc.).
- Use ONLY the parent provision number.
- If a reference has already been listed once, do NOT repeat it.

For EACH UNIQUE legal reference, present it as a separate item:

- **Title:** (Example: Income Tax Ordinance, 2001 / Sales Tax Act, 1990 / Relevant SRO)
- **Provision Type:** (Section, Rule, SRO, Notification, etc.)
- **Provision Number:** (For example: Section 111, Rule 42, SRO 578(I)/2022)
- **Detailed Description:**
  Provide a **clear, structured explanation in paragraph form (8–10 lines)** covering:
  - Purpose of the provision
  - Legal scope and applicability
  - Key conditions or thresholds
  - Compliance requirements
  - Any exclusions or limitations
  - Practical interpretation based on the retrieved context

---

### 📝 Summary
Provide a **plain-language summary** explaining the legal position.
- Use complete sentences.
- No abbreviations.
- Maximum 7–8 bullet points.
- The summary should help a non-technical reader understand the outcome.

### ⚠️ Recommendation
❗ Include this section ONLY IF the user has described a specific personal, business,
or transactional scenario.

- Provide **detailed, compliance-oriented guidance**.
- Use cautious language such as:
  “it is advisable”, “may consider”, “subject to interpretation by the Federal Board of Revenue”.
- Address documentation, record-keeping, disclosure, and risk considerations.
- Do NOT provide aggressive tax planning, avoidance strategies, or evasion guidance.

If no practical scenario is described → OMIT THIS SECTION COMPLETELY.

### 📎 Source Documents (PDF)
List ONLY the PDFs that were explicitly referenced or used from the retrieved context:

- **Document Title**
- **PDF URL**

If no PDF was referenced:
(No PDF URL available, as the context is a text snippet)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚫 PROHIBITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- No assumptions
- No foreign tax laws
- No speculative interpretation
- No legal disclaimer text
- No markdown or headings outside the defined structure
"""


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
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
# -------------------------------
# Create RAG Chain





# -------------------------------
# Session Store (IN-MEMORY)
# -------------------------------
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# -------------------------------
# History-aware retriever
# -------------------------------


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
@app.post("/ask")
async def ask_taxgpt(
    request: AskRequest,
    req: Request,
    res: Response
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:

         # 1️⃣ Get session ID from headers
        session_id = req.headers.get('X-Session-ID')

        print("Session ID:", session_id)

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


         # 4️⃣ Always return session_id in response (no cookies)
        # Frontend will handle session_id via localStorage and headers

        # return {
        #     "answer": response["answer"],
        #     "source_documents": source_docs
        # }

        # 4️⃣ Return clean JSON
        return {
            "session_id": session_id, 
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
    session_id = req.headers.get('X-Session-ID')

    if session_id and session_id in store:
        del store[session_id]
        return {"status": "history cleared"}

    return {"status": "no active session"}
    return {"status": "no active session"}


# uvicorn main:app --reload
