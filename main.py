from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# LangChain & Vector DB
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate




# Load environment variables
load_dotenv()

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(
    title="TaxTro API",
    description="AI Tax Advisor for Pakistan Tax Laws",
    version="1.0.0"
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
    model_kwargs={"device": "cpu"}  # change to "cpu" if no GPU
)

# -------------------------------
# Initialize Pinecone
# -------------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding,
    text_key="text"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# -------------------------------
# Initialize LLM
# -------------------------------


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.5
)

# -------------------------------
# Prompt Template (STRICT)
# -------------------------------
prompt = ChatPromptTemplate.from_template("""
You are **TaxGPT**, an AI Tax Advisor specialized exclusively in **Pakistan Tax Laws**
(Income Tax Ordinance, Sales Tax Act, Federal Excise Act, SROs, Rules, Notifications).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 STRICT OPERATING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. You MUST primarily use the retrieved **Context** to answer.
2. You may ONLY use your general Pakistan tax knowledge if the context is incomplete.
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
For EACH reference, list them separately as clickable items in UI:

- **Title:** (e.g., SRO 123(I)/2023)
- **Provision Type:** (SRO / Rule / Section)
- **Provision Number:** (e.g., Rule 45A, Section 80C)
- **Short Description:** (7–8 lines extracted from context)

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
""")



# -------------------------------
# Create RAG Chain
# -------------------------------
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/ask")
async def ask_taxgpt(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        response = retrieval_chain.invoke({"input": request.question})

        source_docs = []
        seen = set()

        for i, doc in enumerate(response.get("context", []), start=1):
            meta = doc.metadata or {}

            if meta.get("pdf_url") and meta["pdf_url"] not in seen:
                seen.add(meta["pdf_url"])
                source_docs.append({
                    "index": i,
                    "title": meta.get("title", "Unknown"),
                    "file": meta.get("source"),
                    "pdf_url": meta.get("pdf_url")
                })

        return {
            "answer": response["answer"],
            "source_documents": source_docs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# uvicorn main:app --reload
