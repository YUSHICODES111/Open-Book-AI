"""
PDF Q&A RAG Chatbot - Production Backend
Stack: FastAPI + LangChain + FAISS + OpenAI/Gemini
Author: Industry-grade RAG pipeline
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import time
import json
import asyncio
from typing import Optional, AsyncGenerator
import logging

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

# Embeddings — supports both OpenAI and Google
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    PROVIDER = "openai"
except ImportError:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    PROVIDER = "google"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Open-Book-AI API",
    description="AI-powered PDF QProduction-grade RAG pipeline for PDF Q&AA using RAG",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory session store (use Redis in production) ───────────────────────
sessions: dict = {}   # session_id → { vectorstore, chat_history, metadata }
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─── Models ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    session_id: str
    question: str
    stream: bool = False

class SessionResponse(BaseModel):
    session_id: str
    filename: str
    chunks: int
    pages: int
    message: str


# ─── LLM & Embeddings factory ─────────────────────────────────────────────────
def get_llm(streaming: bool = False):
    if PROVIDER == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            streaming=streaming,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

def get_embeddings():
    if PROVIDER == "openai":
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )


# ─── RAG Chain builder ────────────────────────────────────────────────────────
def build_rag_chain(vectorstore: FAISS, streaming: bool = False):
    """
    Full RAG chain with:
    - History-aware retriever (rephrases questions using chat history)
    - Contextual compression
    - Streaming support
    """
    llm = get_llm(streaming=streaming)
    retriever = vectorstore.as_retriever(
        search_type="mmr",           # Maximal Marginal Relevance — reduces redundancy
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )

    # Step 1: Contextualize question based on chat history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question which might reference "
         "context in the chat history, formulate a standalone question which can be "
         "understood without the chat history. Do NOT answer the question, just "
         "reformulate it if needed, otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Step 2: Answer using retrieved context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an expert document analyst and Q&A assistant. Your job is to \
answer questions based STRICTLY on the provided PDF context.

Guidelines:
- Be precise, concise, and factually accurate
- Always cite page numbers when possible (e.g., "According to page 3...")
- If the answer is not found in the context, say "I couldn't find this in the document"
- Format complex answers with bullet points or numbered lists when helpful
- For code/technical content, use proper formatting

Context:
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "provider": PROVIDER, "version": "2.0.0"}


@app.post("/upload", response_model=SessionResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Ingest PDF → chunk → embed → store in FAISS"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    start = time.time()
    session_id = f"sess_{int(time.time() * 1000)}"

    # Save file
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Load PDF
    logger.info(f"Loading PDF: {file.filename}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    total_pages = len(documents)

    # Smart chunking — preserves semantic boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Add metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["session_id"] = session_id

    # Embed + store
    logger.info(f"Embedding {len(chunks)} chunks...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist vectorstore
    vs_path = f"vectorstore/{session_id}"
    vectorstore.save_local(vs_path)

    # Cache session
    sessions[session_id] = {
        "vectorstore": vectorstore,
        "chat_history": [],
        "filename": file.filename,
        "chunks": len(chunks),
        "pages": total_pages,
        "file_path": file_path,
        "created_at": time.time(),
    }

    elapsed = round(time.time() - start, 2)
    logger.info(f"Ingestion complete in {elapsed}s")

    return SessionResponse(
        session_id=session_id,
        filename=file.filename,
        chunks=len(chunks),
        pages=total_pages,
        message=f"PDF processed in {elapsed}s. Ready for Q&A!"
    )


@app.post("/query")
async def query(req: QueryRequest):
    """Standard (non-streaming) RAG query"""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Please upload a PDF first.")

    chain = build_rag_chain(session["vectorstore"])
    result = chain.invoke({
        "input": req.question,
        "chat_history": session["chat_history"]
    })

    answer = result["answer"]
    sources = list({
        doc.metadata.get("page", "N/A")
        for doc in result.get("context", [])
    })

    # Update history
    session["chat_history"].extend([
        HumanMessage(content=req.question),
        AIMessage(content=answer),
    ])

    return {
        "answer": answer,
        "sources": sorted(sources),
        "session_id": req.session_id,
        "chunks_retrieved": len(result.get("context", [])),
    }


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Streaming RAG query using SSE"""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found.")

    async def event_generator() -> AsyncGenerator[str, None]:
        chain = build_rag_chain(session["vectorstore"], streaming=True)
        full_answer = ""

        async for chunk in chain.astream({
            "input": req.question,
            "chat_history": session["chat_history"]
        }):
            if "answer" in chunk:
                token = chunk["answer"]
                full_answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0)

        # Update history after stream completes
        session["chat_history"].extend([
            HumanMessage(content=req.question),
            AIMessage(content=full_answer),
        ])
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session_id,
        "filename": session["filename"],
        "pages": session["pages"],
        "chunks": session["chunks"],
        "messages": len(session["chat_history"]) // 2,
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        # cleanup
        file_path = sessions[session_id].get("file_path")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        del sessions[session_id]
    return {"message": "Session deleted"}


@app.get("/sessions")
async def list_sessions():
    return [
        {"session_id": sid, "filename": s["filename"], "pages": s["pages"]}
        for sid, s in sessions.items()
    ]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
