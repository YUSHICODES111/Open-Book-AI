# 🧠 Open-Book-AI — PDF QDocMind — PDF Q&A Chatbot (Production RAG Pipeline)A Chatbot (Production RAG Pipeline)

> **Industry-grade** RAG system using LangChain v0.3 + FAISS + OpenAI/Gemini  
> Full-stack: FastAPI backend + Modern dark-theme UI

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
PyPDFLoader ──► RecursiveCharacterTextSplitter (1000 tokens, 200 overlap)
    │
    ▼
OpenAI text-embedding-3-small / Gemini embedding-001
    │
    ▼
FAISS VectorStore (MMR search: k=5, fetch_k=20)
    │
    ▼
History-Aware Retriever  ◄── Chat History
    │
    ▼
ChatPromptTemplate + GPT-4o-mini / Gemini-1.5-flash
    │
    ▼
Streaming JSON Response → React UI
```

### Key Design Decisions
| Decision | Choice | Why |
|---|---|---|
| Chunking | Recursive + Semantic separators | Preserves paragraph structure |
| Search | MMR (Maximal Marginal Relevance) | Reduces redundant chunks |
| History | `create_history_aware_retriever` | Rephrases questions with context |
| Streaming | SSE via FastAPI | Real-time token streaming |
| VectorDB | FAISS (local) | Zero infra, swap to Pinecone for scale |

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)
```bash
cp backend/.env.example backend/.env
# Edit .env with your API key
docker-compose up --build
# Open http://localhost:3000
```

### Option 2 — Local
```bash
# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
uvicorn main:app --reload

# Frontend — just open frontend/index.html in browser
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload PDF → returns `session_id` |
| `POST` | `/query` | Ask a question (standard) |
| `POST` | `/query/stream` | Ask a question (streaming SSE) |
| `GET`  | `/session/{id}` | Get session metadata |
| `DELETE` | `/session/{id}` | Delete session + files |
| `GET`  | `/sessions` | List all sessions |
| `GET`  | `/health` | Health check |

---

## 🔧 Configuration

```env
# Choose provider
OPENAI_API_KEY=sk-...        # For GPT-4o-mini + text-embedding-3-small
GOOGLE_API_KEY=...           # For Gemini-1.5-flash + embedding-001

# Optional tuning
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE_MB=50

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
```

---

## 📈 Production Upgrade Path

| Feature | Current | Production |
|---|---|---|
| VectorDB | FAISS (in-memory) | Pinecone / Weaviate |
| Sessions | Python dict | Redis |
| Auth | None | JWT + OAuth2 |
| File storage | Local disk | S3 / GCS |
| Observability | Logs | LangSmith tracing |
| Scaling | Single process | Kubernetes |

---

## 🎯 What This Demonstrates (for Interviewers)

- ✅ **RAG Pipeline**: Full ingestion → retrieval → generation loop
- ✅ **Vector Embeddings**: Semantic search with FAISS + MMR
- ✅ **LangChain v0.3**: Modern modular chains (`create_retrieval_chain`)
- ✅ **Multi-turn memory**: History-aware question rephrasing
- ✅ **Streaming**: SSE token streaming with FastAPI
- ✅ **REST API**: OpenAPI-documented FastAPI endpoints
- ✅ **Production patterns**: Session management, error handling, cleanup
- ✅ **Docker**: Containerized full-stack deployment
