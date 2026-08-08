# AI Chat with Documents (RAG)

A production-ready Retrieval-Augmented Generation (RAG) system built from scratch and extended with FastAPI and Cross-Encoder reranking.

---

## 🚀 Features

### ✅ Core RAG Engine

- Persistent Chroma vector store
- Cosine similarity (`hnsw:space`)
- Normalized embeddings
- Token-aware retrieval tuning
- Two-stage confidence gating
- Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`)
- Source citation support

### ✅ API Layer (FastAPI)

- RESTful endpoint: `POST /ask`
- Structured JSON responses
- Request validation with Pydantic
- Error handling with proper HTTP status codes
- CLI mode for local testing

---

## 📁 Project Structure

```
ragEngine.py        # RAG core logic
api.py              # FastAPI server
main.py             # CLI testing mode
Data/               # PDF documents
chroma_db_langchain/ # Vector database (ignored in git)
```

---

## 🛠 Usage

### 1️⃣ Set Environment Variable

Create `.env` file:

```
NVIDIA_API_KEY=your_api_key_here
```

---

### 2️⃣ Run CLI Mode

```bash
python main.py
```

---

### 3️⃣ Run API Server

```bash
python -m uvicorn api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

---

## 🧠 Architecture Overview

```
Question
   ↓
Embedding Retrieval (k=10)
   ↓
Embedding Confidence Check
   ↓
Cross-Encoder Reranking
   ↓
Dynamic Margin Filtering
   ↓
LLM Generation
   ↓
Answer + Sources
```

---

## 🎯 Engineering Highlights

- Deterministic rejection using confidence thresholds
- Adaptive retrieval based on score distribution
- Production-ready REST API
- Modular and versioned architecture
- Separation of retrieval, ranking, and generation

---

## 📌 Versioning

- `v1.0.0` – Manual RAG
- `v1.1.0` – Margin-based confidence
- `v1.2.0` – Cross-Encoder reranking
- `v1.3.0` – FastAPI production API
