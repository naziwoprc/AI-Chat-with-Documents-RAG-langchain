# AI Chat with Documents (RAG)

A modular Retrieval-Augmented Generation (RAG) system built with Python, LangChain, vector search, Cross-Encoder reranking, and FastAPI.

The system allows users to ask questions about their documents and generates answers using retrieved document context rather than relying only on the language model's internal knowledge.

---

## 🚀 Features

### Core RAG Engine

- Persistent Chroma vector store
- Cosine similarity search
- Normalized embeddings
- Embedding-based retrieval
- Cross-Encoder reranking with `ms-marco-MiniLM-L-6-v2`
- Confidence-based retrieval logic
- Source attribution
- Modular document ingestion pipeline

### 📄 Document Ingestion

The system currently supports:

- PDF
- TXT
- Markdown
- DOCX

The ingestion pipeline automatically detects the file type and selects the appropriate document loader.

Document metadata is preserved throughout the ingestion and chunking process, including:

- Source file
- File type
- PDF page information when available

### ⚡ FastAPI

- REST API with `POST /ask`
- Pydantic request validation
- Structured JSON responses
- HTTP error handling
- Interactive Swagger documentation
- CLI mode for local testing

---

## 🧠 Architecture

```text
                    Document
                       │
                       ▼
              File Type Detection
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
       PDF            TXT/MD         DOCX
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                 Text Splitting
                       │
                       ▼
                  Embeddings
                       │
                       ▼
                  Chroma DB
                       │
                       ▼
                    Query
                       │
                       ▼
             Similarity Retrieval
                       │
                       ▼
             Confidence Evaluation
                       │
                       ▼
             Cross-Encoder Reranking
                       │
                       ▼
                 Relevant Context
                       │
                       ▼
                      LLM
                       │
                       ▼
              Answer + Sources
```

---

## 🛠 Tech Stack

| Technology            | Purpose                                |
| --------------------- | -------------------------------------- |
| Python                | Core application                       |
| LangChain             | RAG components and document processing |
| Chroma                | Vector database                        |
| Sentence Transformers | Embeddings and reranking               |
| Cross-Encoder         | Retrieval reranking                    |
| FastAPI               | REST API                               |
| Pydantic              | Request validation                     |
| PyPDF                 | PDF ingestion                          |
| Docx2txt              | DOCX ingestion                         |

---

## 📁 Project Structure

```text
AI-Chat-with-Documents-RAG-langchain/
│
├── ragEngine.py          # Core RAG engine
├── ingestion.py          # Document loading, metadata and chunking
├── api.py                # FastAPI application
├── main.py               # CLI interface
│
├── Data/                 # Example documents
├── chroma_db_langchain/  # Local vector database (ignored by Git)
│
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/naziwoprc/AI-Chat-with-Documents-RAG-langchain.git

cd AI-Chat-with-Documents-RAG-langchain

python -m venv .venv
```

Activate the virtual environment.

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_api_key_here
```

The LLM provider can be changed independently from the retrieval pipeline.

---

## ▶️ Running the Application

### CLI Mode

Run:

```bash
python main.py
```

Then enter questions about the loaded documents:

```text
Ask a question: What is the liver?
```

Type `exit` or `quit` to stop the application.

### FastAPI

Start the API server:

```bash
python -m uvicorn api:app --reload
```

The interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## 🔎 Retrieval Pipeline

The project uses a multi-stage retrieval process:

```text
User Question
      ↓
Embedding Retrieval
      ↓
Initial Candidate Documents
      ↓
Confidence Evaluation
      ↓
Cross-Encoder Reranking
      ↓
Top Relevant Documents
      ↓
Context Construction
      ↓
LLM Generation
```

The Cross-Encoder is used to rerank the initial vector-search results based on the relationship between the question and retrieved document chunks.

---

## 📚 Source Attribution

Retrieved documents retain their source metadata throughout the ingestion pipeline.

For example:

```text
Sources:
- Data/document.pdf (Page 3)
- Data/document.pdf (Page 7)
```

For formats without page metadata:

```text
Sources:
- Data/example.docx
```

This makes generated answers easier to trace back to the original documents.

---

## 🎯 Engineering Highlights

This project focuses on understanding and implementing the individual components of a RAG system rather than treating RAG as a single library call.

Key engineering areas include:

- Modular document ingestion
- Metadata preservation
- Vector similarity search
- Retrieval confidence handling
- Cross-Encoder reranking
- Source attribution
- Persistent vector storage
- REST API design
- CLI/API separation
- Version-controlled development
- Feature branches and pull requests
- Semantic versioning and GitHub releases

---

## 📈 Version History

### v1.4.0 — Improved Document Ingestion

- Added DOCX support
- Added TXT support
- Added Markdown support
- Added automatic file-type detection
- Added document source metadata
- Added file-type metadata
- Preserved metadata through document splitting
- Improved source display for different document formats

### v1.3.0 — FastAPI

- Added FastAPI application
- Added `POST /ask`
- Added request validation
- Added structured API responses
- Added CLI mode

### v1.2.0 — Cross-Encoder Reranking

- Added Cross-Encoder reranking
- Improved retrieval ordering
- Added reranking-based relevance handling

### v1.1.0 — Confidence-Based Retrieval

- Added embedding confidence logic
- Added margin-based retrieval decisions
- Improved handling of uncertain retrieval results

### v1.0.0 — Initial RAG

- Manual RAG pipeline
- Document embeddings
- Chroma vector database
- Similarity retrieval
- LLM-based answer generation

---

## 🗺️ Roadmap

### v1.5 — Conversational RAG

- Conversation history
- Context-aware follow-up questions
- Chat session management

### v1.6 — Advanced Retrieval

- Improved query transformation
- Hybrid retrieval
- More advanced reranking strategies

### v1.7 — Evaluation

- Retrieval evaluation
- Answer evaluation
- RAG quality metrics
- Test datasets

### v1.8 — Production API

- Improved API architecture
- Authentication
- Logging
- Configuration management
- Better error handling

### v1.9 — Frontend

- React interface
- Document upload
- Chat interface
- Source visualization

### v2.0 — Production-Style RAG Application

A complete end-to-end AI document assistant combining:

```text
React
  ↓
FastAPI
  ↓
RAG Service
  ↓
Retrieval + Reranking
  ↓
Vector Database
  ↓
LLM
```

---

## 👨‍💻 Author

**Nazanin**

Interested in AI engineering, Python, LLM applications, RAG systems, and software development.
