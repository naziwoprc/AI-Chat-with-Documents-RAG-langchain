# AI Chat with Documents (RAG)

A complete end-to-end Retrieval-Augmented Generation pipeline built from scratch.

## Project Stages

1. **Manual RAG Engine** (`rag_engine.py`)
   - Persistent ChromaDB storage (`hnsw:space: cosine`)
   - Token-based chunking with overlap
   - Normalized embeddings + cosine similarity
   - NVIDIA LLM integration via Dependency Injection

2. **LangChain RAG v1** (`ragEngine.py` / `main.py`)
   - LCEL pipeline (`prompt | retriever | llm`)
   - Modular architecture with `LLMClient`

## Usage

```bash
# 1. Set API key in .env
NVIDIA_API_KEY=your_key_here

# 2. Run manual RAG
python rag_engine.py

# 3. Run LangChain RAG
python main.py
```

## Requirements

```bash
pip install -r requirements.txt
```

## Key Learnings

- Embedding, Retrieval, and Generation must be separated
- Chunking must be token-aware (not just characters)
- Normalization + cosine metric improves retrieval quality
- Dependency Injection makes LLM providers swappable
- LangChain abstracts pipeline but hides scores/debug info
